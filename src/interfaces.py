import os
from copy import deepcopy
from mimetypes import init
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.models import get_model
from src.utils.data import cast_floats_by_trainer_precision, normalize, unnormalize
from src.utils.drawing import create_frames_from_trajectory, create_video_from_frames
from src.utils.misc import linear_schedule


class BasePredictionInterface(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config, logger=True)

    def make_model_inputs_and_targets(self, batch: torch.tensor):
        raise NotImplementedError("Subclass must implement this method")

    def forward(self, x: torch.tensor):
        raise NotImplementedError("Subclass must implement this method")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Subclass must implement this method")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Subclass must implement this method")

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Subclass must implement this method")

    def configure_optimizers(
        self,
    ):
        lr = self.hparams.optimizer.lr
        weight_decay = self.hparams.optimizer.weight_decay

        # All parameters in the model
        all_parameters = list(self.model.parameters())

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        optimizer = AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(self.hparams.get("beta1", 0.9), self.hparams.get("beta2", 0.95)),
        )

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group({"params": params, **hp})

        # Print optimizer info
        keys = sorted(set([k for hp in hps for k in hp.keys()]))
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            print(
                " | ".join(
                    [
                        f"Optimizer group {i}",
                        f"{len(g['params'])} tensors",
                    ]
                    + [f"{k} {v}" for k, v in group_hps.items()]
                )
            )
        # Create a lr scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
        if self.hparams.optimizer.lr_schedule:
            total_steps = self.hparams.optimizer.lr_schedule.total_steps or getattr(
                self.trainer, "estimated_stepping_batches", None
            )
            if total_steps is None:
                raise ValueError(
                    "total_steps not set. Pass total_steps=... to the module "
                    "or let Lightning set trainer.estimated_stepping_batches by calling trainer.fit first."
                )
            max_lrs = [g.get("lr", lr) for g in optimizer.param_groups]
            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=self.hparams.optimizer.lr_schedule.pct_start,
                anneal_strategy="cos",
                cycle_momentum=False,
                div_factor=self.hparams.optimizer.lr_schedule.div_factor,
                final_div_factor=self.hparams.optimizer.lr_schedule.final_div_factor,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # OneCycleLR updates every step
                    "frequency": 1,
                },
            }
        else:
            return optimizer


class AutoregressiveMultiplePathPredictionInterface(BasePredictionInterface):
    def __init__(self, config):
        super().__init__(config)

        self.register_buffer("data_mean", torch.tensor(self.hparams.interface.data_mean))
        self.register_buffer("data_std", torch.tensor(self.hparams.interface.data_std))
        self.diff_as_target = self.hparams.interface.diff_as_target
        if self.hparams.interface.diff_as_target:
            self.register_buffer("diff_mean", torch.tensor(self.hparams.interface.diff_mean))
            self.register_buffer("diff_std", torch.tensor(self.hparams.interface.diff_std))
        self.model = get_model(
            name=self.hparams.model.name,
            model_args=self.hparams.model.args,
            device="cuda" if config.trainer.accelerator == "gpu" else "cpu",
        )

    def make_model_inputs_and_targets(self, batch: torch.tensor):
        if self.hparams.interface.diff_in_input or self.hparams.interface.diff_as_target:
            diff = torch.diff(batch, dim=1)
            diff_n = normalize(diff, self.diff_mean, self.diff_std)
            diff_n = torch.cat([torch.zeros_like(diff_n[:, :1]), diff_n], dim=1)

        batch_n = normalize(
            batch,
            self.data_mean,
            self.data_std,
        )
        if self.hparams.interface.diff_in_input:
            input_n = torch.cat([batch_n, diff_n], dim=-1)
        else:
            input_n = batch_n
        x = input_n[:, :-1]
        x = cast_floats_by_trainer_precision(x, precision=self.trainer.precision)

        if self.diff_as_target:
            y = diff_n[:, 1:]
            y = cast_floats_by_trainer_precision(y, precision=self.trainer.precision)
        else:
            y = batch_n[:, 1:]
            y = cast_floats_by_trainer_precision(y, precision=self.trainer.precision)

        return x, y, batch

    def forward(self, x: torch.tensor):
        pred, scene_logits, _ = self.model(x)
        return pred, scene_logits

    def training_step(self, batch, batch_idx):
        x, y, _ = self.make_model_inputs_and_targets(batch)
        num_agents = y.shape[2]
        pred, scene_logits = self.forward(x)
        error = (pred - y.unsqueeze(2)).norm(dim=-1)  # [b,t,k,a]
        error_by_scene = error.sum(dim=-1)  # [b,t,k]

        masking_ratio = linear_schedule(
            p_start=self.hparams.interface.masking_ratio_start,
            p_end=self.hparams.interface.masking_ratio_end,
            step=self.trainer.global_step,
            total_steps=self.trainer.estimated_stepping_batches
            * self.hparams.interface.masking_end_at_training_pct,
        )

        if masking_ratio > 0:
            # random masking n components by batch per time step, make it goes to infinity
            mask = (
                torch.rand(x.shape[0], x.shape[1], self.hparams.model.args.num_scenes)
                < masking_ratio
            )
            # ensure each (b,t) has at least one component not masked
            all_masked = mask.all(dim=-1)
            rand_idx = torch.randint(
                0,
                self.hparams.model.args.num_scenes,
                size=(x.shape[0], x.shape[1]),
                device=error_by_scene.device,
            )
            mask[all_masked, rand_idx[all_masked]] = False
            mask = mask.to(error_by_scene.device)
            error_by_scene = error_by_scene.masked_fill(mask, float("inf"))
            scene_logits = scene_logits.masked_fill(mask, float("-inf"))
        selected_components = error_by_scene.argmin(dim=-1)  # [b,t]
        reg_loss_components = error_by_scene.gather(2, selected_components.unsqueeze(-1))
        reg_loss = reg_loss_components.mean() / num_agents
        scene_loss = F.cross_entropy(
            scene_logits.reshape(-1, scene_logits.shape[-1]), selected_components.reshape(-1)
        )
        loss = (
            self.hparams.interface.loss_weights.reg * reg_loss
            + self.hparams.interface.loss_weights.scene * scene_loss
        )
        record_step = {
            "trainer_loss": loss.item(),
            "trainer_reg_loss": reg_loss.item(),
            "trainer_scene_loss": scene_loss.item(),
            "trainer_masking_ratio": masking_ratio,
        }

        self.log_dict(
            record_step,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            # sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, gt_path_original_scale = self.make_model_inputs_and_targets(batch)
        num_agents = y.shape[2]
        pred, scene_logits = self.forward(x)
        error = (pred - y.unsqueeze(2)).norm(dim=-1)  # [b,t,k,a]
        error_by_scene = error.sum(dim=-1)  # [b,t,k]
        selected_components = error_by_scene.argmin(dim=-1)  # [b,t]

        reg_loss_components = error_by_scene.gather(2, selected_components.unsqueeze(-1))
        reg_loss = reg_loss_components.mean() / num_agents
        scene_loss = F.cross_entropy(
            scene_logits.reshape(-1, scene_logits.shape[-1]), selected_components.reshape(-1)
        )
        loss = (
            self.hparams.interface.loss_weights.reg * reg_loss
            + self.hparams.interface.loss_weights.scene * scene_loss
        )
        record_step = {
            "validation_loss": loss.item(),
            "validation_reg_loss": reg_loss.item(),
            "validation_scene_loss": scene_loss.item(),
        }

        samples_original_scale = self.sample(
            x[:, : self.hparams.interface.validation_prefix_length],
            max_length=self.hparams.interface.validation_max_length,
            num_paths=self.hparams.interface.validation_num_paths,
        )  # [b, num_paths, t, num_agents, 2]
        metric_dict = self.compute_jade_jfde(
            samples_original_scale,
            gt_path_original_scale[
                :,
                self.hparams.interface.validation_prefix_length : self.hparams.interface.validation_prefix_length
                + self.hparams.interface.validation_max_length,
            ],
        )

        record_step.update(metric_dict)

        self.log_dict(
            record_step,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )

        if (
            self.trainer.global_step > 0
            and (self.trainer.current_epoch + 1) % self.hparams.interface.upload_every_n_epochs == 0
            and batch_idx == 0
            and self.hparams.interface.num_id_to_upload > 0
        ):
            print("sampling at validation step")
            sample_prefixes_original_scale = unnormalize(
                x[
                    : self.hparams.interface.num_id_to_upload,
                    self.hparams.interface.validation_prefix_length
                    - 5 : self.hparams.interface.validation_prefix_length,
                    :,
                    :2,  # only keep coordinates
                ],
                self.data_mean,
                self.data_std,
            )  # [b, t, num_agents, 2]
            samples_prefixes_original_scale = sample_prefixes_original_scale.unsqueeze(1).repeat(
                1, self.hparams.interface.num_paths_to_upload, 1, 1, 1
            )
            samples_original_scale = samples_original_scale[
                : self.hparams.interface.num_id_to_upload,
                : self.hparams.interface.num_paths_to_upload,
            ]
            samples_to_upload = (
                torch.cat([samples_prefixes_original_scale, samples_original_scale], dim=2)
                .cpu()
                .numpy()
            )  # [b, num_paths, t, num_agents, 2]

            video_dir = f"{os.path.expanduser(self.hparams.train.results_dir)}/samples"
            Path(video_dir).mkdir(parents=True, exist_ok=True)
            videos = []
            for i in range(samples_to_upload.shape[0]):
                for j in range(samples_to_upload.shape[1]):
                    frames = create_frames_from_trajectory(
                        samples_to_upload[i, j], game=self.hparams.interface.game
                    )
                    video_path = f"{video_dir}/sample_{i}_{j}.mp4"
                    create_video_from_frames(frames, video_path, fps=5)
                    videos.append(wandb.Video(video_path, format="mp4"))
            wandb.log({"sample": videos}, commit=False)

        return loss

    def sample(self, x: torch.tensor, max_length: int, num_paths: int, temperature: float = 1.0):
        """
        x: [b, t, num_agents, 2]
        """
        assert max_length > 1, "max_length must be greater than 1"
        samples = []
        init_reg_out, init_cls_out, init_inference_cache = self.model(x, return_cache=True)
        init_cls_out = init_cls_out.detach()[:, -1]  # (b, k)
        init_cls_dist = F.softmax(init_cls_out / temperature, dim=-1)
        init_reg_out = init_reg_out.detach()[:, -1:]  # [b, 1, k, a, 2]

        samples = []
        init_prev_output = unnormalize(x[:, -1:, :, :2], self.data_mean, self.data_std)
        for _ in range(num_paths):
            selected_scene_idxs = torch.multinomial(init_cls_dist, num_samples=1)  # [b, 1]
            selected_scene_idxs = rearrange(selected_scene_idxs, "b 1 -> b 1 1 1 1")
            selected_scene_idxs = selected_scene_idxs.repeat(
                1, 1, 1, init_reg_out.shape[-2], init_reg_out.shape[-1]
            )
            x = init_reg_out.gather(2, selected_scene_idxs).squeeze(2)  # [b, t, k, a, 2]
            if self.hparams.interface.diff_as_target:
                output = unnormalize(x, self.diff_mean, self.diff_std) + init_prev_output
            else:
                output = unnormalize(x, self.data_mean, self.data_std)

            if self.hparams.interface.diff_in_input:
                input = torch.cat(
                    [
                        normalize(output, self.data_mean, self.data_std),
                        normalize(output - init_prev_output, self.diff_mean, self.diff_std),
                    ],
                    dim=-1,
                )
            else:
                input = normalize(output, self.data_mean, self.data_std)

            inference_cache = deepcopy(init_inference_cache)
            preds = [output]
            prev_output = output

            for t in range(max_length - 1):
                reg_out, cls_out = self.model.generate(input, inference_cache)
                cls_out = cls_out[:, -1].detach()
                cls_out_dist = F.softmax(cls_out / temperature, dim=-1)
                selected_scene_idxs = torch.multinomial(cls_out_dist, num_samples=1)  # [b, 1]
                selected_scene_idxs = rearrange(selected_scene_idxs, "b 1 -> b 1 1 1 1")
                selected_scene_idxs = selected_scene_idxs.repeat(
                    1, 1, 1, reg_out.shape[-2], reg_out.shape[-1]
                )
                x = reg_out.gather(2, selected_scene_idxs).squeeze(2).detach()  # [b, t, k, a, 2]

                if self.hparams.interface.diff_as_target:
                    output = unnormalize(x, self.diff_mean, self.diff_std) + prev_output
                else:
                    output = unnormalize(x, self.data_mean, self.data_std)

                if self.hparams.interface.diff_in_input:
                    input = torch.cat(
                        [
                            normalize(output, self.data_mean, self.data_std),
                            normalize(output - prev_output, self.diff_mean, self.diff_std),
                        ],
                        dim=-1,
                    )
                else:
                    input = normalize(output, self.data_mean, self.data_std)
                preds.append(output)
                prev_output = output
            samples.append(torch.cat(preds, dim=1))  # [b, t, a, 2]
        samples = torch.stack(samples, dim=1)  # [b, num_paths, t, a, 2]
        return samples

    def compute_jade_jfde(self, samples, y):
        """
        samples: [b, num_paths, t, a, 2]
        y: [b, t, num_agents, 2]
        """
        distances = (samples - y.unsqueeze(1)).norm(p=2, dim=-1)  # [b, num_paths, t, num_agents]
        jade_path_agentwise = distances.mean(dim=-2)  # [b, num_paths, num_agents]
        jade_ball = jade_path_agentwise[:, :, -1:].mean(dim=-1)  # [b, num_paths]
        jade_team1 = jade_path_agentwise[:, :, :5].mean(dim=-1)  # [b, num_paths]
        jade_team2 = jade_path_agentwise[:, :, 5:].mean(dim=-1)  # [b, num_paths]
        jade_all = jade_path_agentwise.mean(dim=-1)  # [b, num_paths]

        jade_mean = jade_all.mean()
        jade_min = jade_all.min(dim=-1).values.mean()
        jade_ball_mean = jade_ball.mean()
        jade_ball_min = jade_ball.min(dim=-1).values.mean()
        jade_team1_mean = jade_team1.mean()
        jade_team1_min = jade_team1.min(dim=-1).values.mean()
        jade_team2_mean = jade_team2.mean()
        jade_team2_min = jade_team2.min(dim=-1).values.mean()

        jfde_path_agentwise = distances[:, :, -1]  # [b, num_paths, num_agents]
        jfde_ball = jfde_path_agentwise[:, :, -1:].mean(dim=-1)  # [b, num_paths]
        jfde_team1 = jfde_path_agentwise[:, :, :5].mean(dim=-1)  # [b, num_paths]
        jfde_team2 = jfde_path_agentwise[:, :, 5:].mean(dim=-1)  # [b, num_paths]
        jfde_all = jfde_path_agentwise.mean(dim=-1)  # [b, num_paths]

        jfde_mean = jfde_all.mean()
        jfde_min = jfde_all.min(dim=-1).values.mean()
        jfde_ball_mean = jfde_ball.mean()
        jfde_ball_min = jfde_ball.min(dim=-1).values.mean()
        jfde_team1_mean = jfde_team1.mean()
        jfde_team1_min = jfde_team1.min(dim=-1).values.mean()
        jfde_team2_mean = jfde_team2.mean()
        jfde_team2_min = jfde_team2.min(dim=-1).values.mean()
        return {
            "jade_mean": jade_mean,
            "jade_min": jade_min,
            "jade_ball_mean": jade_ball_mean,
            "jade_ball_min": jade_ball_min,
            "jade_team1_mean": jade_team1_mean,
            "jade_team1_min": jade_team1_min,
            "jade_team2_mean": jade_team2_mean,
            "jade_team2_min": jade_team2_min,
            "jfde_mean": jfde_mean,
            "jfde_min": jfde_min,
            "jfde_ball_mean": jfde_ball_mean,
            "jfde_ball_min": jfde_ball_min,
            "jfde_team1_mean": jfde_team1_mean,
            "jfde_team1_min": jfde_team1_min,
            "jfde_team2_mean": jfde_team2_mean,
            "jfde_team2_min": jfde_team2_min,
        }

    def compute_ade_fde(self, samples, y):
        """
        samples: [b, num_paths, t, a, 2]
        y: [b, t, num_agents, 2]
        """
        distances = (samples - y.unsqueeze(1)).norm(p=2, dim=-1)  # [b, num_paths, t, num_agents]
        ade_path_agentwise = distances.mean(dim=-2)  # [b, num_paths, num_agents]
        ade_agent_pathwise = rearrange(ade_path_agentwise, "b p a -> b a p")
        ade_min = ade_agent_pathwise.min(dim=-1).values.mean()

        fde_path_agentwise = distances[:, :, -1]  # [b, num_paths, num_agents]
        fde_agent_pathwise = rearrange(fde_path_agentwise, "b p a -> b a p")
        fde_min = fde_agent_pathwise.min(dim=-1).values.mean()

        return {
            "ade_min": ade_min,
            "fde_min": fde_min,
        }

    def test_step(self, batch, batch_idx):
        x, y, gt_path_original_scale = self.make_model_inputs_and_targets(batch)

        record_step = {}

        samples_original_scale = self.sample(
            x[:, : self.hparams.test.prefix_length],
            max_length=self.hparams.test.max_length,
            num_paths=self.hparams.test.num_paths,
            temperature=self.hparams.test.temperature,
        )  # [b, num_paths, t, num_agents, 2]
        metric_dict = self.compute_jade_jfde(
            samples_original_scale,
            gt_path_original_scale[
                :,
                self.hparams.test.prefix_length : self.hparams.test.prefix_length
                + self.hparams.test.max_length,
            ],
        )
        record_step.update(metric_dict)

        metric_dict = self.compute_ade_fde(
            samples_original_scale,
            gt_path_original_scale[
                :,
                self.hparams.test.prefix_length : self.hparams.test.prefix_length
                + self.hparams.test.max_length,
            ],
        )
        record_step.update(metric_dict)

        self.log_dict(
            record_step,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )

        if batch_idx == 0 and self.hparams.test.num_id_to_upload > 0:
            sample_prefixes_original_scale = unnormalize(
                x[
                    : self.hparams.test.num_id_to_upload,
                    self.hparams.test.prefix_length - 5 : self.hparams.test.prefix_length,
                    :,
                    :2,
                ],
                self.data_mean,
                self.data_std,
            )  # [b, t, num_agents, 2]
            samples_prefixes_original_scale = sample_prefixes_original_scale.unsqueeze(1).repeat(
                1, self.hparams.test.num_paths_to_upload, 1, 1, 1
            )
            samples_original_scale = samples_original_scale[
                : self.hparams.test.num_id_to_upload, : self.hparams.test.num_paths_to_upload
            ]
            samples_to_upload = (
                torch.cat([samples_prefixes_original_scale, samples_original_scale], dim=2)
                .cpu()
                .numpy()
            )  # [b, num_paths_to_upload, t, num_agents, 2]
            video_dir = os.path.expanduser(self.hparams.test.video_dir)
            Path(video_dir).mkdir(parents=True, exist_ok=True)

            for i in range(samples_to_upload.shape[0]):
                for j in range(samples_to_upload.shape[1]):
                    frames = create_frames_from_trajectory(
                        samples_to_upload[i, j], game=self.hparams.interface.game
                    )
                    video_path = f"{video_dir}/sample_{i}_{j}.mp4"
                    create_video_from_frames(frames, video_path, fps=5)
                    print(f"Saved video to {video_path}")

        return None


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load(
        "/Users/wzteoh/projects/traj-pred/configs/nba50/relativetransformer.yaml"
    )
    interface = AutoregressiveMultiplePathPredictionInterface(config)
    x = torch.rand(10, 20, 11, 2)
    samples = interface.sample(x, max_length=5, num_paths=3)
    print(samples.shape)
