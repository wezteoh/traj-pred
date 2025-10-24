from sqlite3.dbapi2 import converters

import torch
import torch.nn as nn
from einops import rearrange

from src.modules.motion_transformer_encoder import MotionTransformerEncoder
from src.modules.relative_transformer_block import RelativeTransformerBlock


class RelativeTransformer(nn.Module):
    def __init__(
        self,
        d_traj,
        num_relative_transformer_blocks,
        relative_transformer_block_config,
        motion_transformer_encoder_config,
        d_agentwise_mlp,
        d_shared_head_mlp,
        num_scenes,
        num_agents,
        dropout=0.1,
        traj_conditioning=False,
    ):
        super().__init__()
        self.traj_conditioning = traj_conditioning
        self.relative_transformer_block_config = relative_transformer_block_config
        self.motion_transformer_encoder_config = motion_transformer_encoder_config
        self.past_encoder = MotionTransformerEncoder(**motion_transformer_encoder_config)
        self.decoder_blocks = nn.ModuleList(
            [
                RelativeTransformerBlock(**relative_transformer_block_config)
                for _ in range(num_relative_transformer_blocks)
            ]
        )
        self.post_norm = nn.LayerNorm(
            relative_transformer_block_config["d_model"]
            if self.decoder_blocks
            else motion_transformer_encoder_config["d_model"]
        )
        # use num agentwise_mlp layers to set
        agentwise_mlp_layers = [
            nn.Linear(
                (
                    relative_transformer_block_config["d_model"]
                    if not traj_conditioning
                    else relative_transformer_block_config["d_model"] + d_traj
                ),
                d_agentwise_mlp[0],
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for i in range(1, len(d_agentwise_mlp)):
            agentwise_mlp_layers.extend(
                [
                    nn.Linear(d_agentwise_mlp[i - 1], d_agentwise_mlp[i]),
                    nn.ReLU(),
                ]
            )
        self.agentwise_mlp = nn.Sequential(*agentwise_mlp_layers)
        self.shared_head = nn.Sequential(
            nn.Linear(
                num_agents * d_agentwise_mlp[-1],
                d_shared_head_mlp,
            ),
            nn.ReLU(),
            nn.Linear(
                d_shared_head_mlp,
                d_shared_head_mlp,
            ),
            nn.ReLU(),
            nn.Linear(
                d_shared_head_mlp,
                num_scenes + (num_scenes * num_agents * 3) + (num_scenes * num_agents * 2),
            ),
        )
        self.num_scenes = num_scenes
        self.num_agents = num_agents

    def forward(self, x_traj: torch.tensor, return_cache=False):
        inference_cache = {}
        x_embeddings, past_encoder_cache = self.past_encoder(x_traj, return_cache=return_cache)
        if return_cache:
            inference_cache["past_encoder_cache"] = past_encoder_cache

        if self.decoder_blocks:
            for block in self.decoder_blocks:
                x_embeddings = block(x_traj, x_embeddings)  # [b, t, a, d]

        x_embeddings = self.post_norm(x_embeddings)

        if self.traj_conditioning:
            x_embeddings = torch.cat([x_embeddings, x_traj], dim=-1)

        x_embeddings = self.agentwise_mlp(x_embeddings)
        x_embeddings = rearrange(x_embeddings, "b t a d -> b t (a d)")

        out = self.shared_head(x_embeddings)  # [b, t, d]

        cls_out = out[:, :, : self.num_scenes]  # [b, t, num_scenes]
        cov_out = out[
            :, :, self.num_scenes : self.num_scenes + (self.num_scenes * self.num_agents * 3)
        ]  # [b, t, num_scenes * num_agents * 2]
        cov_out = rearrange(
            cov_out, "b t (k a d) -> b t k a d", k=self.num_scenes, a=self.num_agents
        )
        reg_out = out[:, :, self.num_scenes + (self.num_scenes * self.num_agents * 3) :]
        reg_out = rearrange(
            reg_out, "b t (k a d) -> b t k a d", k=self.num_scenes, a=self.num_agents
        )
        return reg_out, cls_out, cov_out, inference_cache

    def generate(self, x: torch.tensor, inference_cache: dict):
        """
        x: [b, 1, num_agents, d]
        inference_cache: dict, will be updated in place
        """
        x_embeddings = self.past_encoder.generate(
            x, inference_cache["past_encoder_cache"]
        )  # x_embeddings: [b, 1, a, d]

        if self.decoder_blocks:
            for block in self.decoder_blocks:
                x_embeddings = block(x, x_embeddings)  # [b, 1, a, d]

        x_embeddings = self.post_norm(x_embeddings)

        if self.traj_conditioning:
            x_embeddings = torch.cat([x_embeddings, x], dim=-1)

        x_embeddings = self.agentwise_mlp(x_embeddings)
        x_embeddings = rearrange(x_embeddings, "b t a d -> b t (a d)")
        out = self.shared_head(x_embeddings)  # [b, t, d]
        cls_out = out[:, :, : self.num_scenes]  # [b, t, num_scenes]
        cov_out = out[
            :, :, self.num_scenes : self.num_scenes + (self.num_scenes * self.num_agents * 3)
        ]  # [b, t, num_scenes * num_agents * 3]
        cov_out = rearrange(
            cov_out, "b t (k a d) -> b t k a d", k=self.num_scenes, a=self.num_agents
        )
        reg_out = out[:, :, self.num_scenes + (self.num_scenes * self.num_agents * 3) :]
        reg_out = rearrange(
            reg_out, "b t (k a d) -> b t k a d", k=self.num_scenes, a=self.num_agents
        )
        return reg_out, cls_out, cov_out


if __name__ == "__main__":
    model = RelativeTransformer(
        num_relative_transformer_blocks=3,
        relative_transformer_block_config={"d_model": 64, "d_mesh": 132, "n_head": 4, "d_ff": 128},
        motion_transformer_encoder_config={
            "pointnet_in_channels": 4,
            "pointnet_hidden_dim": 64,
            "pointnet_num_layers": 3,
            "pointnet_num_pre_layers": 2,
            "pointnet_out_channels": 64,
            "entity_embedding_dim": 64,
            "d_model": 64,
            "use_pre_norm": False,
            "num_attn_layers": 3,
            "num_attn_heads": 4,
        },
        d_agentwise_mlp=[64],
        d_shared_head_mlp=64,
        num_scenes=10,
        num_agents=11,
        d_traj=4,
    )
    model.eval()
    x = torch.rand(10, 20, 11, 4)
    # with torch.no_grad():
    reg_out, cls_out, shrink_out, inference_cache = model(x[:, :-1], return_cache=True)
    print(reg_out.shape)
    print(cls_out.shape)
    print(shrink_out.shape)

    reg_out_gen, cls_out_gen, shrink_out_gen = model.generate(x[:, -1:], inference_cache)
    print(reg_out_gen.shape)
    print(cls_out_gen.shape)
    print(shrink_out_gen.shape)

    reg_out_all, cls_out_all, shrink_out_all, _ = model(x, return_cache=False)
    print(reg_out_all.shape)
    print(cls_out_all.shape)
    print(shrink_out_all.shape)

    assert torch.allclose(reg_out_all[:, -1:], reg_out_gen, atol=1e-6)
    assert torch.allclose(cls_out_all[:, -1:], cls_out_gen, atol=1e-6)
    assert torch.allclose(shrink_out_all[:, -1:], shrink_out_gen, atol=1e-6)
