import torch
import torch.nn as nn
from einops import rearrange

from src.modules.relative_transformer_block import RelativeTransformerBlock
from src.modules.sequential_motion_transformer_encoder import SequentialMotionTransformerEncoder


class RelativeTransformer(nn.Module):
    def __init__(
        self,
        num_relative_transformer_blocks,
        relative_transformer_block_config,
        sequential_motion_transformer_encoder_config,
        d_agentwise_mlp,
        d_reg_head_mlp,
        d_cls_head_mlp,
        num_scenes,
        num_agents,
    ):
        super().__init__()
        self.past_encoder = SequentialMotionTransformerEncoder(
            **sequential_motion_transformer_encoder_config
        )
        self.decoder_blocks = nn.ModuleList(
            [
                RelativeTransformerBlock(**relative_transformer_block_config)
                for _ in range(num_relative_transformer_blocks)
            ]
        )
        self.agentwise_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(relative_transformer_block_config["d_model"], d_agentwise_mlp),
            nn.ReLU(),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(
                num_agents * d_agentwise_mlp,
                d_reg_head_mlp,
            ),
            nn.ReLU(),
            nn.Linear(d_reg_head_mlp, num_scenes * num_agents * 2),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(
                num_agents * d_agentwise_mlp,
                d_cls_head_mlp,
            ),
            nn.ReLU(),
            nn.Linear(d_cls_head_mlp, num_scenes),
        )
        self.num_scenes = num_scenes
        self.num_agents = num_agents

    def forward(self, x_traj: torch.tensor, return_cache=False):
        inference_cache = {}
        x_embeddings, past_encoder_cache = self.past_encoder(x_traj, return_cache=return_cache)
        if return_cache:
            inference_cache["past_encoder_cache"] = past_encoder_cache

        for block in self.decoder_blocks:
            x_embeddings = block(x_traj, x_embeddings)  # [b, t, a, d]

        x_embeddings = self.agentwise_mlp(x_embeddings)
        x_embeddings = rearrange(x_embeddings, "b t a d -> b t (a d)")

        reg_out = self.reg_head(x_embeddings)  # [b, t, num_paths * num_agents * 2]

        reg_out = rearrange(
            reg_out, "b t (k a d) -> b t k a d", k=self.num_scenes, a=self.num_agents
        )
        cls_out = self.cls_head(x_embeddings)  # [b, t, num_paths]
        return reg_out, cls_out, inference_cache

    def generate(self, x: torch.tensor, inference_cache: dict):
        """
        x: [b, 1, num_agents, d]
        inference_cache: dict, will be updated in place
        """
        x_embeddings = self.past_encoder.generate(
            x, inference_cache["past_encoder_cache"]
        )  # x_embeddings: [b, 1, a, d]

        for block in self.decoder_blocks:
            x_embeddings = block(x, x_embeddings)  # [b, 1, a, d]

        x_embeddings = self.agentwise_mlp(x_embeddings)
        x_embeddings = rearrange(x_embeddings, "b t a d -> b t (a d)")
        reg_out = self.reg_head(x_embeddings)  # [b, t, num_paths * num_agents * 2]
        reg_out = rearrange(
            reg_out, "b t (k a d) -> b t k a d", k=self.num_scenes, a=self.num_agents
        )
        cls_out = self.cls_head(x_embeddings)  # [b, t, num_paths]
        return reg_out, cls_out


if __name__ == "__main__":
    model = RelativeTransformer(
        num_relative_transformer_blocks=3,
        relative_transformer_block_config={"d_model": 64, "d_mesh": 130, "n_head": 4, "d_ff": 128},
        motion_transformer_encoder_config={
            "pointnet_in_channels": 2,
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
        d_agentwise_mlp=64,
        d_reg_head_mlp=64,
        d_cls_head_mlp=64,
        num_scenes=10,
        num_agents=11,
    )
    model.eval()
    x = torch.rand(10, 20, 11, 2)
    # with torch.no_grad():
    reg_out, cls_out, inference_cache = model(x[:, :-1], return_cache=True)
    print(reg_out.shape)
    print(cls_out.shape)

    reg_out_gen, cls_out_gen = model.generate(x[:, -1:], inference_cache)
    print(reg_out_gen.shape)
    print(cls_out_gen.shape)

    reg_out_all, cls_out_all, _ = model(x, return_cache=False)
    print(reg_out_all.shape)
    print(cls_out_all.shape)

    assert torch.allclose(reg_out_all[:, -1:], reg_out_gen, atol=1e-6)
    assert torch.allclose(cls_out_all[:, -1:], cls_out_gen, atol=1e-6)
