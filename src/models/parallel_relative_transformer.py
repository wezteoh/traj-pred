import torch
import torch.nn as nn
from einops import rearrange

from src.modules.motion_transformer_encoder import MotionTransformerEncoder
from src.modules.relative_transformer_block import RelativeTransformerBlock


class ParallelRelativeTransformer(nn.Module):
    def __init__(
        self,
        d_traj,
        num_relative_transformer_blocks,
        relative_transformer_block_config,
        motion_transformer_encoder_config,
        d_reg_head_mlp,
        d_cls_head_mlp,
        d_agentwise_compressor_mlp,
        num_paths,
        num_agents,
        num_output_steps,
        traj_conditioning=False,
    ):
        super().__init__()
        self.traj_conditioning = traj_conditioning
        self.past_encoder = MotionTransformerEncoder(**motion_transformer_encoder_config)
        self.decoder_blocks = nn.ModuleList(
            [
                RelativeTransformerBlock(**relative_transformer_block_config)
                for _ in range(num_relative_transformer_blocks)
            ]
        )
        self.post_decoder_act = nn.ReLU()
        self.path_embedding = nn.Embedding(num_paths, motion_transformer_encoder_config["d_model"])
        self.reg_head = nn.Sequential(
            nn.Linear(
                (
                    relative_transformer_block_config["d_model"] + d_traj
                    if traj_conditioning
                    else relative_transformer_block_config["d_model"]
                ),
                d_reg_head_mlp,
            ),
            nn.ReLU(),
            nn.Linear(d_reg_head_mlp, num_output_steps * 2),
        )
        self.agentwise_compressor_mlp = nn.Sequential(
            nn.Linear(
                (
                    relative_transformer_block_config["d_model"] + d_traj
                    if traj_conditioning
                    else relative_transformer_block_config["d_model"]
                ),
                d_agentwise_compressor_mlp,
            ),
            nn.ReLU(),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(
                num_paths * d_agentwise_compressor_mlp,
                d_cls_head_mlp,
            ),
            nn.ReLU(),
            nn.Linear(d_cls_head_mlp, num_paths),
        )
        self.num_paths = num_paths
        self.num_agents = num_agents
        self.num_output_steps = num_output_steps

    def forward(self, x_traj: torch.tensor, return_cache=False):

        x_embeddings = self.past_encoder(x_traj)  # [b, a, d]
        x_embeddings = rearrange(x_embeddings, "b a d -> b 1 a d").repeat(
            1, self.num_paths, 1, 1
        )  # [b, k, a, d]

        p_embeddings = self.path_embedding(torch.arange(self.num_paths).to(x_traj.device))  # [k, d]
        p_embeddings = rearrange(p_embeddings, "k d -> 1 k 1 d").repeat(
            x_embeddings.shape[0], 1, self.num_agents, 1
        )  # [b, k, a, d]

        x_embeddings = x_embeddings + p_embeddings

        for block in self.decoder_blocks:
            x_embeddings = block(
                x_traj[:, -1:].repeat(1, self.num_paths, 1, 1), x_embeddings
            )  # [b, k, a, d]

        x_embeddings = self.post_decoder_act(x_embeddings)

        if self.traj_conditioning:
            x_embeddings = torch.cat(
                [x_embeddings, x_traj[:, -1:].repeat(1, self.num_paths, 1, 1)], dim=-1
            )
        reg_out = self.reg_head(x_embeddings)  # [b, k, a, td]
        reg_out = rearrange(reg_out, "b k a (t d) -> b t k a d", t=self.num_output_steps)

        x_embeddings = self.agentwise_compressor_mlp(x_embeddings)  # [b, k, a, d]
        x_embeddings = rearrange(x_embeddings, "b k a d -> b a (k d)")
        cls_out = self.cls_head(x_embeddings)  # [b, a, k]
        cls_out = rearrange(cls_out, "b a k -> b k a")
        return reg_out, cls_out


if __name__ == "__main__":
    model = ParallelRelativeTransformer(
        d_traj=4,
        num_relative_transformer_blocks=1,
        relative_transformer_block_config={"d_model": 128, "d_mesh": 260, "n_head": 8, "d_ff": 256},
        motion_transformer_encoder_config={
            "pointnet_in_channels": 4,
            "pointnet_hidden_dim": 64,
            "pointnet_num_layers": 3,
            "pointnet_num_pre_layers": 1,
            "pointnet_out_channels": 64,
            "entity_embedding_dim": 64,
            "d_model": 128,
            "use_pre_norm": True,
            "num_attn_layers": 4,
            "num_attn_heads": 8,
        },
        d_agentwise_mlp=[128],
        d_reg_head_mlp=64,
        d_cls_head_mlp=64,
        d_agentwise_compressor_mlp=32,
        num_paths=20,
        num_agents=11,
        num_output_steps=25,
        traj_conditioning=True,
    )
    model.eval()
    x = torch.rand(10, 15, 11, 4)  # [b, t, a, d]
    reg_out, cls_out = model(x)  # [b, t, k, a, d]
    print(reg_out.shape)
    print(cls_out.shape)
