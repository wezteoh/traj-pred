import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from src.modules.pointnet_polyline_encoder import PointNetPolylineEncoder


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MotionTransformerEncoder(nn.Module):
    def __init__(
        self,
        pointnet_in_channels,
        pointnet_hidden_dim,
        pointnet_num_layers,
        pointnet_num_pre_layers,
        pointnet_out_channels,
        entity_embedding_dim,
        d_model,
        use_pre_norm,
        num_attn_layers,
        num_attn_heads,
    ):
        super().__init__()

        # build polyline encoders
        self.agent_polyline_encoder = PointNetPolylineEncoder(
            in_channels=pointnet_in_channels,
            hidden_dim=pointnet_hidden_dim,
            num_layers=pointnet_num_layers,
            num_pre_layers=pointnet_num_pre_layers,
            out_channels=pointnet_out_channels,
        )
        # Positional encoding
        self.team_one_query_embedding = nn.Embedding(1, entity_embedding_dim)
        self.team_two_query_embedding = nn.Embedding(1, entity_embedding_dim)
        self.ball_query_embedding = nn.Embedding(1, entity_embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(entity_embedding_dim + pointnet_out_channels, d_model),
            nn.ReLU(),
        )
        # build transformer encoder layers
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            dropout=0.1,
            nhead=num_attn_heads,
            dim_feedforward=d_model * 2,
            norm_first=use_pre_norm,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=num_attn_layers)
        self.num_out_channels = d_model

    def agent_query_embedding(self, index):
        """
        Distinguish between team one, team two and ball. High level PE
        One team is at index 0-5
        Another team is at index 5-10
        Ball is at index 10
        """
        team_one_query = self.team_one_query_embedding(index)
        team_two_query = self.team_two_query_embedding(index)
        ball_query = self.ball_query_embedding(index)
        agent_query = torch.cat(
            [team_one_query.repeat(5, 1), team_two_query.repeat(5, 1), ball_query], dim=0
        )
        return agent_query  # [A, D]

    def forward(self, past_traj):
        """
        Args: [B, T, A, D]

        """
        past_traj = rearrange(past_traj, "b t a d -> b a t d")
        obj_polylines_feature = self.agent_polyline_encoder(past_traj)  # (b, a, d)
        agent_query = rearrange(
            self.agent_query_embedding(torch.arange(1).to(past_traj.device)), "a d -> 1 a d"
        ).repeat(obj_polylines_feature.shape[0], 1, 1)
        obj_polylines_feature = torch.cat([obj_polylines_feature, agent_query], dim=-1)
        obj_polylines_feature = self.mlp(obj_polylines_feature)
        encoder_out = self.transformer_encoder(obj_polylines_feature)
        return encoder_out  # [b, a, d]


if __name__ == "__main__":
    encoder = MotionTransformerEncoder(
        pointnet_in_channels=2,
        pointnet_hidden_dim=64,
        pointnet_num_layers=3,
        pointnet_num_pre_layers=2,
        pointnet_out_channels=64,
        entity_embedding_dim=64,
        d_model=64,
        use_pre_norm=False,
        num_attn_layers=3,
        num_attn_heads=4,
    )
    encoder.eval()
    past_traj = torch.rand(64, 50, 11, 2)
    encoder_out = encoder(past_traj)
    print(encoder_out.shape)
