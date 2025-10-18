# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PointNetPolylineEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        super().__init__()
        layers = []
        for i in range(num_pre_layers):
            layers.append(nn.Linear(in_channels if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        self.pre_mlps = nn.Sequential(*layers)
        layers = []
        for i in range(num_layers - num_pre_layers):
            layers.append(nn.Linear(hidden_dim * 2 if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        self.mlps = nn.Sequential(*layers)

        if out_channels is not None:
            layers = []
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, out_channels))
            self.out_mlps = nn.Sequential(*layers)
        else:
            self.out_mlps = None

    def forward(self, polylines, return_cache=False):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):

        Returns:
        """
        batch_size, num_polylines, num_points_each_polylines, C = polylines.shape
        polylines = rearrange(polylines, "b a t d -> (b a) t d")
        # pre-mlp
        polylines = rearrange(polylines, "ba t d -> (ba t) d")
        polylines_feature = self.pre_mlps(polylines)
        polylines_feature = rearrange(
            polylines_feature, "(ba t) d -> ba t d", t=num_points_each_polylines
        )
        # get global feature
        pooled_feature = polylines_feature.max(dim=-2)[0]
        pooled_feature = pooled_feature.unsqueeze(1).repeat(1, num_points_each_polylines, 1)
        polylines_feature = torch.cat((polylines_feature, pooled_feature), dim=-1)  # [ba, t, d]

        # mlp
        polylines_feature = rearrange(polylines_feature, "ba t d -> (ba t) d")
        feature_buffers = self.mlps(polylines_feature)
        feature_buffers = rearrange(
            feature_buffers, "(ba t) d -> ba t d", t=num_points_each_polylines
        )

        feature_buffers = feature_buffers.max(dim=-2)[0]  # [ba, d]

        # out-mlp
        if self.out_mlps is not None:
            feature_buffers = self.out_mlps(feature_buffers)

        feature_buffers = rearrange(
            feature_buffers, "(b a) d -> b a d", b=batch_size, a=num_polylines
        )

        return feature_buffers


if __name__ == "__main__":
    encoder = PointNetPolylineEncoder(
        in_channels=2, hidden_dim=64, num_layers=3, num_pre_layers=1, out_channels=64
    )
    encoder.eval()
    polylines = torch.rand(16, 11, 20, 2)
    feature_buffers = encoder(polylines)
    print(feature_buffers.shape)
