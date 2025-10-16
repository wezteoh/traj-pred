import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RelativeDistanceAttention(nn.Module):
    def __init__(
        self,
        d_model,
        d_mesh,
        n_head,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mesh = d_mesh
        self.n_head = n_head
        self.key_mesh_feedforward = nn.Linear(d_mesh, d_model)
        self.value_mesh_feedforward = nn.Linear(d_mesh, d_model)
        self.query_feedforward = nn.Linear(d_model, d_model)

    def forward(
        self,
        x_traj,
        x_embeddings,
    ):
        """
        x_traj: [b, t, a, d]
        x_embeddings: [b, t, a, d]
        """
        diff_mesh = x_traj.unsqueeze(-3) - x_traj.unsqueeze(-2)  # [b, t, a, a, d]
        q_agent_embeddings = rearrange(x_embeddings, "b t a d -> b t a 1 d").repeat(
            1, 1, 1, diff_mesh.shape[3], 1
        )  # [b, t, a, a, d]
        k_agent_embeddings = rearrange(x_embeddings, "b t a d -> b t 1 a d").repeat(
            1, 1, diff_mesh.shape[3], 1, 1
        )  # [b, t, a, a, d]
        x_mesh = torch.cat(
            [diff_mesh, q_agent_embeddings, k_agent_embeddings], dim=-1
        )  # [b, t, a, a, d]
        x_query = self.query_feedforward(x_embeddings)  # [b, t, a, d]
        x_key = self.key_mesh_feedforward(x_mesh)  # [b, t, a, a, d]
        x_value = self.value_mesh_feedforward(x_mesh)  # [b, t, a, a, d]
        x_query = rearrange(x_query, "b t a (h d) -> b t a h 1 d", h=self.n_head)
        x_key = rearrange(x_key, "b t a r (h d) -> b t a h r d", h=self.n_head)
        x_value = rearrange(x_value, "b t a r (h d) -> b t a h r d", h=self.n_head)
        x = F.scaled_dot_product_attention(x_query, x_key, x_value, None)  # [b, t, a, h, 1, d]
        x = rearrange(x, "b t a h 1 d -> b t a (h d)")
        return x


class RelativeTransformerBlock(nn.Module):
    def __init__(self, d_model, d_mesh, n_head, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_mesh = d_mesh
        self.n_head = n_head
        self.relative_distance_attention = RelativeDistanceAttention(d_model, d_mesh, n_head)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x_traj, x_embeddings):
        relative_distance_attention_out = self.relative_distance_attention(
            x_traj, self.ln(x_embeddings)
        )
        x_embeddings = x_embeddings + relative_distance_attention_out
        ffn_out = self.ffn(x_embeddings)
        x_embeddings = x_embeddings + ffn_out
        return x_embeddings


if __name__ == "__main__":
    block = RelativeTransformerBlock(64, 64 + 64 + 2, 4, 128)
    x_traj = torch.rand(64, 50, 11, 2)
    x_embeddings = torch.rand(64, 50, 11, 64)
    x = block(x_traj, x_embeddings)
    print(x.shape)
