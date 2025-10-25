from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm.models.mixer_seq_simple import _init_weights, create_block

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


class Mamba2MixerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        n_encoder_layer: int | None = None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

    def forward(self, inputs, inference_params=None, return_cache=False, **mixer_kwargs):
        """
        Args:
            inputs: [B, A, T, D]
        """
        if return_cache and inference_params is None:
            inference_params = InferenceParams(
                max_seqlen=50,
                max_batch_size=inputs.shape[0],
            )
        batch_size = inputs.shape[0]
        inputs = rearrange(inputs, "b a t d -> (b a) t d")
        hidden_states = inputs
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )
        hidden_states = rearrange(hidden_states, "(b a) t d -> b a t d", b=batch_size)

        if inference_params is not None:
            inference_params.seqlen_offset += inputs.shape[1]
        return hidden_states, inference_params

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }


class MambaMotionTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_traj,
        mamba_mixer_encoder_config,
        entity_embedding_dim,
        d_model,
        use_pre_norm,
        num_attn_layers,
        num_attn_heads,
        team_size,
        d_ffn=None,
    ):
        super().__init__()

        self.team_size = team_size  # build polyline encoders
        self.input_encoder = nn.Sequential(
            nn.Linear(
                d_traj,
                mamba_mixer_encoder_config["d_model"],
            ),
            nn.ReLU(),
            nn.Linear(
                mamba_mixer_encoder_config["d_model"],
                mamba_mixer_encoder_config["d_model"],
            ),
        )
        self.agent_history_encoder = Mamba2MixerEncoder(**mamba_mixer_encoder_config)
        # Positional encoding
        self.team_one_query_embedding = nn.Embedding(1, entity_embedding_dim)
        self.team_two_query_embedding = nn.Embedding(1, entity_embedding_dim)
        self.ball_query_embedding = nn.Embedding(1, entity_embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(entity_embedding_dim + mamba_mixer_encoder_config["d_model"], d_model),
            nn.ReLU(),
        )
        # build transformer encoder layers
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            dropout=0.1,
            nhead=num_attn_heads,
            dim_feedforward=d_model * 4 if d_ffn is None else d_ffn,
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
            [
                team_one_query.repeat(self.team_size, 1),
                team_two_query.repeat(self.team_size, 1),
                ball_query,
            ],
            dim=0,
        )
        return agent_query  # [A, D]

    def forward(self, past_traj, return_cache=False):
        """
        Args: [B, T, A, D]

        """
        inference_cache = {}
        past_traj = rearrange(past_traj, "b t a d -> b a t d")
        history_input = self.input_encoder(past_traj)
        agent_history_feature, agent_history_encoder_cache = self.agent_history_encoder(
            history_input, return_cache=return_cache
        )  # (b, a, t, d)
        agent_history_feature = rearrange(agent_history_feature, "b a t d -> (b t) a d")
        agent_query = rearrange(
            self.agent_query_embedding(torch.arange(1).to(past_traj.device)), "a d -> 1 a d"
        ).repeat(agent_history_feature.shape[0], 1, 1)
        agent_history_feature = torch.cat([agent_history_feature, agent_query], dim=-1)
        agent_history_feature = self.mlp(agent_history_feature)
        encoder_out = self.transformer_encoder(agent_history_feature)
        encoder_out = rearrange(encoder_out, "(b t) a d -> b t a d", b=past_traj.shape[0])
        if return_cache:
            inference_cache["agent_history_encoder_cache"] = agent_history_encoder_cache
        return encoder_out, inference_cache

    def generate(self, x: torch.tensor, inference_cache: dict):
        """
        x: [b, 1, num_agents, 2]
        inference_cache: dict, will be updated in place
        """
        x = rearrange(x, "b t a d -> b a t d")
        history_input = self.input_encoder(x)
        agent_history_feature, _ = self.agent_history_encoder(
            history_input, inference_params=inference_cache["agent_history_encoder_cache"]
        )
        agent_history_feature = rearrange(agent_history_feature, "b a t d -> (b t) a d")
        agent_query = rearrange(
            self.agent_query_embedding(torch.arange(1).to(x.device)), "a d -> 1 a d"
        ).repeat(agent_history_feature.shape[0], 1, 1)
        agent_history_feature = torch.cat([agent_history_feature, agent_query], dim=-1)
        agent_history_feature = self.mlp(agent_history_feature)
        encoder_out = self.transformer_encoder(agent_history_feature)
        encoder_out = rearrange(encoder_out, "(b t) a d -> b t a d", b=x.shape[0])
        return encoder_out


if __name__ == "__main__":
    encoder = MambaMotionTransformerEncoder(
        d_traj=4,
        mamba_mixer_encoder_config={
            "n_layer": 3,
            "d_model": 64,
            "d_intermediate": 0,
            "ssm_cfg": {
                "layer": "Mamba2",
                "d_state": 128,
                "d_conv": 4,
                "expand": 4,
                "headdim": 16,
                "ngroups": 1,
                "chunk_size": 128,
                "bias": False,
                "conv_bias": False,
            },
            "rms_norm": True,
            "fused_add_norm": True,
            "residual_in_fp32": True,
        },
        entity_embedding_dim=64,
        d_model=64,
        use_pre_norm=False,
        num_attn_layers=3,
        num_attn_heads=4,
        team_size=5,
    ).to("cuda")
    encoder.eval()
    past_traj = torch.rand(64, 50, 11, 4).to("cuda")
    encoder_out, inference_cache = encoder(past_traj[:, :-1], return_cache=True)
    print(encoder_out.shape)
    x = past_traj[:, -1:]
    encoder_out_gen = encoder.generate(x, inference_cache)
    print(encoder_out_gen.shape)

    encoder_out_all, _ = encoder(past_traj, return_cache=False)
    print(encoder_out_all.shape)
    assert torch.allclose(encoder_out_all[:, -1:], encoder_out_gen, atol=5e-4)
