import torch


class ParallelRelativeTransformer(nn.Module):
    def __init__(
        self,
        num_relative_transformer_blocks,
        relative_transformer_block_config,
        motion_transformer_encoder_config,
        d_agentwise_mlp,
        d_reg_head_mlp,
        d_cls_head_mlp,
        num_scenes,
        num_agents,
    ):
        super().__init__()
        self.num_relative_transformer_blocks = num_relative_transformer_blocks
        self.relative_transformer_block_config = relative_transformer_block_config
        self.motion_transformer_encoder_config = motion_transformer_encoder_config
        self.d_agentwise_mlp = d_agentwise_mlp
        self.d_reg_head_mlp = d_reg_head_mlp
        self.d_cls_head_mlp = d_cls_head_mlp
        self.num_scenes = num_scenes
        self.num_agents = num_agents

    def forward(self, x_traj: torch.tensor, return_cache=False):
        pass
