import torch
from einops import rearrange


def linear_schedule(p_start: float, p_end: float, step: int, total_steps: int) -> float:
    # bound output to be between 0 and 1
    return max(0, min(1, p_start + (p_end - p_start) * step / total_steps))


def remix_minade_paths(samples_original_scale, gt_path_original_scale):
    """
    Remix the minade paths to the original scale
    samples_original_scale: [b, k, t, a, 2]
    gt_path_original_scale: [b, t, a, 2]
    """
    bsz = samples_original_scale.shape[0]
    samples_original_scale = rearrange(samples_original_scale, "b k t a d -> (b a) k t d")
    gt_path_original_scale = rearrange(gt_path_original_scale, "b t a d -> (b a) 1 t d")
    distances = (samples_original_scale - gt_path_original_scale).norm(p=2, dim=-1)  # [ba, k, t]
    distances_by_path = distances.mean(dim=-1)
    min_distance_path_idxs = distances_by_path.argmin(dim=-1)  # [ba]

    min_distance_path_idxs = rearrange(min_distance_path_idxs, "ba -> ba 1 1 1").repeat(
        1, 1, samples_original_scale.shape[-2], samples_original_scale.shape[-1]
    )
    min_distance_paths = samples_original_scale.gather(1, min_distance_path_idxs).squeeze(
        1
    )  # [ba, t, 2]
    min_distance_paths = rearrange(min_distance_paths, "(b a) t d -> b t a d", b=bsz)

    return min_distance_paths
