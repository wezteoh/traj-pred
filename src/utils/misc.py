import numpy as np
import torch
import torch.nn.functional as F


def linear_schedule(p_start: float, p_end: float, step: int, total_steps: int) -> float:
    # bound output to be between 0 and 1
    return max(0, min(1, p_start + (p_end - p_start) * step / total_steps))


def nll_molaplace_soft_vector_diagonal(
    y: torch.Tensor,  # [B, D]
    mu: torch.Tensor,  # [B, K, D]
    log_pi: torch.Tensor,  # [B, K]  (unnormalized logits)
    beta: torch.Tensor,  # [B, K, D]  (b_{k,d} = softplus(beta)+eps)
    eps: float = 1e-6,
    reduction: str = "mean",  # "mean" | "sum" | "none"
):
    """
    Mixture of diagonal (per-dimension) Laplace negative log-likelihood (soft version).

    Each component k has:
      mean vector mu_k[D], scale vector b_k[D].

    Returns NLL averaged/summed over batch.
    """
    B, D = y.shape
    Bm, K, Dm = mu.shape
    assert (B, D) == (Bm, Dm), "Shapes must match"
    assert beta.shape == mu.shape, "beta must be [B,K,D]"
    assert log_pi.shape == (B, K)

    # Positive per-dim scales
    b = F.softplus(beta) + eps  # [B,K,D]

    # Log mixture weights
    log_pi_sm = log_pi - torch.logsumexp(log_pi, dim=-1, keepdim=True)  # [B,K]

    # Expand y to [B,1,D]
    y_exp = y.unsqueeze(1)

    # log p(y|k) = -Σ_d [ log(2b_{kd}) + |y_d - μ_{kd}| / b_{kd} ]
    log_pdf_k = -(torch.log(2 * b) + torch.abs(y_exp - mu) / b).sum(dim=-1)  # [B,K]

    # Component scores
    s = log_pi_sm + log_pdf_k  # [B,K]

    # NLL = -logsumexp_k s_k
    nll = -torch.logsumexp(s, dim=-1)  # [B]

    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    elif reduction == "none":
        return nll
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")


def sample_mol_laplace_diagonal(
    mu, log_pi, beta, S=1, eps=1e-3, mean_sampling=False, pi_temperature=1.0
):
    """
    Sample S draws from a Mixture of diagonal Laplace components.

    Args:
      mu:      [B, K, D]     component means
      log_pi:  [B, K]        unnormalized mixture logits
      beta:    [B, K, D]     unconstrained; scale = softplus(beta)+eps
      S:       int           number of samples per batch element
      eps:     float         small floor for scale

    Returns:
      samples: [B, S, D]
      k_idx:   [B, S]        sampled component indices (categorical)
    """
    B, K, D = mu.shape
    assert log_pi.shape == (B, K)
    assert beta.shape == (B, K, D)

    # Mixture weights
    pi = torch.softmax(log_pi / pi_temperature, dim=-1)  # [B, K]

    # Sample component indices per draw
    cat = torch.distributions.Categorical(pi)  # per-batch categorical
    k_idx = cat.sample((S,)).transpose(0, 1)  # [B, S]

    # Gather component params for chosen k
    idx3 = k_idx.unsqueeze(-1).unsqueeze(-1).expand(B, S, 1, D)  # for [B,K,D]
    mu_sel = mu.unsqueeze(1).gather(dim=2, index=idx3).squeeze(2)  # [B,S,D]
    if mean_sampling:
        return mu_sel, k_idx
    b = F.softplus(beta) + eps
    b_sel = b.unsqueeze(1).gather(dim=2, index=idx3).squeeze(2)  # [B,S,D]

    # Sample Laplace per-dimension: Laplace = loc + sign * Exp(1) * scale
    # (equivalently, torch.distributions.Laplace works too; we keep a simple explicit form)
    # Rademacher sign ~ {-1, +1}
    sign = (torch.rand_like(mu_sel) < 0.5).to(mu_sel) * 2 - 1  # [B,S,D] in {-1,+1}
    exp1 = torch.distributions.Exponential(rate=torch.ones_like(mu_sel)).sample()  # [B,S,D]

    samples = mu_sel + sign * b_sel * exp1  # [B,S,D]
    return samples, k_idx


def categorical_entropy(log_pi):
    k = log_pi.shape[-1]
    pi = torch.softmax(log_pi, dim=-1)
    return -(pi * (torch.log(pi + 1e-8))).sum(dim=-1).mean() / np.log(float(k))
