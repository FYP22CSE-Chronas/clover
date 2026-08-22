from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from registry import Registry

Objective = Callable[..., Tensor]
LOSSES: Registry[Objective] = Registry("objective")


def _check(y_true: Tensor, samples: Tensor) -> int:
    """Validate the trailing sample axis and return the sample count."""
    if samples.shape[:-1] != y_true.shape:
        raise ValueError(
            f"samples {tuple(samples.shape)} must be y_true {tuple(y_true.shape)} "
            "plus a trailing sample axis"
        )
    n = samples.shape[-1]
    if n < 2:
        raise ValueError(f"need at least 2 Monte Carlo samples, got {n}")
    return n


def _reduce(x: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return x
    if reduction == "sum":
        return x.sum()
    if reduction == "mean":
        return x.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def pairwise_abs_mean(samples: Tensor) -> Tensor:
    """sum_ij |y_i - y_j| / (N(N-1)) via the O(N log N) sorted identity."""
    n = samples.shape[-1]
    ordered, _ = torch.sort(samples, dim=-1)
    k = torch.arange(1, n + 1, device=samples.device, dtype=samples.dtype)
    return 2.0 * (ordered * (2 * k - n - 1)).sum(dim=-1) / (n * (n - 1))


@LOSSES.register("crps")
def crps(y_true: Tensor, samples: Tensor, reduction: str = "sum") -> Tensor:
    """Unbiased U-statistic CRPS: mean_i |y_i - y| - 1/(2N(N-1)) sum_ij |y_i - y_j|."""
    _check(y_true, samples)
    term1 = (samples - y_true.unsqueeze(-1)).abs().mean(dim=-1)
    return _reduce(term1 - 0.5 * pairwise_abs_mean(samples), reduction)


@LOSSES.register("energy")
def energy_score(
    y_true: Tensor, samples: Tensor, reduction: str = "sum", beta: float = 1.0
) -> Tensor:
    """Energy score over the joint (series x horizon) block of each window.

    `y_true` is [B, H, I] and `samples` [B, H, I, N]; the two trailing non-sample axes
    are flattened into the joint dimension. With `reduction="none"` the result is [B].
    """
    _check(y_true, samples)
    b, n = y_true.shape[0], samples.shape[-1]
    y_flat = y_true.reshape(b, -1)
    s_flat = samples.reshape(b, -1, n)
    term1 = (
        torch.linalg.vector_norm(s_flat - y_flat.unsqueeze(-1), ord=2, dim=-2)
        .pow(beta)
        .mean(dim=-1)
    )
    diffs = s_flat.unsqueeze(-1) - s_flat.unsqueeze(-2)
    term2 = torch.linalg.vector_norm(diffs, ord=2, dim=-3).pow(beta).sum(dim=(-1, -2)) / (
        n * (n - 1)
    )
    return _reduce(term1 - 0.5 * term2, reduction)


def gaussian_crps(mu: Tensor, sigma: Tensor, y: Tensor) -> Tensor:
    """Closed-form Gaussian CRPS; a test reference, never part of the objective."""
    z = (y - mu) / sigma
    normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
    cdf = normal.cdf(z)
    pdf = torch.exp(normal.log_prob(z))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1.0 / math.sqrt(math.pi))
