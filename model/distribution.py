from __future__ import annotations

import torch
from torch import Tensor

from contracts import FactorParams, ForecastSamples


def sample_factor_model(
    params: FactorParams, num_samples: int, generator: torch.Generator | None = None
) -> Tensor:
    """Draw `mu + sigma * z + F @ eps`, differentiable through the noise.

    Returns bottom-level samples shaped [..., Nb, num_samples].
    """
    mu, sigma, loadings = params.mu, params.sigma, params.F
    if mu.shape != sigma.shape:
        raise ValueError(f"mu {tuple(mu.shape)} and sigma {tuple(sigma.shape)} differ")
    if loadings.shape[:-1] != mu.shape:
        raise ValueError(
            f"F leading dims {tuple(loadings.shape[:-1])} must match mu {tuple(mu.shape)}"
        )
    lead, n_bottom = mu.shape[:-1], mu.shape[-1]
    n_factors = loadings.shape[-1]
    kw = {"device": mu.device, "dtype": mu.dtype, "generator": generator}
    z = torch.randn(*lead, n_bottom, num_samples, **kw)
    eps = torch.randn(*lead, n_factors, num_samples, **kw)
    factor_term = torch.einsum("...bk,...kn->...bn", loadings, eps)
    return mu.unsqueeze(-1) + sigma.unsqueeze(-1) * z + factor_term


def coherent_aggregate(S: Tensor, bottom: Tensor, clip: bool = True) -> Tensor:
    """Clip to non-negative first, then aggregate through S. The order matters."""
    if S.shape[-1] != bottom.shape[-2]:
        raise ValueError(
            f"S has {S.shape[-1]} columns but samples have {bottom.shape[-2]} series"
        )
    clipped = torch.relu(bottom) if clip else bottom
    return torch.einsum("ij,...jn->...in", S, clipped)


def sample_coherent(
    params: FactorParams,
    S: Tensor,
    num_samples: int,
    generator: torch.Generator | None = None,
    clip: bool = True,
) -> ForecastSamples:
    """Sample the factor model and aggregate through S in one call."""
    bottom = sample_factor_model(params, num_samples, generator=generator)
    return ForecastSamples(
        bottom=bottom, hierarchy=coherent_aggregate(S, bottom, clip=clip)
    )


def aggregate_targets(S: Tensor, target_bottom: Tensor) -> Tensor:
    """Lift observed bottom-level targets [B, H, Nb] to the full hierarchy."""
    return torch.einsum("ij,bhj->bhi", S, target_bottom)
