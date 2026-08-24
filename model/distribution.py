from __future__ import annotations

import torch
from torch import Tensor

from contracts import FactorParams, ForecastSamples, SkewTParams


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


def sample_skew_t_factor_model(
    params: SkewTParams, num_samples: int, generator: torch.Generator | None = None
) -> Tensor:
    """Draw `mu + sigma * t + F @ eps`, `t` a standard skew-t(nu, lam) innovation.

    `t` follows the Azzalini representation `w / sqrt(chi2/nu)`: `w = delta*|u0| +
    sqrt(1-delta^2)*u1` is a skew-normal(lam) draw with `delta = lam/sqrt(1+lam^2)`,
    and `chi2` is a Wilson-Hilferty normal-transform stand-in for chi2(nu). Both
    routes are reparameterized through standard normal draws, so gradients reach
    `nu`/`lam` and the `generator` argument reproduces draws exactly, mirroring
    `sample_factor_model`.
    """
    mu, sigma, nu, lam, loadings = params.mu, params.sigma, params.nu, params.lam, params.F
    if not (mu.shape == sigma.shape == nu.shape == lam.shape):
        raise ValueError(
            f"mu {tuple(mu.shape)}, sigma {tuple(sigma.shape)}, nu {tuple(nu.shape)} "
            f"and lam {tuple(lam.shape)} must all match"
        )
    if loadings.shape[:-1] != mu.shape:
        raise ValueError(
            f"F leading dims {tuple(loadings.shape[:-1])} must match mu {tuple(mu.shape)}"
        )
    lead, n_bottom = mu.shape[:-1], mu.shape[-1]
    n_factors = loadings.shape[-1]
    kw = {"device": mu.device, "dtype": mu.dtype, "generator": generator}

    nu_ = nu.unsqueeze(-1)
    delta = (lam / torch.sqrt(1.0 + lam**2)).unsqueeze(-1)
    u0 = torch.randn(*lead, n_bottom, num_samples, **kw)
    u1 = torch.randn(*lead, n_bottom, num_samples, **kw)
    w = delta * u0.abs() + torch.sqrt(1.0 - delta**2) * u1

    z_chi = torch.randn(*lead, n_bottom, num_samples, **kw)
    wilson_hilferty = 1.0 - 2.0 / (9.0 * nu_) + z_chi * torch.sqrt(2.0 / (9.0 * nu_))
    chi2 = (nu_ * wilson_hilferty.clamp_min(1e-3) ** 3).clamp_min(1e-6)
    t = w / torch.sqrt(chi2 / nu_)

    eps = torch.randn(*lead, n_factors, num_samples, **kw)
    factor_term = torch.einsum("...bk,...kn->...bn", loadings, eps)
    return mu.unsqueeze(-1) + sigma.unsqueeze(-1) * t + factor_term


def coherent_aggregate(S: Tensor, bottom: Tensor, clip: bool = True) -> Tensor:
    """Clip to non-negative first, then aggregate through S. The order matters."""
    if S.shape[-1] != bottom.shape[-2]:
        raise ValueError(
            f"S has {S.shape[-1]} columns but samples have {bottom.shape[-2]} series"
        )
    clipped = torch.relu(bottom) if clip else bottom
    return torch.einsum("ij,...jn->...in", S, clipped)


def sample_coherent(
    params: FactorParams | SkewTParams,
    S: Tensor,
    num_samples: int,
    generator: torch.Generator | None = None,
    clip: bool = True,
) -> ForecastSamples:
    """Sample the bottom-level distribution and aggregate through S in one call.

    Dispatches on `params`' type: `FactorParams` draws Gaussian factor-model
    samples, `SkewTParams` draws skew-t factor-model samples.
    """
    sampler = (
        sample_skew_t_factor_model
        if isinstance(params, SkewTParams)
        else sample_factor_model
    )
    bottom = sampler(params, num_samples, generator=generator)
    return ForecastSamples(
        bottom=bottom, hierarchy=coherent_aggregate(S, bottom, clip=clip)
    )


def aggregate_targets(S: Tensor, target_bottom: Tensor) -> Tensor:
    """Lift observed bottom-level targets [B, H, Nb] to the full hierarchy."""
    return torch.einsum("ij,bhj->bhi", S, target_bottom)
