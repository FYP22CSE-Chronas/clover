from __future__ import annotations

import torch
from torch import Tensor

from contracts import (
    FactorParams,
    ForecastSamples,
    GMMParams,
    SkewTParams,
    SplineCopulaParams,
)
from model.blocks import rq_spline_transform

SCALE_FLOOR = 1e-6


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


def sample_gmm_factor_model(
    params: GMMParams,
    num_samples: int,
    generator: torch.Generator | None = None,
    tau: float = 1.0,
) -> Tensor:
    """Draw `mu_k + sigma_k * z + F @ eps`, `k` a Gumbel-softmax component pick.

    Each Monte Carlo draw independently picks one of the `K` Gaussian components
    per (series, horizon) via a straight-through Gumbel-softmax over `logits`:
    one-hot (a true mixture draw, not a component-average) on the forward pass,
    soft on the backward pass so gradients still reach `mu`/`sigma`/`logits`.
    Both the component pick and the Gaussian noise accept `generator`, mirroring
    `sample_factor_model`.
    """
    mu, sigma, logits, loadings = params.mu, params.sigma, params.logits, params.F
    if not (mu.shape == sigma.shape == logits.shape):
        raise ValueError(
            f"mu {tuple(mu.shape)}, sigma {tuple(sigma.shape)} and logits "
            f"{tuple(logits.shape)} must all match"
        )
    if loadings.shape[:-1] != mu.shape[:-1]:
        raise ValueError(
            f"F leading dims {tuple(loadings.shape[:-1])} must match mu's series "
            f"dims {tuple(mu.shape[:-1])}"
        )
    lead, n_bottom, n_components = mu.shape[:-2], mu.shape[-2], mu.shape[-1]
    n_factors = loadings.shape[-1]
    kw = {"device": mu.device, "dtype": mu.dtype, "generator": generator}

    expanded_logits = logits.unsqueeze(-2).expand(
        *lead, n_bottom, num_samples, n_components
    )
    u = torch.rand(expanded_logits.shape, **kw).clamp(1e-9, 1.0 - 1e-9)
    gumbel = -torch.log(-torch.log(u))
    y_soft = torch.softmax((expanded_logits + gumbel) / tau, dim=-1)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, y_soft.argmax(-1, keepdim=True), 1.0)
    weights = y_hard + (y_soft - y_soft.detach())

    mu_sample = (weights * mu.unsqueeze(-2)).sum(-1)
    sigma_sample = (weights * sigma.unsqueeze(-2)).sum(-1)

    z = torch.randn(*lead, n_bottom, num_samples, **kw)
    eps = torch.randn(*lead, n_factors, num_samples, **kw)
    factor_term = torch.einsum("...bk,...kn->...bn", loadings, eps)
    return mu_sample + sigma_sample * z + factor_term


def sample_spline_copula_factor_model(
    params: SplineCopulaParams,
    num_samples: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw `mu + s * T(g)`: a Gaussian factor copula with learned marginals.

    The factor term `sigma * z + F @ eps` is built exactly as in
    `sample_factor_model`, then divided by its own standard deviation
    `s = sqrt(sigma^2 + sum_k F_k^2)`. The result `g` is marginally standard normal
    while its cross-series dependence is still the low-rank factor structure -- that
    is, `g` carries the Gaussian copula and nothing else. A monotone spline `T` then
    reshapes each marginal, and `mu`/`s` put it back on the series' own location and
    scale.

    Standardizing before the spline is what keeps the parameterization identified
    (`s` cannot be silently absorbed into `T`'s slope) and what keeps `theta`
    scale-free, so `denormalize_params` stays an exact affine inverse. With
    `theta = 0` the spline is the identity and this reduces to `sample_factor_model`
    draw for draw under the same `generator`.
    """
    mu, sigma, loadings, theta = params.mu, params.sigma, params.F, params.theta
    if mu.shape != sigma.shape:
        raise ValueError(f"mu {tuple(mu.shape)} and sigma {tuple(sigma.shape)} differ")
    if loadings.shape[:-1] != mu.shape:
        raise ValueError(
            f"F leading dims {tuple(loadings.shape[:-1])} must match mu {tuple(mu.shape)}"
        )
    if theta.shape[:-1] != mu.shape:
        raise ValueError(
            f"theta leading dims {tuple(theta.shape[:-1])} must match mu "
            f"{tuple(mu.shape)}"
        )
    lead, n_bottom = mu.shape[:-1], mu.shape[-1]
    n_factors = loadings.shape[-1]
    kw = {"device": mu.device, "dtype": mu.dtype, "generator": generator}
    z = torch.randn(*lead, n_bottom, num_samples, **kw)
    eps = torch.randn(*lead, n_factors, num_samples, **kw)
    factor_term = torch.einsum("...bk,...kn->...bn", loadings, eps)

    scale = torch.sqrt(sigma**2 + (loadings**2).sum(dim=-1)).clamp_min(SCALE_FLOOR)
    latent = (sigma.unsqueeze(-1) * z + factor_term) / scale.unsqueeze(-1)
    shaped = rq_spline_transform(latent, theta, bound=params.bound)
    return mu.unsqueeze(-1) + scale.unsqueeze(-1) * shaped


def coherent_aggregate(S: Tensor, bottom: Tensor, clip: bool = True) -> Tensor:
    """Clip to non-negative first, then aggregate through S. The order matters."""
    if S.shape[-1] != bottom.shape[-2]:
        raise ValueError(
            f"S has {S.shape[-1]} columns but samples have {bottom.shape[-2]} series"
        )
    clipped = torch.relu(bottom) if clip else bottom
    return torch.einsum("ij,...jn->...in", S, clipped)


def sample_coherent(
    params: FactorParams | SkewTParams | GMMParams | SplineCopulaParams,
    S: Tensor,
    num_samples: int,
    generator: torch.Generator | None = None,
    clip: bool = True,
) -> ForecastSamples:
    """Sample the bottom-level distribution and aggregate through S in one call.

    Dispatches on `params`' type: `FactorParams` draws Gaussian factor-model
    samples, `SkewTParams` draws skew-t samples, `GMMParams` draws Gaussian-
    mixture samples, and `SplineCopulaParams` draws Gaussian-copula samples with
    learned marginals.
    """
    if isinstance(params, SkewTParams):
        sampler = sample_skew_t_factor_model
    elif isinstance(params, GMMParams):
        sampler = sample_gmm_factor_model
    elif isinstance(params, SplineCopulaParams):
        sampler = sample_spline_copula_factor_model
    else:
        sampler = sample_factor_model
    bottom = sampler(params, num_samples, generator=generator)
    return ForecastSamples(
        bottom=bottom, hierarchy=coherent_aggregate(S, bottom, clip=clip)
    )


def aggregate_targets(S: Tensor, target_bottom: Tensor) -> Tensor:
    """Lift observed bottom-level targets [B, H, Nb] to the full hierarchy."""
    return torch.einsum("ij,bhj->bhi", S, target_bottom)
