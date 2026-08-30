from __future__ import annotations

import torch
from torch import Tensor

from contracts import FactorParams, FlowParams, ForecastSamples, GMMParams, SkewTParams


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
    mu, sigma, nu, lam, loadings = (
        params.mu,
        params.sigma,
        params.nu,
        params.lam,
        params.F,
    )
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


def _dsf_transform(x: Tensor, w: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Deep sigmoidal flow (Huang et al. 2018): a monotone warp of `x`.

    `w` (a convex combination -- positive, summing to 1 along the trailing K_flow
    axis) and `a` (strictly positive) are assumed already activated by the caller
    (see `model.heads._flow_shape`), so every unit's sigmoid is a strictly increasing
    function of `x`, and their convex combination composed with the logit link is
    strictly increasing end to end -- an invertible warp, though only this forward
    (base-noise -> innovation) direction is ever evaluated, since sampling needs no
    inverse or log-det. `w`/`a`/`b` carry a trailing K_flow axis that `x` does not.
    """
    eps = 1e-6
    s = (
        (
            w.unsqueeze(-2)
            * torch.sigmoid(a.unsqueeze(-2) * x.unsqueeze(-1) + b.unsqueeze(-2))
        )
        .sum(-1)
        .clamp(eps, 1.0 - eps)
    )
    return torch.log(s) - torch.log1p(-s)


def sample_flow_factor_model(
    params: FlowParams, num_samples: int, generator: torch.Generator | None = None
) -> Tensor:
    """Draw `mu + sigma * t + F @ eps`, `t` a conditional-normalizing-flow innovation.

    `t = dsf(z; flow_w, flow_a, flow_b)` warps a standard normal `z` through the
    per-(series, horizon) deep sigmoidal flow (see `_dsf_transform`) instead of
    assuming a fixed parametric family like `SkewTParams`'s, so the innovation's
    shape -- skew, tail weight, and (with enough flow units) multimodality -- is
    learned from data. Reparameterized through `z` exactly like `sample_factor_model`,
    so `generator` reproduces draws and gradients reach the flow parameters.
    """
    mu, sigma, loadings = params.mu, params.sigma, params.F
    w, a, b = params.flow_w, params.flow_a, params.flow_b
    if mu.shape != sigma.shape:
        raise ValueError(f"mu {tuple(mu.shape)} and sigma {tuple(sigma.shape)} differ")
    if not (w.shape == a.shape == b.shape):
        raise ValueError(
            f"flow_w {tuple(w.shape)}, flow_a {tuple(a.shape)} and flow_b "
            f"{tuple(b.shape)} must all match"
        )
    if w.shape[:-1] != mu.shape:
        raise ValueError(
            f"flow parameter leading dims {tuple(w.shape[:-1])} must match mu "
            f"{tuple(mu.shape)}"
        )
    if loadings.shape[:-1] != mu.shape:
        raise ValueError(
            f"F leading dims {tuple(loadings.shape[:-1])} must match mu {tuple(mu.shape)}"
        )
    lead, n_bottom = mu.shape[:-1], mu.shape[-1]
    n_factors = loadings.shape[-1]
    kw = {"device": mu.device, "dtype": mu.dtype, "generator": generator}

    z = torch.randn(*lead, n_bottom, num_samples, **kw)
    t = _dsf_transform(z, w, a, b)

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


def coherent_aggregate(S: Tensor, bottom: Tensor, clip: bool = True) -> Tensor:
    """Clip to non-negative first, then aggregate through S. The order matters."""
    if S.shape[-1] != bottom.shape[-2]:
        raise ValueError(
            f"S has {S.shape[-1]} columns but samples have {bottom.shape[-2]} series"
        )
    clipped = torch.relu(bottom) if clip else bottom
    return torch.einsum("ij,...jn->...in", S, clipped)


def sample_coherent(
    params: FactorParams | SkewTParams | FlowParams | GMMParams,
    S: Tensor,
    num_samples: int,
    generator: torch.Generator | None = None,
    clip: bool = True,
) -> ForecastSamples:
    """Sample the bottom-level distribution and aggregate through S in one call.

    Dispatches on `params`' type: `FactorParams` draws Gaussian factor-model
    samples, `SkewTParams` draws skew-t samples, `FlowParams` draws normalizing-
    flow samples, `GMMParams` draws Gaussian-mixture samples.
    """
    if isinstance(params, SkewTParams):
        sampler = sample_skew_t_factor_model
    elif isinstance(params, FlowParams):
        sampler = sample_flow_factor_model
    elif isinstance(params, GMMParams):
        sampler = sample_gmm_factor_model
    else:
        sampler = sample_factor_model
    bottom = sampler(params, num_samples, generator=generator)
    return ForecastSamples(
        bottom=bottom, hierarchy=coherent_aggregate(S, bottom, clip=clip)
    )


def aggregate_targets(S: Tensor, target_bottom: Tensor) -> Tensor:
    """Lift observed bottom-level targets [B, H, Nb] to the full hierarchy."""
    return torch.einsum("ij,bhj->bhi", S, target_bottom)
