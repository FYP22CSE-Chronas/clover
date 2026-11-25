from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from contracts import FlowParams
from model.distribution import sample_coherent

_EPS = 1e-6
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def dsf_log_density(z: Tensor, w: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Log-density of the DSF innovation `t = dsf(z)` at the point `t(z)`.

    Change of variables on the 1-D flow: `log f_t(t) = log phi(z) - log |dt/dz|`,
    with `s = sum_k w_k sigmoid(a_k z + b_k)`, `t = logit(s)` and the closed-form
    `dt/dz = sum_k w_k a_k sigmoid'(.) / (s (1 - s))`. `z` is [..., P]; `w`/`a`/`b`
    carry the trailing K_flow axis that `z` does not.
    """
    u = torch.sigmoid(a.unsqueeze(-2) * z.unsqueeze(-1) + b.unsqueeze(-2))
    wu = w.unsqueeze(-2)
    s = (wu * u).sum(-1).clamp(_EPS, 1.0 - _EPS)
    ds = (wu * a.unsqueeze(-2) * u * (1.0 - u)).sum(-1)
    log_dt_dz = torch.log(ds.clamp_min(1e-12)) - torch.log(s) - torch.log1p(-s)
    return -0.5 * z**2 - _LOG_SQRT_2PI - log_dt_dz


def dsf_gini_mean_difference(params: FlowParams, n_grid: int = 2000) -> Tensor:
    """`E|X - X'|` of the idiosyncratic DSF term `sigma * t`, from the flow density.

    Substituting `x = q_p` in `E|X - X'| = 2 int F(1-F) dx` gives
    `2 int_0^1 p(1-p) / f(q_p) dp`. The flow is monotone, so `q_p = dsf(Phi^{-1}(p))`
    is exact and `f(q_p)` comes from `dsf_log_density`. This is what the density
    criterion `p(1-p)/f(q_p)` measures: an iid ensemble of n draws inflates expected
    CRPS by exactly `E|X - X'| / (2n)` (see `adaptive_sample`). Ignores the factor
    term; `adaptive_sample` estimates the full marginal from the pilot instead.
    Midpoint rule on `n_grid` levels; returns [B, H, Nb].
    """
    mu = params.mu
    p = (torch.arange(n_grid, dtype=mu.dtype, device=mu.device) + 0.5) / n_grid
    z_p = (math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)).expand(*mu.shape, n_grid)
    log_f = dsf_log_density(z_p, params.flow_w, params.flow_a, params.flow_b)
    integrand = p * (1.0 - p) * torch.exp(-log_f)
    return 2.0 * params.sigma * integrand.mean(-1)


def gini_mean_difference(samples: Tensor) -> Tensor:
    """Unbiased `E|X - X'|` along the trailing sample axis (sorted U-statistic)."""
    n = samples.shape[-1]
    ordered = samples.sort(dim=-1).values
    k = torch.arange(1, n + 1, dtype=samples.dtype, device=samples.device)
    return 2.0 * (ordered * (2.0 * k - n - 1.0)).sum(-1) / (n * (n - 1))


@dataclass(frozen=True)
class AdaptiveSamples:
    """Final coherent samples plus what the allocation decided.

    `hierarchy` is [B, H, N, n_total]; `difficulty` the pilot's per-node
    `E|X - X'|` [B, H, N]; `excess_pilot` the predicted sCRPS inflation (worst
    level) left by the pilot alone, `excess_final` the one left by `n_total`.
    """

    hierarchy: Tensor
    n_pilot: int
    n_total: int
    excess_pilot: float
    excess_final: float
    difficulty: Tensor


def adaptive_sample(
    params,
    S: Tensor,
    n_pilot: int = 200,
    n_max: int = 5000,
    tol: float = 2e-4,
    level_slices: Mapping[str, tuple[int, int]] | None = None,
    stratified: bool = False,
    generator: torch.Generator | None = None,
) -> AdaptiveSamples:
    """Pilot draw, sCRPS-inflation estimate, then top-up sampling.

    An ensemble of n iid draws scores, in expectation and for every outcome `y`,
    `CRPS(F_n, y) = CRPS(F, y) + E|X - X'| / (2n)`: Monte Carlo noise only ever
    inflates the score, by an amount set by the predictive spread. Per level, the
    pilot estimates that spread on every hierarchy node -- aggregates and factor term
    included -- and normalizes it like sCRPS by the predicted mass `sum |mean|`:

        excess_level(n) = sum_i E|X_i - X_i'| / (2 n sum_i |mean_i|)

    The budget is the smallest `n` keeping the worst level's excess <= `tol`
    (in sCRPS units), clamped to [n_pilot, n_max]. Forecasts with wide predictive
    distributions relative to their level get more samples; tight ones stop at the
    pilot. Extra draws are independent and appended, so samples keep equal weight
    and the hierarchy stays coherent. `stratified` uses Latin hypercube noise in
    both stages, which lowers the true excess below this iid bound (the budget is
    then conservative). Inference only: no gradients through the allocation.
    """
    if not 2 <= n_pilot <= n_max:
        raise ValueError(f"need 2 <= n_pilot <= n_max, got {n_pilot}, {n_max}")
    if tol <= 0:
        raise ValueError("tol must be > 0")

    pilot = sample_coherent(
        params, S, n_pilot, generator=generator, stratified=stratified
    ).hierarchy.detach()
    gmd = gini_mean_difference(pilot)  # [B, H, N]
    mass = pilot.mean(-1).abs()
    n_series = pilot.shape[-2]
    slices = [(0, n_series), *(level_slices or {}).values()]
    excess_1 = max(
        float(gmd[..., lo:hi].sum() / (2.0 * mass[..., lo:hi].sum().clamp_min(1e-12)))
        for lo, hi in slices
    )
    n_total = min(max(math.ceil(excess_1 / tol), n_pilot), n_max)

    hierarchy = pilot
    if n_total > n_pilot:
        extra = sample_coherent(
            params, S, n_total - n_pilot, generator=generator, stratified=stratified
        ).hierarchy
        hierarchy = torch.cat([pilot, extra], dim=-1)
    return AdaptiveSamples(
        hierarchy=hierarchy,
        n_pilot=n_pilot,
        n_total=n_total,
        excess_pilot=excess_1 / n_pilot,
        excess_final=excess_1 / n_total,
        difficulty=gmd,
    )
