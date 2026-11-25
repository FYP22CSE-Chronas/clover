from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from contracts import FlowParams
from model.adaptive import (
    adaptive_sample,
    dsf_gini_mean_difference,
    dsf_log_density,
    gini_mean_difference,
)
from model.distribution import _dsf_transform, sample_coherent, standard_normal
from pipeline import run
from tests.test_smoke import quick
from training.metrics import crps_from_quantiles, crps_from_samples, quantile_grid


def erfinv_(x: float) -> float:
    return float(torch.erfinv(torch.tensor(x, dtype=torch.float64)))

S4 = torch.tensor(
    [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)


def _params(sigma: float = 1.0, mu: float = 10.0, f_scale: float = 0.0) -> FlowParams:
    g = torch.Generator().manual_seed(0)
    k = 4
    return FlowParams(
        mu=torch.full((1, 2, 3), mu),
        sigma=torch.full((1, 2, 3), sigma),
        flow_w=torch.softmax(torch.randn(1, 2, 3, k, generator=g), -1),
        flow_a=torch.rand(1, 2, 3, k, generator=g) + 0.5,
        flow_b=torch.randn(1, 2, 3, k, generator=g),
        F=torch.randn(1, 2, 3, 2, generator=g) * f_scale,
    )


def test_dsf_log_density_matches_numerical_change_of_variables() -> None:
    p = _params()
    z = torch.linspace(-2, 2, 9, dtype=torch.float64).expand(1, 2, 3, 9)
    w, a, b = (x.double() for x in (p.flow_w, p.flow_a, p.flow_b))
    h = 1e-5
    dt = (_dsf_transform(z + h, w, a, b) - _dsf_transform(z - h, w, a, b)) / (2 * h)
    expected = -0.5 * z**2 - 0.5 * np.log(2 * np.pi) - torch.log(dt)
    assert torch.allclose(dsf_log_density(z, w, a, b), expected, atol=1e-4)


def test_dsf_density_integral_equals_monte_carlo_gini_mean_difference() -> None:
    """`2 int p(1-p)/f(q_p) dp` is the flow's E|X - X'|, the quantity that sets the
    Monte Carlo inflation of CRPS."""
    params = _params(sigma=2.0)
    analytic = dsf_gini_mean_difference(params)
    t = _dsf_transform(
        torch.randn(1, 2, 3, 40_000, generator=torch.Generator().manual_seed(1)),
        params.flow_w,
        params.flow_a,
        params.flow_b,
    )
    mc = gini_mean_difference(params.sigma.unsqueeze(-1) * t)
    assert torch.allclose(analytic, mc, rtol=0.03)


def test_monte_carlo_inflation_of_crps_estimators() -> None:
    """For n iid draws from F: the empirical-CDF ensemble CRPS is inflated by exactly
    E|X - X'| / (2n); the quantile-grid estimator (the reported metric) by less, so
    that is a conservative bound; the repo's `mc` estimator (n(n-1) normalization)
    is the fair, unbiased one."""
    rng = np.random.default_rng(0)
    y, n, reps = 0.3, 20, 40_000
    samples = rng.standard_normal((reps, 1, n))
    y_true = np.full((reps, 1), y)
    phi = math.exp(-0.5 * y**2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + math.erf(y / math.sqrt(2)))
    exact = y * (2 * cdf - 1) + 2 * phi - 1 / math.sqrt(math.pi)
    bound = (2 / math.sqrt(math.pi)) / (2 * n)

    pairwise = np.abs(samples[..., :, None] - samples[..., None, :]).mean((-1, -2))
    ensemble = (np.abs(samples - y).mean(-1) - 0.5 * pairwise).mean()
    assert ensemble - exact == pytest.approx(bound, abs=1.5e-3)
    assert crps_from_samples(y_true, samples).mean() == pytest.approx(exact, abs=1.5e-3)

    q = quantile_grid(0.01)
    z_q = np.array([math.sqrt(2) * erfinv_(2 * p - 1) for p in q])
    d = y - z_q
    grid_exact = 2 * np.maximum(q * d, (q - 1) * d).mean()
    inflation = crps_from_quantiles(y_true, samples, q).mean() - grid_exact
    assert 0 < inflation < bound


def test_gini_mean_difference_matches_pairwise_definition() -> None:
    x = torch.randn(3, 25, dtype=torch.float64)
    pairwise = (x.unsqueeze(-1) - x.unsqueeze(-2)).abs().sum((-1, -2)) / (25 * 24)
    assert torch.allclose(gini_mean_difference(x), pairwise)


def test_stratified_normal_hits_every_stratum_once() -> None:
    n = 64
    z = standard_normal(5, 7, n, stratified=True, generator=torch.Generator().manual_seed(0))
    u = 0.5 * (1 + torch.erf(z.double() / math.sqrt(2)))
    strata = torch.floor(u * n).long().sort(-1).values
    assert torch.equal(strata, torch.arange(n).expand(5, 7, n))


def test_stratified_flow_quantiles_are_far_more_accurate() -> None:
    params = _params(sigma=1.0)
    p = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    z_p = (math.sqrt(2) * torch.erfinv(2 * p - 1)).expand(1, 2, 3, 5)
    exact = 10.0 + _dsf_transform(z_p, params.flow_w, params.flow_a, params.flow_b)

    def err(stratified: bool) -> float:
        g = torch.Generator().manual_seed(3)
        errs = []
        for _ in range(20):
            x = sample_coherent(params, S4, 200, g, stratified=stratified).bottom
            q = torch.quantile(x, p, dim=-1).permute(1, 2, 3, 0)
            errs.append(((q - exact) ** 2).mean())
        return float(torch.stack(errs).mean())

    assert err(True) < 0.2 * err(False)


def test_stratified_sampling_stays_coherent_and_unbiased() -> None:
    params = _params(f_scale=0.5)
    g = torch.Generator().manual_seed(0)
    draws = sample_coherent(params, S4, 20_000, g, stratified=True)
    expected = torch.einsum("ij,bhjn->bhin", S4, torch.relu(draws.bottom))
    torch.testing.assert_close(draws.hierarchy, expected)
    iid = sample_coherent(params, S4, 20_000, g).bottom
    assert torch.allclose(draws.bottom.std(-1), iid.std(-1), rtol=0.05)


def test_stratified_rejects_unsupported_heads() -> None:
    from contracts import SkewTParams

    shape = (1, 2, 3)
    params = SkewTParams(
        mu=torch.zeros(shape),
        sigma=torch.ones(shape),
        nu=torch.full(shape, 5.0),
        lam=torch.zeros(shape),
        F=torch.zeros(*shape, 1),
    )
    with pytest.raises(ValueError, match="stratified"):
        sample_coherent(params, S4, 10, stratified=True)


def test_adaptive_budget_tracks_spread_relative_to_mass() -> None:
    tight = adaptive_sample(_params(sigma=0.01), S4, n_pilot=50, n_max=4000, tol=1e-3)
    wide = adaptive_sample(_params(sigma=5.0), S4, n_pilot=50, n_max=4000, tol=1e-3)
    assert tight.n_total == 50
    assert wide.n_total > tight.n_total
    assert wide.hierarchy.shape[-1] == wide.n_total
    assert wide.excess_final <= 1e-3 + 1e-9
    capped = adaptive_sample(_params(sigma=50.0), S4, n_pilot=50, n_max=300, tol=1e-6)
    assert capped.n_total == 300


def test_adaptive_budget_is_set_by_the_worst_level() -> None:
    params = _params(sigma=2.0)
    g = torch.Generator().manual_seed(0)
    overall = adaptive_sample(params, S4, n_pilot=100, n_max=10**6, tol=1e-4, generator=g)
    g = torch.Generator().manual_seed(0)
    per_level = adaptive_sample(
        params, S4, 100, 10**6, 1e-4, level_slices={"Total": (0, 1), "Bottom": (1, 4)},
        generator=g,
    )
    assert per_level.n_total >= overall.n_total


def test_adaptive_validates_arguments() -> None:
    with pytest.raises(ValueError):
        adaptive_sample(_params(), S4, n_pilot=100, n_max=10)
    with pytest.raises(ValueError):
        adaptive_sample(_params(), S4, tol=0.0)


def test_adaptive_stratified_flow_end_to_end_run() -> None:
    config = quick("tourism_small")
    config = replace(
        config,
        model=replace(config.model, head="normalizing_flow"),
        eval=replace(
            config.eval, adaptive=True, stratified=True, adaptive_pilot=50, adaptive_max=200
        ),
    )
    result = run(config, seed=0, verbose=False)
    assert all(np.isfinite(v) and v >= 0 for v in result.scrps.values())
    assert 50 <= result.samples.shape[-1] <= 200
