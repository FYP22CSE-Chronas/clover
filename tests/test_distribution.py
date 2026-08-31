from __future__ import annotations

import numpy as np
import pytest
import torch

from contracts import (
    CopulaFlowParams,
    CopulaSplineParams,
    FactorParams,
    FlowParams,
    GMMParams,
    SkewTParams,
)
from model.distribution import (
    _dsf_transform,
    _rational_quadratic_spline,
    aggregate_targets,
    coherent_aggregate,
    sample_coherent,
    sample_copula_flow_factor_model,
    sample_copula_spline_factor_model,
    sample_factor_model,
    sample_flow_factor_model,
    sample_gmm_factor_model,
    sample_skew_t_factor_model,
)

B, H, K = 2, 3, 4


def _params(n_bottom: int) -> FactorParams:
    return FactorParams(
        mu=torch.randn(B, H, n_bottom),
        sigma=torch.rand(B, H, n_bottom) + 0.1,
        F=torch.randn(B, H, n_bottom, K) * 0.1,
    )


def _skew_t_params(n_bottom: int, lam: float | None = None) -> SkewTParams:
    lam_t = (
        torch.full((B, H, n_bottom), lam)
        if lam is not None
        else torch.randn(B, H, n_bottom)
    )
    return SkewTParams(
        mu=torch.randn(B, H, n_bottom),
        sigma=torch.rand(B, H, n_bottom) + 0.1,
        nu=torch.rand(B, H, n_bottom) * 20 + 5,
        lam=lam_t,
        F=torch.randn(B, H, n_bottom, K) * 0.1,
    )


def test_sample_shape(S: np.ndarray) -> None:
    n_bottom = S.shape[1]
    samples = sample_factor_model(_params(n_bottom), num_samples=16)
    assert samples.shape == (B, H, n_bottom, 16)


def test_samples_are_coherent(S: np.ndarray) -> None:
    S_t = torch.as_tensor(S)
    draws = sample_coherent(_params(S.shape[1]), S_t, num_samples=32)
    expected = torch.einsum("ij,bhjn->bhin", S_t, torch.relu(draws.bottom))
    torch.testing.assert_close(draws.hierarchy, expected)


def test_clipping_happens_before_aggregation(S: np.ndarray) -> None:
    """Clip-then-sum and sum-then-clip differ; coherence requires the former."""
    S_t = torch.as_tensor(S)
    bottom = torch.tensor([[-5.0, 10.0, 0.0, 0.0]]).reshape(1, 1, 4, 1)
    clipped_first = coherent_aggregate(S_t, bottom, clip=True)
    summed_first = torch.relu(coherent_aggregate(S_t, bottom, clip=False))
    assert clipped_first[0, 0, 0, 0] == pytest.approx(10.0)
    assert summed_first[0, 0, 0, 0] == pytest.approx(5.0)


def test_aggregate_targets_matches_matrix_product(S: np.ndarray) -> None:
    S_t = torch.as_tensor(S)
    target = torch.rand(B, H, S.shape[1])
    torch.testing.assert_close(aggregate_targets(S_t, target), target @ S_t.T)


def test_moments_match_the_factor_model(S: np.ndarray) -> None:
    """mean -> mu and var -> sigma^2 + ||F||^2 as the sample count grows."""
    torch.manual_seed(0)
    params = FactorParams(
        mu=torch.full((1, 1, 2), 3.0),
        sigma=torch.full((1, 1, 2), 0.5),
        F=torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]]),
    )
    samples = sample_factor_model(params, num_samples=200_000)
    torch.testing.assert_close(samples.mean(-1), params.mu, atol=0.02, rtol=0)
    expected_var = params.sigma**2 + (params.F**2).sum(-1)
    torch.testing.assert_close(samples.var(-1), expected_var, atol=0.05, rtol=0)


def test_rejects_mismatched_series_count(S: np.ndarray) -> None:
    bottom = torch.randn(B, H, S.shape[1] + 1, 5)
    with pytest.raises(ValueError, match="columns but samples have"):
        coherent_aggregate(torch.as_tensor(S), bottom)


def test_generator_makes_sampling_reproducible(S: np.ndarray) -> None:
    params = _params(S.shape[1])
    a = sample_factor_model(params, 8, generator=torch.Generator().manual_seed(7))
    b = sample_factor_model(params, 8, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b)


def test_skew_t_sample_shape(S: np.ndarray) -> None:
    n_bottom = S.shape[1]
    samples = sample_skew_t_factor_model(_skew_t_params(n_bottom), num_samples=16)
    assert samples.shape == (B, H, n_bottom, 16)


def test_skew_t_samples_are_coherent(S: np.ndarray) -> None:
    S_t = torch.as_tensor(S)
    draws = sample_coherent(_skew_t_params(S.shape[1]), S_t, num_samples=32)
    expected = torch.einsum("ij,bhjn->bhin", S_t, torch.relu(draws.bottom))
    torch.testing.assert_close(draws.hierarchy, expected)


def test_skew_t_generator_makes_sampling_reproducible(S: np.ndarray) -> None:
    params = _skew_t_params(S.shape[1])
    a = sample_skew_t_factor_model(params, 8, generator=torch.Generator().manual_seed(7))
    b = sample_skew_t_factor_model(params, 8, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b)


def test_skew_t_rejects_mismatched_shape_params(S: np.ndarray) -> None:
    n_bottom = S.shape[1]
    params = _skew_t_params(n_bottom)
    bad = SkewTParams(
        mu=params.mu,
        sigma=params.sigma,
        nu=params.nu[..., :-1],
        lam=params.lam,
        F=params.F,
    )
    with pytest.raises(ValueError, match="must all match"):
        sample_skew_t_factor_model(bad, num_samples=4)


def test_skew_t_zero_lambda_is_symmetric() -> None:
    """lam=0 collapses the skew-normal mixture to a symmetric Student-t innovation."""
    torch.manual_seed(0)
    params = SkewTParams(
        mu=torch.zeros(1, 1, 1),
        sigma=torch.ones(1, 1, 1),
        nu=torch.full((1, 1, 1), 30.0),
        lam=torch.zeros(1, 1, 1),
        F=torch.zeros(1, 1, 1, 1),
    )
    samples = sample_skew_t_factor_model(params, num_samples=200_000)
    torch.testing.assert_close(samples.mean(-1), params.mu, atol=0.02, rtol=0)


def test_skew_t_positive_lambda_skews_right() -> None:
    torch.manual_seed(0)
    n_bottom = 1
    params = SkewTParams(
        mu=torch.zeros(1, 1, n_bottom),
        sigma=torch.ones(1, 1, n_bottom),
        nu=torch.full((1, 1, n_bottom), 10.0),
        lam=torch.full((1, 1, n_bottom), 5.0),
        F=torch.zeros(1, 1, n_bottom, 1),
    )
    samples = sample_skew_t_factor_model(params, num_samples=200_000)
    assert samples.mean().item() > 0.0


K_FLOW = 3


def _flow_params(n_bottom: int) -> FlowParams:
    return FlowParams(
        mu=torch.randn(B, H, n_bottom),
        sigma=torch.rand(B, H, n_bottom) + 0.1,
        flow_w=torch.softmax(torch.randn(B, H, n_bottom, K_FLOW), dim=-1),
        flow_a=torch.rand(B, H, n_bottom, K_FLOW) + 0.1,
        flow_b=torch.randn(B, H, n_bottom, K_FLOW),
        F=torch.randn(B, H, n_bottom, K) * 0.1,
    )


def test_flow_sample_shape(S: np.ndarray) -> None:
    n_bottom = S.shape[1]
    samples = sample_flow_factor_model(_flow_params(n_bottom), num_samples=16)
    assert samples.shape == (B, H, n_bottom, 16)


def test_flow_samples_are_coherent(S: np.ndarray) -> None:
    S_t = torch.as_tensor(S)
    draws = sample_coherent(_flow_params(S.shape[1]), S_t, num_samples=32)
    expected = torch.einsum("ij,bhjn->bhin", S_t, torch.relu(draws.bottom))
    torch.testing.assert_close(draws.hierarchy, expected)


def test_flow_generator_makes_sampling_reproducible(S: np.ndarray) -> None:
    params = _flow_params(S.shape[1])
    a = sample_flow_factor_model(params, 8, generator=torch.Generator().manual_seed(7))
    b = sample_flow_factor_model(params, 8, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b)


def test_flow_rejects_mismatched_shape_params(S: np.ndarray) -> None:
    n_bottom = S.shape[1]
    params = _flow_params(n_bottom)
    bad = FlowParams(
        mu=params.mu,
        sigma=params.sigma,
        flow_w=params.flow_w[..., :-1],
        flow_a=params.flow_a,
        flow_b=params.flow_b,
        F=params.F,
    )
    with pytest.raises(ValueError, match="must all match"):
        sample_flow_factor_model(bad, num_samples=4)


def test_dsf_transform_reduces_to_affine_with_one_component() -> None:
    """logit(sigmoid(a*x+b)) == a*x+b exactly, so a single, unit-weight unit is a
    pure affine warp."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 5) * 0.5
    w = torch.ones(2, 3, 4, 1)
    a = torch.rand(2, 3, 4, 1) + 0.1
    b = torch.randn(2, 3, 4, 1) * 0.5
    t = _dsf_transform(x, w, a, b)
    expected = a * x + b
    torch.testing.assert_close(t, expected, atol=1e-4, rtol=1e-4)


def test_dsf_transform_is_strictly_increasing() -> None:
    """A convex combination of sigmoids composed with logit is strictly increasing
    in `x` whenever every weight and slope is positive, which is what makes the flow
    an invertible warp."""
    torch.manual_seed(0)
    grid = torch.linspace(-2.0, 2.0, 50).reshape(1, 1, 1, 50)
    w = torch.softmax(torch.randn(1, 1, 1, 6), dim=-1)
    a = torch.rand(1, 1, 1, 6) + 0.1
    b = torch.randn(1, 1, 1, 6)
    t = _dsf_transform(grid, w, a, b)
    assert (t.diff(dim=-1) > 0).all()


K_BINS = 4


def _copula_spline_params(n_bottom: int) -> CopulaSplineParams:
    return CopulaSplineParams(
        mu=torch.randn(B, H, n_bottom),
        sigma=torch.rand(B, H, n_bottom) + 0.1,
        spline_w=torch.softmax(torch.randn(B, H, n_bottom, K_BINS), dim=-1),
        spline_h=torch.softmax(torch.randn(B, H, n_bottom, K_BINS), dim=-1),
        spline_d=torch.rand(B, H, n_bottom, K_BINS - 1) + 0.1,
        F=torch.randn(B, H, n_bottom, K) * 0.1,
    )


def test_copula_spline_sample_shape(S: np.ndarray) -> None:
    n_bottom = S.shape[1]
    samples = sample_copula_spline_factor_model(
        _copula_spline_params(n_bottom), num_samples=16
    )
    assert samples.shape == (B, H, n_bottom, 16)


def test_copula_spline_samples_are_coherent(S: np.ndarray) -> None:
    S_t = torch.as_tensor(S)
    draws = sample_coherent(_copula_spline_params(S.shape[1]), S_t, num_samples=32)
    expected = torch.einsum("ij,bhjn->bhin", S_t, torch.relu(draws.bottom))
    torch.testing.assert_close(draws.hierarchy, expected)


def test_copula_spline_generator_makes_sampling_reproducible(S: np.ndarray) -> None:
    params = _copula_spline_params(S.shape[1])
    a = sample_copula_spline_factor_model(
        params, 8, generator=torch.Generator().manual_seed(7)
    )
    b = sample_copula_spline_factor_model(
        params, 8, generator=torch.Generator().manual_seed(7)
    )
    torch.testing.assert_close(a, b)


def test_copula_spline_rejects_mismatched_shape_params(S: np.ndarray) -> None:
    n_bottom = S.shape[1]
    params = _copula_spline_params(n_bottom)
    bad = CopulaSplineParams(
        mu=params.mu,
        sigma=params.sigma,
        spline_w=params.spline_w,
        spline_h=params.spline_h[..., :-1],
        spline_d=params.spline_d,
        F=params.F,
    )
    with pytest.raises(ValueError, match="differ"):
        sample_copula_spline_factor_model(bad, num_samples=4)


def test_copula_spline_rejects_wrong_derivative_count(S: np.ndarray) -> None:
    n_bottom = S.shape[1]
    params = _copula_spline_params(n_bottom)
    bad = CopulaSplineParams(
        mu=params.mu,
        sigma=params.sigma,
        spline_w=params.spline_w,
        spline_h=params.spline_h,
        spline_d=params.spline_d[..., :-1],
        F=params.F,
    )
    with pytest.raises(ValueError, match="interior derivatives"):
        sample_copula_spline_factor_model(bad, num_samples=4)


def test_rqs_single_bin_is_identity() -> None:
    """softmax over a single bin always gives width=height=2*tail_bound, and the
    fixed boundary derivatives of 1 make a lone bin an exact identity map."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 5) * 3.0
    w = torch.softmax(torch.randn(2, 3, 4, 1), dim=-1)
    h = torch.softmax(torch.randn(2, 3, 4, 1), dim=-1)
    d = torch.zeros(2, 3, 4, 0)
    y = _rational_quadratic_spline(x, w, h, d, tail_bound=4.0)
    torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)


def test_rqs_is_strictly_increasing_including_the_tails() -> None:
    torch.manual_seed(0)
    grid = torch.linspace(-6.0, 6.0, 400).reshape(1, 1, 1, 400)
    w = torch.softmax(torch.randn(1, 1, 1, 6), dim=-1)
    h = torch.softmax(torch.randn(1, 1, 1, 6), dim=-1)
    d = torch.rand(1, 1, 1, 5) + 0.3
    y = _rational_quadratic_spline(grid, w, h, d, tail_bound=4.0)
    assert (y.diff(dim=-1) >= 0).all()


def test_copula_spline_preserves_gaussian_factor_models_rank_correlation() -> None:
    """The point of a copula construction: reusing the same underlying `z`/`eps`
    draws, each series' own sample order survives the spline warp, so any rank-based
    correlation between series (e.g. Spearman's rho) that the Gaussian factor model
    induces carries over exactly after reshaping the marginals."""
    n_bottom = 3
    mu, sigma = torch.zeros(1, 1, n_bottom), torch.ones(1, 1, n_bottom)
    loadings = torch.randn(1, 1, n_bottom, 2) * 0.5
    gaussian_params = FactorParams(mu=mu, sigma=sigma, F=loadings)
    copula_params = CopulaSplineParams(
        mu=mu,
        sigma=sigma,
        spline_w=torch.softmax(torch.randn(1, 1, n_bottom, 5), dim=-1),
        spline_h=torch.softmax(torch.randn(1, 1, n_bottom, 5), dim=-1),
        spline_d=torch.rand(1, 1, n_bottom, 4) + 0.3,
        F=loadings,
    )
    gaussian = sample_factor_model(
        gaussian_params, 500, generator=torch.Generator().manual_seed(3)
    )
    copula = sample_copula_spline_factor_model(
        copula_params, 500, generator=torch.Generator().manual_seed(3)
    )
    for i in range(n_bottom):
        mismatches = (
            torch.argsort(gaussian[0, 0, i]) != torch.argsort(copula[0, 0, i])
        ).sum()
        assert mismatches <= 4, f"series {i}: {mismatches} rank mismatches"


def _copula_flow_params(n_bottom: int) -> CopulaFlowParams:
    return CopulaFlowParams(
        mu=torch.randn(B, H, n_bottom),
        sigma=torch.rand(B, H, n_bottom) + 0.1,
        spline_w=torch.softmax(torch.randn(B, H, n_bottom, K_BINS), dim=-1),
        spline_h=torch.softmax(torch.randn(B, H, n_bottom, K_BINS), dim=-1),
        spline_d=torch.rand(B, H, n_bottom, K_BINS - 1) + 0.1,
        flow_w=torch.softmax(torch.randn(B, H, n_bottom, K_BINS), dim=-1),
        flow_a=torch.rand(B, H, n_bottom, K_BINS) + 0.1,
        flow_b=torch.randn(B, H, n_bottom, K_BINS),
        F=torch.randn(B, H, n_bottom, K) * 0.1,
    )


def test_copula_flow_sample_shape(S: np.ndarray) -> None:
    n_bottom = S.shape[1]
    samples = sample_copula_flow_factor_model(
        _copula_flow_params(n_bottom), num_samples=16
    )
    assert samples.shape == (B, H, n_bottom, 16)


def test_copula_flow_samples_are_coherent(S: np.ndarray) -> None:
    S_t = torch.as_tensor(S)
    draws = sample_coherent(_copula_flow_params(S.shape[1]), S_t, num_samples=32)
    expected = torch.einsum("ij,bhjn->bhin", S_t, torch.relu(draws.bottom))
    torch.testing.assert_close(draws.hierarchy, expected)


def test_copula_flow_generator_makes_sampling_reproducible(S: np.ndarray) -> None:
    params = _copula_flow_params(S.shape[1])
    a = sample_copula_flow_factor_model(
        params, 8, generator=torch.Generator().manual_seed(7)
    )
    b = sample_copula_flow_factor_model(
        params, 8, generator=torch.Generator().manual_seed(7)
    )
    torch.testing.assert_close(a, b)


def test_copula_flow_rejects_mismatched_spline_shape_params(S: np.ndarray) -> None:
    n_bottom = S.shape[1]
    params = _copula_flow_params(n_bottom)
    bad = CopulaFlowParams(
        mu=params.mu,
        sigma=params.sigma,
        spline_w=params.spline_w,
        spline_h=params.spline_h[..., :-1],
        spline_d=params.spline_d,
        flow_w=params.flow_w,
        flow_a=params.flow_a,
        flow_b=params.flow_b,
        F=params.F,
    )
    with pytest.raises(ValueError, match="differ"):
        sample_copula_flow_factor_model(bad, num_samples=4)


def test_copula_flow_rejects_mismatched_flow_shape_params(S: np.ndarray) -> None:
    n_bottom = S.shape[1]
    params = _copula_flow_params(n_bottom)
    bad = CopulaFlowParams(
        mu=params.mu,
        sigma=params.sigma,
        spline_w=params.spline_w,
        spline_h=params.spline_h,
        spline_d=params.spline_d,
        flow_w=params.flow_w[..., :-1],
        flow_a=params.flow_a,
        flow_b=params.flow_b,
        F=params.F,
    )
    with pytest.raises(ValueError, match="must all match"):
        sample_copula_flow_factor_model(bad, num_samples=4)


def test_copula_flow_preserves_gaussian_factor_models_rank_correlation() -> None:
    """A composition of two strictly increasing functions (spline then flow) is
    itself strictly increasing, so the copula's rank correlation survives both
    layers exactly, the same guarantee `CopulaSplineParams` gives with just one."""
    n_bottom = 3
    mu, sigma = torch.zeros(1, 1, n_bottom), torch.ones(1, 1, n_bottom)
    loadings = torch.randn(1, 1, n_bottom, 2) * 0.5
    gaussian_params = FactorParams(mu=mu, sigma=sigma, F=loadings)
    copula_flow_params = CopulaFlowParams(
        mu=mu,
        sigma=sigma,
        spline_w=torch.softmax(torch.randn(1, 1, n_bottom, 5), dim=-1),
        spline_h=torch.softmax(torch.randn(1, 1, n_bottom, 5), dim=-1),
        spline_d=torch.rand(1, 1, n_bottom, 4) + 0.3,
        flow_w=torch.softmax(torch.randn(1, 1, n_bottom, 5), dim=-1),
        flow_a=torch.rand(1, 1, n_bottom, 5) + 0.3,
        flow_b=torch.randn(1, 1, n_bottom, 5),
        F=loadings,
    )
    gaussian = sample_factor_model(
        gaussian_params, 500, generator=torch.Generator().manual_seed(3)
    )
    combo = sample_copula_flow_factor_model(
        copula_flow_params, 500, generator=torch.Generator().manual_seed(3)
    )
    for i in range(n_bottom):
        mismatches = (
            torch.argsort(gaussian[0, 0, i]) != torch.argsort(combo[0, 0, i])
        ).sum()
        assert mismatches <= 4, f"series {i}: {mismatches} rank mismatches"


N_COMPONENTS = 2


def _gmm_params(n_bottom: int) -> GMMParams:
    return GMMParams(
        mu=torch.randn(B, H, n_bottom, N_COMPONENTS),
        sigma=torch.rand(B, H, n_bottom, N_COMPONENTS) + 0.1,
        logits=torch.randn(B, H, n_bottom, N_COMPONENTS),
        F=torch.randn(B, H, n_bottom, K) * 0.1,
    )


def test_gmm_sample_shape(S: np.ndarray) -> None:
    n_bottom = S.shape[1]
    samples = sample_gmm_factor_model(_gmm_params(n_bottom), num_samples=16)
    assert samples.shape == (B, H, n_bottom, 16)


def test_gmm_samples_are_coherent(S: np.ndarray) -> None:
    S_t = torch.as_tensor(S)
    draws = sample_coherent(_gmm_params(S.shape[1]), S_t, num_samples=32)
    expected = torch.einsum("ij,bhjn->bhin", S_t, torch.relu(draws.bottom))
    torch.testing.assert_close(draws.hierarchy, expected)


def test_gmm_generator_makes_sampling_reproducible(S: np.ndarray) -> None:
    params = _gmm_params(S.shape[1])
    a = sample_gmm_factor_model(params, 8, generator=torch.Generator().manual_seed(7))
    b = sample_gmm_factor_model(params, 8, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b)


def test_gmm_rejects_mismatched_shape_params(S: np.ndarray) -> None:
    n_bottom = S.shape[1]
    params = _gmm_params(n_bottom)
    bad = GMMParams(
        mu=params.mu,
        sigma=params.sigma,
        logits=params.logits[..., :-1],
        F=params.F,
    )
    with pytest.raises(ValueError, match="must all match"):
        sample_gmm_factor_model(bad, num_samples=4)


def test_gmm_is_a_true_mixture_not_an_averaged_gaussian() -> None:
    """Two far-apart, narrow components: draws must cluster near one mean or the
    other, not smear across the gap the way a single averaged Gaussian would."""
    torch.manual_seed(0)
    params = GMMParams(
        mu=torch.tensor([-10.0, 10.0]).reshape(1, 1, 1, 2),
        sigma=torch.full((1, 1, 1, 2), 0.1),
        logits=torch.zeros(1, 1, 1, 2),
        F=torch.zeros(1, 1, 1, 1),
    )
    samples = sample_gmm_factor_model(params, num_samples=20_000)
    in_gap = ((samples > -2.0) & (samples < 2.0)).float().mean()
    assert in_gap.item() < 0.01
    near_either_mode = ((samples < -8.0) | (samples > 8.0)).float().mean()
    assert near_either_mode.item() > 0.95


def test_gmm_logits_control_which_component_dominates() -> None:
    torch.manual_seed(0)
    params = GMMParams(
        mu=torch.tensor([-10.0, 10.0]).reshape(1, 1, 1, 2),
        sigma=torch.full((1, 1, 1, 2), 0.1),
        logits=torch.tensor([-10.0, 10.0]).reshape(1, 1, 1, 2),
        F=torch.zeros(1, 1, 1, 1),
    )
    samples = sample_gmm_factor_model(params, num_samples=20_000)
    assert (samples > 0).float().mean().item() > 0.95
