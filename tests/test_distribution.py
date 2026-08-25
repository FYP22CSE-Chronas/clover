from __future__ import annotations

import numpy as np
import pytest
import torch

from contracts import FactorParams, GMMParams, SkewTParams
from model.distribution import (
    aggregate_targets,
    coherent_aggregate,
    sample_coherent,
    sample_factor_model,
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
