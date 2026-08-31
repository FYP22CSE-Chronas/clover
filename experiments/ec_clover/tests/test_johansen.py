from __future__ import annotations

import numpy as np
import pytest

from experiments.ec_clover.johansen import estimate_beta_johansen

statsmodels = pytest.importorskip("statsmodels")


def test_estimate_beta_shape() -> None:
    rng = np.random.default_rng(0)
    t, n_bottom, rank = 200, 4, 2
    common_trend = np.cumsum(rng.normal(size=t))
    noise = rng.normal(scale=0.5, size=(t, n_bottom))
    train_bottom = (common_trend[:, None] + noise).astype(np.float64)

    beta = estimate_beta_johansen(train_bottom, rank=rank)
    assert beta.shape == (n_bottom, rank)
    assert beta.dtype == np.float32


def test_rejects_rank_greater_than_n_bottom() -> None:
    rng = np.random.default_rng(0)
    train_bottom = rng.normal(size=(50, 3))
    with pytest.raises(ValueError, match=r"rank must be in \[1, 3\]"):
        estimate_beta_johansen(train_bottom, rank=4)


def test_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match=r"must be \[T, Nb\]"):
        estimate_beta_johansen(np.zeros(10), rank=1)


def test_to_normalized_units_rescales_rows() -> None:
    from experiments.ec_clover.johansen import to_normalized_units

    beta = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scale = np.array([10.0, 0.5], dtype=np.float32)
    scaled = to_normalized_units(beta, scale)
    np.testing.assert_allclose(scaled, np.array([[10.0, 20.0], [1.5, 2.0]]), rtol=1e-6)


def test_to_normalized_units_preserves_the_raw_combination() -> None:
    """beta_raw^T y == (beta_raw*sigma)^T y_norm, up to the per-window constant."""
    rng = np.random.default_rng(0)
    y = rng.normal(loc=[100.0, 2.0, 50.0], scale=[10.0, 0.5, 5.0], size=(40, 3))
    beta_raw = rng.normal(size=(3, 1))
    sigma, mu = y.std(axis=0), y.mean(axis=0)

    from experiments.ec_clover.johansen import to_normalized_units

    raw_combo = y @ beta_raw
    norm_combo = ((y - mu) / sigma) @ to_normalized_units(
        beta_raw.astype(np.float32), sigma.astype(np.float32)
    )
    # The two differ only by a constant offset, so their deviations must coincide.
    np.testing.assert_allclose(
        raw_combo - raw_combo.mean(), norm_combo - norm_combo.mean(), rtol=1e-4, atol=1e-4
    )


def test_to_normalized_units_rejects_mismatched_scale() -> None:
    from experiments.ec_clover.johansen import to_normalized_units

    with pytest.raises(ValueError, match="series_scale must be"):
        to_normalized_units(np.ones((3, 1), np.float32), np.ones(2, np.float32))
