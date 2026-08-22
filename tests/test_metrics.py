from __future__ import annotations

import numpy as np
import pytest

from training.metrics import (
    crps_from_quantiles,
    crps_from_samples,
    per_location_crps,
    quantile_grid,
    scrps,
    scrps_by_level,
)


def test_quantile_grid_is_interior() -> None:
    grid = quantile_grid(0.01)
    assert len(grid) == 99
    assert grid[0] == pytest.approx(0.01)
    assert grid[-1] == pytest.approx(0.99)


def test_estimators_agree_on_a_large_sample() -> None:
    rng = np.random.default_rng(0)
    y = np.array([[1.0]])
    samples = rng.normal(1.2, 0.6, size=(1, 1, 200_000))
    a = crps_from_quantiles(y, samples)
    b = crps_from_samples(y, samples)
    assert a[0, 0] == pytest.approx(b[0, 0], abs=5e-3)


def test_per_location_crps_shape() -> None:
    y = np.zeros((5, 3))
    samples = np.random.default_rng(0).normal(size=(5, 3, 64))
    assert per_location_crps(y, samples).shape == (5, 3)


def test_scrps_is_a_ratio_of_sums() -> None:
    y = np.array([[2.0, 4.0]])
    samples = np.tile(y[..., None], (1, 1, 32))
    assert scrps(y, samples) == pytest.approx(0.0, abs=1e-9)


def test_scrps_by_level_covers_every_level_plus_overall(level_slices: dict) -> None:
    rng = np.random.default_rng(0)
    y = rng.gamma(4.0, 10.0, size=(7, 3))
    samples = y[..., None] + rng.normal(0, 1, size=(7, 3, 64))
    scores = scrps_by_level(y, samples, level_slices)
    assert set(scores) == {"Overall", *level_slices}
    assert all(np.isfinite(v) for v in scores.values())


def test_overall_is_not_the_mean_of_levels(level_slices: dict) -> None:
    """Overall is the whole-hierarchy ratio; a level mean would double-count mass."""
    rng = np.random.default_rng(1)
    y = rng.gamma(4.0, 10.0, size=(7, 3))
    samples = y[..., None] + rng.normal(0, 5, size=(7, 3, 128))
    scores = scrps_by_level(y, samples, level_slices)
    level_mean = np.mean([scores[k] for k in level_slices])
    assert scores["Overall"] != pytest.approx(level_mean)


def test_rejects_unknown_estimator() -> None:
    with pytest.raises(ValueError, match="unknown estimator"):
        per_location_crps(np.zeros((1, 1)), np.zeros((1, 1, 4)), estimator="nope")


def test_zero_mass_level_raises() -> None:
    y = np.zeros((2, 2))
    with pytest.raises(ValueError, match="no target mass"):
        scrps(y, np.zeros((2, 2, 8)))
