"""End-to-end runs on real data, kept short enough for the default test suite."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from config import load_config
from data import load, verify_coherence
from pipeline.runner import prepare, run

QUICK = {
    "max_steps": 6,
    "val_check_steps": 3,
    "train_mc_samples": 8,
    "val_mc_samples": 8,
    "checkpoint_every": 1000,
}


def quick(dataset: str):
    config = load_config(dataset)
    return replace(
        config,
        train=replace(config.train, **QUICK),
        eval=replace(config.eval, num_samples=32),
    )


@pytest.mark.parametrize(
    "dataset",
    ["tourism_small", "labour", "tourism_large", "wiki2", "traffic", "prison", "police"],
)
def test_bundled_hierarchies_are_coherent(dataset: str) -> None:
    config = load_config(dataset)
    verify_coherence(load(dataset, config.data))


@pytest.mark.parametrize("dataset", ["tourism_small", "prison"])
def test_prepare_returns_consistent_shapes(dataset: str) -> None:
    config = load_config(dataset)
    data, features, split = prepare(config)
    assert data.S.shape == (data.n_series, data.n_bottom)
    assert split.train[0].insample.shape == (config.data.input_size, data.n_bottom)
    assert split.test[0].target.shape == (config.data.h, data.n_bottom)
    if features.static is not None:
        assert features.static.shape[0] == data.n_bottom


def test_end_to_end_run_produces_finite_scores() -> None:
    result = run(quick("tourism_small"), seed=0, verbose=False)
    assert set(result.scrps) >= {"Overall"}
    assert all(np.isfinite(v) and v >= 0 for v in result.scrps.values())
    assert result.history.steps_run == QUICK["max_steps"]
    assert result.n_params > 0


def test_forecast_samples_are_coherent() -> None:
    result = run(quick("tourism_small"), seed=0, verbose=False)
    config = load_config("tourism_small")
    dataset, _, _ = prepare(config)
    bottom = result.samples[-dataset.n_bottom :]
    np.testing.assert_allclose(
        np.einsum("ij,jhn->ihn", dataset.S, bottom), result.samples, rtol=1e-4, atol=1e-3
    )


def test_runs_are_reproducible_under_a_seed() -> None:
    a = run(quick("tourism_small"), seed=1, verbose=False)
    b = run(quick("tourism_small"), seed=1, verbose=False)
    assert a.scrps == pytest.approx(b.scrps)
