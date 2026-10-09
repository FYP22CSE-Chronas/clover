"""End-to-end runs on real data, kept short enough for the default test suite."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from config import load_config
from data import load, verify_coherence
from pipeline.results import RunResult
from pipeline.runner import ensemble_scrps, prepare, run
from training.trainer import History

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


def test_skew_t_head_end_to_end_run_produces_finite_scores() -> None:
    """The skew-t distribution head trains through the same pipeline as the default."""
    config = quick("tourism_small")
    config = replace(config, model=replace(config.model, head="skew_t"))
    result = run(config, seed=0, verbose=False)
    assert set(result.scrps) >= {"Overall"}
    assert all(np.isfinite(v) and v >= 0 for v in result.scrps.values())
    assert result.history.steps_run == QUICK["max_steps"]


def test_normalizing_flow_head_end_to_end_run_produces_finite_scores() -> None:
    """The normalizing-flow head trains through the same pipeline as the default."""
    config = quick("tourism_small")
    config = replace(config, model=replace(config.model, head="normalizing_flow"))
    result = run(config, seed=0, verbose=False)
    assert set(result.scrps) >= {"Overall"}
    assert all(np.isfinite(v) and v >= 0 for v in result.scrps.values())
    assert result.history.steps_run == QUICK["max_steps"]


def test_normalizing_flow_shared_head_end_to_end_run_produces_finite_scores() -> None:
    """The shared-flow head (one global flow, not per-output) trains through the
    same pipeline as the default."""
    config = quick("tourism_small")
    config = replace(config, model=replace(config.model, head="normalizing_flow_shared"))
    result = run(config, seed=0, verbose=False)
    assert set(result.scrps) >= {"Overall"}
    assert all(np.isfinite(v) and v >= 0 for v in result.scrps.values())
    assert result.history.steps_run == QUICK["max_steps"]


def test_copula_spline_head_end_to_end_run_produces_finite_scores() -> None:
    """The Gaussian-copula spline-marginal head trains through the same pipeline as
    the default."""
    config = quick("tourism_small")
    config = replace(config, model=replace(config.model, head="copula_spline"))
    result = run(config, seed=0, verbose=False)
    assert set(result.scrps) >= {"Overall"}
    assert all(np.isfinite(v) and v >= 0 for v in result.scrps.values())
    assert result.history.steps_run == QUICK["max_steps"]


def test_copula_flow_head_end_to_end_run_produces_finite_scores() -> None:
    """The copula spline-then-flow head trains through the same pipeline as the
    default."""
    config = quick("tourism_small")
    config = replace(config, model=replace(config.model, head="copula_flow"))
    result = run(config, seed=0, verbose=False)
    assert set(result.scrps) >= {"Overall"}
    assert all(np.isfinite(v) and v >= 0 for v in result.scrps.values())
    assert result.history.steps_run == QUICK["max_steps"]


def test_gmm_head_end_to_end_run_produces_finite_scores() -> None:
    """The Gaussian-mixture head trains through the same pipeline as the default."""
    config = quick("tourism_small")
    config = replace(config, model=replace(config.model, head="gmm"))
    result = run(config, seed=0, verbose=False)
    assert set(result.scrps) >= {"Overall"}
    assert all(np.isfinite(v) and v >= 0 for v in result.scrps.values())
    assert result.history.steps_run == QUICK["max_steps"]


def test_ensemble_pools_seed_forecasts() -> None:
    """Pooling seeds scores the mixture over the same levels as a single run."""
    config = quick("tourism_small")
    a = run(config, seed=0, verbose=False)
    b = run(config, seed=1, verbose=False)
    pooled = ensemble_scrps([a, b])
    assert set(pooled) == set(a.scrps)
    assert all(np.isfinite(v) and v >= 0 for v in pooled.values())
    with pytest.raises(ValueError, match="at least two"):
        ensemble_scrps([a])
    with pytest.raises(ValueError, match="cannot be pooled"):
        ensemble_scrps([a, replace(b, y_true=b.y_true + 1.0)])


def test_ensemble_of_opposite_biases_beats_each_member() -> None:
    """CRPS is convex in the forecast, so mixing two opposite biases must help."""
    rng = np.random.default_rng(0)
    y_true = np.full((3, 4), 10.0)
    noise = rng.standard_normal((3, 4, 2000))

    def member(seed: int, shift: float) -> RunResult:
        return RunResult(
            dataset="toy",
            seed=seed,
            scrps={},
            history=History(),
            samples=y_true[..., None] + shift + noise,
            y_true=y_true,
            level_slices={"Top": (0, 1), "Bottom": (1, 3)},
        )

    low, high = member(0, -2.0), member(1, 2.0)
    pooled = ensemble_scrps([low, high])["Overall"]
    single = ensemble_scrps([low, low])["Overall"]
    assert pooled < 0.8 * single


def test_snapshot_ensemble_pools_every_cycle() -> None:
    """Each cycle ends in a kept snapshot, and the test forecast pools all of them."""
    config = quick("tourism_small")
    config = replace(
        config, train=replace(config.train, max_steps=7, snapshot_count=3)
    )
    result = run(config, seed=0, verbose=False)
    assert result.history.snapshot_steps == [2, 4, 6]
    assert result.history.steps_run == 6
    assert result.samples.shape[-1] == 3 * config.eval.num_samples
    assert set(result.extras["single_scrps"]) == set(result.scrps)
    assert all(np.isfinite(v) and v >= 0 for v in result.scrps.values())


def test_validation_selected_snapshots_are_a_subset() -> None:
    config = quick("tourism_small")
    config = replace(
        config,
        train=replace(
            config.train, max_steps=6, snapshot_count=3, snapshot_select="validation"
        ),
    )
    result = run(config, seed=0, verbose=False)
    kept = result.extras["snapshots_kept"]
    assert kept and set(kept) <= set(result.history.snapshot_steps)
    assert result.samples.shape[-1] == len(kept) * config.eval.num_samples
    assert set(result.extras["all_scrps"]) == set(result.scrps)
