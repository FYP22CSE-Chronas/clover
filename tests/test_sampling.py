from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from config import TrainConfig
from pipeline.runner import run
from tests.test_smoke import QUICK, quick
from training.sampling import AdaptiveWindowSampler, RecencyWindowSampler
from training.trainer import Trainer


def test_sampler_is_uniform_before_any_loss_is_seen() -> None:
    sampler = AdaptiveWindowSampler(5, mix=0.5)
    np.testing.assert_allclose(sampler.probabilities(), np.full(5, 0.2))


def test_sampler_favours_high_loss_windows_but_keeps_a_uniform_floor() -> None:
    sampler = AdaptiveWindowSampler(4, mix=0.5)
    sampler.update([0, 1, 2, 3], np.array([1.0, 1.0, 1.0, 5.0]))
    p = sampler.probabilities()
    assert p.sum() == pytest.approx(1.0)
    assert p[3] > p[0]
    assert p.min() >= 0.5 / 4


def test_unseen_windows_score_the_mean_of_seen_ones() -> None:
    sampler = AdaptiveWindowSampler(3, mix=0.9)
    sampler.update([0, 1], np.array([1.0, 3.0]))
    p = sampler.probabilities()
    assert p[2] == pytest.approx(1.0 / 3.0)


def test_full_importance_weights_keep_the_objective_unbiased() -> None:
    """E_p[w * L] equals the uniform mean of L when beta = 1."""
    losses = np.array([0.5, 1.0, 2.0, 8.0])
    sampler = AdaptiveWindowSampler(4, mix=0.7, beta=1.0)
    sampler.update([0, 1, 2, 3], losses)
    p = sampler.probabilities()
    weights = 1.0 / (4 * p)
    assert (p * weights * losses).sum() == pytest.approx(losses.mean())


def test_zero_beta_drops_the_correction() -> None:
    sampler = AdaptiveWindowSampler(4, mix=0.5, beta=0.0)
    sampler.update([0, 1, 2, 3], np.array([1.0, 2.0, 3.0, 4.0]))
    _, weights = sampler.draw(8)
    np.testing.assert_allclose(weights, 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_sampler": "nope"},
        {"sampler_mix": 1.0},
        {"sampler_beta": 1.5},
        {"sampler_power": 0.0},
        {"recency_halflife": 0.0},
    ],
)
def test_config_rejects_bad_sampler_settings(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_negative_power_favours_easy_windows() -> None:
    sampler = AdaptiveWindowSampler(4, mix=0.5, power=-1.0)
    sampler.update([0, 1, 2, 3], np.array([1.0, 1.0, 1.0, 5.0]))
    p = sampler.probabilities()
    assert p[3] < p[0]
    assert p.min() >= 0.5 / 4


def test_relative_clip_stops_a_near_zero_loss_taking_over() -> None:
    sampler = AdaptiveWindowSampler(4, mix=0.9, power=-1.0)
    sampler.update([0, 1, 2, 3], np.array([1e-9, 1.0, 1.0, 1.0]))
    assert sampler.probabilities().max() < 0.9


def test_recency_weight_halves_every_halflife() -> None:
    q = RecencyWindowSampler([0, 1, 2, 3, 4], halflife=0.25).preference()
    assert np.all(np.diff(q) > 0)
    assert q[3] / q[4] == pytest.approx(0.5)


def test_recency_ignores_window_order() -> None:
    q = RecencyWindowSampler([4, 0, 2]).preference()
    assert q.argmax() == 0 and q.argmin() == 1


@pytest.mark.parametrize(
    "sampler",
    [
        {"window_sampler": "adaptive"},
        {"window_sampler": "adaptive", "sampler_power": -1.0, "sampler_beta": 0.0},
        {"window_sampler": "recency", "sampler_beta": 0.0},
    ],
)
def test_non_uniform_samplers_train_end_to_end(sampler: dict) -> None:
    config = quick("tourism_small")
    config = replace(config, train=replace(config.train, **sampler))
    result = run(config, seed=0, verbose=False)
    assert all(np.isfinite(v) and v >= 0 for v in result.scrps.values())
    assert result.history.steps_run == QUICK["max_steps"]


def test_adaptive_draws_happen_one_step_at_a_time() -> None:
    """Each draw must see the losses recorded by earlier steps of the same pass."""

    class CountingSampler:
        calls = 0

        def draw(self, k: int):
            CountingSampler.calls += 1
            return [0] * k, np.ones(k)

    trainer = Trainer.__new__(Trainer)
    trainer.config = TrainConfig(window_sampler="adaptive")
    trainer.sampler = CountingSampler()
    batches = trainer._epoch(100)
    next(batches)
    assert CountingSampler.calls == 1
