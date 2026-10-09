from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from config import TrainConfig

# Bounds on a window's loss relative to the mean, so a near-zero loss under a
# negative power cannot claim the whole non-uniform share.
RELATIVE_CLIP = (0.1, 10.0)


class WindowSampler:
    """Non-uniform training-window sampling, mixed with uniform.

    Window `i` is drawn with probability `p_i = (1 - mix) / n + mix * q_i`, where `q`
    is the subclass's preference distribution. Mixing in the uniform term keeps every
    window reachable and bounds the importance weight `(1 / (n p_i)) ** beta` by
    `(1 / (1 - mix)) ** beta`. With `beta = 1` the weighted loss is an unbiased
    estimate of the uniform-sampling objective, so only gradient variance changes;
    `beta = 0` drops the correction and shifts the objective toward preferred windows.
    """

    def __init__(self, n_windows: int, mix: float = 0.5, beta: float = 1.0) -> None:
        if n_windows < 1:
            raise ValueError("n_windows must be >= 1")
        if not 0.0 <= mix < 1.0:
            raise ValueError("mix must lie in [0, 1)")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must lie in [0, 1]")
        self.n = n_windows
        self.mix = mix
        self.beta = beta

    def preference(self) -> np.ndarray | None:
        """The non-uniform share `q`, summing to 1, or None for uniform."""
        raise NotImplementedError

    def probabilities(self) -> np.ndarray:
        """Current draw probability of every window."""
        q = self.preference()
        if q is None:
            return np.full(self.n, 1.0 / self.n)
        return (1.0 - self.mix) / self.n + self.mix * q

    def draw(self, k: int) -> tuple[list[int], np.ndarray]:
        """`k` window indices, with replacement, and their importance weights."""
        p = self.probabilities()
        idx = np.random.choice(self.n, size=k, replace=True, p=p)
        weights = (1.0 / (self.n * p[idx])) ** self.beta
        return idx.tolist(), weights

    def update(self, indices: list[int], losses: np.ndarray) -> None:
        """Record the latest per-window losses; a no-op for fixed preferences."""


class AdaptiveWindowSampler(WindowSampler):
    """Prefers windows by their latest training loss raised to `power`.

    `power > 0` favours hard windows (prioritized replay); `power < 0` favours easy
    ones, the self-paced reading that treats persistently high-loss windows as
    anomalies. Losses are taken relative to the mean of those seen and clipped to
    `RELATIVE_CLIP` before the power; a window not yet seen scores the mean.
    """

    def __init__(
        self, n_windows: int, mix: float = 0.5, beta: float = 1.0, power: float = 1.0
    ) -> None:
        super().__init__(n_windows, mix, beta)
        if power == 0:
            raise ValueError("power must be non-zero")
        self.power = power
        self.scores = np.full(n_windows, np.nan)

    def preference(self) -> np.ndarray | None:
        seen = ~np.isnan(self.scores)
        if not seen.any():
            return None
        mean = self.scores[seen].mean()
        if not np.isfinite(mean) or mean <= 0:
            return None
        relative = np.where(seen, self.scores / mean, 1.0)
        q = np.clip(relative, *RELATIVE_CLIP) ** self.power
        return q / q.sum()

    def update(self, indices: list[int], losses: np.ndarray) -> None:
        """Record the latest per-window losses; repeats keep the last value."""
        self.scores[np.asarray(indices)] = np.maximum(np.asarray(losses), 0.0)


class RecencyWindowSampler(WindowSampler):
    """Prefers recent windows: weight halves every `halflife` of the training span."""

    def __init__(
        self,
        window_times: Sequence[int],
        mix: float = 0.5,
        beta: float = 1.0,
        halflife: float = 0.25,
    ) -> None:
        times = np.asarray(window_times, dtype=np.float64)
        super().__init__(len(times), mix, beta)
        if halflife <= 0:
            raise ValueError("halflife must be > 0")
        span = max(times.max() - times.min(), 1.0)
        age = (times.max() - times) / span
        q = 0.5 ** (age / halflife)
        self._q = q / q.sum()

    def preference(self) -> np.ndarray:
        return self._q


def build_sampler(
    config: TrainConfig, window_times: Sequence[int]
) -> WindowSampler | None:
    """The sampler `config.window_sampler` names, or None for uniform shuffling."""
    if config.window_sampler == "adaptive":
        return AdaptiveWindowSampler(
            len(window_times), config.sampler_mix, config.sampler_beta,
            config.sampler_power,
        )
    if config.window_sampler == "recency":
        return RecencyWindowSampler(
            window_times, config.sampler_mix, config.sampler_beta,
            config.recency_halflife,
        )
    return None
