from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def quantile_grid(step: float = 0.01) -> np.ndarray:
    """Interior quantile grid, e.g. 0.01..0.99 for the default 1% grid."""
    return np.round(np.arange(step, 1.0, step), 10)


QUANTILES_1PCT = quantile_grid(0.01)


def crps_from_quantiles(
    y_true: np.ndarray,
    samples: np.ndarray,
    quantiles: Sequence[float] = QUANTILES_1PCT,
) -> np.ndarray:
    """Quantile-integral CRPS on a discrete grid; the default estimator.

    `y_true` is [n_series, h], `samples` [n_series, h, N]; returns [n_series, h].
    """
    q = np.asarray(quantiles, dtype=np.float64)
    q_hat = np.quantile(samples, q, axis=-1)  # [Q, n_series, h]
    delta = y_true[None, ...] - q_hat
    qc = q[:, None, None]
    pinball = np.maximum(qc * delta, (qc - 1.0) * delta)
    return 2.0 * pinball.mean(axis=0)


def crps_from_samples(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Monte Carlo CRPS matching the training objective; returns [n_series, h]."""
    n = samples.shape[-1]
    if n < 2:
        raise ValueError(f"need at least 2 samples, got {n}")
    term1 = np.abs(samples - y_true[..., None]).mean(axis=-1)
    ordered = np.sort(samples, axis=-1)
    k = np.arange(1, n + 1, dtype=np.float64)
    term2 = 2.0 * (ordered * (2.0 * k - n - 1.0)).sum(axis=-1) / (n * (n - 1))
    return term1 - 0.5 * term2


def per_location_crps(
    y_true: np.ndarray,
    samples: np.ndarray,
    estimator: str = "quantile",
    quantile_step: float = 0.01,
) -> np.ndarray:
    """Dispatch to the quantile-integral or Monte Carlo estimator."""
    if estimator == "quantile":
        return crps_from_quantiles(y_true, samples, quantile_grid(quantile_step))
    if estimator == "mc":
        return crps_from_samples(y_true, samples)
    raise ValueError(f"unknown estimator {estimator!r}; expected 'quantile' or 'mc'")


def scrps(
    y_true: np.ndarray,
    samples: np.ndarray,
    mask: np.ndarray | None = None,
    estimator: str = "quantile",
    quantile_step: float = 0.01,
) -> float:
    """Scaled CRPS: a level's total CRPS divided by its total absolute target mass."""
    per_loc = per_location_crps(y_true, samples, estimator, quantile_step)
    if mask is None:
        mask = np.ones(y_true.shape[0], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    denominator = np.abs(y_true[mask]).sum()
    if denominator == 0:
        raise ValueError("sCRPS denominator is zero: the level has no target mass")
    return float(per_loc[mask].sum() / denominator)


def scrps_by_level(
    y_true: np.ndarray,
    samples: np.ndarray,
    level_slices: Mapping[str, tuple[int, int]],
    estimator: str = "quantile",
    quantile_step: float = 0.01,
) -> dict[str, float]:
    """sCRPS per level plus "Overall", the whole-hierarchy ratio rather than a mean."""
    per_loc = per_location_crps(y_true, samples, estimator, quantile_step)
    n_series = y_true.shape[0]

    def ratio(mask: np.ndarray) -> float:
        denominator = np.abs(y_true[mask]).sum()
        if denominator == 0:
            raise ValueError("sCRPS denominator is zero: the level has no target mass")
        return float(per_loc[mask].sum() / denominator)

    out = {"Overall": ratio(np.ones(n_series, dtype=bool))}
    for level, (start, stop) in level_slices.items():
        mask = np.zeros(n_series, dtype=bool)
        mask[start:stop] = True
        out[level] = ratio(mask)
    return out
