from __future__ import annotations

import numpy as np


def estimate_beta_johansen(
    train_bottom: np.ndarray, rank: int, det_order: int = 0, k_ar_diff: int = 1
) -> np.ndarray:
    """Johansen cointegration analysis on bottom-level history, returning beta [Nb, rank].

    `train_bottom` must be the TRAINING-split bottom-level series only, `[T, Nb]` in
    raw units -- see IMPLEMENTATION.md's leakage rule and `experiments/ec_clover/
    NOTES.md` §17: validation/test rows must never influence beta's initialization.
    Callers are responsible for slicing the training rows before calling this (e.g.
    `scripts/estimate_beta.py`, or `dataset.bottom[: split.train_end]`).

    Requires `statsmodels`, an optional dependency of this experiment only (see
    `pyproject.toml`'s `ec_clover` extra) -- the core model and the random-init path
    never need it.
    """
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
    except ImportError as exc:
        raise ImportError(
            "estimate_beta_johansen requires statsmodels; install it with "
            "`pip install statsmodels` or `pip install -e .[ec_clover]`"
        ) from exc

    if train_bottom.ndim != 2:
        raise ValueError(f"train_bottom must be [T, Nb], got shape {train_bottom.shape}")
    t, n_bottom = train_bottom.shape
    if not (1 <= rank <= n_bottom):
        raise ValueError(f"rank must be in [1, {n_bottom}], got {rank}")

    result = coint_johansen(train_bottom, det_order, k_ar_diff)
    beta = result.evec[:, :rank]
    return np.ascontiguousarray(beta, dtype=np.float32)


def to_normalized_units(beta: np.ndarray, series_scale: np.ndarray) -> np.ndarray:
    """Convert a raw-level beta into the per-window-normalized units the model sees.

    Johansen runs on raw levels, but `CLOVER` hands the model
    `y_norm = (y - mu_w) / sigma_w`, so the branch computes
    `sum_i beta_i * (y_i - mu_i) / sigma_i`. Reproducing the raw combination
    `sum_i beta_raw_i * y_i` (up to a per-window constant) therefore needs the stored
    vector to be `beta_raw_i * sigma_i`, not `beta_raw_i`.

    Skipping this rescaling is not a harmless approximation: series whose levels differ
    by orders of magnitude get their loadings distorted by the same factor, which is why
    an unscaled Johansen initialization measured *worse* than a random unit-norm one.

    `series_scale` must be estimated from TRAINING windows only (see
    `experiments/ec_clover/run.py:training_window_scale`).
    """
    if beta.ndim != 2:
        raise ValueError(f"beta must be [Nb, rank], got shape {beta.shape}")
    if series_scale.shape != (beta.shape[0],):
        raise ValueError(
            f"series_scale must be [Nb] = [{beta.shape[0]}], "
            f"got shape {series_scale.shape}"
        )
    return np.ascontiguousarray(beta * series_scale[:, None], dtype=np.float32)
