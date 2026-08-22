from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from config import FeatureSpec
from data.dataset import HierarchicalDataset
from data.features.base import FeatureBlock, register_feature

ANCHOR_FEATURE = "seasonal_naive_anchor"


@register_feature(ANCHOR_FEATURE)
def seasonal_naive_anchor(
    dataset: HierarchicalDataset, lags: Sequence[int]
) -> FeatureBlock:
    """[T, Nb, len(lags)] lagged values, back-filled with each first observation.

    Raw units on purpose: the pipeline re-normalizes these channels with the window's
    own (loc, scale) so anchor and target share a scale.
    """
    if not lags:
        raise ValueError("seasonal_naive_anchor needs at least one lag")
    values = dataset.bottom
    n_steps = values.shape[0]
    out = np.zeros((n_steps, values.shape[1], len(lags)), dtype=np.float32)
    for k, lag in enumerate(lags):
        if lag < 1:
            raise ValueError(f"anchor lag must be >= 1, got {lag}")
        if lag >= n_steps:
            raise ValueError(f"anchor lag {lag} exceeds the series length {n_steps}")
        out[lag:, :, k] = values[: n_steps - lag]
        out[:lag, :, k] = values[0]  # back-fill, never zero-fill
    return FeatureBlock(
        kind="future_series", values=out, names=[f"anchor_lag{lag}" for lag in lags]
    )


def check_anchor_lags(future_specs: Iterable[FeatureSpec], h: int) -> None:
    """Reject anchors whose lag is shorter than the horizon.

    Over the horizon the model sees `values[t-lag : t+h-lag]`, so `lag < h` feeds it
    `h - lag` true future observations.
    """
    for spec in future_specs:
        if spec.type != ANCHOR_FEATURE:
            continue
        bad = sorted(lag for lag in spec.params.get("lags", ()) if lag < h)
        if bad:
            raise ValueError(
                f"seasonal_naive_anchor lag(s) {bad} are shorter than the horizon "
                f"h={h}, leaking {h - min(bad)} true target step(s); use lags >= {h}"
            )
