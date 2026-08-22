from __future__ import annotations

import numpy as np

from data.dataset import HierarchicalDataset
from data.features.base import FeatureBlock, register_feature


def _one_hot(codes: np.ndarray, n_levels: int, prefix: str) -> FeatureBlock:
    """Dense one-hot of zero-based codes, as a shared future block."""
    values = np.zeros((len(codes), n_levels), dtype=np.float32)
    values[np.arange(len(codes)), codes] = 1.0
    return FeatureBlock(
        kind="future_shared",
        values=values,
        names=[f"{prefix}_{i}" for i in range(n_levels)],
    )


@register_feature("quarter_dummies")
def quarter_dummies(dataset: HierarchicalDataset) -> FeatureBlock:
    """[T, 4] one-hot quarter of year."""
    return _one_hot(dataset.dates.quarter.to_numpy() - 1, 4, "quarter")


@register_feature("month_dummies")
def month_dummies(dataset: HierarchicalDataset) -> FeatureBlock:
    """[T, 12] one-hot month of year."""
    return _one_hot(dataset.dates.month.to_numpy() - 1, 12, "month")


@register_feature("dayofweek_dummies")
def dayofweek_dummies(dataset: HierarchicalDataset) -> FeatureBlock:
    """[T, 7] one-hot day of week, Monday first."""
    return _one_hot(dataset.dates.dayofweek.to_numpy(), 7, "dow")


@register_feature("weekend_indicator")
def weekend_indicator(dataset: HierarchicalDataset) -> FeatureBlock:
    """[T, 1] indicator for Saturday or Sunday."""
    dow = dataset.dates.dayofweek.to_numpy()
    values = np.isin(dow, (5, 6)).astype(np.float32)[:, None]
    return FeatureBlock(kind="future_shared", values=values, names=["is_weekend"])


@register_feature("saturday_proximity")
def saturday_proximity(
    dataset: HierarchicalDataset, normalize: bool = True
) -> FeatureBlock:
    """[T, 1] cyclic day distance to the nearest Saturday, 0 on Saturday itself."""
    dow = dataset.dates.dayofweek.to_numpy()
    raw = np.abs(dow - 5)
    distance = np.minimum(raw, 7 - raw).astype(np.float32)
    if normalize:
        distance = distance / 3.0
    return FeatureBlock(
        kind="future_shared", values=distance[:, None], names=["saturday_proximity"]
    )
