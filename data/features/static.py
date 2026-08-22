from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from data.dataset import HierarchicalDataset
from data.features.base import FeatureBlock, register_feature


@register_feature("grouped_static_dummies")
def grouped_static_dummies(
    dataset: HierarchicalDataset,
    keys: Sequence[str],
    sep: str = "-",
    positions: Sequence[int] | None = None,
) -> FeatureBlock:
    """One-hot each grouping key parsed out of the bottom-series names.

    Categories are read off the data, so the block stays correct if the dataset is
    regenerated with different levels.
    """
    positions = list(range(len(keys))) if positions is None else list(positions)
    if len(positions) != len(keys):
        raise ValueError(f"got {len(keys)} keys but {len(positions)} positions")
    parts = [name.split(sep) for name in dataset.bottom_names]
    needed = max(positions) + 1
    short = [
        n for n, p in zip(dataset.bottom_names, parts, strict=True) if len(p) < needed
    ]
    if short:
        raise ValueError(
            f"bottom-series names need at least {needed} {sep!r}-separated fields; "
            f"e.g. {short[0]!r} has {len(short[0].split(sep))}"
        )

    blocks, names = [], []
    for key, position in zip(keys, positions, strict=True):
        levels = sorted({p[position] for p in parts})
        index = {level: i for i, level in enumerate(levels)}
        one_hot = np.zeros((len(parts), len(levels)), dtype=np.float32)
        for row, p in enumerate(parts):
            one_hot[row, index[p[position]]] = 1.0
        blocks.append(one_hot)
        names += [f"{key}_{level}" for level in levels]
    return FeatureBlock(kind="static", values=np.concatenate(blocks, axis=1), names=names)


@register_feature("identity_dummies")
def identity_dummies(dataset: HierarchicalDataset) -> FeatureBlock:
    """[Nb, Nb] per-series identity embedding, for panels with no usable node names."""
    n_bottom = dataset.n_bottom
    return FeatureBlock(
        kind="static",
        values=np.eye(n_bottom, dtype=np.float32),
        names=[f"node_{i}" for i in range(n_bottom)],
    )
