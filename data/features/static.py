from __future__ import annotations

import ast
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


@register_feature("list_static_dummies")
def list_static_dummies(
    dataset: HierarchicalDataset, keys: Sequence[str], positions: Sequence[int]
) -> FeatureBlock:
    """One-hot each grouping key parsed out of list-shaped bottom-series names.

    Several raw hierarchies name a bottom series with a Python list literal, e.g.
    `"['Colombo', 'Western Province']"`. `grouped_static_dummies` splits on a
    separator and cannot read these, so this builder evaluates the literal and
    indexes it. Output shape and naming mirror `grouped_static_dummies`.
    """
    keys = list(keys)
    positions = list(positions)
    if len(positions) != len(keys):
        raise ValueError(f"got {len(keys)} keys but {len(positions)} positions")
    parts = _parse_name_lists(dataset.bottom_names)
    needed = max(positions) + 1
    short = [
        (name, p) for name, p in zip(dataset.bottom_names, parts, strict=True)
        if len(p) < needed
    ]
    if short:
        name, p = short[0]
        raise ValueError(
            f"bottom-series names need at least {needed} fields; "
            f"e.g. {name!r} has {len(p)}"
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


def _parse_name_lists(names: Sequence[str]) -> list[list[str]]:
    """Parse every bottom-series name as a list literal, with a readable error."""
    out: list[list[str]] = []
    for name in names:
        try:
            parsed = ast.literal_eval(name)
        except (ValueError, SyntaxError):
            raise ValueError(
                f"bottom-series name {name!r} is not a Python list literal; "
                "list_static_dummies needs names like \"['Colombo', 'Western']\""
            ) from None
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(
                f"bottom-series name {name!r} parses to {type(parsed).__name__}, "
                "not a list"
            )
        out.append([str(p) for p in parsed])
    return out


@register_feature("identity_dummies")
def identity_dummies(dataset: HierarchicalDataset) -> FeatureBlock:
    """[Nb, Nb] per-series identity embedding, for panels with no usable node names."""
    n_bottom = dataset.n_bottom
    return FeatureBlock(
        kind="static",
        values=np.eye(n_bottom, dtype=np.float32),
        names=[f"node_{i}" for i in range(n_bottom)],
    )
