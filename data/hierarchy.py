from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def validate_level_slices(
    level_slices: Mapping[str, tuple[int, int]], n_series: int
) -> None:
    """Check the slices are contiguous, ordered and cover every hierarchy row."""
    if not level_slices:
        raise ValueError("level_slices is empty; a dataset must declare its levels")
    cursor = 0
    for name, (start, stop) in level_slices.items():
        if (start, stop) != (cursor, max(stop, cursor)) or stop <= start:
            raise ValueError(
                f"level {name!r} spans [{start}, {stop}) but the previous level "
                f"ended at {cursor}; levels must partition [0, {n_series})"
            )
        cursor = stop
    if cursor != n_series:
        raise ValueError(
            f"level slices cover {cursor} rows but the hierarchy has {n_series}"
        )


def level_masks(
    level_slices: Mapping[str, tuple[int, int]], n_series: int
) -> dict[str, np.ndarray]:
    """Boolean [n_series] selector per level, for scoring one level at a time."""
    masks = {}
    for name, (start, stop) in level_slices.items():
        mask = np.zeros(n_series, dtype=bool)
        mask[start:stop] = True
        masks[name] = mask
    return masks


def level_tags(
    level_slices: Mapping[str, tuple[int, int]], series_names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Series names grouped by level, the form reconciliation libraries expect."""
    names = np.asarray(series_names)
    return {level: names[start:stop] for level, (start, stop) in level_slices.items()}


def levels_from_name_depth(
    series_names: Sequence[str], sep: str
) -> dict[str, tuple[int, int]]:
    """Derive level slices from separator-counted name depth; needs contiguous runs."""
    depths = [name.count(sep) for name in series_names]
    slices: dict[str, tuple[int, int]] = {}
    start = 0
    for i in range(1, len(depths) + 1):
        if i == len(depths) or depths[i] != depths[start]:
            name = f"level_{depths[start]}"
            if name in slices:
                raise ValueError(
                    f"depth {depths[start]} appears in more than one run of columns; "
                    "declare level slices explicitly in the dataset config"
                )
            slices[name] = (start, i)
            start = i
    return slices
