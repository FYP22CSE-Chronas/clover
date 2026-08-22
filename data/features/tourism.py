from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from data.dataset import HierarchicalDataset
from data.features.base import FeatureBlock, register_feature

PURPOSES = ("Hol", "Vis", "Bus", "Oth")


@register_feature("tourism_large_static_dummies")
def tourism_large_static_dummies(
    dataset: HierarchicalDataset, purposes: Sequence[str] = PURPOSES
) -> FeatureBlock:
    """[Nb, n_states + n_purposes] one-hot state and purpose-of-travel dummies.

    Tourism-L is the one bundled dataset whose bottom names need parsing rather than
    splitting: "AAAHol" is region "AAA" plus purpose "Hol", and the state is the first
    letter of the region code.
    """
    parsed = [_parse(name, purposes) for name in dataset.bottom_names]
    states = sorted({state for state, _ in parsed})
    present = [p for p in purposes if any(p == purpose for _, purpose in parsed)]
    state_index = {s: i for i, s in enumerate(states)}
    purpose_index = {p: i for i, p in enumerate(present)}

    values = np.zeros((len(parsed), len(states) + len(present)), dtype=np.float32)
    for row, (state, purpose) in enumerate(parsed):
        values[row, state_index[state]] = 1.0
        values[row, len(states) + purpose_index[purpose]] = 1.0
    names = [f"state_{s}" for s in states] + [f"purpose_{p}" for p in present]
    return FeatureBlock(kind="static", values=values, names=names)


def _parse(name: str, purposes: Sequence[str]) -> tuple[str, str]:
    """Split a bottom-series name into (state letter, purpose suffix)."""
    purpose = next((p for p in purposes if name.endswith(p)), None)
    if purpose is None:
        raise ValueError(f"{name!r} does not end in one of {tuple(purposes)}")
    geo = name[: -len(purpose)]
    if not geo:
        raise ValueError(f"{name!r} has no geographic code before its purpose suffix")
    return geo[0], purpose
