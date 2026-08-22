from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelDataset:
    """Many independent hierarchies over a shared calendar and a shared `S`.

    The single-hierarchy datasets are one hierarchy over one time index, which
    `HierarchicalDataset` models. Favorita is not: it is 4,036 grocery items, each with
    its own copy of the same 93-row store geography, so the natural shape is
    `[n_groups, T, Nb]`. `bottom` is usually a memmap -- the Favorita panel is 1.5 GB
    and is never materialized in RAM.
    """

    bottom: np.ndarray  # [n_groups, T, Nb], raw units
    S: np.ndarray  # [Na+Nb, Nb], S = [A; I]
    dates: pd.DatetimeIndex
    series_names: list[str]
    level_slices: dict[str, tuple[int, int]]
    group_names: np.ndarray
    static_group: np.ndarray | None = None  # [n_groups, Sg]
    static_series: np.ndarray | None = None  # [Nb, Ss]
    future_shared: np.ndarray | None = None  # [T, C1]
    future_panel: np.ndarray | None = None  # [n_groups, T, Nb], raw units
    hist_series: np.ndarray | None = None  # [T, Nb, X]
    static_names: list[str] = field(default_factory=list)
    future_names: list[str] = field(default_factory=list)
    hist_names: list[str] = field(default_factory=list)

    @property
    def n_groups(self) -> int:
        """Independent hierarchies (Favorita: items)."""
        return self.bottom.shape[0]

    @property
    def n_steps(self) -> int:
        return self.bottom.shape[1]

    @property
    def n_bottom(self) -> int:
        """Bottom series per hierarchy (Favorita: stores)."""
        return self.bottom.shape[2]

    @property
    def n_series(self) -> int:
        """Rows of `S`: series per hierarchy, aggregates included."""
        return self.S.shape[0]

    @property
    def n_static(self) -> int:
        group = 0 if self.static_group is None else self.static_group.shape[1]
        series = 0 if self.static_series is None else self.static_series.shape[1]
        return group + series

    @property
    def n_future(self) -> int:
        shared = 0 if self.future_shared is None else self.future_shared.shape[1]
        return shared + (0 if self.future_panel is None else 1)

    @property
    def n_hist(self) -> int:
        return 0 if self.hist_series is None else self.hist_series.shape[2]

    def validate(self) -> None:
        """Check the shapes agree before any of them reaches a tensor."""
        n_agg = self.n_series - self.n_bottom
        if n_agg < 0:
            raise ValueError(f"S has {self.n_series} rows, fewer than {self.n_bottom}")
        if not np.allclose(self.S[n_agg:], np.eye(self.n_bottom, dtype=self.S.dtype)):
            raise ValueError("the last Nb rows of S are not an identity block")
        if len(self.dates) != self.n_steps:
            raise ValueError(f"{len(self.dates)} dates for {self.n_steps} steps")
        if len(self.series_names) != self.n_series:
            raise ValueError("series_names does not match the rows of S")
        for name, (start, stop) in self.level_slices.items():
            if not 0 <= start < stop <= self.n_series:
                raise ValueError(f"level {name!r} slice {(start, stop)} out of range")
        flat = [
            i for start, stop in self.level_slices.values() for i in range(start, stop)
        ]
        if sorted(flat) != list(range(self.n_series)):
            raise ValueError("level slices must partition every row of S exactly once")
        if self.future_panel is not None and self.future_panel.shape != self.bottom.shape:
            raise ValueError("future_panel must match bottom's shape")


@dataclass(frozen=True)
class PanelSplit:
    """Train/validation/test windows as `(group, t)` index pairs.

    Favorita's training block alone is ~6M windows, so the pairs are held as two int32
    arrays rather than as objects.
    """

    train: tuple[np.ndarray, np.ndarray]
    val: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": int(len(self.train[0])),
            "val": int(len(self.val[0])),
            "test": int(len(self.test[0])),
        }


def build_panel_split(
    dataset: PanelDataset, input_size: int, h: int, n_val: int, n_test: int
) -> PanelSplit:
    """The last `n_test` steps are test, the `n_val` before them validation.

    Every group contributes the same forecast-creation dates, so each block is the
    cartesian product of the group index with a range of `t`.
    """
    total = dataset.n_steps
    test_start = total - n_test
    val_start = test_start - n_val
    if val_start - input_size < 1:
        raise ValueError(
            f"{total} steps is too short for input_size {input_size} plus "
            f"{n_val} validation and {n_test} test steps"
        )

    groups = np.arange(dataset.n_groups, dtype=np.int32)

    def block(t_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g = np.repeat(groups, len(t_values))
        t = np.tile(t_values.astype(np.int32), len(groups))
        return g, t

    train_t = np.arange(input_size, val_start - h + 1)
    val_t = np.arange(val_start, test_start - h + 1)
    test_t = np.array([test_start])
    if len(train_t) == 0 or len(val_t) == 0:
        raise ValueError("panel split produced an empty train or validation block")
    return PanelSplit(train=block(train_t), val=block(val_t), test=block(test_t))
