from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HierarchicalDataset:
    """A loaded hierarchy in raw units, with the aggregation matrix that defines it."""

    values: np.ndarray  # [T, Na+Nb]
    bottom: np.ndarray  # [T, Nb]
    S: np.ndarray  # [Na+Nb, Nb], S = [A; I]
    dates: pd.DatetimeIndex
    series_names: list[str]
    bottom_names: list[str]
    level_slices: dict[str, tuple[int, int]]

    @property
    def n_bottom(self) -> int:
        return self.bottom.shape[1]

    @property
    def n_series(self) -> int:
        return self.values.shape[1]

    @property
    def n_steps(self) -> int:
        return self.values.shape[0]


@dataclass(frozen=True)
class ExogenousFeatures:
    """Feature blocks built for a dataset; per-series futures stay in raw units."""

    static: np.ndarray | None = None  # [Nb, S]
    future_shared: np.ndarray | None = None  # [T, C1]
    future_series: np.ndarray | None = None  # [T, Nb, C2], raw units
    static_names: list[str] = field(default_factory=list)
    future_names: list[str] = field(default_factory=list)

    @property
    def n_static(self) -> int:
        return 0 if self.static is None else self.static.shape[1]

    @property
    def n_future_shared(self) -> int:
        """Calendar-style future channels, identical across series."""
        return 0 if self.future_shared is None else self.future_shared.shape[1]

    @property
    def n_future_series(self) -> int:
        """Per-series future channels; these are re-normalized per window."""
        return 0 if self.future_series is None else self.future_series.shape[2]

    @property
    def n_future(self) -> int:
        return self.n_future_shared + self.n_future_series
