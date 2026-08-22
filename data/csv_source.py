from __future__ import annotations

import os

import numpy as np
import pandas as pd

from config import DataConfig
from data.dataset import HierarchicalDataset
from data.hierarchy import validate_level_slices


def load_csv_hierarchy(config: DataConfig) -> HierarchicalDataset:
    """Read `<dir>/data.csv` and `<dir>/agg_mat.csv` into a `HierarchicalDataset`."""
    data = pd.read_csv(os.path.join(config.dir, "data.csv"), index_col=0)
    agg = pd.read_csv(os.path.join(config.dir, "agg_mat.csv"), index_col=0)
    if list(data.columns) != list(agg.index):
        raise ValueError(
            f"{config.dir}: data.csv columns must match agg_mat.csv rows exactly; "
            "that ordering is what makes S valid"
        )
    data.index = pd.to_datetime(data.index)
    if config.end_date is not None:
        data = data.loc[: config.end_date]

    series_names = list(data.columns)
    bottom_names = list(agg.columns)
    S = agg.to_numpy(dtype=np.float32)
    _check_identity_block(S, config.dir)

    levels = dict(config.levels)
    validate_level_slices(levels, len(series_names))
    return HierarchicalDataset(
        values=data.to_numpy(dtype=np.float32),
        bottom=data[bottom_names].to_numpy(dtype=np.float32),
        S=S,
        dates=pd.DatetimeIndex(data.index),
        series_names=series_names,
        bottom_names=bottom_names,
        level_slices=levels,
    )


def verify_coherence(dataset: HierarchicalDataset, atol: float = 1e-3) -> None:
    """Assert S @ bottom reproduces every aggregate column of the loaded data."""
    reconstructed = dataset.bottom @ dataset.S.T
    if not np.allclose(reconstructed, dataset.values, atol=atol, rtol=1e-4):
        worst = np.abs(reconstructed - dataset.values).max()
        raise ValueError(f"S @ bottom does not reproduce data.csv (max abs diff {worst})")


def _check_identity_block(S: np.ndarray, where: str) -> None:
    """S must be [A; I]: the trailing Nb rows are the bottom series themselves."""
    n_bottom = S.shape[1]
    if S.shape[0] < n_bottom:
        raise ValueError(f"{where}: S is {S.shape}, expected at least {n_bottom} rows")
    if not np.allclose(S[-n_bottom:], np.eye(n_bottom, dtype=S.dtype)):
        raise ValueError(
            f"{where}: the last {n_bottom} rows of S are not an identity block"
        )
