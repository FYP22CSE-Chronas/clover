from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import ModelConfig, ModelDims
from data.dataset import HierarchicalDataset
from model.network import CLOVER

N_BOTTOM = 4
N_AGG = 3  # total, then two groups of two


@pytest.fixture
def S() -> np.ndarray:
    """[7, 4] aggregation matrix: total, two pairs, then the identity block."""
    A = np.array(
        [[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
        dtype=np.float32,
    )
    return np.vstack([A, np.eye(N_BOTTOM, dtype=np.float32)])


@pytest.fixture
def level_slices() -> dict[str, tuple[int, int]]:
    return {"Total": (0, 1), "Group": (1, 3), "Bottom": (3, 7)}


@pytest.fixture
def dataset(S: np.ndarray, level_slices: dict) -> HierarchicalDataset:
    """A 60-step quarterly hierarchy whose aggregates are exactly coherent."""
    rng = np.random.default_rng(0)
    bottom = rng.gamma(shape=4.0, scale=10.0, size=(60, N_BOTTOM)).astype(np.float32)
    values = (bottom @ S.T).astype(np.float32)
    dates = pd.date_range("2000-01-01", periods=60, freq="QS")
    return HierarchicalDataset(
        values=values,
        bottom=bottom,
        S=S,
        dates=dates,
        series_names=["total", "g1", "g2", "a-x", "a-y", "b-x", "b-y"],
        bottom_names=["a-x", "a-y", "b-x", "b-y"],
        level_slices=level_slices,
    )


@pytest.fixture
def model(S: np.ndarray) -> CLOVER:
    dims = ModelDims(h=4, input_size=8)
    return CLOVER(ModelConfig(temp_conv_channels=6, n_factors=3), dims, S)
