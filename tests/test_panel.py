from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import ModelConfig, ModelDims
from data.panel import PanelDataset, build_panel_split
from model.network import CLOVER
from pipeline.panel_batching import PanelBatcher

N_GROUPS, T, N_BOTTOM = 6, 80, 4
L, H = 8, 4


@pytest.fixture
def panel(S: np.ndarray, level_slices: dict) -> PanelDataset:
    rng = np.random.default_rng(0)
    bottom = rng.gamma(3.0, 5.0, size=(N_GROUPS, T, N_BOTTOM)).astype(np.float32)
    dataset = PanelDataset(
        bottom=bottom,
        S=S,
        dates=pd.date_range("2015-01-01", periods=T, freq="D"),
        series_names=["total", "g1", "g2", "b0", "b1", "b2", "b3"],
        level_slices=level_slices,
        group_names=np.arange(N_GROUPS),
        static_group=rng.random((N_GROUPS, 1)).astype(np.float32),
        static_series=np.eye(N_BOTTOM, dtype=np.float32),
        future_shared=rng.random((T, 7)).astype(np.float32),
        future_panel=(bottom > 10).astype(np.float32),
        hist_series=rng.random((T, N_BOTTOM, 1)).astype(np.float32),
    )
    dataset.validate()
    return dataset


def test_shape_properties(panel: PanelDataset) -> None:
    assert panel.n_groups == N_GROUPS
    assert panel.n_steps == T
    assert panel.n_bottom == N_BOTTOM
    assert panel.n_series == 7
    assert panel.n_static == 1 + N_BOTTOM
    assert panel.n_future == 7 + 1
    assert panel.n_hist == 1


def test_validate_rejects_a_non_identity_tail(panel: PanelDataset) -> None:
    broken = np.array(panel.S)
    broken[-1, -1] = 2.0
    with pytest.raises(ValueError, match="identity block"):
        PanelDataset(**{**panel.__dict__, "S": broken}).validate()


def test_validate_rejects_a_mismatched_promo_panel(panel: PanelDataset) -> None:
    bad = np.zeros((N_GROUPS, T, N_BOTTOM + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="future_panel"):
        PanelDataset(**{**panel.__dict__, "future_panel": bad}).validate()


def test_split_is_the_cartesian_product_of_groups_and_dates(
    panel: PanelDataset,
) -> None:
    split = build_panel_split(panel, input_size=L, h=H, n_val=H, n_test=H)
    assert split.sizes["test"] == N_GROUPS
    assert set(split.train[0]) == set(range(N_GROUPS))


def test_split_blocks_do_not_overlap_in_time(panel: PanelDataset) -> None:
    split = build_panel_split(panel, input_size=L, h=H, n_val=H, n_test=H)

    def targets(pair) -> set[int]:
        return {t for start in pair[1] for t in range(start, start + H)}

    train, val, test = (targets(p) for p in (split.train, split.val, split.test))
    assert not train & val
    assert not train & test
    assert not val & test


def test_split_rejects_a_too_short_panel(panel: PanelDataset) -> None:
    with pytest.raises(ValueError, match="too short"):
        build_panel_split(panel, input_size=T, h=H, n_val=H, n_test=H)


def test_batcher_produces_model_ready_shapes(panel: PanelDataset) -> None:
    split = build_panel_split(panel, input_size=L, h=H, n_val=H, n_test=H)
    batcher = PanelBatcher(panel, split.train, input_size=L, h=H, scaler="mean_floor")
    batch = batcher.batch([0, 1, 2])
    windows = batch.windows
    assert windows.insample_y.shape == (3, L, N_BOTTOM)
    assert windows.futr_exog.shape == (3, panel.n_future, L + H, N_BOTTOM)
    assert windows.hist_exog.shape == (3, panel.n_hist, L, N_BOTTOM)
    assert windows.stat_exog.shape == (3, N_BOTTOM, panel.n_static)
    assert batch.target_bottom.shape == (3, H, N_BOTTOM)


def test_panel_batch_runs_through_the_model(panel: PanelDataset) -> None:
    split = build_panel_split(panel, input_size=L, h=H, n_val=H, n_test=H)
    batcher = PanelBatcher(panel, split.train, input_size=L, h=H, scaler="mean_floor")
    batch = batcher.batch([0, 1])
    dims = ModelDims(
        h=H,
        input_size=L,
        n_stat_features=panel.n_static,
        n_futr_features=panel.n_future,
        n_hist_features=panel.n_hist,
    )
    model = CLOVER(ModelConfig(temp_conv_channels=6, n_factors=3), dims, panel.S)
    assert model(batch.windows).mu.shape == (2, H, N_BOTTOM)


def test_static_features_vary_across_the_batch(panel: PanelDataset) -> None:
    """Item perishability differs per group; store geography does not."""
    split = build_panel_split(panel, input_size=L, h=H, n_val=H, n_test=H)
    batcher = PanelBatcher(panel, split.test, input_size=L, h=H)
    stat = batcher.batch([0, 1]).windows.stat_exog
    assert not np.allclose(stat[0, :, 0], stat[1, :, 0])
    np.testing.assert_allclose(stat[0, :, 1:], stat[1, :, 1:])
