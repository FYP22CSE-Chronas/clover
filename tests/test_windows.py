from __future__ import annotations

import numpy as np
import pytest

from config import SplitConfig
from data.windows import build_split, rolling_windows


@pytest.fixture
def values() -> np.ndarray:
    return np.arange(60 * 3, dtype=np.float32).reshape(60, 3)


def test_rolling_window_alignment(values: np.ndarray) -> None:
    windows = rolling_windows(values, input_size=8, h=4)
    assert len(windows) == 60 - 8 - 4 + 1
    first = windows[0]
    assert first.t == 8
    np.testing.assert_array_equal(first.insample, values[0:8])
    np.testing.assert_array_equal(first.target, values[8:12])


def test_rolling_windows_need_enough_rows() -> None:
    with pytest.raises(ValueError, match="need at least"):
        rolling_windows(np.zeros((5, 2)), input_size=8, h=4)


def test_last_h_split_blocks(values: np.ndarray) -> None:
    split = build_split(values, 8, 4, SplitConfig(strategy="last_h"))
    assert len(split.val) == 1 and len(split.test) == 1
    assert split.test[0].t == 56
    assert split.val[0].t == 52


def test_split_targets_are_disjoint(values: np.ndarray) -> None:
    split = build_split(values, 8, 4, SplitConfig(strategy="last_h"))
    targets = split.target_indices()
    assert not targets["train"] & targets["val"]
    assert not targets["train"] & targets["test"]
    assert not targets["val"] & targets["test"]


def test_training_never_sees_validation_targets(values: np.ndarray) -> None:
    split = build_split(values, 8, 4, SplitConfig(strategy="last_h"))
    latest_train_target = max(split.target_indices()["train"])
    assert latest_train_target < split.val[0].t


def test_block_split_row_counts() -> None:
    values = np.arange(267 * 2, dtype=np.float32).reshape(267, 2)
    config = SplitConfig(strategy="block", train=120, val=120, test=27)
    split = build_split(values, 56, 1, config)
    targets = split.target_indices()
    assert max(targets["train"]) < 120
    assert not targets["train"] & targets["val"]
    assert split.test[0].t == 266


def test_block_split_rejects_wrong_totals() -> None:
    values = np.zeros((100, 2), dtype=np.float32)
    config = SplitConfig(strategy="block", train=50, val=20, test=20)
    with pytest.raises(ValueError, match="block split needs"):
        build_split(values, 8, 4, config)


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown split strategy"):
        SplitConfig(strategy="nope")
