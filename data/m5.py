from __future__ import annotations

import os

import numpy as np
import pandas as pd

from config import DataConfig
from data.panel import PanelDataset

PANEL_FILES = ("sales.npy", "price.npy", "snap.npy", "meta.npz")


def load_m5(config: DataConfig) -> PanelDataset:
    """Open the wrangled M5 panel: 3,049 item hierarchies over 10 Walmart stores.

    Reads the `.npy`/`.npz` files that `scripts/build_m5_panel.py` writes from the raw
    Kaggle M5 Forecasting - Accuracy CSVs. `sales.npy` and `price.npy` are opened as
    memmaps.

    Features: department dummies and store-state dummies (static), sell price
    (future, per item+store -- prices are set in advance, so they are future-known),
    day of week (future, shared), SNAP eligibility (historical, per store -- it varies
    with the calendar, not with which item is being forecast).
    """
    root = config.dir
    missing = [f for f in PANEL_FILES if not os.path.exists(os.path.join(root, f))]
    if missing:
        raise FileNotFoundError(
            f"{root}: missing {missing}. Build the panel from the raw Kaggle CSVs "
            "first: python scripts/build_m5_panel.py"
        )

    def path(name: str) -> str:
        return os.path.join(root, name)

    meta = np.load(path("meta.npz"), allow_pickle=False)
    dates = pd.DatetimeIndex(meta["dates"])
    level_slices = {
        str(name): (int(lo), int(hi))
        for name, (lo, hi) in zip(meta["level_names"], meta["level_bounds"], strict=True)
    }
    dept_dummies = meta["dept_dummies"].astype(np.float32)
    state_dummies = meta["state_dummies"].astype(np.float32)

    dataset = PanelDataset(
        bottom=np.load(path("sales.npy"), mmap_mode="r"),
        S=meta["S"].astype(np.float32),
        dates=dates,
        series_names=[str(n) for n in meta["series_names"]],
        level_slices=level_slices,
        group_names=meta["item_ids"],
        static_group=dept_dummies,
        static_series=state_dummies,
        future_shared=_dayofweek_dummies(dates),
        future_panel=np.load(path("price.npy"), mmap_mode="r"),
        hist_series=np.load(path("snap.npy"))[:, :, None].astype(np.float32),
        static_names=[f"dept_[{d}]" for d in meta["dept_names"]]
        + [f"state_[{s}]" for s in meta["state_names"]],
        future_names=[f"dow_{i}" for i in range(7)] + ["sell_price"],
        hist_names=["snap"],
    )
    dataset.validate()
    return dataset


def _dayofweek_dummies(dates: pd.DatetimeIndex) -> np.ndarray:
    """[T, 7] one-hot day of week."""
    out = np.zeros((len(dates), 7), dtype=np.float32)
    out[np.arange(len(dates)), dates.dayofweek.to_numpy()] = 1.0
    return out
