from __future__ import annotations

import os

import numpy as np
import pandas as pd

from config import DataConfig
from data.panel import PanelDataset

PANEL_FILES = ("sales.npy", "promo.npy", "transactions.npy", "meta.npz")


def load_favorita(config: DataConfig) -> PanelDataset:
    """Open the wrangled Favorita panel: 4,036 item hierarchies over 54 stores.

    Reads the `.npy`/`.npz` files that `scripts/build_favorita_panel.py` writes from the
    raw Kaggle CSVs. `sales.npy` and `promo.npy` are opened as memmaps -- together they
    are ~1.9 GB and are never resident.

    Features: item perishability and store state dummies (static), past unit sales
    and store transactions (historical), promotions and day of week (future-known).
    """
    root = config.dir
    missing = [f for f in PANEL_FILES if not os.path.exists(os.path.join(root, f))]
    if missing:
        raise FileNotFoundError(
            f"{root}: missing {missing}. Build the panel from the raw Kaggle CSVs "
            "first: python scripts/build_favorita_panel.py"
        )

    def path(name: str) -> str:
        return os.path.join(root, name)

    meta = np.load(path("meta.npz"), allow_pickle=False)
    dates = pd.DatetimeIndex(meta["dates"])
    level_slices = {
        str(name): (int(lo), int(hi))
        for name, (lo, hi) in zip(meta["level_names"], meta["level_bounds"], strict=True)
    }
    perishable = meta["perishable"].astype(np.float32).reshape(-1, 1)
    state_dummies = meta["state_dummies"].astype(np.float32)

    dataset = PanelDataset(
        bottom=np.load(path("sales.npy"), mmap_mode="r"),
        S=meta["S"].astype(np.float32),
        dates=dates,
        series_names=[str(n) for n in meta["series_names"]],
        level_slices=level_slices,
        group_names=meta["item_nbrs"],
        static_group=perishable,
        static_series=state_dummies,
        future_shared=_dayofweek_dummies(dates),
        future_panel=np.load(path("promo.npy"), mmap_mode="r"),
        hist_series=np.load(path("transactions.npy"))[:, :, None].astype(np.float32),
        static_names=["perishable"]
        + [f"state_{i}" for i in range(state_dummies.shape[1])],
        future_names=["onpromotion"] + [f"dow_{i}" for i in range(7)],
        hist_names=["transactions"],
    )
    dataset.validate()
    return dataset


def _dayofweek_dummies(dates: pd.DatetimeIndex) -> np.ndarray:
    """[T, 7] one-hot day of week."""
    out = np.zeros((len(dates), 7), dtype=np.float32)
    out[np.arange(len(dates)), dates.dayofweek.to_numpy()] = 1.0
    return out
