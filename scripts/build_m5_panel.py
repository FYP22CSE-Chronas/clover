"""Wrangle the raw M5 CSVs into a balanced item x store x date panel.

    python scripts/build_m5_panel.py

Reads data/raw/m5/*.csv -- the raw Kaggle "M5 Forecasting - Accuracy" files
(`sales_train_evaluation.csv` or `sales_train_validation.csv`, `sell_prices.csv`,
`calendar.csv`) -- and writes memmap-friendly arrays to data/raw/m5/panel/:

    sales.npy   [n_items, T, 10] float32   daily unit sales; the Kaggle wide file is
                                            already dense and zero-filled pre-launch
    price.npy   [n_items, T, 10] float32   sell price, ffill/bfill within each
                                            item-store series, 0 where never sold
    snap.npy    [T, 10]          float32   SNAP eligibility, broadcast from each
                                            store's state to every store in it
    meta.npz                               S, names, dates, dept/state dummies

`sales_train_evaluation.csv` has 1,941 days of real sales (d_1..d_1941) and is
preferred when present; `sales_train_validation.csv` (d_1..d_1913) is the fallback.
Both are already a dense item x store grid, unlike Favorita's sparse transaction
log, so no scatter-fill pass is needed for sales itself.

The bottom-level store geography is M5's own [national; 3 states; 10 stores]
hierarchy -- the same style of S as `build_favorita_panel.py`'s
[national; states; cities; stores], just without a city level.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "m5")
OUT_DIR = os.path.join(RAW_DIR, "panel")

SALES_FILES = ("sales_train_evaluation.csv", "sales_train_validation.csv")
ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def raw(name: str) -> str:
    return os.path.join(RAW_DIR, name)


def out(name: str) -> str:
    return os.path.join(OUT_DIR, name)


def find_sales_file() -> str:
    for name in SALES_FILES:
        if os.path.exists(raw(name)):
            return name
    raise FileNotFoundError(
        f"{RAW_DIR}: found neither {SALES_FILES[0]} nor {SALES_FILES[1]}. "
        "Download the M5 Forecasting - Accuracy CSVs from Kaggle first."
    )


def store_hierarchy(stores: list[str], store_state: dict[str, str]) -> tuple[
    np.ndarray, list[str], dict[str, tuple[int, int]], list[str]
]:
    """S = [national; states; I_10], the 3-level store geography."""
    n_stores = len(stores)
    states = sorted({store_state[s] for s in stores})
    n_agg = 1 + len(states)

    S = np.zeros((n_agg + n_stores, n_stores), dtype=np.float32)
    S[0, :] = 1.0
    for i, state in enumerate(states):
        S[1 + i, [store_state[s] == state for s in stores]] = 1.0
    S[n_agg:, :] = np.eye(n_stores, dtype=np.float32)

    names = (
        ["National"]
        + [f"state_[{s}]" for s in states]
        + [f"store_[{s}]" for s in stores]
    )
    levels = {
        "1 (geo.)": (0, 1),
        "2 (geo.)": (1, n_agg),
        "3 (geo.)": (n_agg, n_agg + n_stores),
    }
    return S, names, levels, states


def load_sales() -> tuple[pd.DataFrame, int]:
    """The wide sales file, plus the day count implied by its d_* columns."""
    sales_name = find_sales_file()
    print(f"reading {sales_name}")
    sales = pd.read_csv(raw(sales_name))
    n_dates = sum(1 for c in sales.columns if c.startswith("d_"))
    print(f"  {len(sales):,} item-store rows, {n_dates} days")
    return sales, n_dates


def build_sales_panel(
    sales: pd.DataFrame, n_dates: int, item_pos: dict[str, int], store_pos: dict[str, int]
) -> np.ndarray:
    """[n_items, T, n_stores] float32, scattered from the dense wide rows."""
    n_items, n_stores = len(item_pos), len(store_pos)
    day_cols = [f"d_{i}" for i in range(1, n_dates + 1)]
    panel = np.lib.format.open_memmap(
        out("sales.npy"), mode="w+", dtype=np.float32, shape=(n_items, n_dates, n_stores)
    )
    item_idx = sales["item_id"].map(item_pos).to_numpy()
    store_idx = sales["store_id"].map(store_pos).to_numpy()
    panel[item_idx, :, store_idx] = sales[day_cols].to_numpy(dtype=np.float32)
    panel.flush()
    return panel


def build_price_panel(
    n_dates: int,
    dates: pd.DatetimeIndex,
    item_pos: dict[str, int],
    store_pos: dict[str, int],
) -> None:
    """[n_items, T, n_stores] float32 sell price, daily via each date's wm_yr_wk."""
    print("reading sell_prices.csv")
    prices = pd.read_csv(raw("sell_prices.csv"))
    calendar = pd.read_csv(raw("calendar.csv"), parse_dates=["date"])
    week_of_date = calendar.set_index("date")["wm_yr_wk"].reindex(dates).to_numpy()

    n_items, n_stores = len(item_pos), len(store_pos)
    weekly = np.full(
        (n_items, prices["wm_yr_wk"].nunique(), n_stores), np.nan, dtype=np.float32
    )
    weeks = np.sort(prices["wm_yr_wk"].unique())
    week_pos = {w: i for i, w in enumerate(weeks)}

    item_idx = prices["item_id"].map(item_pos).to_numpy()
    store_idx = prices["store_id"].map(store_pos).to_numpy()
    week_idx = prices["wm_yr_wk"].map(week_pos).to_numpy()
    good = ~pd.isna(item_idx) & ~pd.isna(store_idx)
    weekly[
        item_idx[good].astype(np.int64),
        week_idx[good].astype(np.int64),
        store_idx[good].astype(np.int64),
    ] = prices["sell_price"].to_numpy(dtype=np.float32)[good]

    daily_week_idx = np.array([week_pos.get(w, -1) for w in week_of_date])
    daily = np.where(
        (daily_week_idx >= 0)[None, :, None],
        weekly[:, np.clip(daily_week_idx, 0, None), :],
        np.nan,
    ).astype(np.float32)

    price = np.lib.format.open_memmap(
        out("price.npy"), mode="w+", dtype=np.float32, shape=(n_items, n_dates, n_stores)
    )
    for start in range(0, n_items, 200):
        block = daily[start : start + 200]  # [b, T, S]
        block = pd.DataFrame(block.reshape(block.shape[0], -1)).ffill(axis=1).to_numpy()
        block = block.reshape(-1, n_dates, n_stores)
        block = (
            pd.DataFrame(block.reshape(block.shape[0], -1))
            .bfill(axis=1)
            .fillna(0.0)
            .to_numpy()
        )
        price[start : start + 200] = block.reshape(-1, n_dates, n_stores).astype(
            np.float32
        )
    price.flush()


def build_snap(
    calendar: pd.DataFrame, n_dates: int, stores: list[str], store_state: dict[str, str]
) -> np.ndarray:
    """[T, n_stores] float32, each store's SNAP flag taken from its own state."""
    cal = calendar.iloc[:n_dates]
    snap_by_state = {
        "CA": cal["snap_CA"].to_numpy(),
        "TX": cal["snap_TX"].to_numpy(),
        "WI": cal["snap_WI"].to_numpy(),
    }
    return np.stack(
        [snap_by_state[store_state[s]] for s in stores], axis=1
    ).astype(np.float32)


def main() -> None:
    started = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    sales, n_dates = load_sales()
    items = sales[["item_id", "dept_id"]].drop_duplicates().sort_values("item_id")
    item_ids = items["item_id"].tolist()
    item_pos = {item: i for i, item in enumerate(item_ids)}

    store_state = dict(
        sales[["store_id", "state_id"]].drop_duplicates().itertuples(index=False)
    )
    stores = sorted(store_state)
    store_pos = {store: i for i, store in enumerate(stores)}
    print(f"catalog: {len(item_ids)} items x {len(stores)} stores x {n_dates} dates")

    S, series_names, levels, states = store_hierarchy(stores, store_state)

    calendar = pd.read_csv(raw("calendar.csv"), parse_dates=["date"])
    dates = pd.DatetimeIndex(calendar["date"].iloc[:n_dates])

    build_sales_panel(sales, n_dates, item_pos, store_pos)
    print(f"  sales panel written  {time.time() - started:.0f}s")
    build_price_panel(n_dates, dates, item_pos, store_pos)
    print(f"  price panel written  {time.time() - started:.0f}s")
    snap = build_snap(calendar, n_dates, stores, store_state)

    dept_names = sorted(items["dept_id"].unique())
    dept_dummies = np.zeros((len(item_ids), len(dept_names)), dtype=np.float32)
    dept_of_item = dict(zip(items["item_id"], items["dept_id"], strict=True))
    for i, item in enumerate(item_ids):
        dept_dummies[i, dept_names.index(dept_of_item[item])] = 1.0

    state_dummies = np.zeros((len(stores), len(states)), dtype=np.float32)
    for i, store in enumerate(stores):
        state_dummies[i, states.index(store_state[store])] = 1.0

    np.save(out("snap.npy"), snap)
    np.savez(
        out("meta.npz"),
        S=S,
        series_names=np.array(series_names),
        level_names=np.array(list(levels)),
        level_bounds=np.array([levels[k] for k in levels]),
        dates=dates.to_numpy(),
        item_ids=np.array(item_ids),
        store_ids=np.array(stores),
        dept_dummies=dept_dummies,
        dept_names=np.array(dept_names),
        state_dummies=state_dummies,
        state_names=np.array(states),
    )
    print(
        f"\nwrote sales {len(item_ids)}x{n_dates}x{len(stores)}, "
        f"bottom series {len(item_ids) * len(stores):,}, "
        f"total series {len(item_ids) * S.shape[0]:,}"
    )
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
