"""Wrangle the raw Favorita CSVs into a balanced item x store x date panel.

    python scripts/build_favorita_panel.py

Reads data/favorita/*.csv and writes memmap-friendly arrays to data/favorita/panel/,
so the loader never holds the 1.5 GB sales panel in RAM:

    sales.npy         [n_items, 1688, 54] float32   zero-filled (absent = no sale)
    promo.npy         [n_items, 1688, 54] int8      ffill/bfill then 0
    transactions.npy  [1688, 54]          float32   store footfall, zero-filled
    meta.npz                                        S, names, dates, static features

Fill conventions: absent sales rows mean no sale (zero-filled); onpromotion is
forward- then back-filled along the date axis.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "favorita")
OUT_DIR = os.path.join(RAW_DIR, "panel")

START, END = np.datetime64("2013-01-01"), np.datetime64("2017-08-15")
N_DATES = int((END - START).astype(int)) + 1  # 1688
N_STORES = 54
CHUNK_ROWS = 8_000_000
ITEM_BLOCK = 200


def raw(name: str) -> str:
    return os.path.join(RAW_DIR, name)


def out(name: str) -> str:
    return os.path.join(OUT_DIR, name)


def store_hierarchy(stores: pd.DataFrame) -> tuple[np.ndarray, list[str], dict]:
    """S = [national; states; cities; I_54], the 4-level store geography."""
    stores = stores.sort_values("store_nbr").reset_index(drop=True)
    states = sorted(stores["state"].unique())
    cities = sorted(stores["city"].unique())
    n_agg = 1 + len(states) + len(cities)

    S = np.zeros((n_agg + N_STORES, N_STORES), dtype=np.float32)
    S[0, :] = 1.0
    for i, state in enumerate(states):
        S[1 + i, (stores["state"] == state).to_numpy()] = 1.0
    for i, city in enumerate(cities):
        S[1 + len(states) + i, (stores["city"] == city).to_numpy()] = 1.0
    S[n_agg:, :] = np.eye(N_STORES, dtype=np.float32)

    names = (
        ["National"]
        + [f"state_[{s}]" for s in states]
        + [f"city_[{c}]" for c in cities]
        + [f"store_[{n}]" for n in stores["store_nbr"]]
    )
    levels = {
        "1 (geo.)": (0, 1),
        "2 (geo.)": (1, 1 + len(states)),
        "3 (geo.)": (1 + len(states), n_agg),
        "4 (geo.)": (n_agg, n_agg + N_STORES),
    }
    return S, names, levels


def open_train_reader() -> pv.CSVStreamingReader:
    """Stream train.csv with pinned column types."""
    return pv.open_csv(
        raw("train.csv"),
        convert_options=pv.ConvertOptions(
            include_columns=[
                "date",
                "store_nbr",
                "item_nbr",
                "unit_sales",
                "onpromotion",
            ],
            # onpromotion is empty until 2014-04 then "True"/"False"; pyarrow infers
            # null from the head of the file and then chokes, so pin it to bool.
            column_types={
                "date": "date32",
                "store_nbr": "int16",
                "item_nbr": "int32",
                "unit_sales": "float32",
                "onpromotion": pa.bool_(),
            },
        ),
        read_options=pv.ReadOptions(block_size=1 << 26),
    )


def fill_promotions(promo: np.ndarray, keep_items: np.ndarray) -> None:
    """Forward-fill then back-fill onpromotion along the date axis, per item block."""
    for start in range(0, len(keep_items), ITEM_BLOCK):
        block = keep_items[start : start + ITEM_BLOCK]
        p = promo[block].astype(np.int16)  # [b, D, 54]
        p = np.where(p < 0, -1, p)
        idx = np.where(p >= 0, np.arange(N_DATES)[None, :, None], 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        p = np.take_along_axis(p, idx, axis=1)
        idx = np.where(p >= 0, np.arange(N_DATES)[None, :, None], N_DATES - 1)
        idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
        p = np.take_along_axis(p, idx, axis=1)
        promo[block] = np.where(p < 0, 0, p).astype(np.int8)
    promo.flush()


def store_transactions(store_pos: np.ndarray) -> np.ndarray:
    """[1688, 54] store footfall, zero-filled onto the panel calendar."""
    trans = pd.read_csv(raw("transactions.csv"), parse_dates=["date"])
    matrix = np.zeros((N_DATES, N_STORES), dtype=np.float32)
    day = (trans["date"].to_numpy().astype("datetime64[D]") - START).astype(np.int64)
    keep = (day >= 0) & (day < N_DATES)
    matrix[day[keep], store_pos[trans["store_nbr"].to_numpy()[keep]]] = trans[
        "transactions"
    ].to_numpy()[keep]
    return matrix


def stream_train(
    sales: np.ndarray,
    promo: np.ndarray,
    item_pos: np.ndarray,
    store_pos: np.ndarray,
    started: float,
) -> np.ndarray:
    """Scatter train.csv into the raw panels; returns a mask of the items that occur."""
    seen = np.zeros(sales.shape[0], dtype=bool)
    rows = 0

    def scatter(frame: pd.DataFrame) -> None:
        nonlocal rows
        day = (frame["date"].to_numpy().astype("datetime64[D]") - START).astype(np.int64)
        keep = (day >= 0) & (day < N_DATES)
        if not keep.any():
            return
        day = day[keep]
        item = item_pos[frame["item_nbr"].to_numpy()[keep]]
        store = store_pos[frame["store_nbr"].to_numpy()[keep]]
        good = (item >= 0) & (store >= 0)
        day, item, store = day[good], item[good], store[good]
        flat = item.astype(np.int64) * (N_DATES * N_STORES) + day * N_STORES + store
        sales.reshape(-1)[flat] = frame["unit_sales"].to_numpy()[keep][good]
        promo.reshape(-1)[flat] = (
            frame["onpromotion"].to_numpy()[keep][good].astype(np.int8)
        )
        seen[np.unique(item)] = True
        rows += len(day)
        print(f"  {rows / 1e6:7.1f}M rows  {time.time() - started:6.0f}s", flush=True)

    pending: list[pd.DataFrame] = []
    pending_rows = 0
    for batch in open_train_reader():
        pending.append(batch.to_pandas())
        pending_rows += batch.num_rows
        if pending_rows >= CHUNK_ROWS:
            scatter(pd.concat(pending, ignore_index=True))
            pending, pending_rows = [], 0
    if pending:
        scatter(pd.concat(pending, ignore_index=True))
    print(f"streamed {rows:,} rows in {time.time() - started:.0f}s")
    return seen


def compact(sales: np.ndarray, promo: np.ndarray, keep_items: np.ndarray) -> None:
    """Copy the items that actually occur into the final, compacted panels."""
    final_sales = np.lib.format.open_memmap(
        out("sales.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(len(keep_items), N_DATES, N_STORES),
    )
    final_promo = np.lib.format.open_memmap(
        out("promo.npy"),
        mode="w+",
        dtype=np.int8,
        shape=(len(keep_items), N_DATES, N_STORES),
    )
    for start in range(0, len(keep_items), ITEM_BLOCK):
        block = keep_items[start : start + ITEM_BLOCK]
        final_sales[start : start + len(block)] = sales[block]
        final_promo[start : start + len(block)] = promo[block]
    final_sales.flush()
    final_promo.flush()


def main() -> None:
    started = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    stores = pd.read_csv(raw("stores.csv")).sort_values("store_nbr")
    items = pd.read_csv(raw("items.csv")).sort_values("item_nbr")
    S, series_names, levels = store_hierarchy(stores)
    print(f"S {S.shape}  levels {levels}")

    store_nbrs = stores["store_nbr"].to_numpy()
    store_pos = np.full(store_nbrs.max() + 1, -1, dtype=np.int16)
    store_pos[store_nbrs] = np.arange(N_STORES)

    item_nbrs = items["item_nbr"].to_numpy()
    item_pos = np.full(item_nbrs.max() + 1, -1, dtype=np.int32)
    item_pos[item_nbrs] = np.arange(len(item_nbrs))
    n_slots = len(item_nbrs)
    print(f"catalog: {n_slots} items x {N_STORES} stores x {N_DATES} dates")

    sales = np.lib.format.open_memmap(
        out("sales_raw.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(n_slots, N_DATES, N_STORES),
    )
    promo = np.lib.format.open_memmap(
        out("promo_raw.npy"),
        mode="w+",
        dtype=np.int8,
        shape=(n_slots, N_DATES, N_STORES),
    )
    promo[:] = -1  # -1 marks "row absent", resolved by the fill pass below

    seen = stream_train(sales, promo, item_pos, store_pos, started)
    keep_items = np.where(seen)[0]
    print(f"items present in train.csv: {len(keep_items)} of {n_slots}")
    fill_promotions(promo, keep_items)
    compact(sales, promo, keep_items)

    del sales, promo
    os.remove(out("sales_raw.npy"))
    os.remove(out("promo_raw.npy"))

    np.save(out("transactions.npy"), store_transactions(store_pos))

    kept = items.iloc[keep_items]
    states = sorted(stores["state"].unique())
    state_dummies = np.zeros((N_STORES, len(states)), dtype=np.float32)
    for i, state in enumerate(states):
        state_dummies[(stores["state"] == state).to_numpy(), i] = 1.0

    np.savez(
        out("meta.npz"),
        S=S,
        series_names=np.array(series_names),
        level_names=np.array(list(levels)),
        level_bounds=np.array([levels[k] for k in levels]),
        dates=START + np.arange(N_DATES).astype("timedelta64[D]"),
        item_nbrs=kept["item_nbr"].to_numpy(),
        store_nbrs=store_nbrs,
        perishable=kept["perishable"].to_numpy().astype(np.float32),
        state_dummies=state_dummies,
    )
    print(
        f"\nwrote sales {len(keep_items)}x{N_DATES}x{N_STORES}, "
        f"bottom series {len(keep_items) * N_STORES:,}, "
        f"total series {len(keep_items) * S.shape[0]:,}"
    )
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
