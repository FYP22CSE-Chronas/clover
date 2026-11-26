"""Univariate windows over every hierarchy series, on the repo's own splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from config import RunConfig
from data import build_split, load


@dataclass
class HintData:
    values: np.ndarray  # [T, N], rows of S in order (aggregates first)
    S: np.ndarray  # [N, Nb]
    level_slices: dict[str, tuple[int, int]]
    h: int
    config_input_size: int  # CLOVER's input window for this dataset
    train_end: int  # training targets lie in rows [0, train_end)
    val_origins: list[int]
    test_origin: int

    @property
    def n_series(self) -> int:
        return self.values.shape[1]


def prepare(config: RunConfig) -> HintData:
    ds = load(config.name, config.data)
    split = build_split(ds.bottom, 1, config.data.h, config.data.split)
    val_origins = sorted(w.t for w in split.val)
    return HintData(
        values=ds.values.astype(np.float32),
        S=ds.S.astype(np.float64),
        level_slices=dict(ds.level_slices),
        h=config.data.h,
        config_input_size=config.data.input_size,
        train_end=val_origins[0],
        val_origins=val_origins,
        test_origin=split.test[0].t,
    )


def robust_scale(x: Tensor) -> tuple[Tensor, Tensor]:
    """Median and median absolute deviation per window, [B, L] -> [B], [B].

    A window with zero MAD falls back to its mean absolute deviation, then to 1, so
    the scale is never zero (a flat window would otherwise blow up).
    """
    loc = x.median(dim=1).values
    dev = (x - loc[:, None]).abs()
    scale = dev.median(dim=1).values
    scale = torch.where(scale > 0, scale, dev.mean(dim=1))
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return loc, scale


class Windows:
    """Input windows of length L ending at an origin, for chosen series."""

    def __init__(self, data: HintData, input_size: int) -> None:
        self.data = data
        self.L = input_size
        self.values = torch.from_numpy(data.values)

    def train_origins(self) -> np.ndarray:
        lo, hi = self.L, self.data.train_end - self.data.h
        if hi < lo:
            raise ValueError(
                f"input {self.L} + h {self.data.h} does not fit in "
                f"{self.data.train_end} training rows"
            )
        return np.arange(lo, hi + 1)

    def batch(self, origins: Tensor, series: Tensor, with_target: bool) -> dict:
        """Windows for every (origin, series) pair, origin-major: [O * S_n, ...]."""
        rows = origins[:, None] + torch.arange(-self.L, 0)[None, :]  # [O, L]
        x = self.values[rows][:, :, series]  # [O, L, S_n]
        x = x.permute(0, 2, 1).reshape(-1, self.L)  # [O * S_n, L]
        loc, scale = robust_scale(x)
        out = {"x": (x - loc[:, None]) / scale[:, None], "loc": loc, "scale": scale}
        if with_target:
            t_rows = origins[:, None] + torch.arange(self.data.h)[None, :]
            y = self.values[t_rows][:, :, series]  # [O, h, S_n]
            out["y"] = y.permute(0, 2, 1).reshape(-1, self.data.h)
        return out
