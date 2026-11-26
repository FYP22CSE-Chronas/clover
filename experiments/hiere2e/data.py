"""Turn a repo dataset into HierE2E inputs: lags, calendar features, splits.

Splits come from the repo's own split strategy, so HierE2E trains on exactly the
rows CLOVER trains on and is scored on the same validation and test origins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from config import RunConfig
from data import build_split, load

# GluonTS get_lags_for_frequency: daily [1, 7, 14], monthly [1, 12], anything else
# (quarterly included) [1].
LAGS = {"D": [1, 7, 14], "M": [1, 12], "Q": [1]}


def infer_freq(dates: pd.DatetimeIndex) -> str:
    """'D', 'M' or 'Q' from the median spacing of the dates."""
    gap = float(np.median(np.diff(dates.values).astype("timedelta64[D]").astype(int)))
    if gap <= 1.5:
        return "D"
    if gap <= 45:
        return "M"
    if gap <= 120:
        return "Q"
    raise ValueError(f"unsupported spacing of {gap} days")


def time_features(dates: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """[T, 2] Fourier (cos, sin) calendar position, as GluonTS FourierDateFeatures.

    Daily uses day of week (GluonTS's choice). For monthly, GluonTS uses week of
    year; for quarterly it has no default and requires explicit features. Here both
    use position in the year (month of 12, quarter of 4), with a fixed period rather
    than GluonTS's data-dependent max+1.
    """
    if freq == "D":
        value, period = dates.dayofweek.values, 7
    elif freq == "M":
        value, period = dates.month.values - 1, 12
    else:
        value, period = dates.quarter.values - 1, 4
    angle = 2 * np.pi * value / period
    return np.stack([np.cos(angle), np.sin(angle)], axis=1).astype(np.float32)


@dataclass
class HierData:
    """One hierarchy in raw units plus everything HierE2E needs to window it."""

    values: np.ndarray  # [T, N], rows of S in order (aggregates first)
    S: np.ndarray  # [N, Nb]
    time_feat: np.ndarray  # [T, F]
    lags: list[int]
    level_slices: dict[str, tuple[int, int]]
    h: int
    train_end: int  # rows [0, train_end) may be used for training targets
    val_origins: list[int]
    test_origin: int

    @property
    def n_series(self) -> int:
        return self.values.shape[1]

    @property
    def max_lag(self) -> int:
        return max(self.lags)


def prepare(config: RunConfig) -> HierData:
    """Load a repo dataset and read the train/val/test origins off its split."""
    ds = load(config.name, config.data)
    h = config.data.h
    split = build_split(ds.bottom, 1, h, config.data.split)
    val_origins = sorted(w.t for w in split.val)
    freq = infer_freq(ds.dates)
    return HierData(
        values=ds.values.astype(np.float32),
        S=ds.S.astype(np.float32),
        time_feat=time_features(ds.dates, freq),
        lags=LAGS[freq],
        level_slices=dict(ds.level_slices),
        h=h,
        train_end=val_origins[0],
        val_origins=val_origins,
        test_origin=split.test[0].t,
    )


def mean_scale(context: Tensor) -> Tensor:
    """GluonTS MeanScaler: mean |y| over the context, [B, C, N] -> [B, 1, N].

    A series that is all zero in a window takes the mean scale of that series over
    the batch, and if that is zero too, a floor -- so the scale is never zero.
    """
    scale = context.abs().mean(dim=1, keepdim=True)
    batch_default = context.abs().sum(dim=(0, 1)) / (context.shape[0] * context.shape[1])
    scale = torch.where(scale > 0, scale, batch_default.expand_as(scale))
    return scale.clamp_min(1e-10)


class Windows:
    """Builds teacher-forced sequences ending at a forecast origin."""

    def __init__(self, data: HierData, context_length: int) -> None:
        self.data = data
        self.ctx = context_length
        self.values = torch.from_numpy(data.values)
        self.time_feat = torch.from_numpy(data.time_feat)
        self.lags = torch.tensor(data.lags)

    def first_origin(self) -> int:
        """Earliest origin with a full context and full lag history."""
        return self.ctx + self.data.max_lag

    def train_origins(self) -> np.ndarray:
        """Origins whose whole horizon lies inside the training rows."""
        lo, hi = self.first_origin(), self.data.train_end - self.data.h
        if hi < lo:
            raise ValueError(
                f"context {self.ctx} + max lag {self.data.max_lag} + h {self.data.h} "
                f"does not fit in {self.data.train_end} training rows"
            )
        return np.arange(lo, hi + 1)

    def sequence(self, origins: Tensor, steps_after: int) -> dict[str, Tensor]:
        """Rows [o - ctx, o + steps_after) for each origin o, teacher forced.

        Returns scaled lags [B, T, L, N], time features [B, T, F], raw targets
        [B, T, N] (only meaningful where they exist) and the scale [B, 1, N].
        """
        steps = torch.arange(-self.ctx, steps_after)  # relative to the origin
        rows = origins[:, None] + steps[None, :]  # [B, T]
        lag_rows = rows[:, :, None] - self.lags[None, None, :]  # [B, T, L]
        ctx_rows = origins[:, None] + torch.arange(-self.ctx, 0)[None, :]
        scale = mean_scale(self.values[ctx_rows])
        lags = self.values[lag_rows] / scale.unsqueeze(1)  # [B, T, L, N]
        target_rows = rows.clamp(max=self.values.shape[0] - 1)
        return {
            "lags": lags,
            "time_feat": self.time_feat[rows.clamp(max=self.time_feat.shape[0] - 1)],
            "target": self.values[target_rows],
            "scale": scale,
            "rows": rows,
        }
