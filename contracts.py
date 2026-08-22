from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


@dataclass
class WindowBatch:
    """Model input; every tensor is already normalized."""

    insample_y: Tensor  # [B, L, Nb]
    futr_exog: Tensor | None = None  # [B, F, L+H, Nb]
    hist_exog: Tensor | None = None  # [B, X, L, Nb]
    stat_exog: Tensor | None = None  # [Nb, S] or [B, Nb, S]


@dataclass(frozen=True)
class FactorParams:
    """Gaussian factor-model parameters at the bottom level."""

    mu: Tensor  # [B, H, Nb]
    sigma: Tensor  # [B, H, Nb]
    F: Tensor  # [B, H, Nb, K]


@dataclass(frozen=True)
class ScaleStats:
    """Per-window, per-series location and scale, broadcastable over time."""

    loc: Tensor  # [B, 1, Nb]
    scale: Tensor  # [B, 1, Nb]


@dataclass(frozen=True)
class ForecastSamples:
    """Draws from the predictive distribution, raw units."""

    bottom: Tensor  # [B, H, Nb, N]
    hierarchy: Tensor  # [B, H, Na+Nb, N]


@dataclass
class TrainingBatch:
    """One optimizer step's worth of windows, plus what is needed to score it."""

    windows: WindowBatch
    target_bottom: Tensor  # [B, H, Nb], raw units
    scale: ScaleStats


class Batcher(Protocol):
    """Supplies training batches to the trainer; implemented by the pipeline layer."""

    n_windows: int

    def batch(self, indices: Sequence[int]) -> TrainingBatch: ...


class Encoder(Protocol):
    """[B, N, L] history (+ optional [B, N, X, L] covariates) -> [B, N, C]."""

    out_channels: int

    def __call__(self, y: Tensor, hist_exog: Tensor | None = None) -> Tensor: ...


class Mixer(Protocol):
    """[B, Na+Nb, C] -> [B, Nb, C]."""

    out_channels: int

    def __call__(self, h: Tensor) -> Tensor: ...


class Decoder(Protocol):
    """[B, Nb, D] context (+ optional [B, Nb, H, F] futures) -> [B, Nb, H, D']."""

    out_dim: int

    def __call__(self, h: Tensor, futr_exog: Tensor | None = None) -> Tensor: ...


class Head(Protocol):
    """[B, Nb, H, D'] -> FactorParams."""

    def __call__(self, z: Tensor) -> FactorParams: ...
