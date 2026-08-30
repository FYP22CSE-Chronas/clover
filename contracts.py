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
class SkewTParams:
    """Skew-t factor-model parameters at the bottom level.

    Adds two scale-free shape parameters to the same mu/sigma/F factor structure
    as `FactorParams`: `nu` (degrees of freedom, > 2 for finite variance) controls
    tail heaviness and `lam` controls skewness (0 recovers a symmetric Student-t).
    """

    mu: Tensor  # [B, H, Nb]
    sigma: Tensor  # [B, H, Nb]
    nu: Tensor  # [B, H, Nb]
    lam: Tensor  # [B, H, Nb]
    F: Tensor  # [B, H, Nb, K]


@dataclass(frozen=True)
class GMMParams:
    """K-component Gaussian-mixture factor-model parameters at the bottom level.

    Extends the mu/sigma/F factor structure with per-component `mu`/`sigma` and
    `logits` (softmax-normalized per series/horizon into K mixture weights); the
    factor loadings `F` stay shared across components rather than duplicated K
    times.
    """

    mu: Tensor  # [B, H, Nb, K]
    sigma: Tensor  # [B, H, Nb, K]
    logits: Tensor  # [B, H, Nb, K]
    F: Tensor  # [B, H, Nb, Kf]


@dataclass(frozen=True)
class SplineCopulaParams:
    """Gaussian factor copula with learned, nonparametric marginals.

    Keeps the same mu/sigma/F factor structure as `FactorParams`, but the
    standardized latent it induces is pushed through a monotone rational-quadratic
    spline before being placed by mu/scale. `theta` holds that spline's raw knot
    parameters -- `3 * n_bins - 1` of them per (series, horizon) -- and is
    *scale-free*: because the spline acts on a latent standardized to unit
    variance, `theta` passes through `denormalize_params` untouched exactly as a
    skew-t's `nu`/`lam` does, and the affine denormalization stays exact.

    `theta = 0` is the identity transform, so a zero-initialized head reproduces
    `FactorParams` sampling exactly.
    """

    mu: Tensor  # [B, H, Nb]
    sigma: Tensor  # [B, H, Nb]
    F: Tensor  # [B, H, Nb, K]
    theta: Tensor  # [B, H, Nb, 3 * n_bins - 1]
    bound: float = 3.0  # spline is the identity outside [-bound, bound]


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
    """[B, Nb, H, D'] -> FactorParams, SkewTParams or GMMParams."""

    def __call__(self, z: Tensor) -> FactorParams | SkewTParams | GMMParams: ...
