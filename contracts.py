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
class FlowParams:
    """Conditional-normalizing-flow factor-model parameters at the bottom level.

    Same mu/sigma/F factor structure as `FactorParams`; `flow_w`/`flow_a`/`flow_b` are
    the already-activated weight/slope/shift of a K_flow-unit deep sigmoidal flow (see
    `sample_flow_factor_model`) that warps a standard-normal draw into the innovation's
    learned shape, in place of a fixed parametric family like `SkewTParams`'s.
    """

    mu: Tensor  # [B, H, Nb]
    sigma: Tensor  # [B, H, Nb]
    flow_w: Tensor  # [B, H, Nb, K_flow], positive, sums to 1 along K_flow
    flow_a: Tensor  # [B, H, Nb, K_flow], positive
    flow_b: Tensor  # [B, H, Nb, K_flow]
    F: Tensor  # [B, H, Nb, K]


@dataclass(frozen=True)
class CopulaSplineParams:
    """Gaussian-copula factor-model parameters with a learned per-series spline marginal.

    The correlation structure comes from the same low-rank Gaussian factor model as
    `FactorParams` (mu, sigma, F); `spline_w`/`spline_h`/`spline_d` are a monotonic
    rational-quadratic spline (see `sample_copula_spline_factor_model`) applied to the
    *entire* correlated Gaussian draw -- idiosyncratic noise and shared factor term
    together -- rather than to the idiosyncratic part alone the way `FlowParams` does,
    so the marginal's shape is learned without disturbing the copula's rank
    correlation.
    """

    mu: Tensor  # [B, H, Nb]
    sigma: Tensor  # [B, H, Nb]
    spline_w: Tensor  # [B, H, Nb, n_bins], positive, sums to 1 along n_bins
    spline_h: Tensor  # [B, H, Nb, n_bins], positive, sums to 1 along n_bins
    spline_d: Tensor  # [B, H, Nb, n_bins-1], positive interior knot derivatives
    F: Tensor  # [B, H, Nb, K]


@dataclass(frozen=True)
class CopulaFlowParams:
    """Gaussian-copula factor-model parameters with a two-layer learned marginal.

    Combines `CopulaSplineParams`'s correlation-preserving construction -- the
    *whole* correlated draw `g = z + F @ eps`, standardized, is what gets warped,
    never just the idiosyncratic part -- with a marginal built from two composed
    monotonic layers instead of one: a rational-quadratic spline
    (`spline_w`/`spline_h`/`spline_d`, as in `CopulaSplineParams`) followed by a
    deep sigmoidal flow (`flow_w`/`flow_a`/`flow_b`, as in `FlowParams`). See
    `sample_copula_flow_factor_model` for why composing them is safe and why the
    copula's rank correlation survives both layers.
    """

    mu: Tensor  # [B, H, Nb]
    sigma: Tensor  # [B, H, Nb]
    spline_w: Tensor  # [B, H, Nb, n_bins], positive, sums to 1 along n_bins
    spline_h: Tensor  # [B, H, Nb, n_bins], positive, sums to 1 along n_bins
    spline_d: Tensor  # [B, H, Nb, n_bins-1], positive interior knot derivatives
    flow_w: Tensor  # [B, H, Nb, K_flow], positive, sums to 1 along K_flow
    flow_a: Tensor  # [B, H, Nb, K_flow], positive
    flow_b: Tensor  # [B, H, Nb, K_flow]
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
    """[B, Nb, H, D'] -> FactorParams, SkewTParams, FlowParams, CopulaSplineParams,
    CopulaFlowParams or GMMParams."""

    def __call__(
        self, z: Tensor
    ) -> (
        FactorParams
        | SkewTParams
        | FlowParams
        | CopulaSplineParams
        | CopulaFlowParams
        | GMMParams
    ): ...
