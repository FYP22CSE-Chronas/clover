from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from contracts import FactorParams, FlowParams, GMMParams, ScaleStats, SkewTParams
from registry import Registry

EPS = 1e-6


class WindowScaler(Protocol):
    """Derives (loc, scale) from an insample block."""

    def __call__(self, insample: Tensor) -> ScaleStats: ...


SCALERS: Registry[WindowScaler] = Registry("scaler")


@SCALERS.register("identity")
def identity_scaler(insample: Tensor) -> ScaleStats:
    """No transform: loc 0, scale 1."""
    return ScaleStats(
        loc=torch.zeros_like(insample[:, :1, :]),
        scale=torch.ones_like(insample[:, :1, :]),
    )


@SCALERS.register("standard")
def standard_scaler(insample: Tensor) -> ScaleStats:
    """Mean and standard deviation over the insample time axis."""
    return ScaleStats(
        loc=insample.mean(dim=1, keepdim=True),
        scale=insample.std(dim=1, keepdim=True) + EPS,
    )


@SCALERS.register("robust")
def robust_scaler(insample: Tensor) -> ScaleStats:
    """Median and median absolute deviation over the insample time axis."""
    loc = insample.median(dim=1, keepdim=True).values
    scale = (insample - loc).abs().median(dim=1, keepdim=True).values
    return ScaleStats(loc=loc, scale=scale + EPS)


@SCALERS.register("mean")
def mean_scaler(insample: Tensor) -> ScaleStats:
    """Scale-only normalization by mean absolute value, preserving non-negativity."""
    return ScaleStats(
        loc=torch.zeros_like(insample[:, :1, :]),
        scale=insample.abs().mean(dim=1, keepdim=True) + EPS,
    )


@SCALERS.register("mean_floor")
def mean_floor_scaler(insample: Tensor) -> ScaleStats:
    """Scale-only normalization floored at one unit, for sparse count panels.

    `mean` divides by `mean|y| + eps`, which collapses to `eps` on a window that is
    entirely zero. Any non-zero target then demands a normalized mean of order `1/eps`,
    and the gradients that follow destabilize training. Flooring the divisor at one
    unit leaves busy series untouched and makes a dormant one merely uninformative.
    """
    return ScaleStats(
        loc=torch.zeros_like(insample[:, :1, :]),
        scale=insample.abs().mean(dim=1, keepdim=True).clamp_min(1.0),
    )


def window_stats(insample: Tensor, scaler: str = "standard") -> ScaleStats:
    """Compute (loc, scale) from the insample block only; never from the horizon."""
    return SCALERS.get(scaler)(insample)


def normalize(x: Tensor, stats: ScaleStats) -> Tensor:
    """(x - loc) / scale, broadcasting [B, 1, Nb] statistics over [B, T, Nb]."""
    return (x - stats.loc) / stats.scale


def denormalize(x: Tensor, stats: ScaleStats) -> Tensor:
    """Inverse of `normalize`."""
    return x * stats.scale + stats.loc


def denormalize_params(
    params: FactorParams | SkewTParams | FlowParams | GMMParams, stats: ScaleStats
) -> FactorParams | SkewTParams | FlowParams | GMMParams:
    """Push (loc, scale) back onto mu/sigma/F so samples land in raw units.

    Every factor model is affine in mu/sigma/F, so this inverse is exact. A skew-t's
    `nu`/`lam`, a flow's `flow_w`/`flow_a`/`flow_b`, and a mixture's `logits` are
    scale-free and pass through unchanged.
    """
    if isinstance(params, SkewTParams):
        return SkewTParams(
            mu=params.mu * stats.scale + stats.loc,
            sigma=params.sigma * stats.scale,
            nu=params.nu,
            lam=params.lam,
            F=params.F * stats.scale.unsqueeze(-1),
        )
    if isinstance(params, FlowParams):
        return FlowParams(
            mu=params.mu * stats.scale + stats.loc,
            sigma=params.sigma * stats.scale,
            flow_w=params.flow_w,
            flow_a=params.flow_a,
            flow_b=params.flow_b,
            F=params.F * stats.scale.unsqueeze(-1),
        )
    if isinstance(params, GMMParams):
        return GMMParams(
            mu=params.mu * stats.scale.unsqueeze(-1) + stats.loc.unsqueeze(-1),
            sigma=params.sigma * stats.scale.unsqueeze(-1),
            logits=params.logits,
            F=params.F * stats.scale.unsqueeze(-1),
        )
    return FactorParams(
        mu=params.mu * stats.scale + stats.loc,
        sigma=params.sigma * stats.scale,
        F=params.F * stats.scale.unsqueeze(-1),
    )
