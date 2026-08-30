from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLPBlock(nn.Module):
    """Linear -> ReLU -> ... -> Linear; the generic block used throughout the model."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_size: int | None = None,
        n_hidden_layers: int = 1,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        hidden = hidden_size if hidden_size is not None else out_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(prev, hidden), activation()]
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DilatedCausalConv(nn.Module):
    """Left-padded dilated conv stack returning the last step: [B, C_in, L] -> [B, C]."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        dilations: Sequence[int] = (1, 2, 4),
        kernel_size: int = 2,
        residual: bool = True,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.out_channels = channels
        self.residual = residual
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.left_pads: list[int] = []
        prev = in_channels
        for d in dilations:
            self.convs.append(nn.Conv1d(prev, channels, kernel_size, dilation=d))
            self.acts.append(activation())
            self.skips.append(
                nn.Identity() if prev == channels else nn.Conv1d(prev, channels, 1)
            )
            self.left_pads.append((kernel_size - 1) * d)
            prev = channels

    def forward(self, x: Tensor) -> Tensor:
        for conv, act, skip, pad in zip(
            self.convs, self.acts, self.skips, self.left_pads, strict=True
        ):
            # Padding on the left only keeps the stack causal.
            out = act(conv(nn.functional.pad(x, (pad, 0))))
            x = out + skip(x) if self.residual else out
        return x[..., -1]


def rq_spline_transform(
    x: Tensor,
    theta: Tensor,
    bound: float = 3.0,
    min_bin: float = 1e-3,
    min_derivative: float = 1e-3,
) -> Tensor:
    """Monotone rational-quadratic spline on `[-bound, bound]`, identity outside.

    The forward direction of a Neural Spline Flow coupling transform (Durkan et al.,
    2019) applied element-wise. `x` is `[..., N]` and `theta` is `[..., 3*n_bins - 1]`
    with matching leading dims: `n_bins` raw widths, `n_bins` raw heights, and
    `n_bins - 1` raw internal derivatives. Widths and heights are softmaxed onto the
    interval and the boundary derivatives are pinned to 1, which makes the linear
    tails outside `[-bound, bound]` exactly the identity -- far-tail draws therefore
    degrade to the untransformed Gaussian rather than diverging.

    `theta = 0` is exactly the identity map: uniform widths and heights give every
    bin slope 1, and the derivative offset is chosen so a zero raw parameter yields
    a unit derivative. Only the forward direction exists here; the inverse and the
    log-determinant are never needed because the training objective scores samples
    rather than densities.
    """
    n_bins = (theta.shape[-1] + 1) // 3
    if theta.shape[-1] != 3 * n_bins - 1:
        raise ValueError(
            f"theta trailing dim {theta.shape[-1]} is not 3*n_bins - 1 for any n_bins"
        )
    if theta.shape[:-1] != x.shape[:-1]:
        raise ValueError(
            f"theta leading dims {tuple(theta.shape[:-1])} must match x "
            f"{tuple(x.shape[:-1])}"
        )
    if n_bins < 2:
        raise ValueError(f"need at least 2 spline bins, got {n_bins}")

    n = x.shape[-1]
    t = theta.reshape(-1, 3 * n_bins - 1)
    flat = x.reshape(-1, n)

    span = 1.0 - min_bin * n_bins
    if span <= 0.0:
        raise ValueError(f"min_bin {min_bin} is too large for {n_bins} bins")
    widths = min_bin + span * torch.softmax(t[:, :n_bins], dim=-1)
    heights = min_bin + span * torch.softmax(t[:, n_bins : 2 * n_bins], dim=-1)
    knots_x = -bound + 2.0 * bound * torch.cumsum(F.pad(widths, (1, 0)), dim=-1)
    knots_y = -bound + 2.0 * bound * torch.cumsum(F.pad(heights, (1, 0)), dim=-1)

    # Offset chosen so a zero raw parameter lands on a derivative of exactly 1.
    offset = math.log(math.expm1(1.0 - min_derivative))
    inner = min_derivative + F.softplus(t[:, 2 * n_bins :] + offset)
    edge = inner.new_ones(inner.shape[0], 1)
    derivs = torch.cat([edge, inner, edge], dim=-1)

    idx = torch.searchsorted(knots_x[:, 1:-1].contiguous(), flat.contiguous())
    x_lo, x_hi = knots_x.gather(1, idx), knots_x.gather(1, idx + 1)
    y_lo, y_hi = knots_y.gather(1, idx), knots_y.gather(1, idx + 1)
    d_lo, d_hi = derivs.gather(1, idx), derivs.gather(1, idx + 1)

    width, height = x_hi - x_lo, y_hi - y_lo
    slope = height / width
    # Clamping keeps the tail branch finite; `torch.where` discards it anyway.
    xi = ((flat - x_lo) / width).clamp(0.0, 1.0)
    one_minus = 1.0 - xi
    numerator = height * (slope * xi * xi + d_lo * xi * one_minus)
    denominator = slope + (d_hi + d_lo - 2.0 * slope) * xi * one_minus
    inside_y = y_lo + numerator / denominator

    inside = (flat >= -bound) & (flat <= bound)
    return torch.where(inside, inside_y, flat).reshape(x.shape)
