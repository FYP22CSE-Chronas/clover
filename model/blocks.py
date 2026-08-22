from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn
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
