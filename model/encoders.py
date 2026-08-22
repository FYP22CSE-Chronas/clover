from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from model.blocks import DilatedCausalConv, MLPBlock
from registry import Registry

ENCODERS: Registry[nn.Module] = Registry("encoder")


def _fold(y: Tensor, hist_exog: Tensor | None) -> tuple[Tensor, int, int]:
    """Stack target and covariate channels, then fold the series axis into the batch."""
    x = y.unsqueeze(2)
    if hist_exog is not None:
        x = torch.cat([x, hist_exog], dim=2)
    b, n, c, length = x.shape
    return x.reshape(b * n, c, length), b, n


@ENCODERS.register("dilated_conv")
class DilatedConvEncoder(nn.Module):
    """Shared-weight dilated causal convolutions applied to every hierarchy series."""

    def __init__(
        self,
        in_channels: int,
        input_size: int,
        channels: int = 10,
        dilations: Sequence[int] = (1, 2, 4),
        kernel_size: int = 2,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.conv = DilatedCausalConv(
            in_channels=in_channels,
            channels=channels,
            dilations=dilations,
            kernel_size=kernel_size,
            residual=residual,
        )
        self.out_channels = channels

    def forward(self, y: Tensor, hist_exog: Tensor | None = None) -> Tensor:
        """[B, N, L] plus optional [B, N, X, L] -> [B, N, C]."""
        folded, b, n = _fold(y, hist_exog)
        return self.conv(folded).reshape(b, n, self.out_channels)


@ENCODERS.register("mlp")
class FlatMLPEncoder(nn.Module):
    """Flattens the window through a shared MLP; a convolution-free baseline."""

    def __init__(
        self,
        in_channels: int,
        input_size: int,
        channels: int = 10,
        hidden_size: int | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.mlp = MLPBlock(in_channels * input_size, channels, hidden_size=hidden_size)
        self.out_channels = channels

    def forward(self, y: Tensor, hist_exog: Tensor | None = None) -> Tensor:
        """[B, N, L] plus optional [B, N, X, L] -> [B, N, C]."""
        folded, b, n = _fold(y, hist_exog)
        return self.mlp(folded.reshape(b * n, -1)).reshape(b, n, self.out_channels)
