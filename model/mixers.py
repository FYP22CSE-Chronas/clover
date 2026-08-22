from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from model.blocks import MLPBlock
from registry import Registry

MIXERS: Registry[nn.Module] = Registry("mixer")


@MIXERS.register("cross_series_mlp")
class CrossSeriesMixer(nn.Module):
    """Residual cross-series mixer; hidden_size=0 makes it an identity passthrough."""

    def __init__(
        self, n_hierarchy: int, n_bottom: int, channels: int, hidden_size: int
    ) -> None:
        super().__init__()
        self.n_bottom = n_bottom
        self.channels = channels
        self.out_channels = channels
        self.enabled = hidden_size > 0
        self.mlp = (
            MLPBlock(
                in_dim=n_hierarchy * channels,
                out_dim=n_bottom * channels,
                hidden_size=hidden_size,
            )
            if self.enabled
            else None
        )

    def forward(self, h: Tensor) -> Tensor:
        """[B, Na+Nb, C] -> [B, Nb, C]."""
        # S = [A; I], so the trailing Nb rows are the bottom series themselves.
        residual = h[:, -self.n_bottom :, :]
        if self.mlp is None:
            return residual
        b = h.shape[0]
        mixed = self.mlp(h.reshape(b, -1)).reshape(b, self.n_bottom, self.channels)
        return residual + mixed


@MIXERS.register("identity")
class IdentityMixer(nn.Module):
    """Drops the aggregate rows without mixing; the disabled-mixer baseline."""

    def __init__(
        self, n_hierarchy: int, n_bottom: int, channels: int, hidden_size: int = 0
    ) -> None:
        super().__init__()
        self.n_bottom = n_bottom
        self.out_channels = channels

    def forward(self, h: Tensor) -> Tensor:
        """[B, Na+Nb, C] -> [B, Nb, C]."""
        return h[:, -self.n_bottom :, :]
