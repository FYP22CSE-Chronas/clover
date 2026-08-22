from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from model.blocks import MLPBlock
from registry import Registry

DECODERS: Registry[nn.Module] = Registry("decoder")


@DECODERS.register("two_stage")
class TwoStageDecoder(nn.Module):
    """Horizon-agnostic and horizon-specific contexts, joined per horizon."""

    def __init__(
        self,
        in_dim: int,
        h: int,
        horizon_agnostic_dim: int,
        horizon_specific_dim: int,
        n_futr_features: int = 0,
    ) -> None:
        super().__init__()
        self.h = h
        self.horizon_specific_dim = horizon_specific_dim
        self.agnostic = MLPBlock(in_dim, horizon_agnostic_dim)
        self.specific = MLPBlock(in_dim, h * horizon_specific_dim)
        self.out_dim = horizon_agnostic_dim + horizon_specific_dim + n_futr_features

    def forward(self, h_b: Tensor, futr_exog: Tensor | None = None) -> Tensor:
        """[B, Nb, D] plus optional [B, Nb, H, F] -> [B, Nb, H, out_dim]."""
        b, n_bottom, _ = h_b.shape
        c_ag = self.agnostic(h_b).unsqueeze(2).expand(b, n_bottom, self.h, -1)
        c_sp = self.specific(h_b).reshape(b, n_bottom, self.h, self.horizon_specific_dim)
        parts = [c_sp, c_ag] if futr_exog is None else [c_sp, c_ag, futr_exog]
        return torch.cat(parts, dim=-1)
