from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from model.blocks import MLPBlock
from registry import Registry

FUSIONS: Registry[nn.Module] = Registry("ec_fusion")


@FUSIONS.register("concat")
class FusionMLP(nn.Module):
    """concat(H_CLOVER, H_EC) -> MLP -> H, with out_dim == H_CLOVER's channel dim.

    Matching `clover_dim` keeps every downstream shape (static/future context
    concatenation, `context_dim`, the decoder, the head) exactly what `CLOVER.__init__`
    already built, so no other part of the baseline network needs to change shape to
    accommodate the extra branch.

    Caveat that motivated `GatedResidualFusion`: this reprojects `h_clover` through
    freshly initialized weights, so at step 0 the encoder/mixer representation the
    baseline relies on is scrambled rather than passed through. The model has to spend
    training re-learning an approximate identity before the EC signal can pay for
    itself.
    """

    def __init__(
        self,
        clover_dim: int,
        ec_dim: int,
        out_dim: int,
        hidden_size: int | None = None,
        n_layers: int = 1,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.mlp = MLPBlock(
            clover_dim + ec_dim,
            out_dim,
            hidden_size=hidden_size,
            n_hidden_layers=n_layers,
        )

    def forward(self, h_clover: Tensor, h_ec: Tensor) -> Tensor:
        """[B, Nb, C] x [B, Nb, C_ec] -> [B, Nb, C]."""
        return self.mlp(torch.cat([h_clover, h_ec], dim=-1))


@FUSIONS.register("gated_residual")
class GatedResidualFusion(nn.Module):
    """`h_clover + sigmoid(gate) * MLP([h_clover; h_ec])`, one gate per channel.

    The residual path means the baseline representation reaches the decoder unchanged,
    and `gate_init` (default -2.0, so `sigmoid(gate) ~ 0.12`) starts the EC contribution
    small: the branch begins as a mild perturbation of baseline CLOVER and has to earn
    its influence through the CRPS gradient rather than being imposed at initialization.
    A per-channel gate lets different context channels admit different amounts of
    equilibrium information at negligible parameter cost.
    """

    def __init__(
        self,
        clover_dim: int,
        ec_dim: int,
        out_dim: int,
        hidden_size: int | None = None,
        n_layers: int = 1,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        if out_dim != clover_dim:
            raise ValueError(
                f"gated_residual fusion needs out_dim == clover_dim; got "
                f"{out_dim} vs {clover_dim}"
            )
        self.mlp = MLPBlock(
            clover_dim + ec_dim,
            out_dim,
            hidden_size=hidden_size,
            n_hidden_layers=n_layers,
        )
        self.gate = nn.Parameter(torch.full((clover_dim,), float(gate_init)))

    def forward(self, h_clover: Tensor, h_ec: Tensor) -> Tensor:
        """[B, Nb, C] x [B, Nb, C_ec] -> [B, Nb, C]."""
        update = self.mlp(torch.cat([h_clover, h_ec], dim=-1))
        return h_clover + torch.sigmoid(self.gate) * update
