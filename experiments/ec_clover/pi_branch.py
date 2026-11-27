from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from config import ModelConfig
from contracts import ScaleStats
from experiments.ec_clover.config import ECConfig
from experiments.ec_clover.ec_branch import EC_BRANCHES
from model.encoders import ENCODERS


@EC_BRANCHES.register("pi")
class PiErrorCorrectionBranch(nn.Module):
    """Negative semi-definite restoring-force branch: u = -L L^T z.

    Operates on bottom-level history `z`:
    - `state_norm="train_ref"`: recovers `y_raw = y * scale.scale + scale.loc` and
      standardizes with fixed training-split references `(mu_ref, sigma_ref)` so the
      long-run coordinate frame does not shift every window.
    - `state_norm="window"`: uses the per-window normalized `y` directly.
    """

    def __init__(
        self,
        n_bottom: int,
        input_size: int,
        ec_config: ECConfig,
        model_config: ModelConfig,
        beta_init: Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        del beta_init  # unused; L replaces alpha/beta
        if ec_config.rank > n_bottom:
            raise ValueError(f"ec.rank {ec_config.rank} must be <= n_bottom {n_bottom}")
        if ec_config.rank == n_bottom:
            warnings.warn(
                f"ec.rank ({ec_config.rank}) == n_bottom ({n_bottom}); full-rank "
                f"operator has no low-rank cointegration bottleneck",
                UserWarning,
                stacklevel=2,
            )

        self.n_bottom = n_bottom
        self.rank = ec_config.rank
        self.state_norm = ec_config.state_norm
        self.ec_input = ec_config.ec_input

        self.L = nn.Parameter(
            torch.randn(n_bottom, ec_config.rank) * ec_config.pi_init_std
        )
        if ec_config.pi_frozen:
            self.L.requires_grad_(False)

        dilations = ec_config.encoder_dilations or model_config.dilations
        receptive_field = 1 + sum((model_config.kernel_size - 1) * d for d in dilations)
        if ec_config.encoder == "dilated_conv" and receptive_field > input_size:
            raise ValueError(
                f"ec encoder receptive field {receptive_field} exceeds input_size "
                f"{input_size}; shorten ec.encoder_dilations {tuple(dilations)}"
            )
        self.encoder = ENCODERS.create(
            ec_config.encoder,
            in_channels=1,
            input_size=input_size,
            channels=ec_config.encoder_channels,
            dilations=dilations,
            kernel_size=model_config.kernel_size,
            residual=model_config.conv_residual,
        )
        self.out_channels = self.encoder.out_channels
        self.register_buffer("mu_ref", torch.zeros(1, 1, n_bottom))
        self.register_buffer("sigma_ref", torch.ones(1, 1, n_bottom))

    def forward(self, y: Tensor, scale: ScaleStats | None = None) -> Tensor:
        """[B, L, Nb] insample history -> [B, Nb, C] EC representation."""
        if self.state_norm == "train_ref":
            y_raw = y * scale.scale + scale.loc if scale is not None else y
            z = (y_raw - self.mu_ref) / self.sigma_ref
        else:
            z = y

        if self.ec_input == "e":
            v = torch.einsum("bln,nr->blr", z, self.L)
            h = self.encoder(v.permute(0, 2, 1))
            return torch.einsum("brc,nr->bnc", h, -self.L)

        pi = -(self.L @ self.L.T)
        u = torch.einsum("bln,mn->blm", z, pi)
        return self.encoder(u.permute(0, 2, 1))
