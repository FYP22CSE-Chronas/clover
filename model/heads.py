from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from contracts import FactorParams
from model.blocks import MLPBlock
from registry import Registry

HEADS: Registry[nn.Module] = Registry("head")


@HEADS.register("factor_model")
class FactorModelHead(nn.Module):
    """Emits 2 + K values per (series, horizon): mu, raw sigma, and K loadings."""

    def __init__(
        self,
        in_dim: int,
        n_factors: int,
        sigma_activation: str = "softplus",
        sigma_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.sigma_activation = sigma_activation
        self.sigma_eps = sigma_eps
        self.proj = MLPBlock(in_dim, 2 + n_factors)

    def forward(self, z: Tensor) -> FactorParams:
        """[B, Nb, H, D] -> mu/sigma [B, H, Nb] and loadings [B, H, Nb, K]."""
        params = self.proj(z)
        mu = params[..., 0].permute(0, 2, 1)
        sigma_raw = params[..., 1].permute(0, 2, 1)
        loadings = params[..., 2:].permute(0, 2, 1, 3)
        return FactorParams(mu=mu, sigma=self._sigma(sigma_raw), F=loadings)

    def _sigma(self, sigma_raw: Tensor) -> Tensor:
        """Positivity constraint on the predicted scale."""
        if self.sigma_activation == "softplus":
            return F.softplus(sigma_raw) + self.sigma_eps
        return torch.exp(sigma_raw.clamp(max=20.0)) + self.sigma_eps
