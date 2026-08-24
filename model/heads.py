from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from contracts import FactorParams, SkewTParams
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


@HEADS.register("skew_t")
class SkewTHead(nn.Module):
    """Emits 4 + K values per (series, horizon): mu, raw sigma, raw nu, lam, loadings."""

    def __init__(
        self,
        in_dim: int,
        n_factors: int,
        sigma_activation: str = "softplus",
        sigma_eps: float = 1e-3,
        nu_floor: float = 2.0,
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.sigma_activation = sigma_activation
        self.sigma_eps = sigma_eps
        self.nu_floor = nu_floor
        self.proj = MLPBlock(in_dim, 4 + n_factors)

    def forward(self, z: Tensor) -> SkewTParams:
        """[B, Nb, H, D] -> mu/sigma/nu/lam [B, H, Nb] and loadings [B, H, Nb, K]."""
        params = self.proj(z)
        mu = params[..., 0].permute(0, 2, 1)
        sigma_raw = params[..., 1].permute(0, 2, 1)
        nu_raw = params[..., 2].permute(0, 2, 1)
        lam = params[..., 3].permute(0, 2, 1)
        loadings = params[..., 4:].permute(0, 2, 1, 3)
        return SkewTParams(
            mu=mu,
            sigma=self._sigma(sigma_raw),
            nu=self._nu(nu_raw),
            lam=lam,
            F=loadings,
        )

    def _sigma(self, sigma_raw: Tensor) -> Tensor:
        """Positivity constraint on the predicted scale."""
        if self.sigma_activation == "softplus":
            return F.softplus(sigma_raw) + self.sigma_eps
        return torch.exp(sigma_raw.clamp(max=20.0)) + self.sigma_eps

    def _nu(self, nu_raw: Tensor) -> Tensor:
        """Floor the degrees of freedom above `nu_floor` so variance stays finite."""
        return F.softplus(nu_raw) + self.nu_floor


@HEADS.register("skew_t_shared")
class SkewTSharedHead(nn.Module):
    """`SkewTHead` with one global (nu, lam) pair instead of one per output.

    Per-series-per-horizon shape parameters need enough windows to estimate; on a
    small panel they add variance faster than they capture real skew/tail
    structure (see `SkewTHead`'s per-output nu/lam). Tying them to two scalars
    shared across every (series, horizon) trades that flexibility for far fewer
    shape parameters to fit, which helps when training windows are scarce.
    """

    def __init__(
        self,
        in_dim: int,
        n_factors: int,
        sigma_activation: str = "softplus",
        sigma_eps: float = 1e-3,
        nu_floor: float = 2.0,
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.sigma_activation = sigma_activation
        self.sigma_eps = sigma_eps
        self.nu_floor = nu_floor
        self.proj = MLPBlock(in_dim, 2 + n_factors)
        self.nu_raw = nn.Parameter(torch.zeros(()))
        self.lam_raw = nn.Parameter(torch.zeros(()))

    def forward(self, z: Tensor) -> SkewTParams:
        """[B, Nb, H, D] -> mu/sigma [B, H, Nb], loadings [B, H, Nb, K], global nu/lam."""
        params = self.proj(z)
        mu = params[..., 0].permute(0, 2, 1)
        sigma_raw = params[..., 1].permute(0, 2, 1)
        loadings = params[..., 2:].permute(0, 2, 1, 3)
        nu = (F.softplus(self.nu_raw) + self.nu_floor).expand_as(mu)
        lam = self.lam_raw.expand_as(mu)
        return SkewTParams(mu=mu, sigma=self._sigma(sigma_raw), nu=nu, lam=lam, F=loadings)

    def _sigma(self, sigma_raw: Tensor) -> Tensor:
        """Positivity constraint on the predicted scale."""
        if self.sigma_activation == "softplus":
            return F.softplus(sigma_raw) + self.sigma_eps
        return torch.exp(sigma_raw.clamp(max=20.0)) + self.sigma_eps
