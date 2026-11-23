from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from config import ModelConfig, ModelDims
from contracts import FactorParams, WindowBatch
from model.blocks import MLPBlock
from model.decoders import DECODERS
from model.encoders import ENCODERS
from model.heads import HEADS
from model.mixers import MIXERS

EPS = 1e-6


class CLOVER(nn.Module):
    """Coherent probabilistic forecaster: a Gaussian factor model over a hierarchy.

    Consumes a `WindowBatch` of already-normalized tensors and returns `FactorParams`
    at the bottom level; coherence is applied downstream by the distribution module.
    """

    def __init__(
        self,
        config: ModelConfig,
        dims: ModelDims,
        S: Tensor | np.ndarray,
        coint_beta: Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        S_t = torch.as_tensor(S, dtype=torch.float32)
        if S_t.dim() != 2 or S_t.shape[0] < S_t.shape[1]:
            raise ValueError(f"S must be [Na+Nb, Nb]; got {tuple(S_t.shape)}")
        self.register_buffer("S", S_t)
        self.config = config
        self.dims = dims
        self.n_hierarchy, self.n_bottom = S_t.shape
        self.n_agg = self.n_hierarchy - self.n_bottom

        self.encoder = ENCODERS.create(
            config.encoder,
            in_channels=1 + (dims.n_hist_features if config.hist_to_encoder else 0),
            input_size=dims.input_size,
            channels=config.temp_conv_channels,
            dilations=config.dilations,
            kernel_size=config.kernel_size,
            residual=config.conv_residual,
        )
        self.mixer = MIXERS.create(
            config.mixer,
            n_hierarchy=self.n_hierarchy,
            n_bottom=self.n_bottom,
            channels=self.encoder.out_channels,
            hidden_size=config.cross_series_hidden,
        )

        self.static_mlp = (
            MLPBlock(dims.n_stat_features, config.static_encoder_dim)
            if dims.n_stat_features > 0
            else None
        )
        self.future_mlp = (
            MLPBlock(dims.n_futr_features * dims.h, config.future_encoder_dim)
            if dims.n_futr_features > 0
            else None
        )
        context_dim = (
            self.mixer.out_channels
            + (config.static_encoder_dim if self.static_mlp is not None else 0)
            + (config.future_encoder_dim if self.future_mlp is not None else 0)
        )

        self.decoder = DECODERS.create(
            config.decoder,
            in_dim=context_dim,
            h=dims.h,
            horizon_agnostic_dim=config.horizon_agnostic_dim,
            horizon_specific_dim=config.horizon_specific_dim,
            n_futr_features=dims.n_futr_features,
        )
        head_factory = HEADS.get(config.head)
        candidate_kwargs = {
            "in_dim": self.decoder.out_dim,
            "n_factors": config.n_factors,
            "sigma_activation": config.sigma_activation,
            "sigma_eps": config.sigma_eps,
            "n_flow_components": config.n_flow_components,
            "a_floor": config.flow_a_floor,
        }
        accepted = inspect.signature(head_factory).parameters
        self.head = head_factory(
            **{k: v for k, v in candidate_kwargs.items() if k in accepted}
        )

        # Zero-initialized so the skip starts as an exact no-op and the adjustment
        # speeds are learned from the CRPS gradient alone.
        self.error_correction: nn.Parameter | None = None
        if config.error_correction_skip:
            if dims.n_hist_features == 0:
                raise ValueError(
                    "error_correction_skip needs n_hist_features >= 1; configure a "
                    "features.history block such as equilibrium_error"
                )
            if config.error_correction_form == "separable":
                self.error_speed: nn.Parameter | None = nn.Parameter(
                    torch.zeros(self.n_bottom, dims.n_hist_features)
                )
                # The response starts at one so the speed alone controls the size.
                self.error_correction = nn.Parameter(
                    torch.ones(dims.h, dims.n_hist_features)
                )
            else:
                self.error_speed = None
                self.error_correction = nn.Parameter(
                    torch.zeros(dims.h, self.n_bottom, dims.n_hist_features)
                )

        self.trend_factors = config.trend_factors
        if config.trend_factors > 0:
            if coint_beta is None:
                raise ValueError(
                    "trend_factors > 0 needs cointegrating vectors; configure an "
                    "equilibrium_error history feature"
                )
            beta_t = torch.as_tensor(coint_beta, dtype=torch.float32)
            if beta_t.dim() != 2 or beta_t.shape[0] != self.n_bottom:
                raise ValueError(
                    f"coint_beta must be [Nb={self.n_bottom}, R]; got "
                    f"{tuple(beta_t.shape)}"
                )
            growth = torch.arange(1, dims.h + 1, dtype=torch.float32)
            if config.trend_growth == "sqrt":
                growth = growth.sqrt()
            elif config.trend_growth == "none":
                growth = torch.ones(dims.h)
            self.register_buffer("coint_beta", beta_t)
            self.register_buffer("trend_growth", growth)
        else:
            self.coint_beta = None
            self.trend_growth = None

    @property
    def h(self) -> int:
        """Forecast horizon."""
        return self.dims.h

    def forward(self, batch: WindowBatch) -> FactorParams:
        b, _, n_bottom = batch.insample_y.shape
        if n_bottom != self.n_bottom:
            raise ValueError(
                f"insample_y has {n_bottom} series but S expects {self.n_bottom}"
            )

        y_hier = torch.einsum("ij,bjl->bil", self.S, batch.insample_y.permute(0, 2, 1))
        h_hist = self.mixer(self.encoder(y_hier, self._hist_channels(batch, b)))

        parts = [h_hist]
        if self.static_mlp is not None and batch.stat_exog is not None:
            static = self.static_mlp(batch.stat_exog)
            # [Nb, S] is shared by every window; [B, Nb, S] lets a static feature vary
            # across the batch, which a panel of hierarchies needs.
            if static.dim() == 2:
                static = static.unsqueeze(0).expand(b, -1, -1)
            parts.append(static)
        futr_h = self._futr_horizon(batch)
        if self.future_mlp is not None and futr_h is not None:
            parts.append(self.future_mlp(futr_h.reshape(b, n_bottom, -1)))

        context = torch.cat(parts, dim=-1)
        params = self.head(self.decoder(context, futr_h))
        return self._structure(params, batch)

    def _structure(self, params, batch: WindowBatch):
        """Cointegration structure: VECM mean adjustment and common-trend loadings.

        The loading constraint is stated in raw units and mapped into the window's
        normalized units through `batch.scale`, so a fixed cointegrating vector means
        the same thing in every window; the mean adjustment acts in normalized units.
        """
        scale = None if batch.scale is None else batch.scale.scale.squeeze(1)  # [B, Nb]
        updates = {}
        if self.error_correction is not None and batch.hist_exog is not None:
            last = batch.hist_exog[:, :, -1, :]  # [B, X, Nb]
            if self.error_speed is not None:
                table = self.error_correction[:, None, :] * self.error_speed[None]
            else:
                table = self.error_correction
            # Normalized units on purpose: a raw-unit table would have to reach the
            # size of a window's standard deviation, which Adam at these learning
            # rates cannot do from zero in one run, leaving the term frozen.
            shift = torch.einsum("bxn,hnx->bhn", last, table)
            if params.mu.dim() == 4:  # a mixture carries a trailing component axis
                shift = shift.unsqueeze(-1)
            updates["mu"] = params.mu + shift
        if self.coint_beta is not None:
            updates["F"] = self._constrain_loadings(params.F, scale)
        return replace(params, **updates) if updates else params

    def _constrain_loadings(self, loadings: Tensor, scale: Tensor | None) -> Tensor:
        """Project the first `trend_factors` loadings onto null(beta^T), raw units.

        With `y_raw = scale * y_norm + loc`, the raw constraint `beta^T F_raw = 0`
        reads `(scale * beta)^T F_norm = 0`, so the projection uses the window-scaled
        vectors. The projected loadings then grow with the horizon, which is where a
        common stochastic trend's variance goes; everything along the equilibrium
        relations is left to the remaining factors and the idiosyncratic term.
        """
        k = self.trend_factors
        b = loadings.shape[0]
        beta = self.coint_beta.unsqueeze(0).expand(b, -1, -1)  # [B, Nb, R]
        if scale is not None:
            beta = beta * scale.unsqueeze(-1)
        trend = loadings[..., :k]  # [B, H, Nb, k]
        gram = beta.transpose(1, 2) @ beta  # [B, R, R]
        ridge = EPS * gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(EPS)
        gram = gram + ridge[:, None, None] * torch.eye(gram.shape[-1], device=gram.device)
        coef = torch.linalg.solve(
            gram.unsqueeze(1), torch.einsum("bnr,bhnk->bhrk", beta, trend)
        )
        trend = trend - torch.einsum("bnr,bhrk->bhnk", beta, coef)
        trend = trend * self.trend_growth.view(1, -1, 1, 1)
        return torch.cat([trend, loadings[..., k:]], dim=-1)

    def _hist_channels(self, batch: WindowBatch, b: int) -> Tensor | None:
        """Lift bottom-level covariates to hierarchy rows, zero-padding the aggregates."""
        if (
            self.dims.n_hist_features == 0
            or batch.hist_exog is None
            or not self.config.hist_to_encoder
        ):
            return None
        hx = batch.hist_exog.permute(0, 3, 1, 2)  # [B, Nb, X, L]
        pad = hx.new_zeros(b, self.n_agg, hx.shape[2], hx.shape[3])
        return torch.cat([pad, hx], dim=1)

    def _futr_horizon(self, batch: WindowBatch) -> Tensor | None:
        """Last H steps of futr_exog as [B, Nb, H, F]."""
        if self.dims.n_futr_features == 0 or batch.futr_exog is None:
            return None
        return batch.futr_exog[:, :, -self.dims.h :, :].permute(0, 3, 2, 1)
