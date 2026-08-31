from __future__ import annotations

import inspect

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


class CLOVER(nn.Module):
    """Coherent probabilistic forecaster: a Gaussian factor model over a hierarchy.

    Consumes a `WindowBatch` of already-normalized tensors and returns `FactorParams`
    at the bottom level; coherence is applied downstream by the distribution module.
    """

    def __init__(
        self, config: ModelConfig, dims: ModelDims, S: Tensor | np.ndarray
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
            in_channels=1 + dims.n_hist_features,
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
        return self.head(self.decoder(context, futr_h))

    def _hist_channels(self, batch: WindowBatch, b: int) -> Tensor | None:
        """Lift bottom-level covariates to hierarchy rows, zero-padding the aggregates."""
        if self.dims.n_hist_features == 0 or batch.hist_exog is None:
            return None
        hx = batch.hist_exog.permute(0, 3, 1, 2)  # [B, Nb, X, L]
        pad = hx.new_zeros(b, self.n_agg, hx.shape[2], hx.shape[3])
        return torch.cat([pad, hx], dim=1)

    def _futr_horizon(self, batch: WindowBatch) -> Tensor | None:
        """Last H steps of futr_exog as [B, Nb, H, F]."""
        if self.dims.n_futr_features == 0 or batch.futr_exog is None:
            return None
        return batch.futr_exog[:, :, -self.dims.h :, :].permute(0, 3, 2, 1)
