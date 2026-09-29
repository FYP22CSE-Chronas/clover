from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from config import ModelConfig, ModelDims
from contracts import FactorParams, GMMParams, SkewTParams, WindowBatch
from experiments.ec_clover.config import ECConfig
from experiments.ec_clover.ec_branch import ErrorCorrectionBranch
from experiments.ec_clover.fusion import FUSIONS
from model.network import CLOVER


class CloverWithErrorCorrection(CLOVER):
    """CLOVER plus an additional error-correction branch over bottom-level history.

    Adds `EC_t = alpha @ (beta^T @ y_t)` as a pathway parallel to the existing
    temporal-encoder + cross-series-mixer representation, fused right before the
    static/future context is assembled and handed to the decoder -- see
    `experiments/ec_clover/NOTES.md` for why that point in `CLOVER.forward` is the
    correct insertion site. The decoder, head, distribution, sampling, hierarchy
    aggregation and CRPS objective are untouched: this subclass only overrides
    `forward`, and every submodule it uses (`self.encoder`, `self.mixer`,
    `self.static_mlp`, `self.future_mlp`, `self.decoder`, `self.head`) is the one
    `CLOVER.__init__` builds via `super().__init__()`, not a reimplementation.

    With `ec_config.enabled=False` (the default), no EC modules are constructed and
    `forward` delegates straight to `CLOVER.forward`, so the baseline is reproduced
    exactly, not approximately.
    """

    def __init__(
        self,
        config: ModelConfig,
        dims: ModelDims,
        S: Tensor | np.ndarray,
        ec_config: ECConfig | None = None,
        beta_init: Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__(config, dims, S)
        self.ec_config = ec_config or ECConfig()
        if self.ec_config.enabled:
            self.ec_branch: ErrorCorrectionBranch | None = ErrorCorrectionBranch(
                n_bottom=self.n_bottom,
                input_size=dims.input_size,
                ec_config=self.ec_config,
                model_config=config,
                beta_init=beta_init,
            )
            self.fusion: nn.Module | None = FUSIONS.create(
                self.ec_config.fusion,
                clover_dim=self.mixer.out_channels,
                ec_dim=self.ec_branch.out_channels,
                out_dim=self.mixer.out_channels,
                hidden_size=self.ec_config.fusion_hidden,
                n_layers=self.ec_config.fusion_layers,
                gate_init=self.ec_config.gate_init,
            )
        else:
            self.ec_branch = None
            self.fusion = None

    def forward(self, batch: WindowBatch) -> FactorParams | SkewTParams | GMMParams:
        if self.ec_branch is None or self.fusion is None:
            return super().forward(batch)

        b, _, n_bottom = batch.insample_y.shape
        if n_bottom != self.n_bottom:
            raise ValueError(
                f"insample_y has {n_bottom} series but S expects {self.n_bottom}"
            )

        y_hier = torch.einsum("ij,bjl->bil", self.S, batch.insample_y.permute(0, 2, 1))
        h_hist = self.mixer(self.encoder(y_hier, self._hist_channels(batch, b)))
        h_hist = self.fusion(h_hist, self.ec_branch(batch.insample_y))

        parts = [h_hist]
        if self.static_mlp is not None and batch.stat_exog is not None:
            static = self.static_mlp(batch.stat_exog)
            if static.dim() == 2:
                static = static.unsqueeze(0).expand(b, -1, -1)
            parts.append(static)
        futr_h = self._futr_horizon(batch)
        if self.future_mlp is not None and futr_h is not None:
            parts.append(self.future_mlp(futr_h.reshape(b, n_bottom, -1)))

        context = torch.cat(parts, dim=-1)
        return self.head(self.decoder(context, futr_h))
