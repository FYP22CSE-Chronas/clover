from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from config import ModelConfig
from experiments.ec_clover.config import ECConfig
from model.encoders import ENCODERS

EPS = 1e-6


class ErrorCorrectionBranch(nn.Module):
    """Historical error-correction representation over bottom-level history.

    For every historical timestep tau, `e_tau = beta^T y_tau` (`beta in R^(Nb x r)`)
    and `ec_tau = alpha @ e_tau` (`alpha in R^(Nb x r)`), computed for the whole window
    at once. The resulting sequence is fed through an ordinary `Encoder` from the shared
    registry (the same contract `CLOVER`'s own temporal encoder uses: `[B, N, L] ->
    [B, N, C]`, treating each channel as an independent series with shared weights),
    which keeps the branch temporally sensitive rather than collapsing to the last value
    -- two histories with the same endpoint but different equilibrium dynamics produce
    different embeddings because the encoder sees the whole causal window.

    `ec_config` selects among several branch-internal variants (see `ECConfig` for the
    reasoning behind each):

    - `ec_input="ec"` encodes `alpha beta^T y` over `Nb` channels; `"e"` encodes
      `beta^T y` over `r` channels and lifts the embedding with `alpha` afterwards.
    - `include_diff` adds the sequence's first difference as a second encoder channel,
      supplying the short-run (`Gamma delta y`) half of a VECM alongside the long-run
      error-correction term.
    - `sequence_norm` centers (or standardizes) the sequence over the window. This
      matters because `CLOVER` hands the branch *per-window normalized* history: with
      `y_norm = (y - mu_w) / sigma_w`, `beta^T y_norm = sum_i (beta_i/sigma_i) y_i +
      const`, so the window-dependent offset and per-series rescaling corrupt the
      equilibrium level even when `beta` is exactly right. Centering removes the offset
      and leaves the deviation dynamics, which is the part that survives normalization.

    Identification note (cointegration vectors have scale/sign indeterminacy: `(beta,
    alpha)` and `(c*beta, alpha/c)` describe the same equilibrium relationship for any
    nonzero `c`). By default this implementation does not renormalize `beta` during
    training: a Johansen-supplied `beta` is used exactly as given, and a randomly
    initialized `beta` starts with unit-norm columns only at construction, not as a
    running constraint. Stability instead comes from initializing `alpha` small
    (`alpha_init_std`, default 1e-2), so the branch starts as a near-zero perturbation
    of the baseline pathway. Setting `beta_normalize="columns"` opts into pinning
    `beta`'s scale on every forward pass; that *does* rescale a supplied Johansen
    vector, which is why it is opt-in rather than the default.
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
        if ec_config.rank > n_bottom:
            raise ValueError(f"ec.rank {ec_config.rank} must be <= n_bottom {n_bottom}")
        self.n_bottom = n_bottom
        self.rank = ec_config.rank
        self.ec_input = ec_config.ec_input
        self.include_diff = ec_config.include_diff
        self.sequence_norm = ec_config.sequence_norm
        self.beta_normalize = ec_config.beta_normalize

        beta = _init_beta(n_bottom, ec_config, beta_init)
        self.beta = nn.Parameter(beta)
        self.alpha = nn.Parameter(
            torch.randn(n_bottom, ec_config.rank) * ec_config.alpha_init_std
        )

        dilations = ec_config.encoder_dilations or model_config.dilations
        receptive_field = 1 + sum((model_config.kernel_size - 1) * d for d in dilations)
        if ec_config.encoder == "dilated_conv" and receptive_field > input_size:
            raise ValueError(
                f"ec encoder receptive field {receptive_field} exceeds input_size "
                f"{input_size}; shorten ec.encoder_dilations {tuple(dilations)}"
            )
        self.encoder = ENCODERS.create(
            ec_config.encoder,
            in_channels=2 if ec_config.include_diff else 1,
            input_size=input_size,
            channels=ec_config.encoder_channels,
            dilations=dilations,
            kernel_size=model_config.kernel_size,
            residual=model_config.conv_residual,
        )
        self.out_channels = self.encoder.out_channels

    def forward(self, y: Tensor) -> Tensor:
        """[B, L, Nb] insample history -> [B, Nb, C] EC representation."""
        beta = self._beta()
        e = torch.einsum("bln,nr->blr", y, beta)  # [B, L, r] equilibrium errors
        seq = e if self.ec_input == "e" else torch.einsum("blr,nr->bln", e, self.alpha)
        seq = self._normalize(seq)

        x = seq.permute(0, 2, 1)  # [B, N, L], N = r or Nb
        h = self.encoder(x, self._diff(x))  # [B, N, C]
        if self.ec_input == "e":
            # Lift the r equilibrium-error embeddings onto the bottom series with the
            # same adjustment matrix that would have scaled the raw signal.
            h = torch.einsum("brc,nr->bnc", h, self.alpha)
        return h

    def _beta(self) -> Tensor:
        if self.beta_normalize == "columns":
            return self.beta / self.beta.norm(dim=0, keepdim=True).clamp_min(EPS)
        return self.beta

    def _normalize(self, seq: Tensor) -> Tensor:
        """Remove the window-dependent offset (and optionally scale) from [B, L, N]."""
        if self.sequence_norm == "none":
            return seq
        centered = seq - seq.mean(dim=1, keepdim=True)
        if self.sequence_norm == "center":
            return centered
        return centered / (seq.std(dim=1, keepdim=True) + EPS)

    def _diff(self, x: Tensor) -> Tensor | None:
        """First difference as an extra encoder channel: [B, N, L] -> [B, N, 1, L]."""
        if not self.include_diff:
            return None
        delta = torch.zeros_like(x)
        delta[..., 1:] = x[..., 1:] - x[..., :-1]
        return delta.unsqueeze(2)


def _init_beta(
    n_bottom: int, ec_config: ECConfig, beta_init: Tensor | np.ndarray | None
) -> Tensor:
    """Resolve the initial beta from config + an optional caller-supplied tensor."""
    if ec_config.beta_init.type == "johansen":
        supplied = beta_init
        if supplied is None:
            if ec_config.beta_init.values is not None:
                supplied = torch.tensor(ec_config.beta_init.values, dtype=torch.float32)
            elif ec_config.beta_init.values_path is not None:
                supplied = torch.as_tensor(
                    np.load(ec_config.beta_init.values_path), dtype=torch.float32
                )
        if supplied is None:
            raise ValueError(
                "beta_init.type == 'johansen' but no beta was supplied via "
                "config values/values_path or the beta_init argument"
            )
        supplied = torch.as_tensor(supplied, dtype=torch.float32)
        expected = (n_bottom, ec_config.rank)
        if tuple(supplied.shape) != expected:
            raise ValueError(
                f"supplied Johansen beta has shape {tuple(supplied.shape)}, "
                f"expected {expected}"
            )
        return supplied.clone()

    raw = torch.randn(n_bottom, ec_config.rank)
    return raw / raw.norm(dim=0, keepdim=True).clamp_min(EPS)
