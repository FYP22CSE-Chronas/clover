from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from config import ModelConfig, ModelDims
from contracts import FactorParams, WindowBatch
from model.network import CLOVER

EPS = 1e-6


@dataclass(frozen=True)
class RICConfig:
    """Realized-innovation-correlation dependence for the factor model.

    `source`: what the window correlation is estimated from. `diff` (first
    differences, the innovations of an I(1) system) is the method; `level` (the
    z-scored levels themselves) is the spurious-correlation ablation.
    `mode`: `ric` replaces the network's dependence with the shrunk window
    correlation, `hybrid` mixes it with the network's own through a learned gate,
    `independent` drops all cross-series dependence (shrinkage pinned at one).
    The network's marginal variance `sigma^2 + |F|^2` is kept exactly in every mode,
    so the variants differ only in the copula.
    """

    source: str = "diff"
    mode: str = "ric"
    shrink_init: float = 0.5
    learn_shrink: bool = True
    hybrid_gate_init: float = -2.0
    demean: bool = True
    # `identity` shrinks toward independence; `tree` toward the window's mean
    # correlation within each class of pairs sharing a lowest common ancestor at the
    # same depth of the hierarchy (siblings, cousins, ...), read off S alone.
    target: str = "identity"
    # `learned`: one logit per horizon trained by the CRPS; `analytic`: the
    # Schafer-Strimmer optimal intensity, recomputed from every window.
    shrink: str = "learned"
    # Differencing lag of the innovations; the seasonal period removes a seasonal
    # unit root as well, so a seasonal pattern is not mistaken for co-movement.
    diff_lag: int = 1
    # Learned per-horizon attenuation rho of the estimated correlation toward I:
    # R -> rho R + (1 - rho) I. A forecast error carries the network's own mean error
    # on top of the innovations, which is largely idiosyncratic, so it is less
    # correlated across series than the innovations the estimate is read from.
    attenuate: bool = False
    attenuate_init: float = 0.8
    # rho = sigmoid(logit(init) + scale * raw): multiplying the raw parameter makes
    # Adam move the logit `scale` times faster; at scale 1 the learned shrinkage
    # never left its initial value inside the step budget.
    attenuate_scale: float = 20.0


def pair_classes(S: Tensor) -> Tensor:
    """[Nb, Nb] class of each bottom pair: the depth of its lowest common ancestor.

    The ancestor is the aggregate row with the fewest members containing both series;
    its depth is how many aggregate rows strictly contain it. Depth classes are
    renumbered 0..C-1; the diagonal is -1.
    """
    n_bottom = S.shape[1]
    agg = S[:-n_bottom] > 0  # [Na, Nb]
    size = agg.sum(1)
    contains = (agg.float() @ agg.float().T) == size[None, :].float()  # a ⊇ b
    strict = contains & (size[:, None] > size[None, :])
    depth = strict.sum(0)  # [Na]
    both = agg[:, :, None] & agg[:, None, :]  # [Na, Nb, Nb]
    masked = torch.where(both, size[:, None, None], size.max() + 1)
    lca = masked.argmin(0)  # [Nb, Nb]
    cls = depth[lca]
    _, cls = torch.unique(cls, return_inverse=True)
    cls = cls.clone()
    cls.fill_diagonal_(-1)
    return cls


class RICClover(CLOVER):
    """CLOVER whose cross-series dependence is re-estimated in every window from the
    realized innovations of that window, instead of being read off the z-scored
    levels by a per-series head.

    The network still predicts every bottom series' marginal scale. The correlation
    between series at horizon `eta` is `(1 - lam_eta) * R_hat + lam_eta * I`, with
    `R_hat` the correlation of the window's first differences and `lam_eta` a learned
    shrinkage per horizon; its top-K eigenpairs become the factor loadings and the
    remainder the idiosyncratic term, so the result is still an exact diag + low-rank
    Gaussian factor model and sampling, coherence and the CRPS objective are unchanged.
    """

    def __init__(
        self,
        config: ModelConfig,
        dims: ModelDims,
        S: Tensor | np.ndarray,
        ric: RICConfig,
        coint_beta: Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__(config, dims, S, coint_beta=coint_beta)
        if ric.source not in ("diff", "level", "multi"):
            raise ValueError(f"unknown RIC source {ric.source!r}")
        if ric.mode not in ("ric", "hybrid", "independent"):
            raise ValueError(f"unknown RIC mode {ric.mode!r}")
        self.ric = ric
        p = min(max(ric.shrink_init, 1e-3), 1 - 1e-3)
        logit = torch.full((dims.h,), math.log(p / (1 - p)))
        if ric.learn_shrink:
            self.shrink_logit = nn.Parameter(logit)
        else:
            self.register_buffer("shrink_logit", logit)
        if ric.mode == "hybrid":
            self.hybrid_gate = nn.Parameter(torch.tensor(float(ric.hybrid_gate_init)))
        if ric.target not in ("identity", "tree") or ric.shrink not in (
            "learned",
            "analytic",
        ):
            raise ValueError(f"bad RIC target/shrink: {ric.target!r}/{ric.shrink!r}")
        cls = pair_classes(self.S)
        n_cls = int(cls.max().item()) + 1
        self.register_buffer(
            "class_masks", torch.stack([(cls == c).float() for c in range(n_cls)])
        )
        self.n_ric = min(max(config.n_factors, 1), self.n_bottom)
        if ric.attenuate:
            p = min(max(ric.attenuate_init, 1e-3), 1 - 1e-3)
            self.register_buffer("attenuate_offset", torch.tensor(math.log(p / (1 - p))))
            self.attenuate_raw = nn.Parameter(torch.zeros(dims.h))

    def attenuation(self) -> Tensor | None:
        """Per-horizon rho in (0, 1), or None when attenuation is off."""
        if not self.ric.attenuate:
            return None
        return torch.sigmoid(
            self.attenuate_offset + self.ric.attenuate_scale * self.attenuate_raw
        )

    def _target_and_intensity(self, x: Tensor, corr: Tensor) -> tuple[Tensor, Tensor]:
        """Shrinkage target [B, Nb, Nb] and Schafer-Strimmer intensity [B]."""
        eye = torch.eye(corr.shape[-1], device=corr.device, dtype=corr.dtype)
        if self.ric.target == "tree":
            masks = self.class_masks  # [C, Nb, Nb]
            count = masks.sum((1, 2)).clamp_min(1)
            mean = torch.einsum("bij,cij->bc", corr, masks) / count
            target = torch.einsum("bc,cij->bij", mean, masks) + eye
        else:
            target = eye.expand_as(corr)
        n = x.shape[1]
        z = x / x.std(dim=1, keepdim=True).clamp_min(EPS)
        w = z.unsqueeze(-1) * z.unsqueeze(-2)  # [B, n, Nb, Nb]
        var_r = n / (n - 1) ** 3 * (w - w.mean(1, keepdim=True)).pow(2).sum(1)
        off = 1 - eye
        num = (var_r * off).sum((1, 2))
        den = ((corr - target).pow(2) * off).sum((1, 2)).clamp_min(EPS)
        return target, (num / den).clamp(0.0, 1.0)

    def window_correlation(self, y: Tensor) -> Tensor:
        """[B, L, Nb] normalized history -> [B, Nb, Nb] correlation matrix.

        Correlation is invariant to the per-series affine window scaling, so the
        normalized history gives the same matrix as the raw one.
        """
        x = y[:, 1:] - y[:, :-1] if self.ric.source == "diff" else y
        if self.ric.demean:
            x = x - x.mean(dim=1, keepdim=True)
        cov = x.transpose(1, 2) @ x / max(x.shape[1] - 1, 1)
        sd = cov.diagonal(dim1=-2, dim2=-1).clamp_min(EPS).sqrt()
        corr = cov / (sd.unsqueeze(-1) * sd.unsqueeze(-2))
        eye = torch.eye(corr.shape[-1], device=corr.device, dtype=corr.dtype)
        # A flat series has no estimable correlation; treat it as independent.
        flat = (cov.diagonal(dim1=-2, dim2=-1) < EPS).unsqueeze(-1)
        corr = torch.where(flat | flat.transpose(1, 2), eye, corr)
        return 0.5 * (corr + corr.transpose(1, 2))

    def _structure(self, params, batch: WindowBatch):
        params = super()._structure(params, batch)
        if not isinstance(params, FactorParams):
            raise TypeError("RIC dependence is defined for the factor_model head only")
        marginal_var = params.sigma.pow(2) + params.F.pow(2).sum(-1)  # [B, H, Nb]
        s = marginal_var.clamp_min(EPS).sqrt()

        if self.ric.mode == "independent":
            return replace(params, sigma=s, F=torch.zeros_like(params.F[..., :1]))

        if self.ric.target == "identity" and self.ric.shrink == "learned":
            with torch.no_grad():
                corr = self.window_correlation(self._dependence_history(batch))
                evals, evecs = torch.linalg.eigh(corr)  # ascending
                k = self.n_ric
                evals = evals[..., -k:].clamp_min(0.0)  # [B, K]
                evecs = evecs[..., -k:]  # [B, Nb, K]
            lam = torch.sigmoid(self.shrink_logit)  # [H]
            # Shrinking toward I keeps the eigenvectors and maps e -> (1-lam) e + lam.
            shrunk = (1 - lam)[None, :, None] * evals[:, None, :] + lam[None, :, None]
            unit_f = evecs[:, None] * shrunk.sqrt()[:, :, None, :]  # [B, H, Nb, K]
        else:
            unit_f = self._structured_unit_loadings(
                self._dependence_history(batch), s.shape[1]
            )
        rho = self.attenuation()
        if rho is not None:
            # Scaling unit loadings by sqrt(rho) maps R to rho R + (1 - rho) I exactly.
            unit_f = unit_f * rho.sqrt()[None, :, None, None]
        # R_lam has a unit diagonal, so the leftover per-series variance is exact.
        unit_resid = (1 - unit_f.pow(2).sum(-1)).clamp_min(EPS)
        loadings = s.unsqueeze(-1) * unit_f
        idio_var = s.pow(2) * unit_resid

        if self.ric.mode == "hybrid":
            w = torch.sigmoid(self.hybrid_gate)
            loadings = torch.cat([(1 - w).sqrt() * loadings, w.sqrt() * params.F], dim=-1)
            idio_var = (1 - w) * idio_var + w * params.sigma.pow(2)
        return replace(params, sigma=idio_var.clamp_min(EPS).sqrt(), F=loadings)

    @staticmethod
    def _dependence_history(batch: WindowBatch) -> Tensor:
        """The longer raw history when the batcher attached one, else the window."""
        extra = getattr(batch, "ric_history", None)
        return batch.insample_y if extra is None else extra

    def _changes(self, y: Tensor, lag: int) -> Tensor:
        """`lag`-step changes of the window (or the levels), demeaned: [B, n, Nb]."""
        x = y if self.ric.source == "level" else y[:, lag:] - y[:, :-lag]
        return x - x.mean(dim=1, keepdim=True) if self.ric.demean else x

    @staticmethod
    def _corr(x: Tensor) -> Tensor:
        cov = x.transpose(1, 2) @ x / max(x.shape[1] - 1, 1)
        var = cov.diagonal(dim1=-2, dim2=-1)
        sd = var.clamp_min(EPS).sqrt()
        corr = cov / (sd.unsqueeze(-1) * sd.unsqueeze(-2))
        eye = torch.eye(corr.shape[-1], device=corr.device, dtype=corr.dtype)
        flat = (var < EPS).unsqueeze(-1)
        corr = torch.where(flat | flat.transpose(1, 2), eye, corr)
        return 0.5 * (corr + corr.transpose(1, 2))

    @torch.no_grad()
    def _structured_unit_loadings(self, y: Tensor, h: int) -> Tensor:
        """Unit-diagonal loadings [B, H, Nb, K] of the shrunk window correlation.

        With `source="multi"` horizon `eta` uses the correlation of the window's
        `eta`-step changes: the dependence of an `eta`-step forecast error, which in a
        cointegrated system drifts away from that of the one-step innovations as the
        error-correcting directions stop accumulating.
        """
        if self.ric.source == "multi":
            max_lag = max(y.shape[1] - 3, 1)
            lags = [min(eta, max_lag) for eta in range(1, h + 1)]
        else:
            lags = [self.ric.diff_lag] * h
        mats = {}
        for lag in sorted(set(lags)):
            x = self._changes(y, lag)
            corr = self._corr(x)
            target, lam = self._target_and_intensity(x, corr)
            mats[lag] = (corr, target, lam)
        corr = torch.stack([mats[g][0] for g in lags], 1)  # [B, H, Nb, Nb]
        target = torch.stack([mats[g][1] for g in lags], 1)
        if self.ric.shrink == "learned":
            lam_h = torch.sigmoid(self.shrink_logit)[None, :, None, None]
        else:
            lam_h = torch.stack([mats[g][2] for g in lags], 1)[..., None, None]
        mixed = (1 - lam_h) * corr + lam_h * target  # [B, H, Nb, Nb]
        evals, evecs = torch.linalg.eigh(mixed)
        k = self.n_ric
        unit_f = evecs[..., -k:] * evals[..., -k:].clamp_min(0.0).sqrt().unsqueeze(-2)
        # A target that is not PSD can push a row past unit variance once its
        # negative eigenvalues are clipped; scale such rows back onto the unit sphere.
        norm = unit_f.pow(2).sum(-1, keepdim=True)
        return unit_f / norm.clamp_min(1 - 1e-4).sqrt() * (1 - 1e-4) ** 0.5
