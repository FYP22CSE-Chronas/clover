"""HINT (Olivares et al., 2023) in PyTorch, with an optional DSF mixture kernel.

A port of neuralforecast's `HINT` wrapping `NHITS` with a `GMM` loss:

- NHITS reads one series' window at a time (channel independent, shared weights)
  and emits, per horizon step, the means and raw scales of K mixture components.
- The mixture couples series through its components: training sums each
  component's log-likelihood across the series of a batch before the log-sum-exp
  over components (neuralforecast's `batch_correlation`), and sampling draws one
  component per (sample path, step) shared by every series -- the joint mixture of
  the paper's Eq. 7.
- Coherence comes afterwards: samples are reconciled, y -> S P y.

`kernel="dsf"` replaces each component's Gaussian bell with a deep sigmoidal flow
shape, one flow per (series, step) shared by the components. The flow is used in
the normalizing direction, z = DSF((y - mu) / sigma), so the density is exact and
the likelihood objective is unchanged; sampling inverts the monotone flow by
bisection. Nothing else differs from the Gaussian model.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.heads import _flow_shape

KERNELS = ("gaussian", "dsf")
RECONCILERS = ("BottomUp", "MinTraceOLS", "MinTraceWLS")
STD_EPS = 0.2  # neuralforecast GMM.scale_decouple default


# --------------------------------------------------------------------- reconciliation
def reconciliation_P(S: np.ndarray, method: str) -> np.ndarray:
    """P [Nb, N] such that S P maps any forecast to a coherent one.

    Ported from neuralforecast.models.hint: BottomUp keeps the bottom rows; the
    MinTrace variants use the U-form with W = I (OLS) or W = diag(S 1) (WLS, the
    number of bottom series under each node).
    """
    n, nb = S.shape
    n_agg = n - nb
    if method == "BottomUp":
        P = np.zeros((nb, n))
        P[:, n_agg:] = np.eye(nb)
        return P
    if method not in ("MinTraceOLS", "MinTraceWLS"):
        raise ValueError(f"unknown reconciliation {method!r}; expected {RECONCILERS}")
    W = np.eye(n) if method == "MinTraceOLS" else np.diag(S @ np.ones(nb))
    A = S[:n_agg]
    U = np.hstack([np.eye(n_agg), -A]).T  # [N, Na]
    J = np.hstack([np.zeros((nb, n_agg)), np.eye(nb)])  # [Nb, N]
    return J - (J @ W @ U) @ np.linalg.pinv(U.T @ W @ U) @ U.T


# --------------------------------------------------------------------------- NHITS
class NHITSBlock(nn.Module):
    """Pool the window, MLP to (backcast, forecast knots), interpolate the knots.

    Layer order follows neuralforecast: Linear(in, u0), then Linear(u0, u1) and the
    activation, then the output Linear -- there is no activation between the first
    two Linears.
    """

    def __init__(
        self,
        input_size: int,
        h: int,
        n_outputs: int,
        pool_kernel: int,
        freq_downsample: int,
        mlp_units: tuple[int, int] = (512, 512),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.h = h
        self.n_outputs = n_outputs
        self.n_knots = max(h // freq_downsample, 1)
        self.pool = nn.MaxPool1d(pool_kernel, stride=pool_kernel, ceil_mode=True)
        pooled = math.ceil(input_size / pool_kernel)
        layers: list[nn.Module] = [
            nn.Linear(pooled, mlp_units[0]),
            nn.Linear(mlp_units[0], mlp_units[1]),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(mlp_units[1], input_size + n_outputs * self.n_knots))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """[B, L] -> backcast [B, L], forecast [B, h, n_outputs]."""
        theta = self.mlp(self.pool(x.unsqueeze(1)).squeeze(1))
        backcast = theta[:, : self.input_size]
        knots = theta[:, self.input_size :].reshape(len(x), self.n_outputs, self.n_knots)
        forecast = F.interpolate(knots, size=self.h, mode="linear")
        return backcast, forecast.permute(0, 2, 1)


class NHITS(nn.Module):
    """neuralforecast NHITS defaults: 3 identity stacks of one block each, pooling
    [2, 2, 1], frequency downsampling [4, 2, 1], 512-unit MLPs, residual backcast.
    """

    def __init__(
        self,
        input_size: int,
        h: int,
        n_outputs: int,
        pool_kernels: tuple[int, ...] = (2, 2, 1),
        freq_downsample: tuple[int, ...] = (4, 2, 1),
        mlp_units: tuple[int, int] = (512, 512),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            NHITSBlock(input_size, h, n_outputs, k, d, mlp_units, dropout)
            for k, d in zip(pool_kernels, freq_downsample, strict=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        """[B, L] scaled window -> [B, h, n_outputs] raw output channels."""
        residual = x.flip(dims=(-1,))
        forecast = 0.0
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast


# ---------------------------------------------------------------------- the model
class HINT(nn.Module):
    """NHITS + a K-component mixture with Gaussian or DSF kernels.

    `params` returns raw-unit mixture parameters for a batch of scaled windows:
    mu, std [B, h, K] and, for the DSF kernel, flow w/a/b [B, h, U].
    """

    def __init__(
        self,
        input_size: int,
        h: int,
        n_components: int = 10,
        kernel: str = "gaussian",
        n_flow_components: int = 8,
        flow_a_floor: float = 0.1,
        mlp_units: tuple[int, int] = (512, 512),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
        self.k = n_components
        self.kernel = kernel
        self.u = n_flow_components
        self.flow_a_floor = flow_a_floor
        n_outputs = 2 * n_components + (3 * n_flow_components if kernel == "dsf" else 0)
        self.nhits = NHITS(input_size, h, n_outputs, mlp_units=mlp_units, dropout=dropout)

    def params(self, x: Tensor, loc: Tensor, scale: Tensor) -> dict[str, Tensor]:
        """Scaled windows [B, L] and their loc/scale [B] -> raw-unit parameters."""
        out = self.nhits(x)
        k, u = self.k, self.u
        # NHITS starts its forecast from the last observed value; added to the means
        # only (neuralforecast adds it to every output channel).
        means = out[..., :k] + x[:, -1, None, None]
        stds = F.softplus(out[..., k : 2 * k]) + STD_EPS
        loc, scale = loc[:, None, None], scale[:, None, None]
        params = {"mu": means * scale + loc, "std": stds * scale}
        if self.kernel == "dsf":
            raw = out[..., 2 * k :]
            w, a, b = _flow_shape(
                raw[..., :u], raw[..., u : 2 * u], raw[..., 2 * u :], self.flow_a_floor
            )
            params.update(flow_w=w, flow_a=a, flow_b=b)
        return params

    def component_log_prob(self, params: dict[str, Tensor], y: Tensor) -> Tensor:
        """log p_k(y) for every component: y [B, h] -> [B, h, K]."""
        mu, std = params["mu"], params["std"]
        u = (y.unsqueeze(-1) - mu) / std
        if self.kernel == "gaussian":
            return -0.5 * u**2 - 0.5 * math.log(2 * math.pi) - std.log()
        z, log_det = dsf_forward(
            u,
            params["flow_w"].unsqueeze(-2),
            params["flow_a"].unsqueeze(-2),
            params["flow_b"].unsqueeze(-2),
        )
        return -0.5 * z**2 - 0.5 * math.log(2 * math.pi) + log_det - std.log()

    @torch.no_grad()
    def sample(
        self,
        params: dict[str, Tensor],
        n_series: int,
        num_samples: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Joint draws for windows laid out [origins * n_series, h, ...].

        One component index per (origin, step, sample) is shared by all n_series
        series of that origin -- the multivariate mixture. Returns raw-unit samples
        [n_origins, n_series, h, num_samples] (not yet reconciled).
        """
        mu, std = params["mu"], params["std"]
        b, h, k = mu.shape
        n_origins = b // n_series
        comp = torch.randint(k, (n_origins, 1, h, num_samples), generator=generator)
        comp = comp.expand(n_origins, n_series, h, num_samples).reshape(b, h, num_samples)
        mu_s = torch.gather(mu, -1, comp)
        std_s = torch.gather(std, -1, comp)
        z = torch.randn(b, h, num_samples, generator=generator)
        if self.kernel == "dsf":
            z = dsf_inverse(
                z,
                params["flow_w"].unsqueeze(-2),
                params["flow_a"].unsqueeze(-2),
                params["flow_b"].unsqueeze(-2),
            )
        y = mu_s + std_s * z
        return y.reshape(n_origins, n_series, h, num_samples)


# ------------------------------------------------------- DSF, normalizing direction
def dsf_forward(u: Tensor, w: Tensor, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """z = logit(sum_j w_j sigmoid(a_j u + b_j)) and log dz/du, computed in logs.

    `w` sums to one and `a` > 0 over the trailing flow axis (see `_flow_shape`), so
    the map is strictly increasing from R onto R -- a valid change of variables.
    """
    x = a * u.unsqueeze(-1) + b
    log_w = w.clamp_min(1e-12).log()
    log_s = torch.logsumexp(log_w + F.logsigmoid(x), dim=-1)
    log_1ms = torch.logsumexp(log_w + F.logsigmoid(-x), dim=-1)
    z = log_s - log_1ms
    log_ds = torch.logsumexp(log_w + a.log() + F.logsigmoid(x) + F.logsigmoid(-x), dim=-1)
    return z, log_ds - log_s - log_1ms


@torch.no_grad()
def dsf_inverse(
    z: Tensor, w: Tensor, a: Tensor, b: Tensor, bisections: int = 16, newton: int = 6
) -> Tensor:
    """Solve dsf_forward(u) = z: widen a bracket, bisect, then safeguarded Newton.

    Newton uses the flow's own derivative; any step leaving the bracket falls back
    to its midpoint, so the root stays bracketed throughout.
    """
    lo = torch.full_like(z, -1.0)
    hi = torch.full_like(z, 1.0)
    for _ in range(60):
        below = dsf_forward(lo, w, a, b)[0] > z
        above = dsf_forward(hi, w, a, b)[0] < z
        if not (below.any() or above.any()):
            break
        lo = torch.where(below, lo * 2, lo)
        hi = torch.where(above, hi * 2, hi)
    for _ in range(bisections):
        mid = 0.5 * (lo + hi)
        too_low = dsf_forward(mid, w, a, b)[0] < z
        lo = torch.where(too_low, mid, lo)
        hi = torch.where(too_low, hi, mid)
    u = 0.5 * (lo + hi)
    for _ in range(newton):
        f, log_det = dsf_forward(u, w, a, b)
        lo = torch.where(f < z, u, lo)
        hi = torch.where(f < z, hi, u)
        step = u - (f - z) / log_det.exp().clamp_min(1e-12)
        inside = (step > lo) & (step < hi)
        u = torch.where(inside, step, 0.5 * (lo + hi))
    return u


def mixture_nll(log_prob: Tensor, n_series: int) -> Tensor:
    """Composite multivariate-mixture NLL, per series and step.

    `log_prob` [origins * n_series, h, K]: component log-likelihoods are summed over
    the series of each origin (neuralforecast's batch_correlation), then mixed with
    equal weights 1/K (its default, weighted=False).
    """
    b, h, k = log_prob.shape
    joint = log_prob.reshape(b // n_series, n_series, h, k).sum(dim=1)
    nll = -(torch.logsumexp(joint, dim=-1) - math.log(k))
    return nll.mean() / n_series
