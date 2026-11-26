"""HierE2E (Rangapuram et al., ICML 2021) in PyTorch, with an optional DSF marginal.

A port of GluonTS's `DeepVARHierarchicalEstimator` (MXNet). One LSTM reads the whole
hierarchy at once -- lagged values of all N = Na+Nb series, a learned per-series
index value, and calendar features -- and emits, per time step, an independent
Gaussian for every series (GluonTS hard-codes rank=0, so the covariance is
diagonal). Samples are projected onto the coherent subspace, y -> M y with
M = S (S^T S)^+ S^T, and the projected samples are scored by CRPS. The projection is
the only source of cross-series dependence.

`marginal="dsf"` swaps sigma * z for sigma * DSF(z), z ~ N(0, 1), using the repo's
deep sigmoidal flow -- the same substitution the CLOVER flow head makes -- and
changes nothing else.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.distribution import _dsf_transform
from model.heads import _flow_shape

MARGINALS = ("gaussian", "dsf")


def projection_matrix(S: np.ndarray | Tensor) -> Tensor:
    """M = S (S^T S)^+ S^T: the orthogonal projection onto coherent vectors."""
    S_t = torch.as_tensor(S, dtype=torch.float64)
    M = S_t @ torch.linalg.pinv(S_t.T @ S_t) @ S_t.T
    return M.float()


def project(M: Tensor, samples: Tensor) -> Tensor:
    """Apply M along the series axis of [..., N, n_samples] samples."""
    return torch.einsum("ij,...js->...is", M, samples)


class ZoneoutLSTM(nn.Module):
    """Stacked LSTM cells as GluonTS builds them for DeepVAR.

    Layers after the first are residual (output + input). Zoneout on the states
    (GluonTS's `dropout_rate`): in training each state unit keeps its previous value
    with probability p; at inference states always update, matching MXNet's
    ZoneoutCell.
    """

    def __init__(self, input_size: int, hidden: int, num_layers: int, zoneout: float):
        super().__init__()
        self.hidden = hidden
        self.zoneout = zoneout
        self.cells = nn.ModuleList(
            nn.LSTMCell(input_size if i == 0 else hidden, hidden)
            for i in range(num_layers)
        )

    def init_state(self, batch: int, ref: Tensor) -> list[tuple[Tensor, Tensor]]:
        zeros = ref.new_zeros(batch, self.hidden)
        return [(zeros, zeros) for _ in self.cells]

    def step(
        self, x: Tensor, state: list[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """One time step: [B, I] -> [B, H]."""
        new_state = []
        inp = x
        for i, (cell, (h_old, c_old)) in enumerate(zip(self.cells, state, strict=True)):
            h_new, c_new = cell(inp, (h_old, c_old))
            if self.training and self.zoneout > 0:
                keep_h = torch.rand_like(h_new) < self.zoneout
                keep_c = torch.rand_like(c_new) < self.zoneout
                h_new = torch.where(keep_h, h_old, h_new)
                c_new = torch.where(keep_c, c_old, c_new)
            new_state.append((h_new, c_new))
            inp = h_new + inp if i > 0 else h_new
        return inp, new_state

    def forward(
        self, x: Tensor, state: list[tuple[Tensor, Tensor]] | None = None
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """[B, T, I] -> [B, T, H] and the final state."""
        if state is None:
            state = self.init_state(x.shape[0], x)
        outputs = []
        for t in range(x.shape[1]):
            out, state = self.step(x[:, t], state)
            outputs.append(out)
        return torch.stack(outputs, dim=1), state


class HierE2E(nn.Module):
    """DeepVAR backbone + per-series marginal + coherent projection.

    Inputs per step: lags of every series (already divided by the window scale),
    one learned value per series (GluonTS's target-dimension embedding, dim 1), and
    calendar features. Outputs per step, in scaled units: mu [.., N], sigma [.., N]
    and, for the DSF marginal, flow weights/slopes/shifts [.., N, K].
    """

    SIGMA_INIT = 1.0  # GluonTS LowrankMultivariateGaussianOutput defaults
    SIGMA_MIN = 1e-3

    def __init__(
        self,
        S: np.ndarray,
        n_lags: int,
        n_time_features: int,
        num_layers: int = 2,
        num_cells: int = 40,
        zoneout: float = 0.1,
        marginal: str = "gaussian",
        n_flow_components: int = 8,
        flow_a_floor: float = 0.1,
    ) -> None:
        super().__init__()
        if marginal not in MARGINALS:
            raise ValueError(f"marginal must be one of {MARGINALS}, got {marginal!r}")
        n = S.shape[0]
        self.n_series = n
        self.marginal = marginal
        self.k = n_flow_components
        self.flow_a_floor = flow_a_floor
        self.register_buffer("M", projection_matrix(S))
        self.index_embedding = nn.Embedding(n, 1)
        input_size = n_lags * n + n + n_time_features
        self.rnn = ZoneoutLSTM(input_size, num_cells, num_layers, zoneout)
        per_series = 2 + (3 * self.k if marginal == "dsf" else 0)
        self.proj = nn.Linear(num_cells, n * per_series)
        # softplus(raw + bias) = sigma_init^2 at raw = 0
        self.d_bias = math.log(math.expm1(self.SIGMA_INIT**2))

    def inputs(self, lags: Tensor, time_feat: Tensor) -> Tensor:
        """[B, T, L, N] scaled lags + [B, T, F] features -> [B, T, I]."""
        b, t = lags.shape[:2]
        idx = self.index_embedding.weight[:, 0].expand(b, t, -1)
        return torch.cat([lags.reshape(b, t, -1), idx, time_feat], dim=-1)

    def params(self, hidden: Tensor) -> dict[str, Tensor]:
        """[..., H] -> distribution parameters in scaled units."""
        n = self.n_series
        out = self.proj(hidden)
        lead = out.shape[:-1]
        mu = out[..., :n]
        variance = F.softplus(out[..., n : 2 * n] + self.d_bias) + self.SIGMA_MIN**2
        params = {"mu": mu, "sigma": variance.sqrt()}
        if self.marginal == "dsf":
            raw = out[..., 2 * n :].reshape(*lead, n, 3 * self.k)
            w, a, b = _flow_shape(
                raw[..., : self.k],
                raw[..., self.k : 2 * self.k],
                raw[..., 2 * self.k :],
                self.flow_a_floor,
            )
            params.update(flow_w=w, flow_a=a, flow_b=b)
        return params

    def sample(
        self,
        params: dict[str, Tensor],
        scale: Tensor,
        num_samples: int,
        coherent: bool = True,
    ) -> Tensor:
        """Reparameterized draws in raw units, [..., N, num_samples].

        `scale` broadcasts against mu ([B, 1, N] for a sequence, [B, N] for a step).
        """
        mu, sigma = params["mu"], params["sigma"]
        z = torch.randn(*mu.shape, num_samples, device=mu.device, dtype=mu.dtype)
        if self.marginal == "dsf":
            z = _dsf_transform(z, params["flow_w"], params["flow_a"], params["flow_b"])
        raw = (mu.unsqueeze(-1) + sigma.unsqueeze(-1) * z) * scale.unsqueeze(-1)
        return project(self.M, raw) if coherent else raw
