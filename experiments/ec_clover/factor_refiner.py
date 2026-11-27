from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from contracts import FactorParams

EPS = 1e-6


def _lca_level_masks(S: Tensor) -> Tensor:
    """Build [L_agg, Nb, Nb] zero-diagonal mixing matrices grouped by LCA group size.

    Each aggregate row of `S` (`[:-Nb]`) defines a group of bottom series of size
    `k > 1`. For every pair `(i, j)` with `i != j`, its lowest common ancestor (LCA)
    is the smallest aggregate containing both `i` and `j`. Grouping LCAs by group size
    yields one symmetric, row-normalized mixing basis `M_l` per hierarchy level.
    """
    n_total, n_bottom = S.shape
    agg = (S[: n_total - n_bottom] > 0).to(torch.float32)  # [Na, Nb]
    sizes = agg.sum(dim=1)  # [Na]
    valid = sizes > 1
    agg = agg[valid]
    sizes = sizes[valid]

    # Sort smallest -> largest so the first aggregate containing (i, j) is the LCA
    order = torch.argsort(sizes)
    agg = agg[order]
    sizes = sizes[order]

    # co[a, i, j] = 1 iff aggregate a contains both i and j (i != j)
    eye = torch.eye(n_bottom, device=S.device, dtype=torch.bool)
    co = (agg.unsqueeze(2) * agg.unsqueeze(1)) > 0  # [Na, Nb, Nb]
    co = co & (~eye.unsqueeze(0))

    # Assign each pair (i, j) to its smallest containing aggregate's size
    assigned = torch.zeros((n_bottom, n_bottom), device=S.device, dtype=torch.bool)
    pair_size = torch.zeros((n_bottom, n_bottom), device=S.device, dtype=torch.float32)
    for a in range(agg.shape[0]):
        take = co[a] & (~assigned)
        pair_size[take] = sizes[a]
        assigned = assigned | take

    unique_sizes = torch.unique(pair_size[assigned])
    masks = []
    for sz in unique_sizes:
        m = (pair_size == sz).to(torch.float32)
        # Row-normalize so alpha_l is O(1) regardless of group size
        row_sum = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        masks.append(m / row_sum)
    if not masks:
        return torch.zeros((0, n_bottom, n_bottom), device=S.device, dtype=torch.float32)
    return torch.stack(masks, dim=0)  # [L_agg, Nb, Nb]


class CrossSeriesFactorRefiner(nn.Module):
    """Hierarchy-structured, variance-preserving cross-series rotation on F.

    Mixes unit-norm factor directions across bottom series using one learned
    scalar weight `alpha_l` per hierarchy level of `S` (initialized to 0 -> exact
    identity at step 0), then rescales back to each series' original ||F_{i,h}||
    so marginal variances `sigma^2 + ||F||^2` are preserved exactly.
    """

    def __init__(self, S: Tensor, n_bottom: int) -> None:
        super().__init__()
        self.n_bottom = n_bottom
        masks = _lca_level_masks(S)
        self.register_buffer("masks", masks)  # [L_agg, Nb, Nb]
        self.alphas = nn.Parameter(torch.zeros(masks.shape[0]))

    def forward(self, params: FactorParams, insample_y: Tensor) -> FactorParams:
        """Mix factor directions across `Nb` while preserving per-series ||F||."""
        del insample_y
        if self.masks.shape[0] == 0:
            return params
        f = params.F  # [B, H, Nb, K]
        norms = f.norm(dim=-1, keepdim=True).clamp_min(EPS)
        f_unit = f / norms

        w = torch.eye(self.n_bottom, device=f.device, dtype=f.dtype) + torch.einsum(
            "l,lnm->nm", self.alphas, self.masks.to(dtype=f.dtype)
        )
        mixed = torch.einsum("nm,bhmk->bhnk", w, f_unit)
        mixed_unit = mixed / mixed.norm(dim=-1, keepdim=True).clamp_min(EPS)
        return FactorParams(mu=params.mu, sigma=params.sigma, F=mixed_unit * norms)
