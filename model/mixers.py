from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from contracts import ScaleStats
from model.blocks import MLPBlock
from registry import Registry

MIXERS: Registry[nn.Module] = Registry("mixer")


@MIXERS.register("cross_series_mlp")
class CrossSeriesMixer(nn.Module):
    """Residual cross-series mixer; hidden_size=0 makes it an identity passthrough."""

    def __init__(
        self, n_hierarchy: int, n_bottom: int, channels: int, hidden_size: int
    ) -> None:
        super().__init__()
        self.n_bottom = n_bottom
        self.channels = channels
        self.out_channels = channels
        self.enabled = hidden_size > 0
        self.mlp = (
            MLPBlock(
                in_dim=n_hierarchy * channels,
                out_dim=n_bottom * channels,
                hidden_size=hidden_size,
            )
            if self.enabled
            else None
        )

    def forward(self, h: Tensor) -> Tensor:
        """[B, Na+Nb, C] -> [B, Nb, C]."""
        # S = [A; I], so the trailing Nb rows are the bottom series themselves.
        residual = h[:, -self.n_bottom :, :]
        if self.mlp is None:
            return residual
        b = h.shape[0]
        mixed = self.mlp(h.reshape(b, -1)).reshape(b, self.n_bottom, self.channels)
        return residual + mixed


def hierarchy_edges(S: Tensor | np.ndarray) -> tuple[Tensor, Tensor]:
    """Directed (src, dst) node indices for every bottom<->aggregate link in S.

    Nodes follow S's row order: aggregates 0..Na-1, then bottoms Na..Na+Nb-1.
    Each nonzero A[i, j] of S = [A; I] gives the edge pair i <-> Na+j; links
    between aggregates are deliberately left out. The first half of the list runs
    bottom -> aggregate, the second half aggregate -> bottom.
    """
    S_t = torch.as_tensor(S)
    n_hierarchy, n_bottom = S_t.shape
    n_agg = n_hierarchy - n_bottom
    agg, bot = torch.nonzero(S_t[:n_agg], as_tuple=True)
    bot = bot + n_agg
    return torch.cat([bot, agg]), torch.cat([agg, bot])


class SAGELayer(nn.Module):
    """GraphSAGE update with mean aggregation: W [h_v ; mean_{u in N(v)} h_u].

    Neighbor sums use index_add over an edge list, so cost is O(edges), not
    O(nodes^2); a node with no neighbors aggregates to zero.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * in_dim, out_dim)

    def forward(self, h: Tensor, src: Tensor, dst: Tensor, inv_deg: Tensor) -> Tensor:
        """[B, N, C_in] -> [B, N, C_out]."""
        neigh = torch.zeros_like(h).index_add_(1, dst, h[:, src, :])
        neigh = neigh * inv_deg[None, :, None]
        return self.linear(torch.cat([h, neigh], dim=-1))


@MIXERS.register("cross_series_gnn")
class CrossSeriesGNNMixer(nn.Module):
    """Residual cross-series mixer passing messages along the hierarchy in S.

    Two GraphSAGE layers over the bipartite bottom<->aggregate graph: the first
    hop carries each bottom series to its parents, the second brings that back
    down, so a series hears from its siblings through every shared aggregate.
    Unlike cross_series_mlp, weights are shared across nodes and only linked
    series interact. hidden_size=0 makes it an identity passthrough.

    Every series is normalized by its own window statistics before encoding, which
    erases how large it is -- yet the raw-unit aggregates the loss scores are
    dominated by the large series. With `scale_features`, each node's features are
    extended, before every layer, with its window magnitude: log scale relative to
    the window's mean over nodes, and asinh(loc / scale). An aggregate's loc and
    scale are the sums of its children's. Without `stats` these features are zero.
    """

    N_SCALE_FEATURES = 2

    def __init__(
        self,
        n_hierarchy: int,
        n_bottom: int,
        channels: int,
        hidden_size: int,
        S: Tensor | np.ndarray,
        scale_features: bool = True,
    ) -> None:
        super().__init__()
        if tuple(S.shape) != (n_hierarchy, n_bottom):
            raise ValueError(
                f"S has shape {tuple(S.shape)}, expected ({n_hierarchy}, {n_bottom})"
            )
        self.n_hierarchy = n_hierarchy
        self.n_bottom = n_bottom
        self.n_agg = n_hierarchy - n_bottom
        self.out_channels = channels
        self.enabled = hidden_size > 0
        self.scale_features = scale_features

        src, dst = hierarchy_edges(S)
        self.n_up = src.numel() // 2
        deg = torch.bincount(dst, minlength=n_hierarchy).clamp(min=1)
        # Rebuilt from S on construction, so kept out of the state dict.
        self.register_buffer("src", src, persistent=False)
        self.register_buffer("dst", dst, persistent=False)
        self.register_buffer("inv_deg", 1.0 / deg.float(), persistent=False)

        extra = self.N_SCALE_FEATURES if scale_features else 0
        self.layers = (
            nn.ModuleList(
                [
                    SAGELayer(channels + extra, hidden_size),
                    SAGELayer(hidden_size + extra, channels),
                ]
            )
            if self.enabled
            else None
        )
        self.act = nn.ReLU()

    def _lift(self, x: Tensor) -> Tensor:
        """Bottom values [B, Nb] -> every node [B, N]; aggregates sum their children."""
        full = torch.cat([x.new_zeros(x.shape[0], self.n_agg), x], dim=1)
        up_src, up_dst = self.src[: self.n_up], self.dst[: self.n_up]
        return full.index_add(1, up_dst, full[:, up_src])

    def magnitude_features(self, h: Tensor, stats: ScaleStats | None) -> Tensor:
        """[B, N, 2] per-node (relative log scale, asinh(loc / scale))."""
        if stats is None:
            return h.new_zeros(h.shape[0], self.n_hierarchy, self.N_SCALE_FEATURES)
        loc = self._lift(stats.loc[:, 0, :])
        scale = self._lift(stats.scale[:, 0, :]).clamp_min(1e-8)
        log_scale = scale.log()
        relative = log_scale - log_scale.mean(dim=1, keepdim=True)
        return torch.stack([relative, torch.asinh(loc / scale)], dim=-1)

    def forward(self, h: Tensor, stats: ScaleStats | None = None) -> Tensor:
        """[B, Na+Nb, C] -> [B, Nb, C]; `stats` are the window's [B, 1, Nb] loc/scale."""
        residual = h[:, -self.n_bottom :, :]
        if self.layers is None:
            return residual
        graph = (self.src, self.dst, self.inv_deg)
        if self.scale_features:
            feats = self.magnitude_features(h, stats)
            x = self.act(self.layers[0](torch.cat([h, feats], dim=-1), *graph))
            x = self.layers[1](torch.cat([x, feats], dim=-1), *graph)
        else:
            x = self.act(self.layers[0](h, *graph))
            x = self.layers[1](x, *graph)
        return residual + x[:, -self.n_bottom :, :]


@MIXERS.register("identity")
class IdentityMixer(nn.Module):
    """Drops the aggregate rows without mixing; the disabled-mixer baseline."""

    def __init__(
        self, n_hierarchy: int, n_bottom: int, channels: int, hidden_size: int = 0
    ) -> None:
        super().__init__()
        self.n_bottom = n_bottom
        self.out_channels = channels

    def forward(self, h: Tensor) -> Tensor:
        """[B, Na+Nb, C] -> [B, Nb, C]."""
        return h[:, -self.n_bottom :, :]
