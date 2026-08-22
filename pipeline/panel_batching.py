from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from contracts import TrainingBatch, WindowBatch
from data.panel import PanelDataset
from model.normalization import normalize, window_stats

HIST_EPS = 1e-6


class PanelBatcher:
    """Implements `Batcher` over a panel; reads memmaps one window at a time.

    The batch axis indexes item-hierarchies as well as dates, so the tensors handed to
    the model are the shapes `WindowBatcher` produces -- with one exception. `stat_exog`
    is `[B, Nb, S]` rather than `[Nb, S]`, because item perishability varies across the
    batch while store geography does not.
    """

    def __init__(
        self,
        dataset: PanelDataset,
        pairs: tuple[np.ndarray, np.ndarray],
        input_size: int,
        h: int,
        scaler: str = "mean",
        device: str | torch.device = "cpu",
        hist_stats: tuple[float, float] | None = None,
    ) -> None:
        self.dataset = dataset
        self.groups, self.ts = pairs
        self.input_size = input_size
        self.h = h
        self.scaler = scaler
        self.device = torch.device(device)
        self.n_windows = int(len(self.groups))

        # Historical covariates are in their own units (store footfall in the
        # thousands), so they get one global z-score rather than the per-window target
        # scaling.
        self.hist_stats = hist_stats
        if dataset.hist_series is not None and hist_stats is None:
            self.hist_stats = (
                float(dataset.hist_series.mean()),
                float(dataset.hist_series.std()) + HIST_EPS,
            )

        self._stat_series = (
            None
            if dataset.static_series is None
            else torch.as_tensor(
                dataset.static_series, dtype=torch.float32, device=self.device
            )
        )

    def batch(self, indices: Sequence[int]) -> TrainingBatch:
        idx = np.asarray(indices, dtype=np.int64)
        groups, ts = self.groups[idx], self.ts[idx]
        bottom = self.dataset.bottom
        L, h = self.input_size, self.h

        pairs = list(zip(groups, ts, strict=True))
        insample = np.stack([bottom[g, t - L : t] for g, t in pairs])
        target = np.stack([bottom[g, t : t + h] for g, t in pairs])
        insample_t = _tensor(insample, self.device)
        stats = window_stats(insample_t, self.scaler)

        return TrainingBatch(
            windows=WindowBatch(
                insample_y=normalize(insample_t, stats),
                futr_exog=self._futr_exog(groups, ts),
                hist_exog=self._hist_exog(ts),
                stat_exog=self._stat_exog(groups),
            ),
            target_bottom=_tensor(target, self.device),
            scale=stats,
        )

    def _futr_exog(self, groups: np.ndarray, ts: np.ndarray) -> Tensor | None:
        """[B, F, L+H, Nb]: day-of-week dummies then the promotion flag."""
        data = self.dataset
        if data.n_future == 0:
            return None
        L, h = self.input_size, self.h
        n_bottom = data.n_bottom
        out = []
        for g, t in zip(groups, ts, strict=True):
            parts = []
            if data.future_shared is not None:
                shared = data.future_shared[t - L : t + h]  # [L+h, C]
                parts.append(
                    np.broadcast_to(
                        shared[:, :, None], (*shared.shape, n_bottom)
                    ).transpose(1, 0, 2)
                )
            if data.future_panel is not None:
                panel = np.asarray(data.future_panel[g, t - L : t + h], np.float32)
                parts.append(panel[None])
            out.append(np.concatenate(parts, axis=0))
        # Both blocks are indicators in [0, 1]; neither carries the target unit, so
        # neither is re-scaled the way the single-hierarchy anchors are.
        return _tensor(np.stack(out), self.device)

    def _hist_exog(self, ts: np.ndarray) -> Tensor | None:
        """[B, X, L, Nb]: store transactions, globally z-scored."""
        data = self.dataset
        if data.hist_series is None or self.hist_stats is None:
            return None
        loc, scale = self.hist_stats
        L = self.input_size
        block = np.stack([data.hist_series[t - L : t] for t in ts])  # [B, L, Nb, X]
        block = (block - loc) / scale
        return _tensor(block.transpose(0, 3, 1, 2), self.device)

    def _stat_exog(self, groups: np.ndarray) -> Tensor | None:
        """[B, Nb, S]: per-item perishability over stores, plus store state dummies."""
        data = self.dataset
        if data.n_static == 0:
            return None
        parts = []
        if data.static_group is not None:
            per_group = np.asarray(data.static_group[groups], np.float32)  # [B, Sg]
            parts.append(
                np.broadcast_to(
                    per_group[:, None, :],
                    (len(groups), data.n_bottom, per_group.shape[1]),
                )
            )
        if data.static_series is not None:
            series = np.asarray(data.static_series, np.float32)
            parts.append(np.broadcast_to(series[None], (len(groups), *series.shape)))
        return _tensor(np.concatenate(parts, axis=-1), self.device)


def _tensor(array: np.ndarray, device: torch.device) -> Tensor:
    """Contiguous float32 tensor on the training device."""
    return torch.as_tensor(np.ascontiguousarray(array, dtype=np.float32), device=device)
