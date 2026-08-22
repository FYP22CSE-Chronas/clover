from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from contracts import ScaleStats, TrainingBatch, WindowBatch
from data.dataset import ExogenousFeatures
from data.windows import Window
from model.normalization import normalize, window_stats


class WindowBatcher:
    """Assembles model input for a fixed list of windows; implements `Batcher`."""

    def __init__(
        self,
        windows: Sequence[Window],
        features: ExogenousFeatures,
        input_size: int,
        h: int,
        scaler: str = "standard",
        device: str | torch.device = "cpu",
    ) -> None:
        self.windows = list(windows)
        self.features = features
        self.input_size = input_size
        self.h = h
        self.scaler = scaler
        self.device = torch.device(device)
        self.n_windows = len(self.windows)
        self._stat = _to_tensor(features.static, self.device)

    def batch(self, indices: Sequence[int]) -> TrainingBatch:
        chosen = [self.windows[i] for i in indices]
        insample = _to_tensor(np.stack([w.insample for w in chosen]), self.device)
        target = _to_tensor(np.stack([w.target for w in chosen]), self.device)
        stats = window_stats(insample, self.scaler)
        return TrainingBatch(
            windows=WindowBatch(
                insample_y=normalize(insample, stats),
                futr_exog=self._futr_exog([w.t for w in chosen], stats),
                stat_exog=self._stat,
            ),
            target_bottom=target,
            scale=stats,
        )

    def _futr_exog(self, ts: Sequence[int], stats: ScaleStats) -> Tensor | None:
        """[B, F, L+H, Nb], shared channels first, per-series anchors re-normalized."""
        n_shared = self.features.n_future_shared
        n_series = self.features.n_future_series
        if n_shared + n_series == 0:
            return None
        futr = _to_tensor(np.stack([self._slice(t) for t in ts]), self.device)
        if n_series == 0:
            return futr
        # Anchors carry the target unit, so they follow the target scaling; calendar
        # dummies must not be shifted by the target location.
        anchors = futr[:, n_shared:]
        scaled = (anchors - stats.loc.unsqueeze(1)) / stats.scale.unsqueeze(1)
        return torch.cat([futr[:, :n_shared], scaled], dim=1)

    def _slice(self, t: int) -> np.ndarray:
        """Future features for one window as [F, L+H, Nb]."""
        start, stop = t - self.input_size, t + self.h
        n_bottom = self.windows[0].insample.shape[1]
        parts = []
        if self.features.future_shared is not None:
            shared = self.features.future_shared[start:stop]  # [L+H, C1]
            parts.append(
                np.broadcast_to(shared[:, :, None], (*shared.shape, n_bottom)).transpose(
                    1, 0, 2
                )
            )
        if self.features.future_series is not None:
            parts.append(self.features.future_series[start:stop].transpose(2, 0, 1))
        return np.concatenate(parts, axis=0).astype(np.float32)


def _to_tensor(array: np.ndarray | None, device: torch.device) -> Tensor | None:
    if array is None:
        return None
    return torch.as_tensor(array, dtype=torch.float32, device=device)
