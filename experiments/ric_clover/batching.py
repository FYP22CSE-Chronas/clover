from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

import numpy as np
import torch
from torch import Tensor

from contracts import TrainingBatch, WindowBatch
from pipeline.batching import WindowBatcher


@dataclass
class RICWindowBatch(WindowBatch):
    """A `WindowBatch` that also carries a longer raw history for the dependence
    estimate only; the encoder still sees `insample_y` alone."""

    ric_history: Tensor | None = None  # [B, M, Nb], raw units


class RICBatcher(WindowBatcher):
    """Attaches the `lookback` rows that end where each window's insample block ends.

    Only rows before the forecast origin `t` are used, so the estimate never sees the
    horizon. Windows near the start of the series get what history exists; a batch
    is cut to its shortest member so it stacks.
    """

    bottom: np.ndarray | None = None
    lookback: int = 0

    def batch(self, indices: Sequence[int]) -> TrainingBatch:
        out = super().batch(indices)
        if self.bottom is None or self.lookback <= 0:
            return out
        ts = [self.windows[i].t for i in indices]
        n = min(min(t, self.lookback) for t in ts)
        block = np.stack([self.bottom[t - n : t] for t in ts]).astype(np.float32)
        base = {f.name: getattr(out.windows, f.name) for f in fields(WindowBatch)}
        windows = RICWindowBatch(
            **base, ric_history=torch.as_tensor(block, device=self.device)
        )
        return replace(out, windows=windows)
