from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from training.trainer import History


@dataclass
class RunResult:
    """One seed's outcome, ready for reporting or plotting."""

    dataset: str
    seed: int
    scrps: dict[str, float]
    history: History
    n_params: int = 0
    samples: np.ndarray | None = None  # [Na+Nb, h, N], raw units
    y_true: np.ndarray | None = None  # [Na+Nb, h], raw units
    series_names: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """JSON-serializable record of the run."""
        return {
            "dataset": self.dataset,
            "seed": self.seed,
            "scrps": self.scrps,
            "steps_run": self.history.steps_run,
            "best_val_scrps": self.history.best_val,
            "best_step": self.history.best_step,
            "n_params": self.n_params,
            **self.extras,
        }
