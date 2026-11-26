"""Training with validation early stopping, reconciled forecasting and scoring.

Defaults follow neuralforecast NHITS: Adam at the chosen learning rate, 1024 windows
per step (32 series x 32 origins), learning rate halved 3 times evenly over
max_steps, validation every 100 steps. Early stopping restores the best checkpoint.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from experiments.hint.data import HintData, Windows
from experiments.hint.model import HINT, RECONCILERS, mixture_nll, reconciliation_P
from training.metrics import crps_from_samples, scrps_by_level


@dataclass(frozen=True)
class FitConfig:
    input_size: int
    learning_rate: float = 1e-3
    max_steps: int = 3000
    series_per_batch: int = 32
    origins_per_batch: int = 32
    num_lr_decays: int = 3
    val_check_steps: int = 100
    patience: int = 10  # validation checks without improvement before stopping
    val_samples: int = 200
    # Early stopping tracks this reconciler; None tracks the best of all three.
    reconciler: str | None = None


@dataclass
class FitResult:
    best_step: int = 0
    best_val: float = float("inf")
    val_history: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    steps_run: int = 0
    seconds: float = 0.0


def fit(
    model: HINT, data: HintData, cfg: FitConfig, seed: int, verbose: bool = False
) -> FitResult:
    windows = Windows(data, cfg.input_size)
    origins = torch.from_numpy(windows.train_origins())
    n_series = data.n_series
    per_batch = min(cfg.series_per_batch, n_series)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    decay_every = max(cfg.max_steps // (cfg.num_lr_decays + 1), 1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_every, gamma=0.5)
    result = FitResult()
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    start = time.time()
    for step in range(1, cfg.max_steps + 1):
        model.train()
        o = origins[torch.randint(len(origins), (cfg.origins_per_batch,), generator=gen)]
        s = torch.randperm(n_series, generator=gen)[:per_batch]
        batch = windows.batch(o, s, with_target=True)
        params = model.params(batch["x"], batch["loc"], batch["scale"])
        loss = mixture_nll(model.component_log_prob(params, batch["y"]), per_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        result.steps_run = step
        if step % cfg.val_check_steps == 0:
            vals = validation_scrps(model, windows, data, cfg.val_samples, seed)
            result.val_history.append((step, vals))
            score = vals[cfg.reconciler] if cfg.reconciler else min(vals.values())
            if verbose:
                shown = " ".join(f"{k}={v:.4f}" for k, v in vals.items())
                print(f"  step {step:5d} loss={loss.item():.3f} {shown}", flush=True)
            if score < result.best_val - 1e-7:
                result.best_val, result.best_step, stale = score, step, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
    model.load_state_dict(best_state)
    result.seconds = time.time() - start
    return result


@torch.no_grad()
def forecast(
    model: HINT, windows: Windows, origins: list[int], num_samples: int, seed: int
) -> np.ndarray:
    """Unreconciled joint samples, [len(origins), N, h, num_samples]."""
    model.eval()
    data = windows.data
    series = torch.arange(data.n_series)
    batch = windows.batch(torch.tensor(origins), series, with_target=False)
    params = model.params(batch["x"], batch["loc"], batch["scale"])
    gen = torch.Generator().manual_seed(seed)
    samples = model.sample(params, data.n_series, num_samples, generator=gen)
    return samples.numpy().astype(np.float64)


def reconcile(S: np.ndarray, samples: np.ndarray, method: str) -> np.ndarray:
    """S P y over the series axis of [..., N, h, n] samples."""
    SP = S @ reconciliation_P(S, method)
    return np.einsum("ij,...jhs->...ihs", SP, samples)


def validation_samples(data: HintData, cap: int, budget: int = 2_000_000) -> int:
    """Samples per validation window: `cap`, cut so one check draws <= `budget`.

    Traffic validates on 120 windows of 207 series, where 200 draws each would
    dominate training time; the floor of 50 keeps the CRPS estimate usable.
    """
    per_window = budget // (len(data.val_origins) * data.n_series * data.h)
    n = max(50, min(cap, per_window))
    return n - n % 2


def validation_scrps(
    model: HINT, windows: Windows, data: HintData, num_samples: int, seed: int
) -> dict[str, float]:
    """Overall sCRPS pooled over validation origins, for each reconciler."""
    num_samples = validation_samples(data, num_samples)
    raw = forecast(model, windows, data.val_origins, num_samples, seed)
    ys = [data.values[o : o + data.h].T.astype(np.float64) for o in data.val_origins]
    denominator = sum(np.abs(y).sum() for y in ys)
    out = {}
    for method in RECONCILERS:
        rec = reconcile(data.S, raw, method)
        numerator = sum(crps_from_samples(y, rec[i]).sum() for i, y in enumerate(ys))
        out[method] = numerator / max(denominator, 1e-12)
    return out


def score_test_window(
    model: HINT,
    data: HintData,
    input_size: int,
    reconciler: str,
    num_samples: int,
    seed: int,
    estimator: str = "quantile",
    quantile_step: float = 0.01,
) -> dict[str, float]:
    """Per-level sCRPS on the test origin, scored exactly as CLOVER's runs are."""
    windows = Windows(data, input_size)
    raw = forecast(model, windows, [data.test_origin], num_samples, seed)
    samples = reconcile(data.S, raw, reconciler)[0]
    o = data.test_origin
    y = data.values[o : o + data.h].T.astype(np.float64)
    return scrps_by_level(
        y, samples, data.level_slices, estimator=estimator, quantile_step=quantile_step
    )
