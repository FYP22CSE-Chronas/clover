"""Training, autoregressive forecasting and scoring for the HierE2E port.

Defaults follow GluonTS's Trainer and DeepVARHierarchicalEstimator: Adam at 1e-3
(weight decay 1e-8), 50 batches per epoch, element-wise gradient clipping at 10,
learning rate halved after 10 epochs without improvement (floor 5e-5), 200 samples
for the CRPS loss, teacher forcing over context + horizon.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from experiments.hiere2e.data import HierData, Windows
from experiments.hiere2e.model import HierE2E
from training.losses import crps
from training.metrics import crps_from_samples, scrps_by_level


@dataclass(frozen=True)
class FitConfig:
    context_length: int
    epochs: int = 100
    batch_size: int = 32
    batches_per_epoch: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-8
    clip_gradient: float = 10.0
    loss_samples: int = 200
    val_samples: int = 200
    # Epochs at which the validation window is scored; the paper's epoch grid.
    checkpoints: tuple[int, ...] = (10, 25, 50, 100)
    # Time steps per CRPS chunk; bounds memory, does not change the gradient.
    loss_chunk: int = 16
    # CPU budget, unbiased either way (see `loss_and_backward`). None scores every
    # context step, as GluonTS does; an int scores that many random context steps,
    # reweighted. "pairs" is the sort-free CRPS estimator; "sorted" the repo's
    # U-statistic.
    loss_context_steps: int | None = 8
    crps_estimator: str = "pairs"


def crps_pairs(y_true: Tensor, samples: Tensor) -> Tensor:
    """Unbiased CRPS from 2m samples without sorting: E|X - y| - E|X - X'| / 2.

    The second term averages |X_i - X_{m+i}| over m independent pairs; the samples
    are i.i.d., so this is unbiased for E|X - X'| at O(n) cost. Same expectation as
    the sorted U-statistic, a little more variance.
    """
    n = samples.shape[-1]
    if n < 2 or n % 2:
        raise ValueError(f"need an even sample count >= 2, got {n}")
    m = n // 2
    term1 = (samples - y_true.unsqueeze(-1)).abs().mean(dim=-1)
    term2 = (samples[..., :m] - samples[..., m:]).abs().mean(dim=-1)
    return term1 - 0.5 * term2


@dataclass
class FitResult:
    val_by_epoch: dict[int, float] = field(default_factory=dict)
    states: dict[int, dict[str, Tensor]] = field(default_factory=dict)
    train_loss: list[float] = field(default_factory=list)
    seconds: float = 0.0


def loss_steps(
    context_length: int, h: int, n_context: int | None, gen: torch.Generator
) -> tuple[Tensor, Tensor]:
    """Sequence positions scored by the loss, and the weight of each.

    Every horizon step has weight 1. With `n_context` set, that many context steps
    are drawn without replacement and weighted context_length / n_context, so the
    weighted sum is an unbiased estimate of the sum over every step.
    """
    horizon = torch.arange(context_length, context_length + h)
    if n_context is None or n_context >= context_length:
        steps = torch.arange(context_length + h)
        return steps, torch.ones(len(steps))
    ctx = torch.randperm(context_length, generator=gen)[:n_context].sort().values
    weights = torch.cat(
        [torch.full((n_context,), context_length / n_context), torch.ones(h)]
    )
    return torch.cat([ctx, horizon]), weights


def loss_and_backward(
    model: HierE2E,
    batch: dict[str, Tensor],
    cfg: FitConfig,
    gen: torch.Generator | None = None,
) -> float:
    """CRPS of projected samples, mean over steps, backpropagated in time chunks.

    The RNN and output layer run once over the whole sequence. The parameters at
    the scored steps are then cut from the graph, the sampling + projection + CRPS
    runs chunk by chunk (each chunk's graph freed after its backward), and the
    gradients collected on the cut parameters are pushed back through the network
    in one pass. That equals a single backward over the full loss.
    """
    gen = gen if gen is not None else torch.Generator().manual_seed(0)
    hidden, _ = model.rnn(model.inputs(batch["lags"], batch["time_feat"]))
    b, t = hidden.shape[:2]
    ctx = cfg.context_length
    steps, weights = loss_steps(ctx, t - ctx, cfg.loss_context_steps, gen)
    params = {k: v[:, steps] for k, v in model.params(hidden).items()}
    target_all = batch["target"][:, steps]
    leaves = {k: v.detach().requires_grad_() for k, v in params.items()}
    score = (
        crps_pairs
        if cfg.crps_estimator == "pairs"
        else (lambda y, s: crps(y, s, reduction="none"))
    )
    total = 0.0
    for start in range(0, len(steps), cfg.loss_chunk):
        sl = slice(start, start + cfg.loss_chunk)
        chunk = {k: v[:, sl] for k, v in leaves.items()}
        samples = model.sample(chunk, batch["scale"], cfg.loss_samples)
        per_step = score(target_all[:, sl], samples).sum(dim=-1)  # [B, c]
        loss = (per_step * weights[sl]).sum() / (b * t)
        loss.backward()
        total += loss.item()
    keys = list(params)
    torch.autograd.backward([params[k] for k in keys], [leaves[k].grad for k in keys])
    return total


def fit(
    model: HierE2E,
    data: HierData,
    cfg: FitConfig,
    seed: int,
    validate: bool = True,
    verbose: bool = False,
) -> FitResult:
    """Train for cfg.epochs; score validation (and keep weights) at checkpoints."""
    windows = Windows(data, cfg.context_length)
    origins = windows.train_origins()
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=10, min_lr=5e-5
    )
    result = FitResult()
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(cfg.batches_per_epoch):
            pick = torch.randint(len(origins), (cfg.batch_size,), generator=gen)
            batch = windows.sequence(torch.from_numpy(origins)[pick], data.h)
            opt.zero_grad()
            epoch_loss += loss_and_backward(model, batch, cfg, gen)
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg.clip_gradient)
            opt.step()
        epoch_loss /= cfg.batches_per_epoch
        if not np.isfinite(epoch_loss):
            raise RuntimeError(f"non-finite training loss at epoch {epoch}")
        result.train_loss.append(epoch_loss)
        sched.step(epoch_loss)
        if epoch in cfg.checkpoints or epoch == cfg.epochs:
            if validate:
                result.val_by_epoch[epoch] = validation_scrps(
                    model, windows, data, cfg.val_samples
                )
            result.states[epoch] = copy.deepcopy(model.state_dict())
            if verbose:
                val = result.val_by_epoch.get(epoch, float("nan"))
                print(
                    f"  epoch {epoch:3d}  train_crps={epoch_loss:.4f}  "
                    f"val_sCRPS={val:.5f}  ({time.time() - start:.0f}s)",
                    flush=True,
                )
    result.seconds = time.time() - start
    return result


@torch.no_grad()
def forecast(
    model: HierE2E,
    windows: Windows,
    origins: Sequence[int],
    num_samples: int,
) -> np.ndarray:
    """Autoregressive coherent sample paths, [len(origins), N, h, num_samples].

    The context is encoded with observed values; then each step draws one value per
    path, projects it onto the coherent subspace, and feeds it back as the next
    step's lag -- GluonTS's `sampling_decoder` with `post_process_samples`.
    """
    model.eval()
    data = windows.data
    o = torch.tensor(list(origins))
    seq = windows.sequence(o, steps_after=0)
    _, state = model.rnn(model.inputs(seq["lags"], seq["time_feat"]))
    n = num_samples
    state = [(hh.repeat_interleave(n, 0), cc.repeat_interleave(n, 0)) for hh, cc in state]
    scale = seq["scale"][:, 0, :].repeat_interleave(n, 0)  # [B*n, N]
    max_lag = data.max_lag
    hist_rows = o[:, None] + torch.arange(-max_lag, 0)[None, :]
    history = windows.values[hist_rows].repeat_interleave(n, 0)  # [B*n, max_lag, N]
    idx = model.index_embedding.weight[:, 0]
    paths = []
    for k in range(data.h):
        lags = torch.stack([history[:, -lag] for lag in data.lags], dim=1)
        tf = windows.time_feat[o + k].repeat_interleave(n, 0)
        x = torch.cat(
            [(lags / scale[:, None, :]).flatten(1), idx.expand(len(scale), -1), tf], dim=1
        )
        hidden, state = model.rnn.step(x, state)
        y = model.sample(model.params(hidden), scale, 1)[..., 0]  # [B*n, N]
        paths.append(y)
        history = torch.cat([history[:, 1:], y[:, None, :]], dim=1)
    out = torch.stack(paths, dim=-1)  # [B*n, N, h]
    out = out.reshape(len(o), n, data.n_series, data.h).permute(0, 2, 3, 1)
    return out.numpy().astype(np.float64)


def validation_scrps(
    model: HierE2E, windows: Windows, data: HierData, num_samples: int
) -> float:
    """Overall sCRPS pooled over every validation origin (ratio of sums)."""
    samples = forecast(model, windows, data.val_origins, num_samples)
    numerator = denominator = 0.0
    for i, o in enumerate(data.val_origins):
        y = data.values[o : o + data.h].T.astype(np.float64)
        numerator += crps_from_samples(y, samples[i]).sum()
        denominator += np.abs(y).sum()
    return numerator / max(denominator, 1e-12)


def test_scores(
    model: HierE2E,
    data: HierData,
    context_length: int,
    num_samples: int,
    estimator: str = "quantile",
    quantile_step: float = 0.01,
) -> tuple[dict[str, float], np.ndarray]:
    """Per-level sCRPS on the test origin, scored exactly as CLOVER's runs are."""
    windows = Windows(data, context_length)
    samples = forecast(model, windows, [data.test_origin], num_samples)[0]
    o = data.test_origin
    y = data.values[o : o + data.h].T.astype(np.float64)
    scores = scrps_by_level(
        y, samples, data.level_slices, estimator=estimator, quantile_step=quantile_step
    )
    return scores, samples
