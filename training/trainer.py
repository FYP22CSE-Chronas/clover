from __future__ import annotations

import math
import os
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from config import TrainConfig
from contracts import Batcher, TrainingBatch
from model.distribution import aggregate_targets, sample_coherent
from model.network import CLOVER
from model.normalization import denormalize_params
from training.losses import LOSSES
from training.sampling import WindowSampler, build_sampler

MASS_FLOOR = 1e-8


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch, and pin cudnn to deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class History:
    """Per-step training losses and the (step, val sCRPS) checkpoints."""

    train_losses: list[float] = field(default_factory=list)
    val_scrps: list[tuple[int, float]] = field(default_factory=list)
    best_val: float = float("inf")
    best_step: int = -1
    steps_run: int = 0
    snapshot_steps: list[int] = field(default_factory=list)


class Trainer:
    """Fits a `CLOVER` model against `Batcher`s of raw-unit windows."""

    def __init__(
        self,
        model: CLOVER,
        config: TrainConfig,
        device: str | torch.device = "cpu",
        verbose: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.verbose = verbose
        self.objective = LOSSES.get(config.objective)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if config.snapshot_cycle:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.snapshot_cycle,
                eta_min=config.learning_rate * config.snapshot_lr_floor,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.lr_step_size,
                gamma=config.lr_decay_gamma,
            )
        self.history = History()
        self.snapshots: list[dict[str, Tensor]] = []
        self._best_state: dict[str, Tensor] | None = None
        self._patience = 0
        self.sampler: WindowSampler | None = None

    def fit(
        self, train: Batcher, val: Batcher, checkpoint_dir: str | None = None
    ) -> History:
        """Train to `max_steps` or early stopping, then restore the best weights.

        With snapshots on, training runs whole cycles only and keeps the weights at
        the end of each in `self.snapshots`.
        """
        cfg = self.config
        cycle = cfg.snapshot_cycle
        limit = cycle * cfg.snapshot_count if cycle else cfg.max_steps
        path = (
            os.path.join(checkpoint_dir, "checkpoint_latest.pt")
            if checkpoint_dir
            else None
        )
        self.sampler = build_sampler(cfg, train.window_times)
        step = 0
        if path is not None and os.path.exists(path):
            step = self._load(path)
            if self.verbose:
                best = self.history.best_val
                print(f"resumed at step {step} (best val sCRPS {best:.5f})")

        stop = step >= limit
        while step < limit and not stop:
            for indices, weights in self._epoch(train.n_windows):
                if step >= limit or stop:
                    break
                loss, priority = self._train_step(train.batch(indices), weights)
                if self.sampler is not None:
                    self.sampler.update(indices, priority)
                step += 1
                self.history.train_losses.append(loss)
                self.history.steps_run = step
                self.scheduler.step()
                if cycle and step % cycle == 0:
                    self.snapshots.append(self._state_copy())
                    self.history.snapshot_steps.append(step)

                if step % cfg.val_check_steps == 0 or step == limit:
                    stop = self._validate(val, step)
                if path is not None and (
                    step % cfg.checkpoint_every == 0 or stop or step >= limit
                ):
                    self._save(path, step)

        if cfg.restore_best and self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return self.history

    def _epoch(
        self, n_windows: int
    ) -> Iterator[tuple[list[int], np.ndarray | None]]:
        """One pass of batches: a shuffled partition, or adaptive draws of equal count.

        Adaptive draws are made lazily, one per step, so each sees the losses the
        previous steps recorded rather than a snapshot from the start of the pass.
        """
        size = self.config.batch_size
        if self.sampler is None:
            order = np.random.permutation(n_windows)
            for start in range(0, n_windows, size):
                yield order[start : start + size].tolist(), None
            return
        for _ in range(math.ceil(n_windows / size)):
            yield self.sampler.draw(size)

    def loss(self, batch: TrainingBatch, num_samples: int) -> Tensor:
        """Score one batch in raw units: denormalize, sample, clip, aggregate, score."""
        return self.window_losses(batch, num_samples)[0].sum()

    def window_losses(
        self, batch: TrainingBatch, num_samples: int
    ) -> tuple[Tensor, Tensor]:
        """Each window's share of the batch loss, plus its own mass-scaled loss.

        The shares sum to the batch loss. The second tensor is every window's loss
        divided by its own target mass, a scale-free priority for adaptive sampling.
        """
        params = denormalize_params(self.model(batch.windows), batch.scale)
        samples = sample_coherent(params, self.model.S, num_samples).hierarchy
        target = aggregate_targets(self.model.S, batch.target_bottom)
        per = self.objective(target, samples, reduction="none")
        b = target.shape[0]
        own_mass = target.abs().reshape(b, -1).sum(-1).detach().clamp_min(MASS_FLOOR)

        if self.config.loss_weighting == "per_series":
            mass = target.abs().sum(dim=(0, 1)).detach().clamp_min(MASS_FLOOR)
            shares = (per / mass).reshape(b, -1).sum(-1)
        else:
            raw = per.reshape(b, -1).sum(-1)
            shares = raw
            if self.config.normalize_loss:
                shares = raw / target.abs().sum().detach().clamp_min(MASS_FLOOR)
        priority = per.detach().reshape(b, -1).sum(-1) / own_mass
        return shares, priority

    @torch.no_grad()
    def validation_scrps(self, val: Batcher, num_samples: int) -> float:
        """Overall sCRPS across validation windows, accumulated as a ratio of sums."""
        self.model.eval()
        numerator = 0.0
        denominator = 0.0
        for i in range(val.n_windows):
            batch = val.batch([i])
            params = denormalize_params(self.model(batch.windows), batch.scale)
            samples = sample_coherent(params, self.model.S, num_samples).hierarchy
            target = aggregate_targets(self.model.S, batch.target_bottom)
            numerator += self.objective(target, samples, reduction="none").sum().item()
            denominator += target.abs().sum().item()
        return numerator / max(denominator, MASS_FLOOR)

    @torch.no_grad()
    def validation_draws(
        self, val: Batcher, num_samples: int
    ) -> list[tuple[Tensor, Tensor]]:
        """(target, samples) for every validation window under the current weights."""
        self.model.eval()
        draws = []
        for i in range(val.n_windows):
            batch = val.batch([i])
            params = denormalize_params(self.model(batch.windows), batch.scale)
            samples = sample_coherent(params, self.model.S, num_samples).hierarchy
            draws.append((aggregate_targets(self.model.S, batch.target_bottom), samples))
        return draws

    @torch.no_grad()
    def pooled_validation_scrps(
        self, members: Sequence[list[tuple[Tensor, Tensor]]]
    ) -> float:
        """Validation sCRPS of the equal-weight mixture of `validation_draws` sets."""
        numerator = 0.0
        denominator = 0.0
        for window in zip(*members, strict=True):
            target = window[0][0]
            samples = torch.cat([s for _, s in window], dim=-1)
            numerator += self.objective(target, samples, reduction="none").sum().item()
            denominator += target.abs().sum().item()
        return numerator / max(denominator, MASS_FLOOR)

    def select_snapshots(self, val: Batcher, num_samples: int) -> list[int]:
        """Indices of the snapshots to pool, per `config.snapshot_select`.

        "validation" ranks snapshots by their own validation sCRPS and keeps the
        best k, choosing k by the pooled validation sCRPS of each prefix. Every
        candidate reuses the same draws, so the comparison is free of resampling
        noise. Leaves the model holding the last snapshot's weights.
        """
        everything = list(range(len(self.snapshots)))
        if self.config.snapshot_select == "all" or len(everything) < 2:
            return everything
        draws = []
        for state in self.snapshots:
            self.model.load_state_dict(state)
            draws.append(self.validation_draws(val, num_samples))
        own = [self.pooled_validation_scrps([d]) for d in draws]
        ranked = sorted(everything, key=own.__getitem__)
        scores = [
            self.pooled_validation_scrps([draws[i] for i in ranked[:k]])
            for k in range(1, len(ranked) + 1)
        ]
        best_k = 1 + int(np.argmin(scores))
        return sorted(ranked[:best_k])

    def _train_step(
        self, batch: TrainingBatch, weights: np.ndarray | None = None
    ) -> tuple[float, np.ndarray]:
        """One optimizer step; returns the loss and each window's priority.

        `weights` are the adaptive sampler's importance weights, one per window.
        Raises if the loss goes non-finite.
        """
        self.model.train()
        self.optimizer.zero_grad()
        shares, priority = self.window_losses(batch, self.config.train_mc_samples)
        if weights is not None:
            shares = shares * torch.as_tensor(
                weights, dtype=shares.dtype, device=shares.device
            )
        loss = shares.sum()
        loss.backward()
        self.optimizer.step()
        value = loss.item()
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite training loss: {value}")
        return value, priority.cpu().numpy()

    def _validate(self, val: Batcher, step: int) -> bool:
        """Record validation sCRPS and return True when patience is exhausted."""
        score = self.validation_scrps(val, self.config.val_mc_samples)
        if not math.isfinite(score):
            raise RuntimeError(f"non-finite validation sCRPS at step {step}: {score}")
        self.history.val_scrps.append((step, score))
        if self.verbose:
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"step {step:5d}  val_sCRPS={score:8.5f}  lr={lr:.1e}")

        if score < self.history.best_val - 1e-6:
            self.history.best_val = score
            self.history.best_step = step
            self._patience = 0
            self._best_state = self._state_copy()
            return False
        self._patience += 1
        # Snapshot cycles fix the schedule, so patience never ends them early.
        patience = 0 if self.config.snapshot_cycle else self.config.early_stop_patience
        stop = 0 < patience <= self._patience
        if stop and self.verbose:
            print(f"early stop at step {step} (best {self.history.best_val:.5f})")
        return stop

    def _state_copy(self) -> dict[str, Tensor]:
        """A detached CPU copy of the model weights."""
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _save(self, path: str, step: int) -> None:
        """Atomically write model, optimizer, schedule, history and RNG state."""
        state = {
            "step": step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "history": self.history,
            "best_state": self._best_state,
            "patience": self._patience,
            "snapshots": self.snapshots,
            "sampler_scores": getattr(self.sampler, "scores", None),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)

    def _load(self, path: str) -> int:
        """Restore everything `_save` wrote and return the step to resume from."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.history = state["history"]
        self._best_state = state["best_state"]
        self._patience = state["patience"]
        self.snapshots = state.get("snapshots", [])
        if hasattr(self.sampler, "scores") and state.get("sampler_scores") is not None:
            self.sampler.scores = state["sampler_scores"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        return int(state["step"])
