from __future__ import annotations

import math
import os
import random
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
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config.lr_step_size, gamma=config.lr_decay_gamma
        )
        self.history = History()
        self._best_state: dict[str, Tensor] | None = None
        self._patience = 0

    def fit(
        self, train: Batcher, val: Batcher, checkpoint_dir: str | None = None
    ) -> History:
        """Train to `max_steps` or early stopping, then restore the best weights."""
        cfg = self.config
        path = (
            os.path.join(checkpoint_dir, "checkpoint_latest.pt")
            if checkpoint_dir
            else None
        )
        step = 0
        if path is not None and os.path.exists(path):
            step = self._load(path)
            if self.verbose:
                best = self.history.best_val
                print(f"resumed at step {step} (best val sCRPS {best:.5f})")

        stop = step >= cfg.max_steps
        while step < cfg.max_steps and not stop:
            order = np.random.permutation(train.n_windows)
            for start in range(0, len(order), cfg.batch_size):
                if step >= cfg.max_steps or stop:
                    break
                batch = train.batch(order[start : start + cfg.batch_size].tolist())
                loss = self._train_step(batch)
                step += 1
                self.history.train_losses.append(loss)
                self.history.steps_run = step
                self.scheduler.step()

                if step % cfg.val_check_steps == 0 or step == cfg.max_steps:
                    stop = self._validate(val, step)
                if path is not None and (
                    step % cfg.checkpoint_every == 0 or stop or step >= cfg.max_steps
                ):
                    self._save(path, step)

        if cfg.restore_best and self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return self.history

    def loss(self, batch: TrainingBatch, num_samples: int) -> Tensor:
        """Score one batch in raw units: denormalize, sample, clip, aggregate, score."""
        params = denormalize_params(self.model(batch.windows), batch.scale)
        samples = sample_coherent(params, self.model.S, num_samples).hierarchy
        target = aggregate_targets(self.model.S, batch.target_bottom)
        per = self.objective(target, samples, reduction="none")

        if self.config.loss_weighting == "per_series":
            mass = target.abs().sum(dim=(0, 1)).detach().clamp_min(MASS_FLOOR)
            return (per / mass).sum()
        total = per.sum()
        if self.config.normalize_loss:
            total = total / target.abs().sum().detach().clamp_min(MASS_FLOOR)
        return total

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

    def _train_step(self, batch: TrainingBatch) -> float:
        """One optimizer step; raises if the loss goes non-finite."""
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.loss(batch, self.config.train_mc_samples)
        loss.backward()
        self.optimizer.step()
        value = loss.item()
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite training loss: {value}")
        return value

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
            self._best_state = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }
            return False
        self._patience += 1
        stop = 0 < self.config.early_stop_patience <= self._patience
        if stop and self.verbose:
            print(f"early stop at step {step} (best {self.history.best_val:.5f})")
        return stop

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
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        return int(state["step"])
