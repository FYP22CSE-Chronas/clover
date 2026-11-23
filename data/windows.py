from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from config import SplitConfig


@dataclass(frozen=True)
class Window:
    """One forecast-creation point: `insample` ends at t, `target` starts at t."""

    insample: np.ndarray  # [L, Nb]
    target: np.ndarray  # [h, Nb]
    t: int


@dataclass(frozen=True)
class WindowSplit:
    """Train/validation/test windows for one dataset."""

    train: list[Window]
    val: list[Window]
    test: list[Window]

    @property
    def train_end(self) -> int:
        """First row no training target reaches; features may be fitted on [0, it)."""
        return max(w.t + len(w.target) for w in self.train)

    def target_indices(self) -> dict[str, set[int]]:
        """Target row indices touched by each split, for leakage checks."""
        return {
            name: {t for w in windows for t in range(w.t, w.t + len(w.target))}
            for name, windows in (
                ("train", self.train),
                ("val", self.val),
                ("test", self.test),
            )
        }


SplitStrategy = Callable[[np.ndarray, int, int, SplitConfig], WindowSplit]
SPLITS: dict[str, SplitStrategy] = {}


def register_split(name: str) -> Callable[[SplitStrategy], SplitStrategy]:
    def wrap(fn: SplitStrategy) -> SplitStrategy:
        if name in SPLITS:
            raise KeyError(f"split strategy {name!r} is already registered")
        SPLITS[name] = fn
        return fn

    return wrap


def rolling_windows(values: np.ndarray, input_size: int, h: int) -> list[Window]:
    """Every window with a full history and a full horizon inside `values`."""
    n_steps = values.shape[0]
    if n_steps < input_size + h:
        raise ValueError(
            f"need at least input_size + h = {input_size + h} rows, got {n_steps}"
        )
    return [
        Window(values[t - input_size : t], values[t : t + h], t)
        for t in range(input_size, n_steps - h + 1)
    ]


@register_split("last_h")
def last_h_split(
    values: np.ndarray, input_size: int, h: int, config: SplitConfig
) -> WindowSplit:
    """Test is the final h steps, validation the h before it, training the rest."""
    n_steps = values.shape[0]
    windows = rolling_windows(values, input_size, h)
    # A training window may not reach into the validation block, which starts at T - 2h.
    train = [w for w in windows if w.t + h <= n_steps - 2 * h]
    val = [w for w in windows if w.t == n_steps - 2 * h]
    test = [w for w in windows if w.t == n_steps - h]
    if not train or not val or not test:
        raise ValueError("last_h split produced an empty block; series is too short")
    return WindowSplit(train=train, val=val, test=test)


@register_split("rolling")
def rolling_split(
    values: np.ndarray, input_size: int, h: int, config: SplitConfig
) -> WindowSplit:
    """`origins` non-overlapping test origins at the end, as many validation origins
    before them, training before that. `origins=1` is exactly `last_h`.

    One test window per dataset cannot separate a small real gain from seed noise;
    scoring several consecutive origins (without re-fitting) is the cheapest way to
    get that power while keeping every test target out of training and validation.
    """
    k = config.origins
    assert k is not None
    n_steps = values.shape[0]
    windows = rolling_windows(values, input_size, h)
    test_start = n_steps - k * h
    val_start = test_start - k * h
    test = [w for w in windows if w.t >= test_start and (w.t - test_start) % h == 0]
    val = [
        w for w in windows if val_start <= w.t < test_start and (w.t - val_start) % h == 0
    ]
    train = [w for w in windows if w.t + h <= val_start]
    if len(test) != k or len(val) != k or not train:
        raise ValueError(
            f"rolling split with {k} origins of h={h} needs more than "
            f"{input_size + 2 * k * h + h} rows, got {n_steps}"
        )
    return WindowSplit(train=train, val=val, test=test)


@register_split("block")
def block_split(
    values: np.ndarray, input_size: int, h: int, config: SplitConfig
) -> WindowSplit:
    """Fixed chronological row counts; validation rolls, test is the final window."""
    n_steps = values.shape[0]
    n_train, n_val, n_test = config.train, config.val, config.test
    assert n_train is not None and n_val is not None and n_test is not None
    if n_train + n_val + n_test != n_steps:
        raise ValueError(
            f"block split needs {n_train + n_val + n_test} rows, data has {n_steps}"
        )
    windows = rolling_windows(values, input_size, h)
    val_end = n_train + n_val
    train = [w for w in windows if w.t + h <= n_train]
    val = [w for w in windows if n_train <= w.t and w.t + h <= val_end]
    test = [w for w in windows if w.t + h == n_steps]
    if not train or not val or not test:
        raise ValueError("block split produced an empty block; check the row counts")
    return WindowSplit(train=train, val=val, test=test)


def build_split(
    values: np.ndarray, input_size: int, h: int, config: SplitConfig
) -> WindowSplit:
    """Dispatch to the configured split strategy."""
    try:
        strategy = SPLITS[config.strategy]
    except KeyError:
        raise KeyError(
            f"unknown split strategy {config.strategy!r}; available: {sorted(SPLITS)}"
        ) from None
    return strategy(values, input_size, h, config)
