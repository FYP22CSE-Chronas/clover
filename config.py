from __future__ import annotations

import copy
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_ROOT = os.path.join(PROJECT_ROOT, "configs")

TOP_LEVEL_KEYS = frozenset({"name", "data", "features", "model", "train", "eval"})


def _strict(cls: type[T], raw: Mapping[str, Any], where: str) -> T:
    """Build a dataclass from a mapping, rejecting unknown keys."""
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; expected {sorted(known)}")
    return cls(**raw)  # type: ignore[call-arg]


@dataclass(frozen=True)
class ModelDims:
    """Shapes the model reads off the data rather than off a config file."""

    h: int
    input_size: int
    n_stat_features: int = 0
    n_futr_features: int = 0
    n_hist_features: int = 0


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters; component names resolve through the registries."""

    temp_conv_channels: int = 10
    dilations: tuple[int, ...] = (1, 2, 4)
    kernel_size: int = 2
    conv_residual: bool = True
    cross_series_hidden: int = 100
    static_encoder_dim: int = 10
    future_encoder_dim: int = 10
    horizon_specific_dim: int = 5
    horizon_agnostic_dim: int = 10
    n_factors: int = 10
    sigma_activation: str = "softplus"
    sigma_eps: float = 1e-3
    encoder: str = "dilated_conv"
    mixer: str = "cross_series_mlp"
    decoder: str = "two_stage"
    head: str = "factor_model"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dilations", tuple(self.dilations))
        if self.sigma_activation not in ("softplus", "exp"):
            raise ValueError(
                f"sigma_activation must be 'softplus' or 'exp', "
                f"got {self.sigma_activation!r}"
            )
        if not self.dilations:
            raise ValueError("dilations must be non-empty")
        if self.kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if self.n_factors < 1:
            raise ValueError("n_factors must be >= 1")
        if self.cross_series_hidden < 0:
            raise ValueError("cross_series_hidden must be >= 0 (0 disables the mixer)")

    @property
    def receptive_field(self) -> int:
        """Time steps the dilated stack can see: 1 + sum_d (k-1) * d."""
        return 1 + sum((self.kernel_size - 1) * d for d in self.dilations)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ModelConfig:
        return _strict(cls, raw, "model")


@dataclass(frozen=True)
class TrainConfig:
    """Optimization schedule, early stopping and objective selection."""

    learning_rate: float = 5e-4
    max_steps: int = 1000
    batch_size: int = 1
    early_stop_patience: int = 5
    val_check_steps: int = 100
    val_mc_samples: int = 500
    train_mc_samples: int = 50
    num_lr_decays: int = 4
    lr_decay_gamma: float = 0.5
    weight_decay: float = 0.0
    scaler: str = "standard"
    objective: str = "crps"
    loss_weighting: str = "none"
    normalize_loss: bool = True
    checkpoint_every: int = 100
    restore_best: bool = True

    def __post_init__(self) -> None:
        if self.loss_weighting not in ("none", "per_series"):
            raise ValueError(
                f"loss_weighting must be 'none' or 'per_series', "
                f"got {self.loss_weighting!r}"
            )
        if self.loss_weighting == "per_series" and self.objective != "crps":
            raise ValueError("loss_weighting='per_series' requires objective='crps'")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.train_mc_samples < 2 or self.val_mc_samples < 2:
            raise ValueError("Monte Carlo sample counts must be >= 2")

    @property
    def lr_step_size(self) -> int:
        """StepLR period: max_steps // num_lr_decays, floored at 1."""
        return max(self.max_steps // max(self.num_lr_decays, 1), 1)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TrainConfig:
        return _strict(cls, raw, "train")


@dataclass(frozen=True)
class EvalConfig:
    """Test-time sampling and the sCRPS estimator to report."""

    num_samples: int = 1000
    estimator: str = "quantile"
    quantile_step: float = 0.01

    def __post_init__(self) -> None:
        if self.estimator not in ("quantile", "mc"):
            raise ValueError(
                f"estimator must be 'quantile' or 'mc', got {self.estimator!r}"
            )
        if not 0.0 < self.quantile_step < 1.0:
            raise ValueError("quantile_step must lie in (0, 1)")
        if self.num_samples < 2:
            raise ValueError("num_samples must be >= 2")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EvalConfig:
        return _strict(cls, raw, "eval")


@dataclass(frozen=True)
class SplitConfig:
    """Which split strategy to use, plus the row counts the block strategy needs."""

    strategy: str = "last_h"
    train: int | None = None
    val: int | None = None
    test: int | None = None

    def __post_init__(self) -> None:
        if self.strategy not in ("last_h", "block"):
            raise ValueError(f"unknown split strategy {self.strategy!r}")
        if self.strategy == "block" and None in (self.train, self.val, self.test):
            raise ValueError("block split requires train, val and test row counts")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SplitConfig:
        return _strict(cls, raw, "data.split")


@dataclass(frozen=True)
class DataConfig:
    """Everything needed to load and window one dataset."""

    dir: str
    h: int
    input_size: int
    split: SplitConfig = field(default_factory=SplitConfig)
    end_date: str | None = None
    levels: dict[str, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.h < 1 or self.input_size < 1:
            raise ValueError("h and input_size must be >= 1")
        object.__setattr__(
            self, "levels", {k: tuple(v) for k, v in dict(self.levels).items()}
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DataConfig:
        raw = dict(raw)
        if "split" in raw:
            raw["split"] = SplitConfig.from_dict(raw["split"])
        return _strict(cls, raw, "data")


@dataclass(frozen=True)
class FeatureSpec:
    """One entry of the `features.static` or `features.future` config list."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FeatureSpec:
        raw = dict(raw)
        try:
            kind = raw.pop("type")
        except KeyError:
            raise ValueError(f"feature entry {raw} is missing a 'type' key") from None
        return cls(type=str(kind), params=raw)


@dataclass(frozen=True)
class RunConfig:
    """One dataset run: data, features, architecture, training and evaluation."""

    name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig
    static_features: tuple[FeatureSpec, ...] = ()
    future_features: tuple[FeatureSpec, ...] = ()

    def validate(self) -> None:
        if self.data.input_size < self.model.receptive_field:
            raise ValueError(
                f"{self.name}: input_size {self.data.input_size} is shorter than the "
                f"encoder receptive field {self.model.receptive_field}"
            )


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}


def load_config(
    dataset: str, root: str = CONFIG_ROOT, overrides: Sequence[str] = ()
) -> RunConfig:
    """Merge `datasets/<name>.yaml` over `default.yaml`, apply overrides, validate."""
    merged = _deep_merge(
        _read_yaml(os.path.join(root, "default.yaml")),
        _read_yaml(os.path.join(root, "datasets", f"{dataset}.yaml")),
    )
    for override in overrides:
        _apply_override(merged, override)

    unknown = sorted(set(merged) - TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"{dataset}: unknown top-level config key(s) {unknown}")

    features = merged.get("features") or {}
    unknown_sections = sorted(set(features) - {"static", "future"})
    if unknown_sections:
        raise ValueError(f"{dataset}: unknown features section(s) {unknown_sections}")

    data = DataConfig.from_dict(merged["data"])
    config = RunConfig(
        name=merged.get("name", dataset),
        data=replace(data, dir=_resolve(data.dir)),
        model=ModelConfig.from_dict(merged.get("model") or {}),
        train=TrainConfig.from_dict(merged.get("train") or {}),
        eval=EvalConfig.from_dict(merged.get("eval") or {}),
        static_features=tuple(
            FeatureSpec.from_dict(f) for f in features.get("static") or []
        ),
        future_features=tuple(
            FeatureSpec.from_dict(f) for f in features.get("future") or []
        ),
    )
    config.validate()
    return config


def available_datasets(root: str = CONFIG_ROOT) -> list[str]:
    """Dataset names with a config file under `configs/datasets/`."""
    directory = os.path.join(root, "datasets")
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith(".yaml")
    )


def _read_yaml(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve(path: str) -> str:
    """Interpret a `data.dir` relative to the project root."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; lists and scalars are replaced wholesale."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _apply_override(target: dict[str, Any], override: str) -> None:
    """Apply one `section.key=value` override, typed by the target dataclass field."""
    if "=" not in override:
        raise ValueError(f"override {override!r} must look like section.key=value")
    path, raw = override.split("=", 1)
    keys = path.split(".")
    node = target
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"override path {path!r} traverses a non-mapping value")
    node[keys[-1]] = _coerce(keys, raw)


def _coerce(keys: Sequence[str], raw: str) -> Any:
    """Parse an override value, checking the field exists on its dataclass."""
    value = _parse_scalar(raw)
    section = _SECTIONS.get(keys[0])
    if section is None or len(keys) != 2:
        return value
    known = {f.name for f in fields(section) if is_dataclass(section)}
    if keys[1] not in known:
        name = ".".join(keys)
        raise ValueError(
            f"override {name} targets no field of {section.__name__}; "
            f"expected one of {sorted(known)}"
        )
    return value


def _parse_scalar(raw: str) -> Any:
    """YAML scalar parsing, widened to accept `1e-3` (YAML 1.1 needs `1.0e-3`)."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value
