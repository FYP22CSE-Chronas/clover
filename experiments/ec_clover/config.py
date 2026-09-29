from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import yaml


def _strict(cls: type, raw: Mapping[str, Any], where: str):
    """Build a dataclass from a mapping, rejecting unknown keys (mirrors config.py)."""
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; expected {sorted(known)}")
    return cls(**raw)


@dataclass(frozen=True)
class BetaInitConfig:
    """How to initialize the cointegrating matrix beta ∈ R^(Nb x r).

    `type="random"`: unit-norm random columns (see `ec_branch.py` for why unit norm).
    `type="johansen"`: use a beta estimated offline from TRAINING-split data only
    (`experiments/ec_clover/johansen.py` / `scripts/estimate_beta.py`), supplied either
    inline as `values` ([Nb, r] nested list) or via `values_path` to a `.npy` file.
    Supplying both `values` and `values_path` is rejected as ambiguous.

    `scale_to_normalized` converts that raw-level beta into the per-window-normalized
    units the model is actually fed, by multiplying row `i` by the mean training-window
    scale of series `i` (see `johansen.to_normalized_units`). Without it the loadings of
    series with very different levels are distorted by exactly that ratio. It defaults
    to False only to keep earlier runs reproducible; True is the correct setting for a
    Johansen beta, and it is ignored for `type="random"`.
    """

    type: str = "random"
    values: tuple[tuple[float, ...], ...] | None = None
    values_path: str | None = None
    scale_to_normalized: bool = False

    def __post_init__(self) -> None:
        if self.type not in ("random", "johansen"):
            raise ValueError(
                f"beta_init.type must be 'random' or 'johansen', got {self.type!r}"
            )
        if self.scale_to_normalized and self.type != "johansen":
            raise ValueError(
                "beta_init.scale_to_normalized only applies to type='johansen'"
            )
        if self.type == "johansen" and self.values is None and self.values_path is None:
            raise ValueError(
                "beta_init.type == 'johansen' requires 'values' or 'values_path'"
            )
        if self.values is not None and self.values_path is not None:
            raise ValueError("beta_init: set only one of 'values' or 'values_path'")
        if self.values is not None:
            object.__setattr__(
                self, "values", tuple(tuple(float(x) for x in row) for row in self.values)
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BetaInitConfig:
        return _strict(cls, raw, "ec.beta_init")


@dataclass(frozen=True)
class ECConfig:
    """Configuration for the error-correction branch of `CloverWithErrorCorrection`.

    `enabled=False` (the default) makes the model numerically identical to baseline
    CLOVER: no EC modules are constructed and `forward` delegates to `CLOVER.forward`.

    Every field below defaults to the behaviour of the first implementation, so an
    existing config keeps reproducing its earlier numbers; the non-default values are
    the enhancements swept in `docs`/NOTES.md.

    Branch-internal knobs (none of these touch a CLOVER module):

    - `fusion`: `"concat"` reprojects `[h_clover; h_ec]` through a fresh MLP, which
      *discards* the baseline representation at initialization. `"gated_residual"`
      instead returns `h_clover + sigmoid(gate) * MLP([h_clover; h_ec])` with `gate`
      initialized at `gate_init`, so the branch starts as a near-no-op perturbation of
      the baseline and can only earn influence if it lowers the loss.
    - `fusion_layers` / `fusion_hidden`: depth and width of the fusion MLP.
    - `ec_input`: `"ec"` encodes the `[B, L, Nb]` signal `alpha @ beta^T y`; `"e"`
      encodes the `[B, L, r]` equilibrium errors `beta^T y` directly and lifts the
      resulting embedding to the bottom level with `alpha`. Since `EC` is a rank-`r`
      linear map of `E`, `"e"` gives the encoder the same information in `r` channels
      instead of `Nb` redundant ones.
    - `include_diff`: append the first difference of the encoded sequence as a second
      encoder channel. A textbook VECM has both a long-run term (`alpha beta^T y`) and
      short-run dynamics (`Gamma_i @ delta y`); the first implementation modelled only
      the long-run half.
    - `sequence_norm`: `"center"` subtracts each window's own mean from the
      equilibrium-error sequence, `"standardize"` also divides by its standard
      deviation. `CLOVER` normalizes `insample_y` per window per series, so
      `beta^T y_norm` carries a window-dependent offset and per-series scale distortion
      that has nothing to do with the equilibrium relationship; removing the offset
      leaves the branch to model deviation *dynamics*, which is the part that survives
      normalization.
    - `encoder_dilations`: EC-specific dilations, so the branch can look further back
      than CLOVER's own encoder without changing CLOVER's encoder. `None` reuses the
      model config's.
    - `beta_normalize`: `"columns"` rescales `beta`'s columns to unit norm on every
      forward pass, pinning the scale half of the identification problem. Note this
      *does* alter a supplied Johansen normalization; it is opt-in for exactly that
      reason (see the identification note in `ec_branch.py`).
    """

    enabled: bool = False
    rank: int = 1
    encoder: str = "dilated_conv"
    encoder_channels: int = 8
    encoder_dilations: tuple[int, ...] | None = None
    fusion: str = "concat"
    fusion_hidden: int | None = None
    fusion_layers: int = 1
    gate_init: float = -2.0
    ec_input: str = "ec"
    include_diff: bool = False
    sequence_norm: str = "none"
    beta_normalize: str = "none"
    alpha_init_std: float = 1e-2
    beta_init: BetaInitConfig = field(default_factory=BetaInitConfig)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"ec.rank must be >= 1, got {self.rank}")
        if self.encoder_channels < 1:
            raise ValueError(
                f"ec.encoder_channels must be >= 1, got {self.encoder_channels}"
            )
        if self.alpha_init_std < 0:
            raise ValueError(f"ec.alpha_init_std must be >= 0, got {self.alpha_init_std}")
        if self.fusion not in ("concat", "gated_residual"):
            raise ValueError(
                f"ec.fusion must be 'concat' or 'gated_residual', got {self.fusion!r}"
            )
        if self.fusion_layers < 1:
            raise ValueError(f"ec.fusion_layers must be >= 1, got {self.fusion_layers}")
        if self.ec_input not in ("ec", "e"):
            raise ValueError(f"ec.ec_input must be 'ec' or 'e', got {self.ec_input!r}")
        if self.sequence_norm not in ("none", "center", "standardize"):
            raise ValueError(
                f"ec.sequence_norm must be 'none', 'center' or 'standardize', "
                f"got {self.sequence_norm!r}"
            )
        if self.beta_normalize not in ("none", "columns"):
            raise ValueError(
                f"ec.beta_normalize must be 'none' or 'columns', "
                f"got {self.beta_normalize!r}"
            )
        if self.encoder_dilations is not None:
            dilations = tuple(int(d) for d in self.encoder_dilations)
            if not dilations:
                raise ValueError("ec.encoder_dilations must be non-empty when given")
            if any(d < 1 for d in dilations):
                raise ValueError(
                    f"ec.encoder_dilations must all be >= 1, got {dilations}"
                )
            object.__setattr__(self, "encoder_dilations", dilations)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ECConfig:
        raw = dict(raw)
        if "beta_init" in raw:
            raw["beta_init"] = BetaInitConfig.from_dict(raw["beta_init"])
        return _strict(cls, raw, "ec")


def load_ec_config(path: str | None) -> ECConfig:
    """Load the `ec:` section of a standalone YAML file; `None` -> EC disabled.

    Kept out of `config.load_config` deliberately -- `RunConfig`'s top-level keys are
    a closed set (`config.py:TOP_LEVEL_KEYS`) that must keep rejecting typos for the
    shipped model, so this experiment's config lives in its own file instead of a new
    top-level key on the core schema (see IMPLEMENTATION.md §3, "experiment configs
    live beside the experiment").
    """
    if path is None:
        return ECConfig()
    if not os.path.exists(path):
        raise FileNotFoundError(f"ec config file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    unknown = sorted(set(raw) - {"ec"})
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s) {unknown}; expected ['ec']")
    return ECConfig.from_dict(raw.get("ec") or {})
