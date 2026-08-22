from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from config import FeatureSpec
from data.dataset import ExogenousFeatures, HierarchicalDataset

FeatureKind = Literal["static", "future_shared", "future_series"]


@dataclass(frozen=True)
class FeatureBlock:
    """One builder's output: an array, the axis convention it follows, its names."""

    kind: FeatureKind
    values: np.ndarray
    names: list[str]


FeatureBuilder = Callable[..., FeatureBlock]
FEATURE_REGISTRY: dict[str, FeatureBuilder] = {}


def register_feature(name: str) -> Callable[[FeatureBuilder], FeatureBuilder]:
    def wrap(fn: FeatureBuilder) -> FeatureBuilder:
        if name in FEATURE_REGISTRY:
            raise KeyError(f"feature builder {name!r} is already registered")
        FEATURE_REGISTRY[name] = fn
        return fn

    return wrap


def build_feature(spec: FeatureSpec, dataset: HierarchicalDataset) -> FeatureBlock:
    """Resolve one `FeatureSpec` against the registry and run it."""
    try:
        builder = FEATURE_REGISTRY[spec.type]
    except KeyError:
        raise KeyError(
            f"unknown feature {spec.type!r}; available: {sorted(FEATURE_REGISTRY)}"
        ) from None
    return builder(dataset, **spec.params)


def build_features(
    dataset: HierarchicalDataset,
    static: Iterable[FeatureSpec] = (),
    future: Iterable[FeatureSpec] = (),
) -> ExogenousFeatures:
    """Assemble every configured block into one `ExogenousFeatures`.

    Future channels are ordered shared-first, then per-series, because the pipeline
    re-normalizes only the trailing per-series block per window.
    """
    static_blocks = [build_feature(spec, dataset) for spec in static]
    future_blocks = [build_feature(spec, dataset) for spec in future]
    _reject(static_blocks, {"static"}, "features.static")
    _reject(future_blocks, {"future_shared", "future_series"}, "features.future")

    shared = [b for b in future_blocks if b.kind == "future_shared"]
    series = [b for b in future_blocks if b.kind == "future_series"]
    return ExogenousFeatures(
        static=_concat(static_blocks, axis=1),
        future_shared=_concat(shared, axis=1),
        future_series=_concat(series, axis=2),
        static_names=[n for b in static_blocks for n in b.names],
        future_names=[n for b in shared + series for n in b.names],
    )


def _reject(blocks: list[FeatureBlock], allowed: set[str], where: str) -> None:
    """Fail when a builder is listed under the wrong config section."""
    bad = [b.kind for b in blocks if b.kind not in allowed]
    if bad:
        raise ValueError(
            f"{where}: builders of kind {sorted(set(bad))} do not belong here"
        )


def _concat(blocks: list[FeatureBlock], axis: int) -> np.ndarray | None:
    if not blocks:
        return None
    return np.concatenate([b.values for b in blocks], axis=axis).astype(np.float32)
