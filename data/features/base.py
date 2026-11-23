from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from config import FeatureSpec
from data.dataset import ExogenousFeatures, HierarchicalDataset

FeatureKind = Literal["static", "future_shared", "future_series", "hist_series"]


@dataclass(frozen=True)
class FeatureBlock:
    """One builder's output: an array, the axis convention it follows, its names."""

    kind: FeatureKind
    values: np.ndarray
    names: list[str]
    # A history builder that found cointegrating vectors also hands them over as
    # [Nb, R], so the model can constrain its factor loadings with them.
    beta: np.ndarray | None = None


FeatureBuilder = Callable[..., FeatureBlock]
FEATURE_REGISTRY: dict[str, FeatureBuilder] = {}


def register_feature(name: str) -> Callable[[FeatureBuilder], FeatureBuilder]:
    def wrap(fn: FeatureBuilder) -> FeatureBuilder:
        if name in FEATURE_REGISTRY:
            raise KeyError(f"feature builder {name!r} is already registered")
        FEATURE_REGISTRY[name] = fn
        return fn

    return wrap


def build_feature(
    spec: FeatureSpec, dataset: HierarchicalDataset, fit_end: int | None = None
) -> FeatureBlock:
    """Resolve one `FeatureSpec` against the registry and run it.

    A builder that estimates something from the data (a cointegrating vector, say)
    declares a `fit_end` keyword and receives the first row it must not look at, so
    validation and test rows never shape a feature. Builders that only transform the
    calendar or the series names do not declare it and are called as before.
    """
    try:
        builder = FEATURE_REGISTRY[spec.type]
    except KeyError:
        raise KeyError(
            f"unknown feature {spec.type!r}; available: {sorted(FEATURE_REGISTRY)}"
        ) from None
    if "fit_end" in spec.params:
        raise ValueError(
            f"feature {spec.type!r}: fit_end is set by the pipeline from the training "
            "split, not by config"
        )
    kwargs = dict(spec.params)
    if "fit_end" in inspect.signature(builder).parameters:
        kwargs["fit_end"] = fit_end
    return builder(dataset, **kwargs)


def build_features(
    dataset: HierarchicalDataset,
    static: Iterable[FeatureSpec] = (),
    future: Iterable[FeatureSpec] = (),
    history: Iterable[FeatureSpec] = (),
    fit_end: int | None = None,
) -> ExogenousFeatures:
    """Assemble every configured block into one `ExogenousFeatures`.

    Future channels are ordered shared-first, then per-series, because the pipeline
    re-normalizes only the trailing per-series block per window. `fit_end` is the
    end of the training span, handed to any builder that fits parameters.
    """
    static_blocks = [build_feature(spec, dataset, fit_end) for spec in static]
    future_blocks = [build_feature(spec, dataset, fit_end) for spec in future]
    hist_blocks = [build_feature(spec, dataset, fit_end) for spec in history]
    _reject(static_blocks, {"static"}, "features.static")
    _reject(future_blocks, {"future_shared", "future_series"}, "features.future")
    _reject(hist_blocks, {"hist_series"}, "features.history")

    shared = [b for b in future_blocks if b.kind == "future_shared"]
    series = [b for b in future_blocks if b.kind == "future_series"]
    return ExogenousFeatures(
        static=_concat(static_blocks, axis=1),
        future_shared=_concat(shared, axis=1),
        future_series=_concat(series, axis=2),
        hist_series=_concat(hist_blocks, axis=2),
        coint_beta=_concat_beta([b.beta for b in hist_blocks if b.beta is not None]),
        static_names=[n for b in static_blocks for n in b.names],
        future_names=[n for b in shared + series for n in b.names],
        hist_names=[n for b in hist_blocks for n in b.names],
    )


def _concat_beta(betas: list[np.ndarray]) -> np.ndarray | None:
    if not betas:
        return None
    return np.concatenate(betas, axis=1).astype(np.float32)


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
