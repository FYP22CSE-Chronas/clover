from __future__ import annotations

import numpy as np
import pytest

from config import FeatureSpec
from data.dataset import HierarchicalDataset
from data.features import FEATURE_REGISTRY, build_features, check_anchor_lags


def spec(kind: str, **params) -> FeatureSpec:
    return FeatureSpec(type=kind, params=params)


def test_quarter_dummies_are_one_hot(dataset: HierarchicalDataset) -> None:
    features = build_features(dataset, future=[spec("quarter_dummies")])
    block = features.future_shared
    assert block.shape == (dataset.n_steps, 4)
    np.testing.assert_array_equal(block.sum(axis=1), np.ones(dataset.n_steps))


def test_grouped_static_dummies_read_categories_off_the_data(
    dataset: HierarchicalDataset,
) -> None:
    features = build_features(
        dataset, static=[spec("grouped_static_dummies", keys=["state"], sep="-")]
    )
    assert features.static.shape == (dataset.n_bottom, 2)  # "a" and "b"
    assert features.static_names == ["state_a", "state_b"]


def test_identity_dummies_are_the_identity(dataset: HierarchicalDataset) -> None:
    features = build_features(dataset, static=[spec("identity_dummies")])
    np.testing.assert_array_equal(features.static, np.eye(dataset.n_bottom))


def test_anchor_is_a_lagged_copy_in_raw_units(dataset: HierarchicalDataset) -> None:
    features = build_features(dataset, future=[spec("seasonal_naive_anchor", lags=[4])])
    anchor = features.future_series[..., 0]
    np.testing.assert_allclose(anchor[4:], dataset.bottom[:-4], rtol=1e-6)
    # Back-filled, never zero-filled.
    np.testing.assert_allclose(anchor[:4], np.tile(dataset.bottom[0], (4, 1)), rtol=1e-6)


def test_future_blocks_are_ordered_shared_then_series(
    dataset: HierarchicalDataset,
) -> None:
    features = build_features(
        dataset,
        future=[spec("quarter_dummies"), spec("seasonal_naive_anchor", lags=[4])],
    )
    assert features.n_future_shared == 4
    assert features.n_future_series == 1
    assert features.future_names == [
        "quarter_0",
        "quarter_1",
        "quarter_2",
        "quarter_3",
        "anchor_lag4",
    ]


def test_anchor_leakage_guard_rejects_short_lags() -> None:
    with pytest.raises(ValueError, match="shorter than the horizon"):
        check_anchor_lags([spec("seasonal_naive_anchor", lags=[2])], h=4)


def test_anchor_leakage_guard_allows_lag_equal_to_h() -> None:
    check_anchor_lags([spec("seasonal_naive_anchor", lags=[4, 8])], h=4)


def test_static_builder_rejected_in_the_future_section(
    dataset: HierarchicalDataset,
) -> None:
    with pytest.raises(ValueError, match="do not belong here"):
        build_features(dataset, future=[spec("identity_dummies")])


def test_unknown_feature_lists_alternatives(dataset: HierarchicalDataset) -> None:
    with pytest.raises(KeyError, match="available"):
        build_features(dataset, static=[spec("nope")])


def test_registry_contains_every_shipped_builder() -> None:
    assert set(FEATURE_REGISTRY) >= {
        "dayofweek_dummies",
        "grouped_static_dummies",
        "identity_dummies",
        "month_dummies",
        "quarter_dummies",
        "saturday_proximity",
        "seasonal_naive_anchor",
        "tourism_large_static_dummies",
        "weekend_indicator",
    }
