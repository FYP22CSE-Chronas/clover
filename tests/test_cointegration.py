from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from config import FeatureSpec, ModelConfig, ModelDims, SplitConfig
from contracts import WindowBatch
from data.dataset import HierarchicalDataset
from data.features import build_features, johansen_beta
from data.windows import build_split
from model.network import CLOVER
from pipeline.batching import WindowBatcher, hist_feature_stats

N_BOTTOM = 4
T = 400


def subspace_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Largest principal angle (degrees) between the column spaces of a and b."""
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    cosines = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(cosines.min(), -1.0, 1.0))))


def cointegrated_levels(
    rng: np.random.Generator, n_steps: int, n_series: int, n_trends: int
) -> tuple[np.ndarray, np.ndarray]:
    """`n_series` levels driven by `n_trends` random walks plus stationary noise.

    Returns the levels and an orthonormal basis of the true cointegrating space,
    the orthogonal complement of the trend loadings.
    """
    loadings = rng.normal(size=(n_series, n_trends))
    trends = np.cumsum(rng.normal(size=(n_steps, n_trends)), axis=0)
    stationary = np.zeros((n_steps, n_series))
    for t in range(1, n_steps):
        stationary[t] = 0.5 * stationary[t - 1] + rng.normal(scale=0.3, size=n_series)
    levels = 1000.0 + trends @ loadings.T + stationary
    _, _, vt = np.linalg.svd(loadings.T)
    return levels, vt[n_trends:].T


@pytest.fixture
def coint_dataset(S: np.ndarray, level_slices: dict) -> HierarchicalDataset:
    """The conftest hierarchy, but with cointegrated bottom series."""
    rng = np.random.default_rng(1)
    levels, _ = cointegrated_levels(rng, T, N_BOTTOM, n_trends=2)
    bottom = levels.astype(np.float32)
    return HierarchicalDataset(
        values=(bottom @ S.T).astype(np.float32),
        bottom=bottom,
        S=S,
        dates=pd.date_range("1990-01-01", periods=T, freq="MS"),
        series_names=["total", "g1", "g2", "a-x", "a-y", "b-x", "b-y"],
        bottom_names=["a-x", "a-y", "b-x", "b-y"],
        level_slices=level_slices,
    )


def test_johansen_recovers_the_cointegrating_space() -> None:
    rng = np.random.default_rng(0)
    levels, truth = cointegrated_levels(rng, T, n_series=5, n_trends=2)
    beta, eigenvalues = johansen_beta(levels, rank=3)
    assert beta.shape == (5, 3)
    np.testing.assert_allclose(np.linalg.norm(beta, axis=0), 1.0, rtol=1e-5)
    assert np.all(np.diff(eigenvalues) <= 0)
    assert subspace_angle(beta, truth) < 10.0
    # The combinations found are stationary: their variance is far below the levels'.
    assert (levels @ beta).var(axis=0).max() < 0.05 * levels.var(axis=0).mean()


def test_johansen_matches_statsmodels_direction() -> None:
    statsmodels = pytest.importorskip("statsmodels.tsa.vector_ar.vecm")
    rng = np.random.default_rng(2)
    levels, _ = cointegrated_levels(rng, T, n_series=4, n_trends=2)
    ours, _ = johansen_beta(levels, rank=2, k_ar_diff=1, deterministic="const")
    theirs = statsmodels.coint_johansen(levels, det_order=0, k_ar_diff=1).evec[:, :2]
    assert subspace_angle(ours, theirs) < 5.0


def test_johansen_needs_ridge_when_wider_than_long() -> None:
    rng = np.random.default_rng(3)
    levels = np.cumsum(rng.normal(size=(20, 30)), axis=0)
    with pytest.raises(ValueError, match="singular"):
        johansen_beta(levels, rank=1)
    beta, _ = johansen_beta(levels, rank=1, ridge=0.1)
    assert np.isfinite(beta).all()


def test_johansen_rejects_bad_arguments() -> None:
    levels = np.cumsum(np.ones((50, 3)), axis=0)
    with pytest.raises(ValueError, match="rank"):
        johansen_beta(levels, rank=4)
    with pytest.raises(ValueError, match="deterministic"):
        johansen_beta(levels, rank=1, deterministic="trend")
    with pytest.raises(ValueError, match=r"\[T, m\]"):
        johansen_beta(levels[:, 0], rank=1)


def test_equilibrium_error_is_a_fixed_combination_of_levels(
    coint_dataset: HierarchicalDataset,
) -> None:
    features = build_features(
        coint_dataset,
        history=[FeatureSpec("equilibrium_error", {"rank": 2})],
        fit_end=300,
    )
    assert features.n_hist == 2
    assert features.hist_names == ["eq_error_0", "eq_error_1"]
    block = features.hist_series
    assert block.shape == (T, N_BOTTOM, 2)
    # Every series carries the same two relations when the whole bottom is one group.
    np.testing.assert_allclose(block[:, 0], block[:, 3], rtol=1e-6)
    beta, _ = johansen_beta(coint_dataset.bottom[:300], rank=2)
    np.testing.assert_allclose(block[:, 0], coint_dataset.bottom @ beta, rtol=1e-4)


def test_equilibrium_error_never_looks_past_fit_end(
    coint_dataset: HierarchicalDataset,
) -> None:
    spec = FeatureSpec("equilibrium_error", {"rank": 1})
    clean = build_features(coint_dataset, history=[spec], fit_end=300).hist_series
    tampered = coint_dataset.bottom.copy()
    tampered[300:] *= 7.0
    dirty = build_features(
        replace(coint_dataset, bottom=tampered), history=[spec], fit_end=300
    ).hist_series
    np.testing.assert_allclose(clean[:300], dirty[:300], rtol=1e-6)


def test_equilibrium_error_per_sibling_group(
    coint_dataset: HierarchicalDataset,
) -> None:
    features = build_features(
        coint_dataset,
        history=[FeatureSpec("equilibrium_error", {"rank": 2, "level": "Group"})],
        fit_end=300,
    )
    block = features.hist_series
    # Each pair supports one relation; the second channel is zero everywhere.
    assert np.all(block[..., 1] == 0)
    assert np.allclose(block[:, 0, 0], block[:, 1, 0])
    assert not np.allclose(block[:, 0, 0], block[:, 2, 0])


def test_equilibrium_error_rejects_non_partitioning_level_and_config_fit_end(
    coint_dataset: HierarchicalDataset,
) -> None:
    with pytest.raises(KeyError, match="unknown level"):
        build_features(
            coint_dataset,
            history=[FeatureSpec("equilibrium_error", {"level": "Nope"})],
            fit_end=300,
        )
    with pytest.raises(ValueError, match="set by the pipeline"):
        build_features(
            coint_dataset,
            history=[FeatureSpec("equilibrium_error", {"fit_end": 10})],
            fit_end=300,
        )
    with pytest.raises(ValueError, match="do not belong here"):
        build_features(coint_dataset, future=[FeatureSpec("equilibrium_error", {})])


def test_batcher_z_scores_history_with_training_rows_only(
    coint_dataset: HierarchicalDataset,
) -> None:
    input_size, h = 12, 4
    split = build_split(coint_dataset.bottom, input_size, h, SplitConfig())
    features = build_features(
        coint_dataset,
        history=[FeatureSpec("equilibrium_error", {"rank": 1})],
        fit_end=split.train_end,
    )
    stats = hist_feature_stats(features, split.train_end)
    with pytest.raises(ValueError, match="hist_stats"):
        WindowBatcher(split.train, features, input_size, h)
    batcher = WindowBatcher(split.train, features, input_size, h, hist_stats=stats)
    batch = batcher.batch([0, 1])
    hist = batch.windows.hist_exog
    assert hist.shape == (2, 1, input_size, N_BOTTOM)
    loc, scale = stats
    expected = (features.hist_series[0:input_size, :, 0] - loc[:, 0]) / scale[:, 0]
    torch.testing.assert_close(hist[0, 0], torch.as_tensor(expected, dtype=torch.float32))


def test_error_correction_skip_starts_as_a_no_op_then_moves_mu(S: np.ndarray) -> None:
    n_bottom = S.shape[1]
    dims = ModelDims(h=4, input_size=8, n_hist_features=2)
    config = ModelConfig(temp_conv_channels=6, n_factors=3, error_correction_skip=True)
    torch.manual_seed(0)
    with_skip = CLOVER(config, dims, S)
    torch.manual_seed(0)
    without = CLOVER(replace(config, error_correction_skip=False), dims, S)
    batch = WindowBatch(
        insample_y=torch.randn(2, 8, n_bottom), hist_exog=torch.randn(2, 2, 8, n_bottom)
    )
    torch.testing.assert_close(with_skip(batch).mu, without(batch).mu)

    with torch.no_grad():
        with_skip.error_correction[1, 0, 0] = 2.0
    shifted = with_skip(batch)
    expected = without(batch).mu.clone()
    expected[:, 1, 0] += 2.0 * batch.hist_exog[:, 0, -1, 0]
    torch.testing.assert_close(shifted.mu, expected)
    torch.testing.assert_close(shifted.sigma, without(batch).sigma)


def test_error_correction_skip_gets_a_gradient(S: np.ndarray) -> None:
    n_bottom = S.shape[1]
    dims = ModelDims(h=4, input_size=8, n_hist_features=1)
    model = CLOVER(ModelConfig(n_factors=2, error_correction_skip=True), dims, S)
    batch = WindowBatch(
        insample_y=torch.randn(2, 8, n_bottom), hist_exog=torch.randn(2, 1, 8, n_bottom)
    )
    model(batch).mu.sum().backward()
    assert model.error_correction.grad is not None
    assert model.error_correction.grad.abs().sum() > 0


def test_error_correction_skip_requires_a_history_channel(S: np.ndarray) -> None:
    with pytest.raises(ValueError, match="n_hist_features"):
        CLOVER(ModelConfig(error_correction_skip=True), ModelDims(h=4, input_size=8), S)


def test_rolling_split_scores_several_origins() -> None:
    values = np.arange(60 * 3, dtype=np.float32).reshape(60, 3)
    split = build_split(values, 8, 4, SplitConfig(strategy="rolling", origins=3))
    assert [w.t for w in split.test] == [48, 52, 56]
    assert [w.t for w in split.val] == [36, 40, 44]
    assert max(w.t for w in split.train) + 4 <= 36
    targets = split.target_indices()
    assert not targets["train"] & targets["val"]
    assert not targets["val"] & targets["test"]
    assert split.train_end == 36
    with pytest.raises(ValueError, match="needs more than"):
        build_split(values, 8, 4, SplitConfig(strategy="rolling", origins=7))
    with pytest.raises(ValueError, match="origins"):
        SplitConfig(strategy="rolling")


def test_trend_loadings_are_orthogonal_to_beta_in_raw_units(S: np.ndarray) -> None:
    from contracts import ScaleStats

    n_bottom = S.shape[1]
    rng = np.random.default_rng(0)
    beta = rng.normal(size=(n_bottom, 2)).astype(np.float32)
    dims = ModelDims(h=4, input_size=8)
    config = ModelConfig(temp_conv_channels=6, n_factors=5, trend_factors=3)
    model = CLOVER(config, dims, S, coint_beta=beta)
    scale = torch.rand(2, 1, n_bottom) + 0.5
    batch = WindowBatch(
        insample_y=torch.randn(2, 8, n_bottom),
        scale=ScaleStats(loc=torch.zeros(2, 1, n_bottom), scale=scale),
    )
    loadings = model(batch).F  # [B, H, Nb, 5], normalized units
    raw = loadings * scale.squeeze(1)[:, None, :, None]
    along_beta = torch.einsum("nr,bhnk->bhrk", torch.as_tensor(beta), raw[..., :3])
    assert along_beta.abs().max() < 1e-4
    # The free factors are untouched by the projection and the trend ones grow.
    free = CLOVER(replace(config, trend_factors=0), dims, S)
    free.load_state_dict(
        {
            k: v
            for k, v in model.state_dict().items()
            if "coint" not in k and "trend" not in k
        }
    )
    torch.testing.assert_close(free(batch).F[..., 3:], loadings[..., 3:])
    norms = raw[..., :3].norm(dim=(2, 3))  # [B, H]
    assert (norms[:, 1:] > norms[:, :-1] * 0.9).all()


def test_trend_factors_need_beta_and_bounds(S: np.ndarray) -> None:
    with pytest.raises(ValueError, match="cointegrating vectors"):
        CLOVER(ModelConfig(trend_factors=1), ModelDims(h=4, input_size=8), S)
    with pytest.raises(ValueError, match="trend_factors"):
        ModelConfig(n_factors=2, trend_factors=3)


def test_feature_block_carries_block_structured_beta(
    coint_dataset: HierarchicalDataset,
) -> None:
    features = build_features(
        coint_dataset,
        history=[FeatureSpec("equilibrium_error", {"rank": 1, "level": "Group"})],
        fit_end=300,
    )
    beta = features.coint_beta
    assert beta.shape == (N_BOTTOM, 2)
    assert np.all(beta[2:, 0] == 0) and np.all(beta[:2, 1] == 0)
    np.testing.assert_allclose(np.linalg.norm(beta, axis=0), 1.0, rtol=1e-5)


def test_separable_skip_is_a_no_op_at_init_and_hides_channels_from_encoder(
    S: np.ndarray,
) -> None:
    n_bottom = S.shape[1]
    dims = ModelDims(h=4, input_size=8, n_hist_features=2)
    config = ModelConfig(
        temp_conv_channels=6,
        n_factors=3,
        error_correction_skip=True,
        error_correction_form="separable",
        hist_to_encoder=False,
    )
    torch.manual_seed(0)
    model = CLOVER(config, dims, S)
    torch.manual_seed(0)
    plain = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3), ModelDims(h=4, input_size=8), S
    )
    batch = WindowBatch(
        insample_y=torch.randn(2, 8, n_bottom), hist_exog=torch.randn(2, 2, 8, n_bottom)
    )
    # Channels bypass the encoder, so the plain model with no history sees the same input.
    torch.testing.assert_close(model(batch).mu, plain(batch).mu)
    assert model.error_speed.shape == (n_bottom, 2)
    assert model.error_correction.shape == (4, 2)
    with torch.no_grad():
        model.error_speed[0, 1] = 0.5
    shifted = model(batch).mu
    expected = plain(batch).mu.clone()
    expected[:, :, 0] += 0.5 * batch.hist_exog[:, 1, -1, 0][:, None]
    torch.testing.assert_close(shifted, expected)
    with pytest.raises(ValueError, match="error_correction_form"):
        ModelConfig(error_correction_form="lowrank")


def test_share_method_recovers_shares_and_zero_sum_deviations(
    coint_dataset: HierarchicalDataset,
) -> None:
    from data.features.cointegration import share_deviation

    levels = coint_dataset.bottom[:, :2].astype(np.float64)
    deviations, shares, beta = share_deviation(levels, fit_end=300)
    assert shares.sum() == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(deviations[:300].sum(axis=1), 0.0, atol=1e-2)
    assert beta.shape == (2, 1)
    assert abs(shares @ beta[:, 0]) < 1e-6
    features = build_features(
        coint_dataset,
        history=[FeatureSpec("equilibrium_error", {"level": "Group", "method": "share"})],
        fit_end=300,
    )
    assert features.n_hist == 1
    assert features.hist_names == ["share_deviation"]
    assert features.coint_beta.shape == (N_BOTTOM, 2)
    with pytest.raises(ValueError, match="needs a level"):
        build_features(
            coint_dataset,
            history=[FeatureSpec("equilibrium_error", {"method": "share"})],
            fit_end=300,
        )


def test_ols_warm_start_matches_a_direct_regression(
    coint_dataset: HierarchicalDataset,
) -> None:
    from pipeline.runner import warm_start_error_correction

    input_size, h = 12, 4
    split = build_split(coint_dataset.bottom, input_size, h, SplitConfig())
    features = build_features(
        coint_dataset,
        history=[FeatureSpec("equilibrium_error", {"level": "Group", "method": "share"})],
        fit_end=split.train_end,
    )
    stats = hist_feature_stats(features, split.train_end)
    train = WindowBatcher(split.train, features, input_size, h, hist_stats=stats)
    dims = ModelDims(h=h, input_size=input_size, n_hist_features=1)
    config = ModelConfig(
        temp_conv_channels=6,
        n_factors=3,
        error_correction_skip=True,
        error_correction_init="ols",
    )
    model = CLOVER(config, dims, coint_dataset.S)
    warm_start_error_correction(model, train)
    batch = train.batch(list(range(train.n_windows)))
    x = batch.windows.hist_exog[:, 0, -1, 0].numpy()
    y = (
        (batch.target_bottom[:, 2, 0] - batch.scale.loc[:, 0, 0])
        / batch.scale.scale[:, 0, 0]
        - batch.windows.insample_y[:, -1, 0]
    ).numpy()
    slope = np.polyfit(x, y, 1)[0]
    assert model.error_correction[2, 0, 0].item() == pytest.approx(slope, rel=1e-3)
    with pytest.raises(ValueError, match="requires error_correction_form"):
        ModelConfig(error_correction_init="ols", error_correction_form="separable")
