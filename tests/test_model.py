from __future__ import annotations

import numpy as np
import pytest
import torch

from config import ModelConfig, ModelDims, SplitConfig
from contracts import ScaleStats, WindowBatch
from data import build_features, build_split
from model.mixers import MIXERS, CrossSeriesGNNMixer, hierarchy_edges
from model.network import CLOVER
from pipeline.batching import WindowBatcher

B, L, H = 2, 8, 4


def _batch(n_bottom: int, **kwargs) -> WindowBatch:
    return WindowBatch(insample_y=torch.randn(B, L, n_bottom), **kwargs)


def test_forward_shapes(model: CLOVER) -> None:
    params = model(_batch(model.n_bottom))
    assert params.mu.shape == (B, H, model.n_bottom)
    assert params.sigma.shape == (B, H, model.n_bottom)
    assert params.F.shape == (B, H, model.n_bottom, model.config.n_factors)


def test_sigma_is_positive(model: CLOVER) -> None:
    assert (model(_batch(model.n_bottom)).sigma > 0).all()


def test_rejects_wrong_series_count(model: CLOVER) -> None:
    with pytest.raises(ValueError, match="series but S expects"):
        model(_batch(model.n_bottom + 1))


def test_rejects_non_hierarchical_S() -> None:
    with pytest.raises(ValueError, match=r"S must be \[Na\+Nb, Nb\]"):
        CLOVER(ModelConfig(), ModelDims(h=H, input_size=L), np.ones((2, 5), np.float32))


@pytest.mark.parametrize(
    "n_stat,n_futr,n_hist", [(3, 0, 0), (0, 2, 0), (0, 0, 2), (3, 2, 1)]
)
def test_forward_with_exogenous_blocks(
    S: np.ndarray, n_stat: int, n_futr: int, n_hist: int
) -> None:
    n_bottom = S.shape[1]
    dims = ModelDims(
        h=H,
        input_size=L,
        n_stat_features=n_stat,
        n_futr_features=n_futr,
        n_hist_features=n_hist,
    )
    model = CLOVER(ModelConfig(temp_conv_channels=6, n_factors=3), dims, S)
    batch = _batch(
        n_bottom,
        stat_exog=torch.randn(n_bottom, n_stat) if n_stat else None,
        futr_exog=torch.randn(B, n_futr, L + H, n_bottom) if n_futr else None,
        hist_exog=torch.randn(B, n_hist, L, n_bottom) if n_hist else None,
    )
    assert model(batch).mu.shape == (B, H, n_bottom)


def test_batched_static_features(S: np.ndarray) -> None:
    """A panel varies static features across the batch, so [B, Nb, S] must work."""
    n_bottom = S.shape[1]
    dims = ModelDims(h=H, input_size=L, n_stat_features=3)
    model = CLOVER(ModelConfig(temp_conv_channels=6), dims, S)
    batch = _batch(n_bottom, stat_exog=torch.randn(B, n_bottom, 3))
    assert model(batch).mu.shape == (B, H, n_bottom)


def test_identity_mixer_is_a_passthrough(S: np.ndarray) -> None:
    n_hier, n_bottom = S.shape
    mixer = MIXERS.create(
        "identity", n_hierarchy=n_hier, n_bottom=n_bottom, channels=5, hidden_size=0
    )
    h = torch.randn(B, n_hier, 5)
    torch.testing.assert_close(mixer(h), h[:, -n_bottom:])


def test_disabled_cross_series_mlp_matches_identity(S: np.ndarray) -> None:
    n_hier, n_bottom = S.shape
    mixer = MIXERS.create(
        "cross_series_mlp",
        n_hierarchy=n_hier,
        n_bottom=n_bottom,
        channels=5,
        hidden_size=0,
    )
    h = torch.randn(B, n_hier, 5)
    torch.testing.assert_close(mixer(h), h[:, -n_bottom:])


def test_hierarchy_edges_link_bottoms_to_their_aggregates(S: np.ndarray) -> None:
    src, dst = hierarchy_edges(S)
    edges = set(zip(src.tolist(), dst.tolist(), strict=True))
    # Total (0) touches every bottom (3..6); group 1 has 3,4; group 2 has 5,6.
    expected = {(0, b) for b in range(3, 7)} | {(1, 3), (1, 4), (2, 5), (2, 6)}
    expected |= {(d, s) for s, d in expected}
    assert edges == expected


def test_disabled_cross_series_gnn_matches_identity(S: np.ndarray) -> None:
    n_hier, n_bottom = S.shape
    mixer = MIXERS.create(
        "cross_series_gnn",
        n_hierarchy=n_hier,
        n_bottom=n_bottom,
        channels=5,
        hidden_size=0,
        S=S,
    )
    h = torch.randn(B, n_hier, 5)
    torch.testing.assert_close(mixer(h), h[:, -n_bottom:])


def test_cross_series_gnn_only_mixes_linked_series() -> None:
    """Two disjoint groups and no total: a change in one never reaches the other."""
    A = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32)
    S = np.vstack([A, np.eye(4, dtype=np.float32)])
    torch.manual_seed(0)
    mixer = MIXERS.create(
        "cross_series_gnn", n_hierarchy=6, n_bottom=4, channels=5, hidden_size=8, S=S
    )
    h = torch.randn(B, 6, 5)
    bumped = h.clone()
    bumped[:, 2] += 1.0  # bottom series 0 (node 2), in the first group
    delta = (mixer(bumped) - mixer(h)).abs().amax(dim=(0, 2))
    assert delta[1] > 0  # its sibling hears it through the shared aggregate
    torch.testing.assert_close(delta[2:], torch.zeros(2))


def test_cross_series_gnn_rejects_mismatched_S(S: np.ndarray) -> None:
    with pytest.raises(ValueError, match="S has shape"):
        MIXERS.create(
            "cross_series_gnn",
            n_hierarchy=6,
            n_bottom=4,
            channels=5,
            hidden_size=8,
            S=S,
        )


def _gnn(S: np.ndarray, **kwargs) -> CrossSeriesGNNMixer:
    n_hier, n_bottom = S.shape
    return MIXERS.create(
        "cross_series_gnn",
        n_hierarchy=n_hier,
        n_bottom=n_bottom,
        channels=5,
        hidden_size=8,
        S=S,
        **kwargs,
    )


def _stats(n_bottom: int) -> ScaleStats:
    return ScaleStats(
        loc=torch.rand(B, 1, n_bottom) * 10, scale=torch.rand(B, 1, n_bottom) + 0.5
    )


def test_magnitude_features_sum_children_into_aggregates(S: np.ndarray) -> None:
    mixer = _gnn(S)
    stats = _stats(S.shape[1])
    feats = mixer.magnitude_features(torch.zeros(B, S.shape[0], 5), stats)
    S_t = torch.as_tensor(S)
    loc = stats.loc[:, 0] @ S_t.T
    log_scale = (stats.scale[:, 0] @ S_t.T).log()
    torch.testing.assert_close(feats[..., 0], log_scale - log_scale.mean(1, keepdim=True))
    torch.testing.assert_close(feats[..., 1], torch.asinh(loc / log_scale.exp()))


def test_magnitude_features_ignore_a_common_change_of_units(S: np.ndarray) -> None:
    """Measuring every series in other units must not change what the GNN sees."""
    mixer = _gnn(S)
    stats = _stats(S.shape[1])
    rescaled = ScaleStats(loc=stats.loc * 1000.0, scale=stats.scale * 1000.0)
    h = torch.zeros(B, S.shape[0], 5)
    torch.testing.assert_close(
        mixer.magnitude_features(h, stats), mixer.magnitude_features(h, rescaled)
    )


def test_scale_features_make_the_gnn_depend_on_stats(S: np.ndarray) -> None:
    torch.manual_seed(0)
    h, stats = torch.randn(B, S.shape[0], 5), _stats(S.shape[1])
    on = _gnn(S)
    assert not torch.allclose(on(h, stats), on(h, None))
    off = _gnn(S, scale_features=False)
    torch.testing.assert_close(off(h, stats), off(h, None))


def test_scale_features_travel_mainly_along_the_graph() -> None:
    """A series' magnitude reaches its sibling through their shared aggregate.

    The relative log scale is centered on a window-wide mean, so a disjoint group
    still sees a small shift through that mean -- but only through it, so its
    response must be far weaker than the sibling's.
    """
    A = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32)
    S = np.vstack([A, np.eye(4, dtype=np.float32)])
    torch.manual_seed(0)
    mixer = _gnn(S)
    h, stats = torch.randn(B, 6, 5), _stats(4)
    bumped = ScaleStats(loc=stats.loc.clone(), scale=stats.scale.clone())
    bumped.scale[:, :, 0] *= 5.0
    delta = (mixer(h, bumped) - mixer(h, stats)).abs().amax(dim=(0, 2))
    assert delta[1] > 2 * delta[2:].max()


def test_batcher_hands_the_model_its_normalization_stats(dataset) -> None:
    features = build_features(dataset)
    split = build_split(dataset.bottom, L, H, SplitConfig())
    batch = WindowBatcher(split.train, features, input_size=L, h=H).batch([0, 1])
    assert batch.windows.stats is batch.scale


def test_clover_passes_stats_to_the_gnn(S: np.ndarray) -> None:
    torch.manual_seed(0)
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, mixer="cross_series_gnn"),
        ModelDims(h=H, input_size=L),
        S,
    )
    batch = _batch(S.shape[1])
    with_stats = WindowBatch(insample_y=batch.insample_y, stats=_stats(S.shape[1]))
    assert not torch.allclose(model(batch).mu, model(with_stats).mu)


@pytest.mark.parametrize("mixer", ["cross_series_gnn", "cross_series_mlp", "identity"])
def test_forward_with_each_mixer(S: np.ndarray, mixer: str) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(ModelConfig(temp_conv_channels=6, n_factors=3, mixer=mixer), dims, S)
    assert model(_batch(S.shape[1])).mu.shape == (B, H, S.shape[1])


def test_gradients_reach_every_parameter(model: CLOVER) -> None:
    params = model(_batch(model.n_bottom))
    (params.mu.sum() + params.sigma.sum() + params.F.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


def test_skew_t_head_forward_shapes(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(ModelConfig(temp_conv_channels=6, n_factors=3, head="skew_t"), dims, S)
    params = model(_batch(model.n_bottom))
    assert params.mu.shape == (B, H, model.n_bottom)
    assert params.sigma.shape == (B, H, model.n_bottom)
    assert params.nu.shape == (B, H, model.n_bottom)
    assert params.lam.shape == (B, H, model.n_bottom)
    assert params.F.shape == (B, H, model.n_bottom, model.config.n_factors)
    assert (params.sigma > 0).all()
    assert (params.nu > 2).all()


def test_skew_t_head_gradients_reach_every_parameter(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(ModelConfig(temp_conv_channels=6, n_factors=3, head="skew_t"), dims, S)
    params = model(_batch(model.n_bottom))
    total = params.mu.sum() + params.sigma.sum() + params.nu.sum() + params.lam.sum()
    (total + params.F.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


def test_skew_t_shared_head_forward_shapes(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, head="skew_t_shared"), dims, S
    )
    params = model(_batch(model.n_bottom))
    assert params.mu.shape == (B, H, model.n_bottom)
    assert params.sigma.shape == (B, H, model.n_bottom)
    assert params.nu.shape == (B, H, model.n_bottom)
    assert params.lam.shape == (B, H, model.n_bottom)
    assert params.F.shape == (B, H, model.n_bottom, model.config.n_factors)
    assert (params.sigma > 0).all()
    assert (params.nu > 2).all()


def test_skew_t_shared_head_nu_and_lam_are_global() -> None:
    """A single (nu, lam) pair broadcasts across every series and horizon."""
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, head="skew_t_shared"),
        dims,
        np.eye(4, dtype=np.float32),
    )
    params = model(_batch(model.n_bottom))
    assert params.nu.unique().numel() == 1
    assert params.lam.unique().numel() == 1


def test_skew_t_shared_head_gradients_reach_every_parameter(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, head="skew_t_shared"), dims, S
    )
    params = model(_batch(model.n_bottom))
    total = params.mu.sum() + params.sigma.sum() + params.nu.sum() + params.lam.sum()
    (total + params.F.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


def test_flow_head_forward_shapes(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, head="normalizing_flow"), dims, S
    )
    params = model(_batch(model.n_bottom))
    k_flow = model.head.n_flow_components
    assert params.mu.shape == (B, H, model.n_bottom)
    assert params.sigma.shape == (B, H, model.n_bottom)
    assert params.flow_w.shape == (B, H, model.n_bottom, k_flow)
    assert params.flow_a.shape == (B, H, model.n_bottom, k_flow)
    assert params.flow_b.shape == (B, H, model.n_bottom, k_flow)
    assert params.F.shape == (B, H, model.n_bottom, model.config.n_factors)
    assert (params.sigma > 0).all()
    assert (params.flow_a > model.head.a_floor).all()
    torch.testing.assert_close(params.flow_w.sum(-1), torch.ones(B, H, model.n_bottom))


def test_flow_head_rejects_fewer_than_one_component() -> None:
    with pytest.raises(ValueError, match="n_flow_components must be >= 1"):
        from model.heads import NormalizingFlowHead

        NormalizingFlowHead(in_dim=4, n_factors=2, n_flow_components=0)


def test_model_config_n_flow_components_and_a_floor_reach_the_flow_head(
    S: np.ndarray,
) -> None:
    """`ModelConfig.n_flow_components`/`flow_a_floor` are only consumed by heads
    whose constructor declares them; CLOVER must filter, not pass unconditionally."""
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(
        ModelConfig(
            temp_conv_channels=6,
            n_factors=3,
            head="normalizing_flow",
            n_flow_components=3,
            flow_a_floor=0.25,
        ),
        dims,
        S,
    )
    assert model.head.n_flow_components == 3
    assert model.head.a_floor == 0.25
    params = model(_batch(model.n_bottom))
    assert params.flow_w.shape[-1] == 3


def test_model_config_flow_fields_do_not_break_unrelated_heads(S: np.ndarray) -> None:
    """A head that doesn't declare `n_flow_components`/`a_floor` (e.g. `gmm`) must
    still build, ignoring those config fields rather than raising a TypeError."""
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, head="gmm", n_flow_components=99),
        dims,
        S,
    )
    assert model.head.n_components == 2


def test_flow_head_gradients_reach_every_parameter(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, head="normalizing_flow"), dims, S
    )
    params = model(_batch(model.n_bottom))
    total = (
        params.mu.sum()
        + params.sigma.sum()
        + params.flow_w.sum()
        + params.flow_a.sum()
        + params.flow_b.sum()
    )
    (total + params.F.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


def test_flow_shared_head_forward_shapes(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, head="normalizing_flow_shared"),
        dims,
        S,
    )
    params = model(_batch(model.n_bottom))
    k_flow = model.head.n_flow_components
    assert params.mu.shape == (B, H, model.n_bottom)
    assert params.sigma.shape == (B, H, model.n_bottom)
    assert params.flow_w.shape == (B, H, model.n_bottom, k_flow)
    assert params.flow_a.shape == (B, H, model.n_bottom, k_flow)
    assert params.flow_b.shape == (B, H, model.n_bottom, k_flow)
    assert params.F.shape == (B, H, model.n_bottom, model.config.n_factors)
    assert (params.sigma > 0).all()
    assert (params.flow_a > model.head.a_floor).all()
    torch.testing.assert_close(params.flow_w.sum(-1), torch.ones(B, H, model.n_bottom))


def test_flow_shared_head_flow_is_global() -> None:
    """A single (w, a, b) flow broadcasts across every series and horizon."""
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, head="normalizing_flow_shared"),
        dims,
        np.eye(4, dtype=np.float32),
    )
    params = model(_batch(model.n_bottom))
    for k in range(model.head.n_flow_components):
        assert params.flow_w[..., k].unique().numel() == 1
        assert params.flow_a[..., k].unique().numel() == 1
        assert params.flow_b[..., k].unique().numel() == 1


def test_flow_shared_head_rejects_fewer_than_one_component() -> None:
    with pytest.raises(ValueError, match="n_flow_components must be >= 1"):
        from model.heads import NormalizingFlowSharedHead

        NormalizingFlowSharedHead(in_dim=4, n_factors=2, n_flow_components=0)


def test_flow_shared_head_gradients_reach_every_parameter(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, head="normalizing_flow_shared"),
        dims,
        S,
    )
    params = model(_batch(model.n_bottom))
    total = (
        params.mu.sum()
        + params.sigma.sum()
        + params.flow_w.sum()
        + params.flow_a.sum()
        + params.flow_b.sum()
    )
    (total + params.F.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


def test_copula_spline_head_forward_shapes(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, head="copula_spline"), dims, S
    )
    params = model(_batch(model.n_bottom))
    n_bins = model.head.n_spline_bins
    assert params.mu.shape == (B, H, model.n_bottom)
    assert params.sigma.shape == (B, H, model.n_bottom)
    assert params.spline_w.shape == (B, H, model.n_bottom, n_bins)
    assert params.spline_h.shape == (B, H, model.n_bottom, n_bins)
    assert params.spline_d.shape == (B, H, model.n_bottom, n_bins - 1)
    assert params.F.shape == (B, H, model.n_bottom, model.config.n_factors)
    assert (params.sigma > 0).all()
    assert (params.spline_d > model.head.slope_floor).all()
    torch.testing.assert_close(params.spline_w.sum(-1), torch.ones(B, H, model.n_bottom))
    torch.testing.assert_close(params.spline_h.sum(-1), torch.ones(B, H, model.n_bottom))


def test_copula_spline_head_rejects_fewer_than_one_bin() -> None:
    with pytest.raises(ValueError, match="n_spline_bins must be >= 1"):
        from model.heads import CopulaSplineHead

        CopulaSplineHead(in_dim=4, n_factors=2, n_spline_bins=0)


def test_copula_spline_head_gradients_reach_every_parameter(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, head="copula_spline"), dims, S
    )
    params = model(_batch(model.n_bottom))
    total = (
        params.mu.sum()
        + params.sigma.sum()
        + params.spline_w.sum()
        + params.spline_h.sum()
        + params.spline_d.sum()
    )
    (total + params.F.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


def test_copula_flow_head_forward_shapes(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, head="copula_flow"), dims, S
    )
    params = model(_batch(model.n_bottom))
    n_bins, n_flow = model.head.n_spline_bins, model.head.n_flow_components
    assert params.mu.shape == (B, H, model.n_bottom)
    assert params.sigma.shape == (B, H, model.n_bottom)
    assert params.spline_w.shape == (B, H, model.n_bottom, n_bins)
    assert params.spline_h.shape == (B, H, model.n_bottom, n_bins)
    assert params.spline_d.shape == (B, H, model.n_bottom, n_bins - 1)
    assert params.flow_w.shape == (B, H, model.n_bottom, n_flow)
    assert params.flow_a.shape == (B, H, model.n_bottom, n_flow)
    assert params.flow_b.shape == (B, H, model.n_bottom, n_flow)
    assert params.F.shape == (B, H, model.n_bottom, model.config.n_factors)
    assert (params.sigma > 0).all()
    assert (params.spline_d > model.head.slope_floor).all()
    assert (params.flow_a > model.head.a_floor).all()
    torch.testing.assert_close(params.spline_w.sum(-1), torch.ones(B, H, model.n_bottom))
    torch.testing.assert_close(params.flow_w.sum(-1), torch.ones(B, H, model.n_bottom))


def test_copula_flow_head_rejects_fewer_than_one_bin() -> None:
    with pytest.raises(ValueError, match="n_spline_bins must be >= 1"):
        from model.heads import CopulaFlowHead

        CopulaFlowHead(in_dim=4, n_factors=2, n_spline_bins=0)


def test_copula_flow_head_rejects_fewer_than_one_flow_component() -> None:
    with pytest.raises(ValueError, match="n_flow_components must be >= 1"):
        from model.heads import CopulaFlowHead

        CopulaFlowHead(in_dim=4, n_factors=2, n_flow_components=0)


def test_copula_flow_head_gradients_reach_every_parameter(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, head="copula_flow"), dims, S
    )
    params = model(_batch(model.n_bottom))
    total = (
        params.mu.sum()
        + params.sigma.sum()
        + params.spline_w.sum()
        + params.spline_h.sum()
        + params.spline_d.sum()
        + params.flow_w.sum()
        + params.flow_a.sum()
        + params.flow_b.sum()
    )
    (total + params.F.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


def test_ddsf_head_forward_shapes(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(ModelConfig(temp_conv_channels=6, n_factors=3, head="ddsf"), dims, S)
    params = model(_batch(model.n_bottom))
    k_flow, n_layers = model.head.n_flow_components, model.head.n_flow_layers
    assert params.mu.shape == (B, H, model.n_bottom)
    assert params.sigma.shape == (B, H, model.n_bottom)
    assert params.ddsf_w.shape == (B, H, model.n_bottom, n_layers, k_flow)
    assert params.ddsf_a.shape == (B, H, model.n_bottom, n_layers, k_flow)
    assert params.ddsf_b.shape == (B, H, model.n_bottom, n_layers, k_flow)
    assert params.ddsf_skip.shape == (B, H, model.n_bottom, n_layers - 1, 2)
    assert params.F.shape == (B, H, model.n_bottom, model.config.n_factors)
    assert (params.sigma > 0).all()
    assert (params.ddsf_a > model.head.a_floor).all()
    torch.testing.assert_close(
        params.ddsf_w.sum(-1), torch.ones(B, H, model.n_bottom, n_layers)
    )
    torch.testing.assert_close(
        params.ddsf_skip.sum(-1), torch.ones(B, H, model.n_bottom, n_layers - 1)
    )


def test_ddsf_head_rejects_fewer_than_one_component() -> None:
    with pytest.raises(ValueError, match="n_flow_components must be >= 1"):
        from model.heads import DDSFHead

        DDSFHead(in_dim=4, n_factors=2, n_flow_components=0)


def test_ddsf_head_rejects_fewer_than_one_layer() -> None:
    with pytest.raises(ValueError, match="n_flow_layers must be >= 1"):
        from model.heads import DDSFHead

        DDSFHead(in_dim=4, n_factors=2, n_flow_layers=0)


def test_ddsf_head_gradients_reach_every_parameter(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(ModelConfig(temp_conv_channels=6, n_factors=3, head="ddsf"), dims, S)
    params = model(_batch(model.n_bottom))
    total = (
        params.mu.sum()
        + params.sigma.sum()
        + params.ddsf_w.sum()
        + params.ddsf_a.sum()
        + params.ddsf_b.sum()
        + params.ddsf_skip.sum()
    )
    (total + params.F.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


def test_gmm_head_forward_shapes(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(ModelConfig(temp_conv_channels=6, n_factors=3, head="gmm"), dims, S)
    params = model(_batch(model.n_bottom))
    k = model.head.n_components
    assert k == 2
    assert params.mu.shape == (B, H, model.n_bottom, k)
    assert params.sigma.shape == (B, H, model.n_bottom, k)
    assert params.logits.shape == (B, H, model.n_bottom, k)
    assert params.F.shape == (B, H, model.n_bottom, model.config.n_factors)
    assert (params.sigma > 0).all()


def test_gmm_head_rejects_fewer_than_two_components() -> None:
    with pytest.raises(ValueError, match="n_components must be >= 2"):
        from model.heads import GMMHead

        GMMHead(in_dim=4, n_factors=2, n_components=1)


def test_gmm_head_gradients_reach_every_parameter(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(ModelConfig(temp_conv_channels=6, n_factors=3, head="gmm"), dims, S)
    params = model(_batch(model.n_bottom))
    total = params.mu.sum() + params.sigma.sum() + params.logits.sum()
    (total + params.F.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"
