from __future__ import annotations

import numpy as np
import pytest
import torch

from config import ModelConfig, ModelDims
from contracts import WindowBatch
from model.mixers import MIXERS
from model.network import CLOVER

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


def test_gradients_reach_every_parameter(model: CLOVER) -> None:
    params = model(_batch(model.n_bottom))
    (params.mu.sum() + params.sigma.sum() + params.F.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


def test_skew_t_head_forward_shapes(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, head="skew_t"), dims, S
    )
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
    model = CLOVER(
        ModelConfig(temp_conv_channels=6, n_factors=3, head="skew_t"), dims, S
    )
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
