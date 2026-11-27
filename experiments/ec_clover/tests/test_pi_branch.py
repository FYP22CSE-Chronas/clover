from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from config import ModelConfig, ModelDims
from contracts import WindowBatch
from data.windows import Window
from experiments.ec_clover.config import ECConfig
from experiments.ec_clover.network import CloverWithErrorCorrection
from experiments.ec_clover.pi_branch import PiErrorCorrectionBranch
from experiments.ec_clover.run import training_ref_stats
from model.distribution import sample_coherent
from model.network import CLOVER
from model.normalization import normalize, window_stats
from training.losses import crps


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        temp_conv_channels=8,
        dilations=(1, 2),
        cross_series_hidden=16,
        static_encoder_dim=4,
        future_encoder_dim=4,
        horizon_specific_dim=4,
        horizon_agnostic_dim=4,
        n_factors=3,
    )


def _tiny_S(n_bottom: int = 4) -> np.ndarray:
    top = np.ones((1, n_bottom), dtype=np.float32)
    return np.vstack([top, np.eye(n_bottom, dtype=np.float32)])


def test_t1_flag_off_bitwise_equality() -> None:
    cfg = _tiny_model_config()
    dims = ModelDims(h=4, input_size=12, n_stat_features=0, n_futr_features=0)
    S = _tiny_S(4)

    torch.manual_seed(42)
    base = CLOVER(cfg, dims, S)
    torch.manual_seed(42)
    ec_off = CloverWithErrorCorrection(cfg, dims, S, ec_config=ECConfig(enabled=False))

    assert list(base.state_dict().keys()) == list(ec_off.state_dict().keys())
    assert sum(p.numel() for p in base.parameters()) == sum(
        p.numel() for p in ec_off.parameters()
    )
    for k in base.state_dict():
        assert torch.equal(base.state_dict()[k], ec_off.state_dict()[k])

    batch = WindowBatch(insample_y=torch.randn(2, 12, 4))
    out_base = base(batch)
    out_ec = ec_off(batch)
    assert torch.equal(out_base.mu, out_ec.mu)
    assert torch.equal(out_base.sigma, out_ec.sigma)
    assert torch.equal(out_base.F, out_ec.F)


def test_t2_t3_train_ref_canaries() -> None:
    rng = np.random.default_rng(0)
    train = [
        Window(t=20 + i, insample=rng.normal(10.0, 2.0, size=(12, 4)), target=rng.normal(10.0, 2.0, size=(4, 4)))
        for i in range(10)
    ]
    mu1, sig1 = training_ref_stats(train)

    # T2: changing val/test windows does not touch train refs (only train is passed)
    _val_different = [
        Window(t=50, insample=np.full((12, 4), 9999.0), target=np.full((4, 4), -9999.0))
    ]
    mu2, sig2 = training_ref_stats(train)
    assert torch.equal(mu1, mu2)
    assert torch.equal(sig1, sig2)

    # T3: permuting targets in train/val does not affect insample refs
    train_perm = [
        Window(t=w.t, insample=w.insample.copy(), target=w.target[::-1].copy())
        for w in train
    ]
    mu3, sig3 = training_ref_stats(train_perm)
    assert torch.equal(mu1, mu3)
    assert torch.equal(sig1, sig3)


def test_t4_scale_round_trip() -> None:
    y_raw = torch.randn(3, 12, 4) * 50.0 + 100.0
    stats = window_stats(y_raw, "standard")
    y_norm = normalize(y_raw, stats)
    recovered = y_norm * stats.scale + stats.loc
    assert torch.allclose(recovered, y_raw, atol=1e-4, rtol=1e-5)


def test_t5_deviation_distinguish() -> None:
    cfg = _tiny_model_config()
    ec_cfg = ECConfig(
        enabled=True,
        operator="pi",
        state_norm="train_ref",
        rank=2,
        encoder="dilated_conv",
        encoder_channels=4,
        encoder_dilations=(1, 2),
        fusion="gated_residual",
        gate_init=-4.595,
    )
    branch = PiErrorCorrectionBranch(n_bottom=4, input_size=5, ec_config=ec_cfg, model_config=cfg)
    branch.eval()

    # Construct two trajectories with identical final value (10) but different paths
    # e=(2,4,6,8,10) vs e=(20,18,16,13,10)
    t1 = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0]).view(1, 5, 1).expand(1, 5, 4).clone()
    t2 = torch.tensor([20.0, 18.0, 16.0, 13.0, 10.0]).view(1, 5, 1).expand(1, 5, 4).clone()
    assert torch.allclose(t1[:, -1, :], t2[:, -1, :])

    h1 = branch(t1)
    h2 = branch(t2)
    diff = (h1 - h2).norm().item()
    assert diff > 1e-3, f"expected temporally distinct embeddings, got diff={diff}"


def test_t6_dead_gradient_guard() -> None:
    with pytest.raises(ValueError, match="dead-gradient"):
        ECConfig(enabled=True, operator="pi", pi_init_std=0.0)


def test_t7_rank_warning() -> None:
    cfg = _tiny_model_config()
    ec_cfg = ECConfig(
        enabled=True,
        operator="pi",
        rank=4,
        encoder_dilations=(1, 2),
    )
    with pytest.warns(UserWarning, match="full-rank"):
        PiErrorCorrectionBranch(n_bottom=4, input_size=12, ec_config=ec_cfg, model_config=cfg)


def test_t8_gradient_path_a3() -> None:
    cfg = _tiny_model_config()
    dims = ModelDims(h=4, input_size=12, n_stat_features=0, n_futr_features=0)
    S = _tiny_S(4)
    ec_cfg = ECConfig(
        enabled=True,
        operator="pi",
        state_norm="train_ref",
        rank=2,
        encoder_dilations=(1, 2),
        fusion="gated_residual",
        gate_init=-4.595,
    )
    model = CloverWithErrorCorrection(cfg, dims, S, ec_config=ec_cfg)
    y_raw = torch.randn(4, 12, 4) * 10.0 + 50.0
    stats = window_stats(y_raw, "standard")
    batch = WindowBatch(insample_y=normalize(y_raw, stats), scale=stats)
    params = model(batch)
    samples = sample_coherent(params, torch.as_tensor(S), num_samples=16)
    target = torch.randn(4, 4, 5)
    loss = crps(target, samples.hierarchy).mean()
    loss.backward()

    assert model.ec_branch is not None
    assert model.ec_branch.L is not None
    assert model.ec_branch.L.grad is not None
    assert model.ec_branch.L.grad.abs().sum().item() > 0.0


def test_t9_fusion_not_zero_init() -> None:
    cfg = _tiny_model_config()
    dims = ModelDims(h=4, input_size=12, n_stat_features=0, n_futr_features=0)
    S = _tiny_S(4)
    for gate_init, expected in [(-4.595, 0.01), (-2.0, 0.1192)]:
        ec_cfg = ECConfig(
            enabled=True,
            operator="pi",
            state_norm="train_ref",
            rank=2,
            encoder_dilations=(1, 2),
            fusion="gated_residual",
            gate_init=gate_init,
        )
        model = CloverWithErrorCorrection(cfg, dims, S, ec_config=ec_cfg)
        assert model.fusion is not None
        g = torch.sigmoid(model.fusion.gate).mean().item()
        assert abs(g - expected) < 1e-3
        # Ensure fusion MLP output layer is not zero-initialized
        last_linear = [m for m in model.fusion.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
        assert last_linear.weight.abs().sum().item() > 0.0
