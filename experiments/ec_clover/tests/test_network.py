from __future__ import annotations

import numpy as np
import pytest
import torch

from config import ModelConfig, ModelDims
from contracts import WindowBatch
from experiments.ec_clover.config import ECConfig
from experiments.ec_clover.network import CloverWithErrorCorrection
from model.network import CLOVER

B, L, H = 2, 8, 4


def _batch(n_bottom: int) -> WindowBatch:
    return WindowBatch(insample_y=torch.randn(B, L, n_bottom))


def test_disabled_ec_builds_no_extra_modules(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CloverWithErrorCorrection(
        ModelConfig(temp_conv_channels=6, n_factors=3), dims, S, ECConfig(enabled=False)
    )
    assert model.ec_branch is None
    assert model.fusion is None


def test_disabled_ec_matches_baseline_clover_exactly(S: np.ndarray) -> None:
    """Same seed, same architecture, disabled EC -> byte-for-byte the baseline."""
    dims = ModelDims(h=H, input_size=L)
    batch = _batch(S.shape[1])

    torch.manual_seed(0)
    baseline = CLOVER(ModelConfig(temp_conv_channels=6, n_factors=3), dims, S)
    torch.manual_seed(0)
    ec_model = CloverWithErrorCorrection(
        ModelConfig(temp_conv_channels=6, n_factors=3), dims, S, ECConfig(enabled=False)
    )

    p1, p2 = baseline(batch), ec_model(batch)
    torch.testing.assert_close(p1.mu, p2.mu)
    torch.testing.assert_close(p1.sigma, p2.sigma)
    torch.testing.assert_close(p1.F, p2.F)


def test_enabled_ec_forward_shapes(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CloverWithErrorCorrection(
        ModelConfig(temp_conv_channels=6, n_factors=3),
        dims,
        S,
        ECConfig(enabled=True, rank=2, encoder_channels=5),
    )
    params = model(_batch(model.n_bottom))
    assert params.mu.shape == (B, H, model.n_bottom)
    assert params.sigma.shape == (B, H, model.n_bottom)
    assert params.F.shape == (B, H, model.n_bottom, model.config.n_factors)
    assert (params.sigma > 0).all()


def test_enabled_ec_gradients_reach_every_parameter(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    model = CloverWithErrorCorrection(
        ModelConfig(temp_conv_channels=6, n_factors=3),
        dims,
        S,
        ECConfig(enabled=True, rank=2, encoder_channels=5),
    )
    params = model(_batch(model.n_bottom))
    (params.mu.sum() + params.sigma.sum() + params.F.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


def test_enabled_ec_changes_predictions_vs_disabled(S: np.ndarray) -> None:
    """Sanity check that the EC branch is actually wired into the forward pass."""
    dims = ModelDims(h=H, input_size=L)
    batch = _batch(S.shape[1])

    torch.manual_seed(1)
    disabled = CloverWithErrorCorrection(
        ModelConfig(temp_conv_channels=6, n_factors=3), dims, S, ECConfig(enabled=False)
    )
    torch.manual_seed(1)
    enabled = CloverWithErrorCorrection(
        ModelConfig(temp_conv_channels=6, n_factors=3),
        dims,
        S,
        ECConfig(enabled=True, rank=2, encoder_channels=5),
    )
    assert not torch.allclose(disabled(batch).mu, enabled(batch).mu)


def test_rank_exceeding_n_bottom_raises(S: np.ndarray) -> None:
    dims = ModelDims(h=H, input_size=L)
    n_bottom = S.shape[1]
    with pytest.raises(ValueError, match="must be <= n_bottom"):
        CloverWithErrorCorrection(
            ModelConfig(temp_conv_channels=6),
            dims,
            S,
            ECConfig(enabled=True, rank=n_bottom + 1),
        )
