from __future__ import annotations

import numpy as np
import pytest
import torch

from config import ModelConfig, ModelDims
from contracts import WindowBatch
from experiments.ec_clover.config import ECConfig
from experiments.ec_clover.ec_branch import ErrorCorrectionBranch
from experiments.ec_clover.fusion import FUSIONS
from experiments.ec_clover.network import CloverWithErrorCorrection

B, L, H, N_BOTTOM = 2, 8, 4, 4


def _ec_config(**kwargs) -> ECConfig:
    return ECConfig(**{"enabled": True, "rank": 2, "encoder_channels": 5, **kwargs})


def _branch(**kwargs) -> ErrorCorrectionBranch:
    return ErrorCorrectionBranch(
        n_bottom=N_BOTTOM,
        input_size=L,
        ec_config=_ec_config(**kwargs),
        model_config=ModelConfig(temp_conv_channels=6),
    )


def _model(S: np.ndarray, **kwargs) -> CloverWithErrorCorrection:
    return CloverWithErrorCorrection(
        ModelConfig(temp_conv_channels=6, n_factors=3),
        ModelDims(h=H, input_size=L),
        S,
        _ec_config(**kwargs),
    )


def _batch(n_bottom: int) -> WindowBatch:
    return WindowBatch(insample_y=torch.randn(B, L, n_bottom))


@pytest.mark.parametrize("ec_input", ["ec", "e"])
@pytest.mark.parametrize("include_diff", [False, True])
@pytest.mark.parametrize("sequence_norm", ["none", "center", "standardize"])
def test_branch_variants_keep_output_shape(
    ec_input: str, include_diff: bool, sequence_norm: str
) -> None:
    branch = _branch(
        ec_input=ec_input, include_diff=include_diff, sequence_norm=sequence_norm
    )
    out = branch(torch.randn(B, L, N_BOTTOM))
    assert out.shape == (B, N_BOTTOM, branch.out_channels)


@pytest.mark.parametrize("ec_input", ["ec", "e"])
@pytest.mark.parametrize("include_diff", [False, True])
def test_branch_variants_reach_alpha_and_beta(ec_input: str, include_diff: bool) -> None:
    branch = _branch(ec_input=ec_input, include_diff=include_diff)
    branch(torch.randn(B, L, N_BOTTOM)).sum().backward()
    assert branch.alpha.grad is not None and (branch.alpha.grad != 0).any()
    assert branch.beta.grad is not None and (branch.beta.grad != 0).any()


def test_centering_removes_window_offset() -> None:
    """A constant shift of the equilibrium level must not change a centered embedding."""
    branch = _branch(sequence_norm="center")
    y = torch.randn(B, L, N_BOTTOM)
    shifted = y + 5.0  # a pure level shift of every series
    torch.testing.assert_close(branch(y), branch(shifted), atol=1e-5, rtol=1e-4)


def test_uncentered_branch_is_offset_sensitive() -> None:
    branch = _branch(sequence_norm="none")
    y = torch.randn(B, L, N_BOTTOM)
    assert not torch.allclose(branch(y), branch(y + 5.0), atol=1e-5)


def test_beta_normalize_columns_pins_scale() -> None:
    branch = _branch(beta_normalize="columns")
    with torch.no_grad():
        branch.beta.mul_(17.0)  # a rescaled beta is the same cointegrating space
    torch.testing.assert_close(
        branch._beta().norm(dim=0), torch.ones(2), atol=1e-5, rtol=1e-5
    )


def test_diff_channel_distinguishes_reversed_dynamics() -> None:
    """The short-run (delta) channel must separate histories a level-only view ties."""
    branch = _branch(include_diff=True, sequence_norm="center", rank=1)
    with torch.no_grad():
        branch.beta.copy_(torch.ones(N_BOTTOM, 1))
        branch.alpha.copy_(torch.ones(N_BOTTOM, 1))
    rising = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    falling = rising.flip(0)
    y = torch.stack(
        [
            rising.unsqueeze(-1).expand(L, N_BOTTOM),
            falling.unsqueeze(-1).expand(L, N_BOTTOM),
        ]
    )
    out = branch(y)
    assert not torch.allclose(out[0], out[1])


def test_encoder_dilations_reject_overlong_receptive_field() -> None:
    with pytest.raises(ValueError, match="receptive field .* exceeds input_size"):
        _branch(encoder_dilations=(1, 2, 4, 8, 16, 32))


def test_gated_residual_starts_near_baseline(S: np.ndarray) -> None:
    """A very negative gate closes the EC path, recovering the baseline representation."""
    fusion = FUSIONS.create(
        "gated_residual",
        clover_dim=6,
        ec_dim=5,
        out_dim=6,
        hidden_size=None,
        n_layers=1,
        gate_init=-30.0,
    )
    h_clover = torch.randn(B, N_BOTTOM, 6)
    h_ec = torch.randn(B, N_BOTTOM, 5)
    torch.testing.assert_close(fusion(h_clover, h_ec), h_clover, atol=1e-6, rtol=1e-5)


def test_gated_residual_rejects_mismatched_out_dim() -> None:
    with pytest.raises(ValueError, match="out_dim == clover_dim"):
        FUSIONS.create(
            "gated_residual", clover_dim=6, ec_dim=5, out_dim=8, hidden_size=None
        )


@pytest.mark.parametrize("fusion", ["concat", "gated_residual"])
def test_network_fusion_variants_forward_and_backward(S: np.ndarray, fusion: str) -> None:
    model = _model(S, fusion=fusion, fusion_layers=2)
    params = model(_batch(model.n_bottom))
    assert params.mu.shape == (B, H, model.n_bottom)
    (params.mu.sum() + params.sigma.sum() + params.F.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


def test_enhanced_stack_forward_and_backward(S: np.ndarray) -> None:
    """Every enhancement switched on at once still trains end to end."""
    model = _model(
        S,
        fusion="gated_residual",
        fusion_layers=2,
        ec_input="e",
        include_diff=True,
        sequence_norm="center",
        beta_normalize="columns",
    )
    params = model(_batch(model.n_bottom))
    (params.mu.sum() + params.sigma.sum() + params.F.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


def test_config_rejects_bad_enhancement_values() -> None:
    for kwargs, match in [
        ({"fusion": "attention"}, "fusion must be"),
        ({"ec_input": "y"}, "ec_input must be"),
        ({"sequence_norm": "zscore"}, "sequence_norm must be"),
        ({"beta_normalize": "rows"}, "beta_normalize must be"),
        ({"fusion_layers": 0}, "fusion_layers must be"),
        ({"encoder_dilations": ()}, "must be non-empty"),
    ]:
        with pytest.raises(ValueError, match=match):
            ECConfig(**kwargs)
