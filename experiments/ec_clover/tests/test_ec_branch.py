from __future__ import annotations

import numpy as np
import pytest
import torch

from config import ModelConfig
from experiments.ec_clover.config import BetaInitConfig, ECConfig
from experiments.ec_clover.ec_branch import ErrorCorrectionBranch

B, L, N_BOTTOM = 3, 8, 4


def _branch(rank: int = 2, **kwargs) -> ErrorCorrectionBranch:
    ec_config = ECConfig(rank=rank, encoder_channels=5, **kwargs)
    return ErrorCorrectionBranch(
        n_bottom=N_BOTTOM,
        input_size=L,
        ec_config=ec_config,
        model_config=ModelConfig(temp_conv_channels=6),
    )


def test_forward_shape() -> None:
    branch = _branch()
    out = branch(torch.randn(B, L, N_BOTTOM))
    assert out.shape == (B, N_BOTTOM, branch.out_channels)


def test_beta_and_alpha_shapes() -> None:
    branch = _branch(rank=3)
    assert branch.beta.shape == (N_BOTTOM, 3)
    assert branch.alpha.shape == (N_BOTTOM, 3)


def test_rejects_rank_greater_than_n_bottom() -> None:
    with pytest.raises(ValueError, match="ec.rank .* must be <= n_bottom"):
        _branch(rank=N_BOTTOM + 1)


def test_gradients_reach_alpha_and_beta() -> None:
    branch = _branch()
    out = branch(torch.randn(B, L, N_BOTTOM))
    out.sum().backward()
    assert branch.alpha.grad is not None and (branch.alpha.grad != 0).any()
    assert branch.beta.grad is not None and (branch.beta.grad != 0).any()


def test_random_beta_columns_start_unit_norm() -> None:
    branch = _branch(rank=3)
    norms = branch.beta.norm(dim=0)
    torch.testing.assert_close(norms, torch.ones(3), atol=1e-5, rtol=1e-5)


def test_alpha_init_is_small() -> None:
    branch = _branch(alpha_init_std=1e-3)
    assert branch.alpha.abs().max().item() < 0.05


def test_johansen_beta_init_used_exactly() -> None:
    beta = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5], [2.0, 3.0]], dtype=torch.float32
    )
    ec_config = ECConfig(
        rank=2, beta_init=BetaInitConfig(type="johansen", values=beta.tolist())
    )
    branch = ErrorCorrectionBranch(
        n_bottom=N_BOTTOM,
        input_size=L,
        ec_config=ec_config,
        model_config=ModelConfig(),
    )
    torch.testing.assert_close(branch.beta.detach(), beta)


def test_johansen_beta_init_rejects_wrong_shape() -> None:
    ec_config = ECConfig(
        rank=2, beta_init=BetaInitConfig(type="johansen", values=((1.0, 2.0),))
    )
    with pytest.raises(ValueError, match="expected"):
        ErrorCorrectionBranch(
            n_bottom=N_BOTTOM,
            input_size=L,
            ec_config=ec_config,
            model_config=ModelConfig(),
        )


def test_johansen_beta_init_from_values_path(tmp_path) -> None:
    beta = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5], [2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "beta.npy"
    np.save(path, beta)
    ec_config = ECConfig(
        rank=2, beta_init=BetaInitConfig(type="johansen", values_path=str(path))
    )
    branch = ErrorCorrectionBranch(
        n_bottom=N_BOTTOM,
        input_size=L,
        ec_config=ec_config,
        model_config=ModelConfig(),
    )
    torch.testing.assert_close(branch.beta.detach(), torch.as_tensor(beta))


def test_distinguishes_histories_with_same_endpoint() -> None:
    """Two series ending at the same value but with different dynamics must not
    collapse to the same EC embedding -- the branch must see the whole window."""
    branch = _branch(rank=1)
    torch.manual_seed(0)
    branch.beta.data = torch.ones(N_BOTTOM, 1)
    branch.alpha.data = torch.ones(N_BOTTOM, 1)

    rising = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 10.0, 10.0, 10.0])
    falling = torch.tensor([20.0, 18.0, 16.0, 13.0, 10.0, 10.0, 10.0, 10.0])
    y = torch.zeros(2, L, N_BOTTOM)
    y[0] = rising.unsqueeze(-1).expand(L, N_BOTTOM)
    y[1] = falling.unsqueeze(-1).expand(L, N_BOTTOM)

    out = branch(y)
    assert not torch.allclose(out[0], out[1])
