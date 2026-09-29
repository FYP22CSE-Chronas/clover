from __future__ import annotations

import torch

from experiments.ec_clover.fusion import FusionMLP

B, N_BOTTOM = 3, 4


def test_output_matches_clover_dim() -> None:
    fusion = FusionMLP(clover_dim=6, ec_dim=5, out_dim=6)
    h_clover = torch.randn(B, N_BOTTOM, 6)
    h_ec = torch.randn(B, N_BOTTOM, 5)
    out = fusion(h_clover, h_ec)
    assert out.shape == (B, N_BOTTOM, 6)


def test_gradients_reach_both_inputs() -> None:
    fusion = FusionMLP(clover_dim=6, ec_dim=5, out_dim=6)
    h_clover = torch.randn(B, N_BOTTOM, 6, requires_grad=True)
    h_ec = torch.randn(B, N_BOTTOM, 5, requires_grad=True)
    fusion(h_clover, h_ec).sum().backward()
    assert h_clover.grad is not None and (h_clover.grad != 0).any()
    assert h_ec.grad is not None and (h_ec.grad != 0).any()
