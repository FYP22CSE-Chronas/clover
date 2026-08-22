from __future__ import annotations

import pytest
import torch

from training.losses import LOSSES, crps, energy_score, gaussian_crps


def test_crps_matches_closed_form_gaussian() -> None:
    torch.manual_seed(0)
    mu, sigma = torch.tensor([[[1.5]]]), torch.tensor([[[0.7]]])
    y = torch.tensor([[[2.0]]])
    samples = mu.unsqueeze(-1) + sigma.unsqueeze(-1) * torch.randn(1, 1, 1, 400_000)
    estimate = crps(y, samples, reduction="none")
    torch.testing.assert_close(estimate, gaussian_crps(mu, sigma, y), atol=5e-3, rtol=0)


def test_crps_is_zero_for_a_perfect_point_forecast() -> None:
    y = torch.tensor([[[3.0]]])
    samples = y.unsqueeze(-1).expand(1, 1, 1, 64)
    assert crps(y, samples, reduction="none").item() == pytest.approx(0.0, abs=1e-6)


def test_crps_rewards_the_closer_forecast() -> None:
    y = torch.zeros(1, 1, 1)
    near = torch.randn(1, 1, 1, 5000) * 0.5
    far = torch.randn(1, 1, 1, 5000) * 0.5 + 4.0
    assert crps(y, near) < crps(y, far)


@pytest.mark.parametrize(
    "reduction,shape", [("none", (2, 3, 4)), ("sum", ()), ("mean", ())]
)
def test_crps_reductions(reduction: str, shape: tuple) -> None:
    y = torch.randn(2, 3, 4)
    samples = torch.randn(2, 3, 4, 16)
    assert crps(y, samples, reduction=reduction).shape == shape


def test_energy_score_reduces_over_the_joint_block() -> None:
    y = torch.randn(2, 3, 4)
    samples = torch.randn(2, 3, 4, 16)
    assert energy_score(y, samples, reduction="none").shape == (2,)


def test_rejects_bad_sample_axis() -> None:
    with pytest.raises(ValueError, match="trailing sample axis"):
        crps(torch.randn(2, 3, 4), torch.randn(2, 3, 5, 16))


def test_rejects_single_sample() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        crps(torch.randn(1, 1, 1), torch.randn(1, 1, 1, 1))


def test_registry_exposes_both_objectives() -> None:
    assert LOSSES.names() == ["crps", "energy"]
