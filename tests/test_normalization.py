from __future__ import annotations

import pytest
import torch

from contracts import FactorParams
from model.normalization import (
    SCALERS,
    denormalize,
    denormalize_params,
    normalize,
    window_stats,
)

B, L, N = 2, 12, 3


@pytest.mark.parametrize("scaler", SCALERS.names())
def test_normalize_roundtrip(scaler: str) -> None:
    x = torch.rand(B, L, N) * 50 + 1
    stats = window_stats(x, scaler)
    torch.testing.assert_close(
        denormalize(normalize(x, stats), stats), x, atol=1e-4, rtol=0
    )


def test_standard_scaler_centres_and_scales() -> None:
    x = torch.randn(B, L, N) * 7 + 3
    z = normalize(x, window_stats(x, "standard"))
    torch.testing.assert_close(z.mean(1), torch.zeros(B, N), atol=1e-5, rtol=0)
    torch.testing.assert_close(z.std(1), torch.ones(B, N), atol=1e-3, rtol=0)


def test_identity_scaler_is_a_noop() -> None:
    x = torch.randn(B, L, N)
    torch.testing.assert_close(normalize(x, window_stats(x, "identity")), x)


def test_mean_scalers_preserve_sign() -> None:
    x = torch.rand(B, L, N) * 10
    for scaler in ("mean", "mean_floor"):
        stats = window_stats(x, scaler)
        assert (stats.loc == 0).all()
        assert (normalize(x, stats) >= 0).all()


def test_mean_floor_survives_an_all_zero_window() -> None:
    """`mean` would divide by eps and explode; `mean_floor` bottoms out at 1."""
    zeros = torch.zeros(1, L, 1)
    assert window_stats(zeros, "mean_floor").scale.item() == pytest.approx(1.0)
    assert window_stats(zeros, "mean").scale.item() < 1e-5


def test_denormalize_params_is_exact() -> None:
    """The factor model is affine, so sampling then denormalizing must agree."""
    torch.manual_seed(0)
    stats = window_stats(torch.rand(1, L, N) * 20 + 5, "standard")
    params = FactorParams(
        mu=torch.randn(1, 4, N),
        sigma=torch.rand(1, 4, N) + 0.1,
        F=torch.randn(1, 4, N, 2),
    )
    raw = denormalize_params(params, stats)

    z = torch.randn(1, 4, N, 500)
    eps = torch.randn(1, 4, 2, 500)

    def draw(p: FactorParams) -> torch.Tensor:
        factor = torch.einsum("...bk,...kn->...bn", p.F, eps)
        return p.mu.unsqueeze(-1) + p.sigma.unsqueeze(-1) * z + factor

    scale = stats.scale.permute(0, 1, 2).unsqueeze(-1)
    loc = stats.loc.unsqueeze(-1)
    torch.testing.assert_close(
        draw(raw), draw(params) * scale + loc, atol=1e-4, rtol=1e-4
    )


def test_unknown_scaler_lists_alternatives() -> None:
    with pytest.raises(KeyError, match="available"):
        window_stats(torch.randn(B, L, N), "nope")
