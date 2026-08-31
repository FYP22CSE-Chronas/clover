from __future__ import annotations

import pytest
import torch

from contracts import (
    CopulaFlowParams,
    CopulaSplineParams,
    FactorParams,
    FlowParams,
    GMMParams,
    SkewTParams,
)
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


def test_denormalize_skew_t_params_scales_shape_free_terms() -> None:
    """mu/sigma/F scale affinely; nu/lam are scale-free and must pass through."""
    torch.manual_seed(0)
    stats = window_stats(torch.rand(1, L, N) * 20 + 5, "standard")
    params = SkewTParams(
        mu=torch.randn(1, 4, N),
        sigma=torch.rand(1, 4, N) + 0.1,
        nu=torch.rand(1, 4, N) * 20 + 5,
        lam=torch.randn(1, 4, N),
        F=torch.randn(1, 4, N, 2),
    )
    raw = denormalize_params(params, stats)
    torch.testing.assert_close(raw.mu, params.mu * stats.scale + stats.loc)
    torch.testing.assert_close(raw.sigma, params.sigma * stats.scale)
    torch.testing.assert_close(raw.nu, params.nu)
    torch.testing.assert_close(raw.lam, params.lam)
    torch.testing.assert_close(raw.F, params.F * stats.scale.unsqueeze(-1))


def test_denormalize_flow_params_scales_shape_free_terms() -> None:
    """mu/sigma/F scale affinely; the flow parameters are scale-free and pass through."""
    torch.manual_seed(0)
    stats = window_stats(torch.rand(1, L, N) * 20 + 5, "standard")
    params = FlowParams(
        mu=torch.randn(1, 4, N),
        sigma=torch.rand(1, 4, N) + 0.1,
        flow_w=torch.randn(1, 4, N, 3),
        flow_a=torch.randn(1, 4, N, 3),
        flow_b=torch.randn(1, 4, N, 3),
        F=torch.randn(1, 4, N, 2),
    )
    raw = denormalize_params(params, stats)
    torch.testing.assert_close(raw.mu, params.mu * stats.scale + stats.loc)
    torch.testing.assert_close(raw.sigma, params.sigma * stats.scale)
    torch.testing.assert_close(raw.flow_w, params.flow_w)
    torch.testing.assert_close(raw.flow_a, params.flow_a)
    torch.testing.assert_close(raw.flow_b, params.flow_b)
    torch.testing.assert_close(raw.F, params.F * stats.scale.unsqueeze(-1))


def test_denormalize_copula_spline_params_scales_shape_free_terms() -> None:
    """mu/sigma/F scale affinely; the spline parameters are scale-free and pass
    through."""
    torch.manual_seed(0)
    stats = window_stats(torch.rand(1, L, N) * 20 + 5, "standard")
    params = CopulaSplineParams(
        mu=torch.randn(1, 4, N),
        sigma=torch.rand(1, 4, N) + 0.1,
        spline_w=torch.softmax(torch.randn(1, 4, N, 5), dim=-1),
        spline_h=torch.softmax(torch.randn(1, 4, N, 5), dim=-1),
        spline_d=torch.rand(1, 4, N, 4) + 0.1,
        F=torch.randn(1, 4, N, 2),
    )
    raw = denormalize_params(params, stats)
    torch.testing.assert_close(raw.mu, params.mu * stats.scale + stats.loc)
    torch.testing.assert_close(raw.sigma, params.sigma * stats.scale)
    torch.testing.assert_close(raw.spline_w, params.spline_w)
    torch.testing.assert_close(raw.spline_h, params.spline_h)
    torch.testing.assert_close(raw.spline_d, params.spline_d)
    torch.testing.assert_close(raw.F, params.F * stats.scale.unsqueeze(-1))


def test_denormalize_copula_flow_params_scales_shape_free_terms() -> None:
    """mu/sigma/F scale affinely; the spline and flow parameters are scale-free and
    pass through."""
    torch.manual_seed(0)
    stats = window_stats(torch.rand(1, L, N) * 20 + 5, "standard")
    params = CopulaFlowParams(
        mu=torch.randn(1, 4, N),
        sigma=torch.rand(1, 4, N) + 0.1,
        spline_w=torch.softmax(torch.randn(1, 4, N, 5), dim=-1),
        spline_h=torch.softmax(torch.randn(1, 4, N, 5), dim=-1),
        spline_d=torch.rand(1, 4, N, 4) + 0.1,
        flow_w=torch.softmax(torch.randn(1, 4, N, 3), dim=-1),
        flow_a=torch.rand(1, 4, N, 3) + 0.1,
        flow_b=torch.randn(1, 4, N, 3),
        F=torch.randn(1, 4, N, 2),
    )
    raw = denormalize_params(params, stats)
    torch.testing.assert_close(raw.mu, params.mu * stats.scale + stats.loc)
    torch.testing.assert_close(raw.sigma, params.sigma * stats.scale)
    torch.testing.assert_close(raw.spline_w, params.spline_w)
    torch.testing.assert_close(raw.spline_h, params.spline_h)
    torch.testing.assert_close(raw.spline_d, params.spline_d)
    torch.testing.assert_close(raw.flow_w, params.flow_w)
    torch.testing.assert_close(raw.flow_a, params.flow_a)
    torch.testing.assert_close(raw.flow_b, params.flow_b)
    torch.testing.assert_close(raw.F, params.F * stats.scale.unsqueeze(-1))


def test_denormalize_gmm_params_scales_shape_free_terms() -> None:
    """mu/sigma/F scale affinely; logits are scale-free and must pass through."""
    torch.manual_seed(0)
    stats = window_stats(torch.rand(1, L, N) * 20 + 5, "standard")
    params = GMMParams(
        mu=torch.randn(1, 4, N, 2),
        sigma=torch.rand(1, 4, N, 2) + 0.1,
        logits=torch.randn(1, 4, N, 2),
        F=torch.randn(1, 4, N, 2),
    )
    raw = denormalize_params(params, stats)
    scale = stats.scale.unsqueeze(-1)
    loc = stats.loc.unsqueeze(-1)
    torch.testing.assert_close(raw.mu, params.mu * scale + loc)
    torch.testing.assert_close(raw.sigma, params.sigma * scale)
    torch.testing.assert_close(raw.logits, params.logits)
    torch.testing.assert_close(raw.F, params.F * stats.scale.unsqueeze(-1))


def test_unknown_scaler_lists_alternatives() -> None:
    with pytest.raises(KeyError, match="available"):
        window_stats(torch.randn(B, L, N), "nope")
