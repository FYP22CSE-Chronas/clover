from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from experiments.hint.data import HintData, Windows, robust_scale
from experiments.hint.model import (
    HINT,
    RECONCILERS,
    dsf_forward,
    dsf_inverse,
    mixture_nll,
    reconciliation_P,
)
from experiments.hint.train import FitConfig, fit, forecast, reconcile, score_test_window
from model.heads import _flow_shape

H = 4


@pytest.fixture
def hdata(dataset) -> HintData:
    T = dataset.n_steps
    return HintData(
        values=dataset.values.astype(np.float32),
        S=dataset.S.astype(np.float64),
        level_slices=dict(dataset.level_slices),
        h=H,
        config_input_size=8,
        train_end=T - 2 * H,
        val_origins=[T - 2 * H],
        test_origin=T - H,
    )


@pytest.mark.parametrize("method", RECONCILERS)
def test_reconciled_forecasts_are_coherent_and_fix_coherent_ones(
    S: np.ndarray, method: str
) -> None:
    SP = S @ reconciliation_P(S, method)
    y = np.random.default_rng(0).normal(size=(S.shape[0], 5))
    rec = SP @ y
    nb = S.shape[1]
    np.testing.assert_allclose(S @ rec[-nb:], rec, atol=1e-10)
    coherent = S @ np.random.default_rng(1).normal(size=(nb, 5))
    np.testing.assert_allclose(SP @ coherent, coherent, atol=1e-10)


@pytest.mark.parametrize("method", ["MinTraceOLS", "MinTraceWLS"])
def test_mintrace_matches_the_textbook_formula(S: np.ndarray, method: str) -> None:
    """The U-form neuralforecast uses equals (S' W^-1 S)^-1 S' W^-1."""
    W = np.eye(S.shape[0]) if method == "MinTraceOLS" else np.diag(S.sum(axis=1))
    Winv = np.linalg.inv(W)
    textbook = np.linalg.inv(S.T @ Winv @ S) @ S.T @ Winv
    np.testing.assert_allclose(reconciliation_P(S, method), textbook, atol=1e-10)


def test_bottom_up_keeps_the_bottom_forecasts(S: np.ndarray) -> None:
    y = np.arange(S.shape[0], dtype=float)
    rec = S @ reconciliation_P(S, "BottomUp") @ y
    nb = S.shape[1]
    np.testing.assert_allclose(rec[-nb:], y[-nb:])


def test_one_component_nll_is_the_gaussian_nll() -> None:
    torch.manual_seed(0)
    model = HINT(input_size=8, h=H, n_components=1)
    x, loc, scale = torch.randn(6, 8), torch.zeros(6), torch.ones(6)
    y = torch.randn(6, H)
    params = model.params(x, loc, scale)
    nll = mixture_nll(model.component_log_prob(params, y), n_series=1)
    normal = torch.distributions.Normal(params["mu"][..., 0], params["std"][..., 0])
    torch.testing.assert_close(nll, -normal.log_prob(y).mean())


def test_mixture_couples_series_through_shared_components() -> None:
    """Summing over series before the log-sum-exp is not the per-series mixture."""
    log_prob = torch.log(torch.tensor([[[0.9, 0.1]], [[0.1, 0.9]]]))  # 2 series, K=2
    joint = mixture_nll(log_prob, n_series=2)
    separate = mixture_nll(log_prob, n_series=1)
    expected_joint = -(math.log(0.5 * 0.09 + 0.5 * 0.09)) / 2
    torch.testing.assert_close(joint, torch.tensor(expected_joint))
    assert separate < joint  # each series free to pick its own component


def test_samples_share_one_component_across_series() -> None:
    torch.manual_seed(0)
    model = HINT(input_size=8, h=H, n_components=3)
    n_series = 5
    # components far apart; every series gets the same component means
    mu = torch.tensor([-100.0, 0.0, 100.0]).expand(n_series, H, 3)
    params = {"mu": mu, "std": torch.full_like(mu, 1e-3)}
    draws = model.sample(params, n_series, 200)[0]  # [series, h, samples]
    assert (draws.round(decimals=0) == draws[:1].round(decimals=0)).all()


def test_dsf_density_integrates_to_one_and_inverts() -> None:
    torch.manual_seed(0)
    w, a, b = _flow_shape(torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 8), 0.1)
    u = torch.linspace(-300, 300, 600001, dtype=torch.float64)[:, None].expand(-1, 3)
    z, log_det = dsf_forward(u, w.double(), a.double(), b.double())
    density = torch.exp(-0.5 * z**2 - 0.5 * math.log(2 * math.pi) + log_det)
    mass = torch.trapezoid(density, u[:, 0], dim=0)
    torch.testing.assert_close(
        mass, torch.ones(3, dtype=torch.float64), atol=1e-3, rtol=0
    )
    zt = torch.randn(3, 1000) * 2
    back = dsf_forward(
        dsf_inverse(zt, w[:, None], a[:, None], b[:, None]),
        w[:, None],
        a[:, None],
        b[:, None],
    )[0]
    torch.testing.assert_close(back, zt, atol=1e-4, rtol=0)


def test_robust_scale_handles_flat_windows() -> None:
    x = torch.tensor([[5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 10.0]])
    loc, scale = robust_scale(x)
    assert (scale > 0).all()
    torch.testing.assert_close(loc[0], torch.tensor(5.0))


@pytest.mark.parametrize("kernel", ["gaussian", "dsf"])
def test_end_to_end_tiny_fit_scores_finite_and_coherent(
    hdata: HintData, kernel: str
) -> None:
    torch.manual_seed(0)
    model = HINT(input_size=8, h=H, kernel=kernel, mlp_units=(16, 16))
    cfg = FitConfig(input_size=8, max_steps=20, val_check_steps=10, val_samples=20)
    result = fit(model, hdata, cfg, seed=0)
    assert np.isfinite(result.best_val)
    raw = forecast(model, Windows(hdata, 8), [hdata.test_origin], 50, seed=0)
    rec = reconcile(hdata.S, raw, "MinTraceWLS")[0]
    nb = hdata.S.shape[1]
    np.testing.assert_allclose(
        np.einsum("ij,jhs->ihs", hdata.S, rec[-nb:]), rec, rtol=1e-6, atol=1e-6
    )
    scores = score_test_window(model, hdata, 8, "BottomUp", 50, seed=0)
    assert all(np.isfinite(v) for v in scores.values())


def test_training_windows_stay_before_validation(hdata: HintData) -> None:
    origins = Windows(hdata, 8).train_origins()
    assert origins.max() + H <= hdata.train_end and origins.min() >= 8
