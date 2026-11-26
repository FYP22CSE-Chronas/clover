from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.hiere2e.data import HierData, Windows, mean_scale, time_features
from experiments.hiere2e.model import HierE2E, project, projection_matrix
from experiments.hiere2e.train import (
    FitConfig,
    crps_pairs,
    forecast,
    loss_and_backward,
    loss_steps,
)
from training.losses import gaussian_crps

H = 4


@pytest.fixture
def hdata(dataset) -> HierData:
    """The conftest hierarchy, windowed as HierE2E sees it (last_h split)."""
    T = dataset.n_steps
    return HierData(
        values=dataset.values.astype(np.float32),
        S=dataset.S,
        time_feat=time_features(dataset.dates, "Q"),
        lags=[1],
        level_slices=dict(dataset.level_slices),
        h=H,
        train_end=T - 2 * H,
        val_origins=[T - 2 * H],
        test_origin=T - H,
    )


def _model(hdata: HierData, marginal: str = "gaussian") -> HierE2E:
    torch.manual_seed(0)
    return HierE2E(hdata.S, len(hdata.lags), hdata.time_feat.shape[1], marginal=marginal)


def test_projection_is_idempotent_and_fixes_coherent_vectors(S: np.ndarray) -> None:
    M = projection_matrix(S)
    torch.testing.assert_close(M @ M, M, atol=1e-5, rtol=0)
    bottom = torch.randn(S.shape[1], 3)
    coherent = torch.as_tensor(S) @ bottom
    torch.testing.assert_close(M @ coherent, coherent, atol=1e-5, rtol=0)


def test_projected_samples_are_coherent(S: np.ndarray) -> None:
    y = project(projection_matrix(S), torch.randn(2, S.shape[0], 50))
    nb = S.shape[1]
    torch.testing.assert_close(
        torch.einsum("ij,bjs->bis", torch.as_tensor(S), y[:, -nb:]), y, atol=1e-5, rtol=0
    )


@pytest.mark.parametrize("marginal", ["gaussian", "dsf"])
def test_forecast_paths_are_coherent_and_shaped(hdata: HierData, marginal: str) -> None:
    model = _model(hdata, marginal)
    windows = Windows(hdata, context_length=8)
    out = forecast(model, windows, [hdata.test_origin], num_samples=20)
    assert out.shape == (1, hdata.n_series, H, 20)
    nb = hdata.S.shape[1]
    agg = np.einsum("ij,jhs->ihs", hdata.S, out[0, -nb:])
    np.testing.assert_allclose(agg, out[0], rtol=1e-4, atol=1e-3)


def test_training_never_touches_validation_or_test_rows(hdata: HierData) -> None:
    windows = Windows(hdata, context_length=8)
    origins = windows.train_origins()
    assert origins.max() + hdata.h <= hdata.train_end
    assert origins.min() - 8 - hdata.max_lag >= 0


def test_mean_scale_never_returns_zero() -> None:
    context = torch.zeros(2, 5, 3)
    context[0, :, 1] = 4.0
    scale = mean_scale(context)
    assert (scale > 0).all()
    torch.testing.assert_close(scale[0, 0, 1], torch.tensor(4.0))
    # an all-zero series in one window borrows that series' batch mean
    torch.testing.assert_close(scale[1, 0, 1], torch.tensor(2.0))


def test_crps_pairs_is_unbiased_for_a_gaussian() -> None:
    torch.manual_seed(0)
    y = torch.tensor([0.3, -1.0, 2.0])
    samples = torch.randn(3, 200_000)
    expected = gaussian_crps(torch.zeros(3), torch.ones(3), y)
    torch.testing.assert_close(crps_pairs(y, samples), expected, atol=5e-3, rtol=0)


def test_context_subsampling_weights_sum_to_the_full_step_count() -> None:
    gen = torch.Generator().manual_seed(0)
    steps, weights = loss_steps(context_length=40, h=7, n_context=8, gen=gen)
    assert len(steps) == 15 and len(set(steps.tolist())) == 15
    assert steps[-7:].tolist() == list(range(40, 47))
    torch.testing.assert_close(weights.sum(), torch.tensor(47.0))
    all_steps, all_w = loss_steps(context_length=40, h=7, n_context=None, gen=gen)
    assert all_steps.tolist() == list(range(47)) and (all_w == 1).all()


@pytest.mark.parametrize("marginal", ["gaussian", "dsf"])
def test_chunked_backward_matches_a_single_backward(
    hdata: HierData, marginal: str, monkeypatch
) -> None:
    """Chunking the loss in time must not change the gradient."""
    windows = Windows(hdata, context_length=8)
    batch = windows.sequence(torch.from_numpy(windows.train_origins()[:4]), hdata.h)

    def grads(chunk: int) -> list[torch.Tensor]:
        model = _model(hdata, marginal)
        model.eval()  # no zoneout noise
        fixed = torch.Generator().manual_seed(1)
        z_bank = torch.randn(4, 64, hdata.n_series, 10, generator=fixed)
        calls = {"start": 0}
        original = model.sample

        def deterministic(params, scale, n, coherent=True):
            # the same noise for a step whichever chunk it falls in
            c = params["mu"].shape[1]
            z = z_bank[:, calls["start"] : calls["start"] + c]
            calls["start"] += c
            monkeypatch.setattr(torch, "randn", lambda *a, **k: z.clone())
            try:
                return original(params, scale, n, coherent)
            finally:
                monkeypatch.undo()

        model.sample = deterministic
        cfg = FitConfig(
            context_length=8, loss_chunk=chunk, loss_samples=10, loss_context_steps=None
        )
        loss_and_backward(model, batch, cfg)
        return [p.grad.clone() for p in model.parameters() if p.grad is not None]

    for full, chunked in zip(grads(chunk=64), grads(chunk=3), strict=True):
        torch.testing.assert_close(full, chunked, atol=1e-4, rtol=1e-4)


def test_dsf_changes_only_the_marginal(hdata: HierData) -> None:
    gaussian, dsf = _model(hdata, "gaussian"), _model(hdata, "dsf")
    rnn_g = sum(p.numel() for p in gaussian.rnn.parameters())
    rnn_d = sum(p.numel() for p in dsf.rnn.parameters())
    assert rnn_g == rnn_d
    torch.testing.assert_close(gaussian.M, dsf.M)
    assert dsf.proj.out_features == hdata.n_series * (2 + 3 * dsf.k)


def test_daily_time_features_cycle_weekly() -> None:
    dates = pd.date_range("2020-01-06", periods=14, freq="D")
    feats = time_features(dates, "D")
    np.testing.assert_allclose(feats[:7], feats[7:], atol=1e-6)


def test_hierdata_is_a_plain_dataclass(hdata: HierData) -> None:
    assert dataclasses.is_dataclass(hdata) and hdata.max_lag == 1
