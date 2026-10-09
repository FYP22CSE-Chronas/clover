from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from config import ModelDims, RunConfig
from data import build_features, build_split, load
from data.dataset import ExogenousFeatures, HierarchicalDataset
from data.features import check_anchor_lags
from data.windows import WindowSplit
from model.distribution import sample_coherent
from model.network import CLOVER
from model.normalization import denormalize_params
from pipeline.batching import WindowBatcher
from pipeline.results import RunResult
from training.metrics import scrps_by_level
from training.trainer import Trainer, seed_everything


def prepare(
    config: RunConfig,
) -> tuple[HierarchicalDataset, ExogenousFeatures, WindowSplit]:
    """Load the dataset, build its features and split it into windows."""
    check_anchor_lags(config.future_features, config.data.h)
    dataset = load(config.name, config.data)
    features = build_features(
        dataset, static=config.static_features, future=config.future_features
    )
    split = build_split(
        dataset.bottom, config.data.input_size, config.data.h, config.data.split
    )
    return dataset, features, split


def build_model(
    config: RunConfig, dataset: HierarchicalDataset, features: ExogenousFeatures
) -> CLOVER:
    """Instantiate CLOVER with the dimensions implied by the data and features."""
    dims = ModelDims(
        h=config.data.h,
        input_size=config.data.input_size,
        n_stat_features=features.n_static,
        n_futr_features=features.n_future,
    )
    return CLOVER(config.model, dims, dataset.S)


def run(
    config: RunConfig,
    seed: int = 0,
    device: str = "cpu",
    checkpoint_dir: str | None = None,
    verbose: bool = True,
) -> RunResult:
    """Train one seed end to end and score the test window."""
    seed_everything(seed)
    dataset, features, split = prepare(config)
    model = build_model(config, dataset, features)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(
            f"{config.name}: T={dataset.n_steps} Nb={dataset.n_bottom} "
            f"N={dataset.n_series} h={config.data.h} L={config.data.input_size} "
            f"train={len(split.train)} val={len(split.val)} params={n_params:,}"
        )

    def batcher(windows) -> WindowBatcher:
        return WindowBatcher(
            windows,
            features,
            input_size=config.data.input_size,
            h=config.data.h,
            scaler=config.train.scaler,
            device=device,
        )

    trainer = Trainer(model, config.train, device=device, verbose=verbose)
    history = trainer.fit(
        batcher(split.train), batcher(split.val), checkpoint_dir=checkpoint_dir
    )

    test = batcher(split.test)
    samples = forecast(model, test, config.eval.num_samples)
    test_t = split.test[0].t
    y_true = dataset.values[test_t : test_t + config.data.h].T.astype(np.float64)

    def score(x: np.ndarray) -> dict[str, float]:
        return scrps_by_level(
            y_true,
            x,
            dataset.level_slices,
            estimator=config.eval.estimator,
            quantile_step=config.eval.quantile_step,
        )

    extras = {}
    if trainer.snapshots:
        # The headline score pools every snapshot; the best-validation model's own
        # score is kept alongside for comparison.
        extras["single_scrps"] = score(samples)
        best = {k: v.clone() for k, v in model.state_dict().items()}
        pooled = []
        for state in trainer.snapshots:
            model.load_state_dict(state)
            pooled.append(forecast(model, test, config.eval.num_samples))
        model.load_state_dict(best)
        samples = np.concatenate(pooled, axis=-1)
    return RunResult(
        dataset=config.name,
        seed=seed,
        scrps=score(samples),
        history=history,
        n_params=n_params,
        samples=samples,
        y_true=y_true,
        series_names=dataset.series_names,
        level_slices=dataset.level_slices,
        extras=extras,
    )


def ensemble_scrps(
    results: Sequence[RunResult],
    estimator: str = "quantile",
    quantile_step: float = 0.01,
) -> dict[str, float]:
    """sCRPS per level of the equal-weight mixture of several seeds' forecasts.

    Concatenating every seed's samples is an exact draw from the mixture. CRPS is
    convex in the forecast distribution, so the mixture scores no worse than the
    seeds' mean, up to estimator error.
    """
    if len(results) < 2:
        raise ValueError("an ensemble needs at least two runs")
    first = results[0]
    if any(r.samples is None or r.y_true is None for r in results):
        raise ValueError("every run must keep its test samples and targets")
    for r in results[1:]:
        if r.dataset != first.dataset or not np.array_equal(r.y_true, first.y_true):
            raise ValueError("runs score different test targets and cannot be pooled")
    pooled = np.concatenate([r.samples for r in results], axis=-1)
    return scrps_by_level(
        first.y_true,
        pooled,
        first.level_slices,
        estimator=estimator,
        quantile_step=quantile_step,
    )


@torch.no_grad()
def forecast(model: CLOVER, batcher: WindowBatcher, num_samples: int) -> np.ndarray:
    """Coherent predictive samples for the first window of `batcher`, [Na+Nb, h, N]."""
    model.eval()
    batch = batcher.batch([0])
    params = denormalize_params(model(batch.windows), batch.scale)
    hierarchy = sample_coherent(params, model.S, num_samples).hierarchy
    return hierarchy.squeeze(0).permute(1, 0, 2).cpu().numpy()
