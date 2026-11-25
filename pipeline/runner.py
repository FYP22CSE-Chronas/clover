from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from config import EvalConfig, ModelDims, RunConfig
from data import build_features, build_split, load
from data.dataset import ExogenousFeatures, HierarchicalDataset
from data.features import check_anchor_lags
from data.windows import WindowSplit
from model.adaptive import adaptive_sample
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

    samples = forecast(
        model,
        batcher(split.test),
        config.eval.num_samples,
        config.eval,
        dataset.level_slices,
    )
    test_t = split.test[0].t
    y_true = dataset.values[test_t : test_t + config.data.h].T.astype(np.float64)
    scores = scrps_by_level(
        y_true,
        samples,
        dataset.level_slices,
        estimator=config.eval.estimator,
        quantile_step=config.eval.quantile_step,
    )
    return RunResult(
        dataset=config.name,
        seed=seed,
        scrps=scores,
        history=history,
        n_params=n_params,
        samples=samples,
        y_true=y_true,
        series_names=dataset.series_names,
    )


@torch.no_grad()
def forecast(
    model: CLOVER,
    batcher: WindowBatcher,
    num_samples: int,
    eval_config: EvalConfig | None = None,
    level_slices: Mapping[str, tuple[int, int]] | None = None,
) -> np.ndarray:
    """Coherent predictive samples for the first window of `batcher`, [Na+Nb, h, N].

    `eval_config.stratified` draws Latin hypercube noise; `eval_config.adaptive`
    lets `adaptive_sample` choose the sample count (checked per `level_slices`
    level) instead of `num_samples`.
    """
    model.eval()
    batch = batcher.batch([0])
    params = denormalize_params(model(batch.windows), batch.scale)
    stratified = eval_config is not None and eval_config.stratified
    if eval_config is not None and eval_config.adaptive:
        hierarchy = adaptive_sample(
            params,
            model.S,
            n_pilot=eval_config.adaptive_pilot,
            n_max=eval_config.adaptive_max,
            tol=eval_config.adaptive_tol,
            level_slices=level_slices,
            stratified=stratified,
        ).hierarchy
    else:
        hierarchy = sample_coherent(
            params, model.S, num_samples, stratified=stratified
        ).hierarchy
    return hierarchy.squeeze(0).permute(1, 0, 2).cpu().numpy()
