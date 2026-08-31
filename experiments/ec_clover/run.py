from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from config import ModelDims, RunConfig
from data.dataset import ExogenousFeatures, HierarchicalDataset
from data.windows import Window
from experiments.ec_clover.config import ECConfig
from experiments.ec_clover.johansen import to_normalized_units
from experiments.ec_clover.network import CloverWithErrorCorrection
from model.distribution import sample_coherent
from model.normalization import denormalize_params, window_stats
from pipeline.batching import WindowBatcher
from pipeline.results import RunResult
from pipeline.runner import prepare
from training.metrics import scrps_by_level
from training.trainer import Trainer, seed_everything


def training_window_scale(
    windows: Sequence[Window], scaler: str = "standard"
) -> np.ndarray:
    """Mean per-series window scale over TRAINING windows only, as [Nb].

    This is the `sigma_i` a Johansen beta has to be multiplied by to live in the
    per-window-normalized units the model is actually fed (see
    `johansen.to_normalized_units`). Averaging the configured scaler's own `scale` over
    the training windows keeps the estimate consistent with what `WindowBatcher`
    computes at train time, and touching only `windows` keeps validation and test rows
    out of it.
    """
    insample = torch.as_tensor(
        np.stack([w.insample for w in windows]), dtype=torch.float32
    )
    return window_stats(insample, scaler).scale.mean(dim=0).squeeze(0).numpy()


def build_model(
    config: RunConfig,
    dataset: HierarchicalDataset,
    features: ExogenousFeatures,
    ec_config: ECConfig,
    beta_init: np.ndarray | None = None,
) -> CloverWithErrorCorrection:
    """Instantiate `CloverWithErrorCorrection` with the dims the data implies.

    Mirrors `pipeline.runner.build_model` exactly, plus the extra `ec_config` /
    `beta_init` arguments -- everything about how dims are read off the dataset and
    features is identical to the baseline path.
    """
    dims = ModelDims(
        h=config.data.h,
        input_size=config.data.input_size,
        n_stat_features=features.n_static,
        n_futr_features=features.n_future,
    )
    return CloverWithErrorCorrection(
        config.model, dims, dataset.S, ec_config=ec_config, beta_init=beta_init
    )


def run(
    config: RunConfig,
    ec_config: ECConfig,
    seed: int = 0,
    device: str = "cpu",
    checkpoint_dir: str | None = None,
    verbose: bool = True,
    beta_init: np.ndarray | None = None,
) -> RunResult:
    """Train one seed end to end and score the test window.

    Identical control flow to `pipeline.runner.run`; the only difference is the model
    class and the extra EC configuration, so results are directly comparable to a
    baseline `pipeline.runner.run` call on the same `config`.
    """
    seed_everything(seed)
    dataset, features, split = prepare(config)
    if ec_config.beta_init.scale_to_normalized:
        beta_init = _resolve_beta(ec_config, beta_init)
        beta_init = to_normalized_units(
            beta_init, training_window_scale(split.train, config.train.scaler)
        )
    model = build_model(config, dataset, features, ec_config, beta_init=beta_init)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(
            f"{config.name} [ec.enabled={ec_config.enabled}]: T={dataset.n_steps} "
            f"Nb={dataset.n_bottom} N={dataset.n_series} h={config.data.h} "
            f"L={config.data.input_size} train={len(split.train)} val={len(split.val)} "
            f"params={n_params:,}"
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

    samples = forecast(model, batcher(split.test), config.eval.num_samples)
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
        extras={"ec_enabled": ec_config.enabled},
    )


def _resolve_beta(ec_config: ECConfig, beta_init: np.ndarray | None) -> np.ndarray:
    """The supplied beta, from the argument or from the config, as a numpy array."""
    if beta_init is not None:
        return np.asarray(beta_init, dtype=np.float32)
    if ec_config.beta_init.values is not None:
        return np.asarray(ec_config.beta_init.values, dtype=np.float32)
    if ec_config.beta_init.values_path is not None:
        return np.asarray(np.load(ec_config.beta_init.values_path), dtype=np.float32)
    raise ValueError(
        "ec.beta_init.scale_to_normalized needs a supplied beta; set beta_init.values, "
        "beta_init.values_path, or pass beta_init to run()"
    )


@torch.no_grad()
def forecast(
    model: CloverWithErrorCorrection, batcher: WindowBatcher, num_samples: int
) -> np.ndarray:
    """Coherent predictive samples for the first window of `batcher`, [Na+Nb, h, N]."""
    model.eval()
    batch = batcher.batch([0])
    params = denormalize_params(model(batch.windows), batch.scale)
    hierarchy = sample_coherent(params, model.S, num_samples).hierarchy
    return hierarchy.squeeze(0).permute(1, 0, 2).cpu().numpy()
