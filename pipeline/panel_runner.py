from __future__ import annotations

import numpy as np
import torch

from config import ModelDims, RunConfig
from data import build_panel_split, load_panel
from data.panel import PanelDataset, PanelSplit
from model.distribution import sample_coherent
from model.network import CLOVER
from model.normalization import denormalize_params
from pipeline.panel_batching import PanelBatcher
from pipeline.results import RunResult
from training.metrics import per_location_crps
from training.trainer import Trainer, seed_everything

MASS_FLOOR = 1e-12


def prepare_panel(config: RunConfig) -> tuple[PanelDataset, PanelSplit]:
    """Load the panel and cut its train/validation/test blocks."""
    dataset = load_panel(config.name, config.data)
    split = build_panel_split(
        dataset,
        input_size=config.data.input_size,
        h=config.data.h,
        n_val=config.data.split.val or config.data.h,
        n_test=config.data.split.test or config.data.h,
    )
    return dataset, split


def build_panel_model(config: RunConfig, dataset: PanelDataset) -> CLOVER:
    """Instantiate CLOVER with the dimensions the panel implies."""
    dims = ModelDims(
        h=config.data.h,
        input_size=config.data.input_size,
        n_stat_features=dataset.n_static,
        n_futr_features=dataset.n_future,
        n_hist_features=dataset.n_hist,
    )
    return CLOVER(config.model, dims, dataset.S)


def run_panel(
    config: RunConfig,
    seed: int = 0,
    device: str = "cpu",
    checkpoint_dir: str | None = None,
    verbose: bool = True,
    val_groups: int = 256,
    eval_chunk: int = 20,
    eval_samples: int | None = None,
) -> RunResult:
    """Train one seed on a panel and score the test window across every group.

    Mirrors `runner.run` over a `PanelDataset`. The model and the trainer are the same
    objects the single-hierarchy path uses; only the data plumbing differs.
    """
    seed_everything(seed)
    dataset, split = prepare_panel(config)
    model = build_panel_model(config, dataset)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        sizes = split.sizes
        print(
            f"{config.name}: groups={dataset.n_groups} T={dataset.n_steps} "
            f"Nb={dataset.n_bottom} N={dataset.n_series} h={config.data.h} "
            f"L={config.data.input_size} params={n_params:,}\n"
            f"  windows train={sizes['train']:,} val={sizes['val']:,} "
            f"test={sizes['test']:,}  "
            f"bottom series={dataset.n_groups * dataset.n_bottom:,} "
            f"total series={dataset.n_groups * dataset.n_series:,}"
        )

    def batcher(pairs) -> PanelBatcher:
        return PanelBatcher(
            dataset,
            pairs,
            input_size=config.data.input_size,
            h=config.data.h,
            scaler=config.train.scaler,
            device=device,
        )

    train_batcher = batcher(split.train)
    val_batcher = batcher(_subsample(split.val, val_groups, seed))
    if verbose:
        print(
            f"  validation subsampled to {val_batcher.n_windows} of "
            f"{split.sizes['val']:,} windows"
        )

    trainer = Trainer(model, config.train, device=device, verbose=verbose)
    history = trainer.fit(train_batcher, val_batcher, checkpoint_dir=checkpoint_dir)

    scores = evaluate_panel(
        model,
        dataset,
        split.test,
        input_size=config.data.input_size,
        h=config.data.h,
        scaler=config.train.scaler,
        device=device,
        num_samples=eval_samples or config.eval.num_samples,
        chunk=eval_chunk,
        quantile_step=config.eval.quantile_step,
        verbose=verbose,
    )
    return RunResult(
        dataset=config.name,
        seed=seed,
        scrps=scores,
        history=history,
        n_params=n_params,
        extras={"n_groups": dataset.n_groups},
    )


@torch.no_grad()
def evaluate_panel(
    model: CLOVER,
    dataset: PanelDataset,
    test_pairs: tuple[np.ndarray, np.ndarray],
    input_size: int,
    h: int,
    scaler: str,
    device: str,
    num_samples: int,
    chunk: int = 20,
    quantile_step: float = 0.01,
    verbose: bool = True,
) -> dict[str, float]:
    """sCRPS per level, accumulated over item chunks as an exact ratio of sums.

    Scoring in one shot would need 4,036 x 93 x 34 x 1000 floats -- about 51 GB -- and
    a ratio of sums is exact when accumulated blockwise, so chunking approximates
    nothing.
    """
    model.eval()
    batcher = PanelBatcher(
        dataset, test_pairs, input_size=input_size, h=h, scaler=scaler, device=device
    )
    numerator = dict.fromkeys(dataset.level_slices, 0.0)
    denominator = dict.fromkeys(dataset.level_slices, 0.0)

    n = batcher.n_windows
    for start in range(0, n, chunk):
        batch = batcher.batch(list(range(start, min(start + chunk, n))))
        params = denormalize_params(model(batch.windows), batch.scale)
        samples = sample_coherent(params, model.S, num_samples).hierarchy
        samples = samples.permute(0, 2, 1, 3).cpu().numpy()  # [b, N, h, S]
        y_true = (
            torch.einsum("ij,bhj->bhi", model.S, batch.target_bottom)
            .permute(0, 2, 1)
            .cpu()
            .numpy()
        )  # [b, N, h]

        # `per_location_crps` takes [n_series, h]; fold the group axis into it and
        # unfold the result, so one estimator serves both dataset shapes.
        b, n_series, horizon = y_true.shape
        crps = per_location_crps(
            y_true.reshape(b * n_series, horizon),
            samples.reshape(b * n_series, horizon, -1),
            estimator="quantile",
            quantile_step=quantile_step,
        ).reshape(b, n_series, horizon)
        for level, (lo, hi) in dataset.level_slices.items():
            numerator[level] += float(crps[:, lo:hi].sum())
            denominator[level] += float(np.abs(y_true[:, lo:hi]).sum())
        if verbose and (start // chunk) % 25 == 0:
            print(f"  eval {min(start + chunk, n):>5}/{n} groups", flush=True)

    scores = {
        level: numerator[level] / max(denominator[level], MASS_FLOOR)
        for level in dataset.level_slices
    }
    total_num, total_den = sum(numerator.values()), sum(denominator.values())
    return {"Overall": total_num / max(total_den, MASS_FLOOR), **scores}


def _subsample(
    pairs: tuple[np.ndarray, np.ndarray], limit: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """A fixed random subset of validation windows; the full block is far too slow."""
    groups, ts = pairs
    if limit <= 0 or limit >= len(groups):
        return pairs
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(groups), size=limit, replace=False))
    return groups[pick], ts[pick]
