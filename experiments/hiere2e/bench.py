"""Benchmark HierE2E and HierE2E + DSF under HierE2E's own protocol.

    python -m experiments.hiere2e.bench select --datasets tourism_small labour
    python -m experiments.hiere2e.bench run --datasets tourism_small labour
    python -m experiments.hiere2e.bench report

`select` follows the paper's supplementary: a grid over context-length factor and
epochs, scored on the validation window, best by lowest CRPS. It uses the Gaussian
model and seed 0, and writes `selected.json`. One 100-epoch run per factor covers
the whole epoch grid, because the training trajectory does not depend on the
total epoch count; the weights are scored at epochs 10, 25, 50 and 100.

`run` trains both marginals with the selected settings for each seed and scores the
test window with the repo's own sCRPS (1% quantile grid, 1000 samples), so the DSF
variant differs from plain HierE2E in its marginal and nothing else. Results are
appended to JSONL as they finish; finished (dataset, marginal, seed) cells are
skipped on a re-run.

`report` prints mean +/- 95% CI per level, the DSF-vs-Gaussian delta, seed wins and
a paired Wilcoxon signed-rank test.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SELECT_JSONL = os.path.join(HERE, "results", "select.jsonl")
SELECTED = os.path.join(HERE, "selected.json")
RUN_JSONL = os.path.join(HERE, "results", "run.jsonl")

# The paper's grid: context-length factors {2,3,4} for the long-horizon datasets
# and {15,25,40,60} for the short-horizon ones; factors whose context does not fit
# in the training rows are dropped. Batch size 32, except 4 for Tourism-L.
FACTORS = {
    "labour": [2, 3, 4],
    "tourism_small": [2, 3, 4],
    "tourism_large": [2, 3, 4],
    "prison": [2, 3, 4],
    "traffic": [15, 25, 40, 60],
    "wiki2": [15, 25, 40, 60],
}
BATCH_SIZE = {"tourism_large": 4}
EPOCH_GRID = (10, 25, 50, 100)
DATASETS = ["tourism_small", "labour", "traffic", "wiki2"]
MARGINALS = ["gaussian", "dsf"]


def _setup(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)


def _data(dataset: str):
    from config import load_config
    from experiments.hiere2e.data import prepare

    config = load_config(dataset)
    return config, prepare(config)


def _model(data, marginal: str):
    from experiments.hiere2e.model import HierE2E

    return HierE2E(data.S, len(data.lags), data.time_feat.shape[1], marginal=marginal)


def feasible_factors(dataset: str, data) -> list[int]:
    """Grid factors whose context leaves at least one training origin."""
    limit = data.train_end - data.h - data.max_lag
    return [f for f in FACTORS[dataset] if f * data.h <= limit]


def _select_one(job: tuple[str, int, int]) -> dict:
    dataset, factor, threads = job
    _setup(threads)
    from experiments.hiere2e.train import FitConfig, fit
    from training.trainer import seed_everything

    _, data = _data(dataset)
    seed_everything(0)
    model = _model(data, "gaussian")
    cfg = FitConfig(
        context_length=factor * data.h,
        epochs=max(EPOCH_GRID),
        checkpoints=EPOCH_GRID,
        batch_size=BATCH_SIZE.get(dataset, 32),
    )
    result = fit(model, data, cfg, seed=0)
    return {
        "dataset": dataset,
        "factor": factor,
        "context_length": cfg.context_length,
        "val_by_epoch": result.val_by_epoch,
        "seconds": result.seconds,
    }


def _run_one(job: tuple[str, str, int, dict, int]) -> dict:
    dataset, marginal, seed, chosen, threads = job
    _setup(threads)
    from experiments.hiere2e.train import FitConfig, fit, test_scores
    from training.trainer import seed_everything

    config, data = _data(dataset)
    seed_everything(seed)
    model = _model(data, marginal)
    cfg = FitConfig(
        context_length=chosen["context_length"],
        epochs=chosen["epochs"],
        checkpoints=(),
        batch_size=BATCH_SIZE.get(dataset, 32),
    )
    result = fit(model, data, cfg, seed=seed, validate=False)
    scores, _ = test_scores(
        model,
        data,
        cfg.context_length,
        config.eval.num_samples,
        estimator=config.eval.estimator,
        quantile_step=config.eval.quantile_step,
    )
    return {
        "dataset": dataset,
        "marginal": marginal,
        "seed": seed,
        "context_length": cfg.context_length,
        "epochs": cfg.epochs,
        "scrps": scores,
        "n_params": sum(p.numel() for p in model.parameters()),
        "seconds": result.seconds,
    }


def _read(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _pool(fn, jobs: list, workers: int, path: str, label) -> None:
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(fn, jobs):
            _append(path, record)
            print(
                f"    {label(record)}  ({record['seconds']:.0f}s, "
                f"wall {time.time() - started:.0f}s)",
                flush=True,
            )


def select(datasets: list[str], workers: int, threads: int) -> None:
    done = {(r["dataset"], r["factor"]) for r in _read(SELECT_JSONL)}
    jobs = []
    for dataset in datasets:
        _, data = _data(dataset)
        for factor in feasible_factors(dataset, data):
            if (dataset, factor) not in done:
                jobs.append((dataset, factor, threads))
    print(f">>> select: {len(jobs)} runs", flush=True)

    def label(r: dict) -> str:
        vals = " ".join(f"e{e}={v:.4f}" for e, v in r["val_by_epoch"].items())
        return f"{r['dataset']:<14} factor {r['factor']:<3} {vals}"

    _pool(_select_one, jobs, workers, SELECT_JSONL, label)
    write_selected()


def write_selected() -> dict:
    """Best (context length, epochs) per dataset by validation sCRPS."""
    best: dict[str, dict] = {}
    for r in _read(SELECT_JSONL):
        for epoch, val in r["val_by_epoch"].items():
            cand = {
                "context_length": r["context_length"],
                "factor": r["factor"],
                "epochs": int(epoch),
                "val_scrps": val,
            }
            if r["dataset"] not in best or val < best[r["dataset"]]["val_scrps"]:
                best[r["dataset"]] = cand
    with open(SELECTED, "w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)
    for dataset, cand in best.items():
        print(
            f"    selected {dataset:<14} context {cand['context_length']:<4} "
            f"(factor {cand['factor']}) epochs {cand['epochs']:<4} "
            f"val sCRPS {cand['val_scrps']:.5f}",
            flush=True,
        )
    return best


def run(datasets: list[str], seeds: int, workers: int, threads: int) -> None:
    with open(SELECTED, encoding="utf-8") as handle:
        selected = json.load(handle)
    done = {(r["dataset"], r["marginal"], r["seed"]) for r in _read(RUN_JSONL)}
    jobs = [
        (dataset, marginal, seed, selected[dataset], threads)
        for dataset in datasets
        for seed in range(seeds)
        for marginal in MARGINALS
        if (dataset, marginal, seed) not in done
    ]
    # Longest first, so the slow DSF jobs do not all start last.
    jobs.sort(key=lambda j: (j[1] != "dsf", -selected[j[0]]["epochs"]))
    print(f">>> run: {len(jobs)} runs", flush=True)

    def label(r: dict) -> str:
        return (
            f"{r['dataset']:<14} {r['marginal']:<9} seed {r['seed']:<2} "
            f"overall={r['scrps']['Overall']:.5f}"
        )

    _pool(_run_one, jobs, workers, RUN_JSONL, label)
    report(datasets)


def _ci95(values: list[float]) -> tuple[float, float]:
    import numpy as np
    from scipy import stats

    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(half)


def report(datasets: list[str] | None = None) -> None:
    from scipy import stats

    records = _read(RUN_JSONL)
    datasets = datasets or sorted({r["dataset"] for r in records})
    for dataset in datasets:
        rows = [r for r in records if r["dataset"] == dataset]
        if not rows:
            continue
        by = {m: {r["seed"]: r for r in rows if r["marginal"] == m} for m in MARGINALS}
        levels = list(rows[0]["scrps"])
        print(f"\n=== {dataset}  (sCRPS, mean +/- 95% CI over seeds; lower is better)")
        header = "".join(f"{lvl[:18]:>22}" for lvl in levels)
        print(f"{'model':<16}{'n':>3}{header}")
        for m in MARGINALS:
            if not by[m]:
                continue
            cells = []
            for lvl in levels:
                mean, half = _ci95([r["scrps"][lvl] for r in by[m].values()])
                cells.append(f"{mean:.4f} +/- {half:.4f}")
            print(
                f"{'HierE2E' if m == 'gaussian' else 'HierE2E + DSF':<16}"
                f"{len(by[m]):>3}" + "".join(f"{c:>22}" for c in cells)
            )
        common = sorted(set(by["gaussian"]) & set(by["dsf"]))
        if len(common) >= 2:
            g = [by["gaussian"][s]["scrps"]["Overall"] for s in common]
            d = [by["dsf"][s]["scrps"]["Overall"] for s in common]
            delta = 100 * (sum(d) / sum(g) - 1)
            wins = sum(di < gi for di, gi in zip(d, g, strict=True))
            p = stats.wilcoxon(d, g).pvalue if len(common) >= 5 else float("nan")
            print(
                f"DSF vs Gaussian, Overall, {len(common)} paired seeds: "
                f"delta {delta:+.1f}%  wins {wins}/{len(common)}  Wilcoxon p={p:.3f}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["select", "run", "report"])
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "select":
        select(args.datasets, args.workers, args.threads)
    elif args.command == "run":
        run(args.datasets, args.seeds, args.workers, args.threads)
    else:
        report(args.datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
