"""Benchmark HINT and HINT + DSF on the repo's hierarchies.

    python -m experiments.hint.bench select   # Gaussian HINT grid, then DSF floor
    python -m experiments.hint.bench run      # 5 seeds x {gaussian, dsf}
    python -m experiments.hint.bench report

`select` (seed 0, validation window only):
  1. Gaussian HINT over input length x learning rate, each run early-stopped on
     validation; the best (input, lr, reconciler) by validation sCRPS wins.
  2. HINT + DSF at those settings over its one extra knob, the flow slope floor
     {0.1, 1.0}, scored with the same reconciler.
The choices land in `selected.json`.

`run` trains both kernels for every seed with the selected settings (validation
early stopping, restore best) and scores the test window with the repo's sCRPS
(1% quantile grid, 1000 samples). Finished cells are skipped on a re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SELECT_JSONL = os.path.join(RESULTS, "select.jsonl")
FLOOR_JSONL = os.path.join(RESULTS, "select_dsf.jsonl")
RUN_JSONL = os.path.join(RESULTS, "run.jsonl")
SELECTED = os.path.join(HERE, "selected.json")

DATASETS = ["tourism_small", "labour", "traffic", "wiki2"]
KERNELS = ["gaussian", "dsf"]
LEARNING_RATES = [1e-3, 5e-4, 1e-4]  # HINT paper's search space
FLOORS = [0.1, 1.0]


def input_sizes(data) -> list[int]:
    """L = m * h for m in {2, 3, 4}, plus CLOVER's own window for the dataset.

    With h = 1 (traffic) m * h is too short to read anything, so that dataset uses
    CLOVER's window and its quarter and half instead.
    """
    if data.h == 1:
        base = data.config_input_size
        sizes = {base // 4, base // 2, base}
    else:
        sizes = {2 * data.h, 3 * data.h, 4 * data.h, data.config_input_size}
    limit = data.train_end - data.h
    return sorted(s for s in sizes if 2 <= s <= limit)


def _data(dataset: str):
    from config import load_config
    from experiments.hint.data import prepare

    config = load_config(dataset)
    return config, prepare(config)


def _train(
    dataset: str,
    kernel: str,
    seed: int,
    input_size: int,
    lr: float,
    floor: float,
    reconciler: str | None,
    threads: int,
):
    import torch

    torch.set_num_threads(threads)
    from experiments.hint.model import HINT
    from experiments.hint.train import FitConfig, fit
    from training.trainer import seed_everything

    config, data = _data(dataset)
    seed_everything(seed)
    model = HINT(input_size, data.h, kernel=kernel, flow_a_floor=floor)
    cfg = FitConfig(input_size=input_size, learning_rate=lr, reconciler=reconciler)
    result = fit(model, data, cfg, seed=seed)
    return config, data, model, result


def _best_by_reconciler(result) -> dict[str, float]:
    best: dict[str, float] = {}
    for _, vals in result.val_history:
        for method, v in vals.items():
            best[method] = min(best.get(method, float("inf")), v)
    return best


def _select_one(job: tuple) -> dict:
    dataset, input_size, lr, threads = job
    _, _, _, result = _train(dataset, "gaussian", 0, input_size, lr, 0.1, None, threads)
    return {
        "dataset": dataset,
        "input_size": input_size,
        "learning_rate": lr,
        "best_val": _best_by_reconciler(result),
        "best_step": result.best_step,
        "steps_run": result.steps_run,
        "seconds": result.seconds,
    }


def _floor_one(job: tuple) -> dict:
    dataset, chosen, floor, threads = job
    _, _, _, result = _train(
        dataset,
        "dsf",
        0,
        chosen["input_size"],
        chosen["learning_rate"],
        floor,
        chosen["reconciler"],
        threads,
    )
    return {
        "dataset": dataset,
        "floor": floor,
        "best_val": result.best_val,
        "best_step": result.best_step,
        "seconds": result.seconds,
    }


def _run_one(job: tuple) -> dict:
    dataset, kernel, seed, chosen, threads = job
    floor = chosen["dsf_floor"] if kernel == "dsf" else 0.1
    config, data, model, result = _train(
        dataset,
        kernel,
        seed,
        chosen["input_size"],
        chosen["learning_rate"],
        floor,
        chosen["reconciler"],
        threads,
    )
    from experiments.hint.train import score_test_window

    scores = score_test_window(
        model,
        data,
        chosen["input_size"],
        chosen["reconciler"],
        config.eval.num_samples,
        seed,
        estimator=config.eval.estimator,
        quantile_step=config.eval.quantile_step,
    )
    return {
        "dataset": dataset,
        "kernel": kernel,
        "seed": seed,
        "scrps": scores,
        "best_val": result.best_val,
        "best_step": result.best_step,
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


def _pool(fn, jobs, workers: int, path: str, label) -> None:
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(fn, jobs):
            _append(path, record)
            print(
                f"    {label(record)}  ({record['seconds']:.0f}s, "
                f"wall {time.time() - started:.0f}s)",
                flush=True,
            )


def _load_selected() -> dict:
    if not os.path.exists(SELECTED):
        return {}
    with open(SELECTED, encoding="utf-8") as handle:
        return json.load(handle)


def _save_selected(selected: dict) -> None:
    with open(SELECTED, "w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)


def select(datasets: list[str], workers: int, threads: int) -> None:
    # Stage 1: Gaussian grid.
    done = {
        (r["dataset"], r["input_size"], r["learning_rate"]) for r in _read(SELECT_JSONL)
    }
    jobs = []
    for dataset in datasets:
        _, data = _data(dataset)
        for size in input_sizes(data):
            for lr in LEARNING_RATES:
                if (dataset, size, lr) not in done:
                    jobs.append((dataset, size, lr, threads))
    print(f">>> select (gaussian grid): {len(jobs)} runs", flush=True)
    _pool(
        _select_one,
        jobs,
        workers,
        SELECT_JSONL,
        lambda r: (
            f"{r['dataset']:<14} L={r['input_size']:<4} "
            f"lr={r['learning_rate']:<7} best={min(r['best_val'].values()):.5f}"
        ),
    )

    selected = _load_selected()
    for dataset in datasets:
        best = None
        for r in _read(SELECT_JSONL):
            if r["dataset"] != dataset:
                continue
            for method, v in r["best_val"].items():
                if best is None or v < best["val_scrps"]:
                    best = {
                        "input_size": r["input_size"],
                        "learning_rate": r["learning_rate"],
                        "reconciler": method,
                        "val_scrps": v,
                    }
        selected[dataset] = {**selected.get(dataset, {}), **best}
    _save_selected(selected)

    # Stage 2: the DSF kernel's slope floor, at the Gaussian-selected settings.
    done = {(r["dataset"], r["floor"]) for r in _read(FLOOR_JSONL)}
    jobs = [
        (d, selected[d], f, threads)
        for d in datasets
        for f in FLOORS
        if (d, f) not in done
    ]
    print(f">>> select (dsf floor): {len(jobs)} runs", flush=True)
    _pool(
        _floor_one,
        jobs,
        workers,
        FLOOR_JSONL,
        lambda r: f"{r['dataset']:<14} floor={r['floor']:<4} best={r['best_val']:.5f}",
    )
    for dataset in datasets:
        rows = [r for r in _read(FLOOR_JSONL) if r["dataset"] == dataset]
        best = min(rows, key=lambda r: r["best_val"])
        selected[dataset]["dsf_floor"] = best["floor"]
        selected[dataset]["dsf_val_scrps"] = best["best_val"]
    _save_selected(selected)
    for dataset in datasets:
        s = selected[dataset]
        print(
            f"    selected {dataset:<14} L={s['input_size']:<4} "
            f"lr={s['learning_rate']:<7} {s['reconciler']:<12} "
            f"val={s['val_scrps']:.5f} | dsf floor {s['dsf_floor']} "
            f"val={s['dsf_val_scrps']:.5f}",
            flush=True,
        )


def run(datasets: list[str], seeds: int, workers: int, threads: int) -> None:
    selected = _load_selected()
    done = {(r["dataset"], r["kernel"], r["seed"]) for r in _read(RUN_JSONL)}
    jobs = [
        (d, k, s, selected[d], threads)
        for d in datasets
        for s in range(seeds)
        for k in KERNELS
        if (d, k, s) not in done
    ]
    jobs.sort(key=lambda j: j[1] != "dsf")  # slow DSF jobs first
    print(f">>> run: {len(jobs)} runs", flush=True)
    _pool(
        _run_one,
        jobs,
        workers,
        RUN_JSONL,
        lambda r: (
            f"{r['dataset']:<14} {r['kernel']:<9} seed {r['seed']:<2} "
            f"overall={r['scrps']['Overall']:.5f}"
        ),
    )
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
        by = {k: {r["seed"]: r for r in rows if r["kernel"] == k} for k in KERNELS}
        levels = list(rows[0]["scrps"])
        print(f"\n=== {dataset}  (sCRPS, mean +/- 95% CI over seeds; lower is better)")
        print(f"{'model':<12}{'n':>3}" + "".join(f"{lvl[:18]:>22}" for lvl in levels))
        for k in KERNELS:
            if not by[k]:
                continue
            cells = []
            for lvl in levels:
                mean, half = _ci95([r["scrps"][lvl] for r in by[k].values()])
                cells.append(f"{mean:.4f} +/- {half:.4f}")
            name = "HINT" if k == "gaussian" else "HINT + DSF"
            print(f"{name:<12}{len(by[k]):>3}" + "".join(f"{c:>22}" for c in cells))
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
