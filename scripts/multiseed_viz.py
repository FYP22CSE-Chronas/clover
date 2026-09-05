"""Train N seeds of each head on one dataset, then draw a seed-averaged fan chart
per head over the test window.

    python scripts/multiseed_viz.py --dataset traffic --seeds 15 \
        --heads factor_model normalizing_flow copula_spline

`pipeline.runner.run` already returns the predictive samples alongside the score,
so one training pass yields both the per-seed sCRPS table and the draws the fan
charts need -- benchmarking and plotting separately would train everything twice.

Seeds are combined by Vincentization: each seed's draws are sorted and averaged
elementwise across seeds, which is the same as averaging every quantile. The
result is the *typical* predictive distribution of the head, so band widths stay
comparable to a single seed. Pooling the raw draws instead would widen the bands
by however much the seeds disagree, which is a different quantity. `--combine
pool` does that if you want it.

Writes to `results/figures/<dataset>_multiseed/`:

    fan_charts_test_<head>.png   seed-averaged test window, one per head
    seed_samples_<head>.npz      per-seed draws, to replot without refitting
    multiseed_<dataset>.jsonl    one record per (head, seed), bench schema
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

LABELS = {
    "factor_model": "Gaussian factor model",
    "normalizing_flow": "Normalizing flow",
    "copula_spline": "Gaussian-copula spline",
    "copula_flow": "Gaussian-copula flow",
}


def _one(job: tuple[str, str, int, int]) -> dict:
    """Train and score one (dataset, head, seed); keep the predictive draws."""
    dataset, head, seed, threads = job
    import torch

    torch.set_num_threads(threads)
    from config import load_config
    from pipeline.runner import run

    config = load_config(dataset, overrides=[f"model.head={head}"])
    start = time.time()
    result = run(config, seed=seed, device="cpu", verbose=False)
    return {
        "dataset": dataset,
        "head": head,
        "seed": seed,
        "scrps": result.scrps,
        "n_params": result.n_params,
        "steps_run": result.history.steps_run,
        "best_val_scrps": result.history.best_val,
        "seconds": time.time() - start,
        # float32 keeps the pickle back to the parent small; plotting is unaffected.
        "samples": result.samples.astype(np.float32),
        "y_true": result.y_true,
    }


def combine(per_seed: np.ndarray, how: str) -> np.ndarray:
    """[n_seeds, N, h, S] -> [N, h, S], the seed-averaged predictive sample.

    `vincent` sorts each seed's draws and averages them elementwise, which
    averages the quantile functions -- the average forecast. `pool` concatenates
    the draws into one mixture over seeds, which also absorbs seed disagreement
    into the band width.
    """
    if how == "pool":
        return np.concatenate(list(per_seed), axis=-1)
    return np.sort(per_seed, axis=-1).mean(axis=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="traffic")
    parser.add_argument("--seeds", type=int, default=15)
    parser.add_argument(
        "--heads", nargs="+",
        default=["factor_model", "normalizing_flow", "copula_spline"],
    )
    parser.add_argument("--combine", choices=["vincent", "pool"], default="vincent")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--n-cols", type=int, default=3,
                        help="series drawn per level")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument(
        "--replot-from", default=None, metavar="DIR",
        help="redraw from a previous run's seed_samples_<head>.npz instead of "
             "training. Seeding is deterministic, so the first --seeds seeds of a "
             "longer run are exactly what a fresh shorter run would produce.",
    )
    parser.add_argument(
        "--suffix", default="", metavar="TAG",
        help="appended to the output directory name, to keep a shorter re-draw "
             "from overwriting the run it was drawn from",
    )
    args = parser.parse_args(argv)

    from config import load_config
    from forecast_viz import plot_fan_charts
    from pipeline.runner import prepare

    out_dir = os.path.join(
        PROJECT_ROOT, args.out, f"{args.dataset}_multiseed{args.suffix}"
    )
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, f"multiseed_{args.dataset}.jsonl")

    # The dataset itself is head-independent: load it once for the plot axes.
    config = load_config(args.dataset)
    data, _, split = prepare(config)
    t0 = split.test[0].t
    h = config.data.h
    print(
        f"{args.dataset}: N={data.n_series} h={h} test window "
        f"{data.dates[t0].date()} to {data.dates[t0 + h - 1].date()}   "
        f"{len(args.heads)} heads x {args.seeds} seeds",
        flush=True,
    )

    by_head: dict[str, list[dict]] = {head: [] for head in args.heads}
    if args.replot_from:
        src = os.path.join(PROJECT_ROOT, args.replot_from)
        scores = {}
        with open(os.path.join(src, f"multiseed_{args.dataset}.jsonl"),
                  encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rec = json.loads(line)
                    scores[(rec["head"], rec["seed"])] = rec
        for head in args.heads:
            blob = np.load(os.path.join(src, f"seed_samples_{head}.npz"),
                           allow_pickle=True)
            keep = [i for i, s in enumerate(blob["seeds"]) if s < args.seeds]
            if len(keep) < args.seeds:
                raise SystemExit(
                    f"{head}: {args.replot_from} holds {len(blob['seeds'])} seeds, "
                    f"need {args.seeds}"
                )
            for i in keep:
                seed = int(blob["seeds"][i])
                by_head[head].append(
                    {**scores[(head, seed)],
                     "samples": blob["samples"][i], "y_true": blob["y_true"]}
                )
                with open(jsonl_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(scores[(head, seed)]) + "\n")
            print(f"    {head:<18} reused {len(keep)} seeds from {args.replot_from}",
                  flush=True)
        jobs = []
    else:
        jobs = [
            (args.dataset, head, seed, args.threads)
            for head in args.heads
            for seed in range(args.seeds)
        ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(_one, jobs):
            by_head[record["head"]].append(record)
            with open(jsonl_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    {k: v for k, v in record.items()
                     if k not in ("samples", "y_true")}
                ) + "\n")
            print(
                f"    {record['head']:<18} seed {record['seed']:<3} "
                f"overall={record['scrps']['Overall']:.6f} "
                f"({record['seconds']:.0f}s)",
                flush=True,
            )

    for head in args.heads:
        records = sorted(by_head[head], key=lambda r: r["seed"])
        per_seed = np.stack([r["samples"] for r in records])
        y_true = records[0]["y_true"]
        averaged = combine(per_seed, args.combine)
        fan_path = os.path.join(out_dir, f"fan_charts_test_{head}.png")
        plot_fan_charts(
            data, averaged, y_true, t0, h, fan_path,
            n_cols=args.n_cols, history=False,
            label=f"{LABELS.get(head, head)} -- {len(records)}-seed "
                  f"{args.combine} average",
        )
        np.savez_compressed(
            os.path.join(out_dir, f"seed_samples_{head}.npz"),
            samples=per_seed,
            y_true=y_true,
            seeds=np.array([r["seed"] for r in records]),
            series_names=np.array(data.series_names),
        )
        overall = [r["scrps"]["Overall"] for r in records]
        mean = sum(overall) / len(overall)
        std = (sum((v - mean) ** 2 for v in overall) / len(overall)) ** 0.5
        print(f"\n{LABELS.get(head, head)}: Overall sCRPS "
              f"{mean:.6f} +/-{std:.6f} over {len(records)} seeds")
        print(f"  wrote {fan_path}")

    print(f"\nwrote {len(os.listdir(out_dir))} files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
