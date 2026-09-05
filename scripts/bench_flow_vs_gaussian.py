"""Benchmark the pure normalizing-flow head against the Gaussian factor model.

Runs `--seeds` seeds of `model.head=factor_model` and `model.head=normalizing_flow`
on each dataset, smallest first, appending one JSON line per run to a JSONL file so
partial progress survives an interrupt. After every dataset it writes a summary block
and prints the per-level comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Smallest first so the cheap datasets report before police finishes.
DATASETS = [
    "labour",
    "prison",
    "tourism_small",
    "wiki2",
    "traffic",
    "tourism_large",
    "police",
]
HEADS = ["factor_model", "normalizing_flow"]


def _one(job: tuple[str, str, int, int]) -> dict:
    """Train and score a single (dataset, head, seed) in a worker process."""
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
    }


def _stats(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var**0.5


def summarize(records: list[dict], heads: list[str], baseline: str) -> dict:
    """Per-level mean/std for each head, plus the candidate-vs-baseline delta."""
    levels = list(records[0]["scrps"])
    summary: dict = {"levels": {}, "n_params": {}, "seconds": {}}
    for head in heads:
        rows = [r for r in records if r["head"] == head]
        if not rows:
            continue
        summary["n_params"][head] = rows[0]["n_params"]
        summary["seconds"][head] = sum(r["seconds"] for r in rows)
        for level in levels:
            mean, std = _stats([r["scrps"][level] for r in rows])
            summary["levels"].setdefault(level, {})[head] = {
                "mean": mean,
                "std": std,
                "n_seeds": len(rows),
            }
    candidates = [h for h in heads if h != baseline]
    for level, per_head in summary["levels"].items():
        if baseline not in per_head:
            continue
        base = per_head[baseline]["mean"]
        for head in candidates:
            if head not in per_head:
                continue
            cand = per_head[head]["mean"]
            key = "delta_pct" if len(candidates) == 1 else f"delta_pct.{head}"
            per_head[key] = 100.0 * (cand - base) / base if base else float("nan")
    return summary


def print_summary(
    dataset: str, summary: dict, heads: list[str], baseline: str
) -> None:
    print(f"\n{'=' * 78}\n{dataset}\n{'=' * 78}")
    candidates = [h for h in heads if h != baseline]
    header = f"{'level':<34}" + "".join(f"{h[:21]:>22}" for h in heads)
    header += "".join(f"{'delta':>10}" for _ in candidates)
    print(header)
    print("-" * len(header))
    for level, per_head in summary["levels"].items():
        row = f"{level:<34}"
        for head in heads:
            cell = per_head.get(head)
            text = f"{cell['mean']:.6f} +/-{cell['std']:.6f}" if cell else "-"
            row += f"{text:>22}"
        for head in candidates:
            key = "delta_pct" if len(candidates) == 1 else f"delta_pct.{head}"
            delta = per_head.get(key)
            row += f"{delta:+.2f}%".rjust(10) if delta is not None else f"{'-':>10}"
        print(row)
    params = summary["n_params"]
    print("\nparams:  " + "   ".join(f"{h}={params.get(h, 0):,}" for h in heads))
    print("cpu-seconds:  " + "   ".join(
        f"{h}={summary['seconds'].get(h, 0):.0f}" for h in heads))
    print(f"(delta < 0 means the candidate beats {baseline}; sCRPS is lower-is-better)")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--heads", nargs="+", default=HEADS)
    parser.add_argument("--baseline", default="factor_model",
                        help="head the delta column is measured against")
    parser.add_argument(
        "--merge-jsonl", nargs="+", default=[],
        help="JSONL files of earlier runs; records for heads not in --heads are "
             "folded into the summary instead of being recomputed",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--jsonl", default="bench_flow_vs_gaussian.jsonl")
    parser.add_argument("--out", default="bench_flow_vs_gaussian.json")
    args = parser.parse_args(argv)

    jsonl_path = os.path.join(PROJECT_ROOT, args.jsonl)
    out_path = os.path.join(PROJECT_ROOT, args.out)
    # Merge into any summary written by an earlier invocation so running the
    # datasets in separate batches still leaves one complete file.
    summaries: dict[str, dict] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as handle:
            summaries = json.load(handle)

    merged: list[dict] = []
    for path in args.merge_jsonl:
        full = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
        with open(full, encoding="utf-8") as handle:
            merged.extend(json.loads(line) for line in handle if line.strip())

    report_heads = list(args.heads)
    if args.baseline not in report_heads:
        report_heads.insert(0, args.baseline)

    for dataset in args.datasets:
        jobs = [
            (dataset, head, seed, args.threads)
            for head in args.heads
            for seed in range(args.seeds)
        ]
        print(f"\n>>> {dataset}: {len(jobs)} runs "
              f"({args.seeds} seeds x {len(args.heads)} heads)", flush=True)
        # Heads not being re-run come from --merge-jsonl, so a baseline already
        # measured under the same config/seeds is reused rather than recomputed.
        records: list[dict] = [
            r for r in merged
            if r["dataset"] == dataset
            and r["head"] in report_heads
            and r["head"] not in args.heads
            and r["seed"] < args.seeds
        ]
        if records:
            print(f"    reusing {len(records)} prior runs from --merge-jsonl",
                  flush=True)
        started = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for record in pool.map(_one, jobs):
                records.append(record)
                with open(jsonl_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(
                    f"    {record['head']:<18} seed {record['seed']:<3} "
                    f"overall={record['scrps']['Overall']:.6f} "
                    f"({record['seconds']:.0f}s)",
                    flush=True,
                )
        summaries[dataset] = summarize(records, report_heads, args.baseline)
        summaries[dataset]["wall_seconds"] = time.time() - started
        print_summary(dataset, summaries[dataset], report_heads, args.baseline)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2)

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
