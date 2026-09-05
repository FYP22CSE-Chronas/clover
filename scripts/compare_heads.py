"""Fold several bench JSONL files into one table: every head against the Gaussian
factor model, per dataset and per aggregation level.

The bench script writes one JSON line per (dataset, head, seed); this reads any
number of those files, groups them, and prints mean +/- std with the percentage
delta each non-baseline head scores against the baseline. No training happens
here -- it is pure re-reporting of runs already on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Smallest first, matching the order the benchmarks run in.
DATASET_ORDER = [
    "labour", "prison", "tourism_small", "wiki2",
    "traffic", "tourism_large", "police",
]
LABELS = {
    "factor_model": "gaussian",
    "normalizing_flow": "flow",
    "copula_spline": "spline",
    "copula_flow": "copula_flow",
}


def _stats(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var**0.5


def load(paths: list[str]) -> dict:
    """(dataset, head) -> list of records, de-duplicated on seed.

    Later files win on a repeated seed so a re-run supersedes an earlier one.
    """
    by_key: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for path in paths:
        full = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
        if not os.path.exists(full):
            print(f"skipping missing {path}", file=sys.stderr)
            continue
        with open(full, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = json.loads(line)
                by_key[(rec["dataset"], rec["head"])][rec["seed"]] = rec
    return {key: list(seeds.values()) for key, seeds in by_key.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl", nargs="+",
        default=["bench_flow_vs_gaussian.jsonl", "bench_spline_vs_gaussian.jsonl"],
    )
    parser.add_argument("--baseline", default="factor_model")
    parser.add_argument("--levels", action="store_true",
                        help="also print the per-level breakdown for each dataset")
    args = parser.parse_args(argv)

    data = load(args.jsonl)
    datasets = sorted(
        {d for d, _ in data},
        key=lambda d: DATASET_ORDER.index(d) if d in DATASET_ORDER else 99,
    )
    heads = [args.baseline] + sorted(
        {h for _, h in data if h != args.baseline},
        key=lambda h: list(LABELS).index(h) if h in LABELS else 99,
    )

    def cell(dataset: str, head: str, level: str) -> tuple[str, float | None, int]:
        rows = [r for r in data.get((dataset, head), []) if level in r["scrps"]]
        if not rows:
            return "-", None, 0
        mean, std = _stats([r["scrps"][level] for r in rows])
        return f"{mean:.6f} +/-{std:.6f}", mean, len(rows)

    print(f"\nOverall sCRPS (lower is better), delta vs {LABELS.get(args.baseline, args.baseline)}")
    width = 22
    header = f"{'dataset':<16}" + "".join(
        f"{LABELS.get(h, h):>{width}}" + ("" if h == args.baseline else f"{'delta':>10}")
        for h in heads
    )
    print(header)
    print("-" * len(header))
    seed_notes: list[str] = []
    for dataset in datasets:
        row = f"{dataset:<16}"
        _, base, base_n = cell(dataset, args.baseline, "Overall")
        counts = {}
        for head in heads:
            text, mean, n = cell(dataset, head, "Overall")
            counts[head] = n
            row += f"{text:>{width}}"
            if head == args.baseline:
                continue
            if mean is None or not base:
                row += f"{'-':>10}"
            else:
                row += f"{100.0 * (mean - base) / base:+.2f}%".rjust(10)
        print(row)
        if len(set(n for n in counts.values() if n)) > 1:
            seed_notes.append(
                f"  {dataset}: " + ", ".join(
                    f"{LABELS.get(h, h)}={counts[h]}" for h in heads if counts[h]
                )
            )
    if seed_notes:
        print("\nuneven seed counts (delta not on equal footing):")
        print("\n".join(seed_notes))

    if args.levels:
        for dataset in datasets:
            levels = next(
                (list(r["scrps"]) for (d, _), rows in data.items()
                 if d == dataset for r in rows), []
            )
            print(f"\n{'=' * len(header)}\n{dataset}\n{'=' * len(header)}")
            print(header.replace("dataset", "level", 1)
                  if len(header) else header)
            print("-" * len(header))
            for level in levels:
                row = f"{level[:15]:<16}"
                _, base, _ = cell(dataset, args.baseline, level)
                for head in heads:
                    text, mean, _ = cell(dataset, head, level)
                    row += f"{text:>{width}}"
                    if head == args.baseline:
                        continue
                    if mean is None or not base:
                        row += f"{'-':>10}"
                    else:
                        row += f"{100.0 * (mean - base) / base:+.2f}%".rjust(10)
                print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
