from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

# Backtest-gated selection. For every dataset the gate picks, among `base` and the
# candidate variants, the one with the lowest mean score on the pre-test backtest
# (`bench.py --backtest K [--joint]`). The choice is then scored on the real test, in
# every metric, against the baseline on the same seeds. The backtest never touches the
# test horizon, so the gated rows are honest results.

DATASETS = ["labour", "prison", "tourism_small", "wiki2", "traffic", "tourism_large"]
METRICS = ["scrps", "energy", "variogram"]


def value(record: dict, metric: str) -> float | None:
    if metric == "scrps":
        return record["scrps"]["Overall"]
    return record.get("joint", {}).get(metric)


def load(paths: list[str]) -> dict:
    """(dataset, variant) -> seed -> {metric: value}."""
    out = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                r = json.loads(line)
                for m in METRICS:
                    v = value(r, m)
                    if v is not None:
                        out[(r["dataset"], r["variant"])][r["seed"]][m] = v
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--select-on", nargs="+", default=METRICS, choices=METRICS)
    parser.add_argument("--candidates", nargs="+", default=["ric_tree", "ric_tree_att1"])
    args = parser.parse_args(argv)
    gate, test = load([args.backtest]), load(args.test)

    for select in args.select_on:
        print(f"\n== gate selects on backtest {select}  (pool: base + {args.candidates})")
        summary = {m: [] for m in METRICS}
        for d in DATASETS:
            pool = ["base", *args.candidates]
            scores = {}
            for v in pool:
                vals = [s[select] for s in gate.get((d, v), {}).values() if select in s]
                if vals:
                    scores[v] = float(np.mean(vals))
            if "base" not in scores:
                continue
            pick = min(scores, key=scores.get)
            margin = scores[pick] / scores["base"] - 1
            parts = [f"  {d:14s} picks {pick:14s} (backtest {margin:+.2%})  test:"]
            base = test.get((d, "base"), {})
            chosen = test.get((d, pick), {})
            for m in METRICS:
                seeds = [
                    s for s in base if s in chosen and m in base[s] and m in chosen[s]
                ]
                if not seeds:
                    parts.append(f"{m}=n/a")
                    continue
                rel = np.array([chosen[s][m] / base[s][m] - 1 for s in seeds])
                summary[m].append(rel.mean())
                parts.append(f"{m} {rel.mean():+.2%} ({(rel < 0).sum()}/{len(seeds)})")
            print(" ".join(parts))
        means = "  ".join(f"{m} {np.mean(v):+.2%}" for m, v in summary.items() if v)
        print(f"  mean over datasets: {means}")


if __name__ == "__main__":
    main()
