from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy import stats

# Paired comparison of each variant against `base` on the same seeds: mean and
# median relative change in overall sCRPS, wins, a two-sided Wilcoxon signed-rank
# test, an exact sign test, and Holm-adjusted p-values across every comparison made
# in one call.


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument(
        "--metric",
        default="scrps",
        choices=["scrps", "energy", "variogram"],
        help="overall sCRPS, or a joint score from `bench.py --joint`",
    )
    args = parser.parse_args(argv)

    tables = []
    for path in args.files:
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        by = defaultdict(dict)
        for r in rows:
            if args.metric == "scrps":
                value = r["scrps"]["Overall"]
            elif "joint" in r:
                value = r["joint"][args.metric]
            else:
                continue
            by[(r["dataset"], r["variant"])][r["seed"]] = value
        for dataset in sorted({d for d, _ in by}):
            if args.datasets and dataset not in args.datasets:
                continue
            base = by.get((dataset, "base"), {})
            for variant in args.variants:
                cand = by.get((dataset, variant), {})
                seeds = sorted(set(base) & set(cand))
                if len(seeds) < 3:
                    continue
                b = np.array([base[s] for s in seeds])
                c = np.array([cand[s] for s in seeds])
                rel = c / b - 1
                wins = int((rel < 0).sum())
                p_w = stats.wilcoxon(c, b).pvalue
                p_s = stats.binomtest(wins, len(seeds), 0.5).pvalue
                tables.append(
                    [path, dataset, variant, len(seeds), b.mean(), c.mean(), rel, wins]
                    + [p_w, p_s]
                )
    p = np.array([t[8] for t in tables])
    order = np.argsort(p)
    holm = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        holm[i] = running
    for t, h in zip(tables, holm, strict=True):
        path, dataset, variant, n, bm, cm, rel, wins, p_w, p_s = t
        tag = path.split("/")[-1].replace(".jsonl", "")
        print(
            f"{tag:16s} {dataset:13s} {variant:11s} n={n:2d} base={bm:.5f} "
            f"var={cm:.5f} mean={rel.mean() * 100:+.2f}% "
            f"median={np.median(rel) * 100:+.2f}% wins={wins}/{n} "
            f"wilcoxon p={p_w:.4f} sign p={p_s:.4f} holm p={h:.4f}"
        )


if __name__ == "__main__":
    main()
