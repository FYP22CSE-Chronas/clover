from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def main(path: str) -> None:
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["variant"])].append(r)
    for dataset in sorted({d for d, _ in groups}):
        base_rows = groups.get((dataset, "base"), [])
        base = {r["seed"]: r["scrps"]["Overall"] for r in base_rows}
        print(f"\n== {dataset}")
        for (d, variant), rs in sorted(groups.items()):
            if d != dataset:
                continue
            levels = list(rs[0]["scrps"])
            mean = {k: np.mean([r["scrps"][k] for r in rs]) for k in levels}
            sd = np.std([r["scrps"]["Overall"] for r in rs])
            paired = [
                r["scrps"]["Overall"] / base[r["seed"]] - 1
                for r in rs
                if r["seed"] in base
            ]
            wins = sum(p < 0 for p in paired)
            cal = rs[0]["calibration"]
            top = list(cal)[0]
            sr = np.mean([r["calibration"][top]["spread_rmse"] for r in rs])
            bot = list(cal)[-1]
            cb = np.mean([r["calibration"][bot]["cov80"] for r in rs])
            lv = " ".join(f"{k[:6]}={mean[k]:.5f}" for k in levels)
            print(
                f"{variant:12s} n={len(rs)} {lv} sd={sd:.5f} "
                f"vs_base={np.mean(paired) * 100:+.2f}% wins={wins}/{len(paired)} "
                f"top_spread/rmse={sr:.2f} bottom_cov80={cb:.2f}"
            )


if __name__ == "__main__":
    main(sys.argv[1])
