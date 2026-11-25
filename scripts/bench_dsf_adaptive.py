"""Gaussian factor model vs DSF vs DSF with adaptive stratified sampling.

Per (dataset, seed), trains `factor_model` and `normalizing_flow` in the same
process. The Gaussian and the DSF are scored with `eval.num_samples` iid draws; the
same trained DSF is also scored with `adaptive_sample(stratified=True)`. Each method
is averaged over `--repeats` sampling seeds. Writes one JSON line per (dataset,
seed, method); the summary reports mean +- sd per level, the paired difference to
the Gaussian, wins and a two-sided paired Wilcoxon p-value.
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

METHODS = ["gaussian", "dsf", "dsf_adaptive"]


def _one(job: tuple[str, int, int, int]) -> list[dict]:
    dataset, seed, threads, repeats = job
    import numpy as np
    import torch

    torch.set_num_threads(threads)
    from config import load_config
    from model.adaptive import adaptive_sample
    from model.distribution import sample_coherent
    from model.normalization import denormalize_params
    from pipeline.batching import WindowBatcher
    from pipeline.runner import build_model, prepare
    from training.metrics import scrps_by_level
    from training.trainer import Trainer, seed_everything

    rows = []
    for head in ("factor_model", "normalizing_flow"):
        config = load_config(dataset, overrides=[f"model.head={head}"])
        ev = config.eval
        seed_everything(seed)
        ds, features, split = prepare(config)
        model = build_model(config, ds, features)

        def batcher(w):
            return WindowBatcher(
                w, features, input_size=config.data.input_size, h=config.data.h,
                scaler=config.train.scaler, device="cpu",
            )

        start = time.time()
        Trainer(model, config.train, device="cpu", verbose=False).fit(
            batcher(split.train), batcher(split.val)
        )
        train_seconds = time.time() - start
        model.eval()
        t0 = split.test[0].t
        y = ds.values[t0 : t0 + config.data.h].T.astype(np.float64)
        with torch.no_grad():
            batch = batcher(split.test).batch([0])
            params = denormalize_params(model(batch.windows), batch.scale)

            def draw(method, g):
                if method == "dsf_adaptive":
                    out = adaptive_sample(
                        params, model.S, ev.adaptive_pilot, ev.adaptive_max,
                        ev.adaptive_tol, ds.level_slices, stratified=True, generator=g,
                    )
                    return out.hierarchy, out.n_total
                return sample_coherent(params, model.S, ev.num_samples, g).hierarchy, ev.num_samples

            methods = ["gaussian"] if head == "factor_model" else ["dsf", "dsf_adaptive"]
            for method in methods:
                scores, ns, secs = [], [], []
                for r in range(repeats):
                    g = torch.Generator().manual_seed(1000 * seed + r)
                    s0 = time.time()
                    hierarchy, n = draw(method, g)
                    samples = hierarchy.squeeze(0).permute(1, 0, 2).cpu().numpy()
                    scores.append(scrps_by_level(
                        y, samples, ds.level_slices, ev.estimator, ev.quantile_step
                    ))
                    ns.append(n)
                    secs.append(time.time() - s0)
                rows.append({
                    "dataset": dataset, "seed": seed, "method": method,
                    "scrps": {k: float(np.mean([s[k] for s in scores])) for k in scores[0]},
                    "n_samples": float(np.mean(ns)),
                    "sample_seconds": float(np.mean(secs)),
                    "train_seconds": train_seconds,
                })
    return rows


def summarize(path: str) -> None:
    import numpy as np
    from scipy.stats import wilcoxon

    recs: dict = {}
    for line in open(path):
        r = json.loads(line)
        recs[(r["dataset"], r["method"], r["seed"])] = r
    for d in sorted({k[0] for k in recs}):
        seeds = sorted(
            s for s in {k[2] for k in recs if k[0] == d}
            if all((d, m, s) in recs for m in METHODS)
        )
        levels = list(recs[(d, "gaussian", seeds[0])]["scrps"])
        print(f"\n{d} ({len(seeds)} seeds): sCRPS mean +- sd; delta = paired mean vs gaussian")
        for lv in levels:
            base = np.array([recs[(d, "gaussian", s)]["scrps"][lv] for s in seeds])
            print(f"  [{lv}]")
            for m in METHODS:
                x = np.array([recs[(d, m, s)]["scrps"][lv] for s in seeds])
                line = f"    {m:<13} {x.mean():.5f} +- {x.std():.5f}"
                if m != "gaussian":
                    delta = 100 * np.mean((x - base) / base)
                    wins = int((x < base).sum())
                    p = wilcoxon(x, base).pvalue if np.any(x != base) else 1.0
                    line += f"   delta {delta:+6.2f}%  wins {wins}/{len(seeds)}  p={p:.3f}"
                print(line)
        for m in METHODS:
            n = np.mean([recs[(d, m, s)]["n_samples"] for s in seeds])
            t = np.mean([recs[(d, m, s)]["sample_seconds"] for s in seeds])
            print(f"  {m:<13} samples {n:7.0f}  sampling+scoring {t:.2f}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out", default="bench_dsf_adaptive.jsonl")
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    if not a.summary_only:
        jobs = [
            (d, s, a.threads, a.repeats)
            for d in a.dataset or ["labour", "traffic", "wiki2"]
            for s in range(a.seed_start, a.seed_start + a.seeds)
        ]
        with ProcessPoolExecutor(a.workers) as pool, open(a.out, "a") as f:
            for rows in pool.map(_one, jobs):
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"done {rows[0]['dataset']} seed {rows[0]['seed']}", flush=True)
    summarize(a.out)


if __name__ == "__main__":
    main()
