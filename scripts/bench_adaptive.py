"""Sampling-quality benchmark for iid, stratified and adaptive sampling.

Per seed, trains `normalizing_flow` once, then scores the *same* model under each
sampling variant `--repeats` times with different sampling seeds. Training noise is
held fixed, so each variant is judged by its sCRPS inflation over a near-exact
reference (`--ref` iid draws), its Monte-Carlo spread across repeats, the sample
count it used, and its sampling+scoring time. Writes one JSON line per (seed,
variant, repeat).
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

# name -> (stratified, adaptive, num_samples)
VARIANTS = {
    "iid_200": (False, False, 200),
    "iid_1000": (False, False, 1000),
    "iid_5000": (False, False, 5000),
    "lhs_200": (True, False, 200),
    "lhs_1000": (True, False, 1000),
    "adaptive_iid": (False, True, None),
    "adaptive_lhs": (True, True, None),
}


def _one(job: tuple[str, int, int, int, int, list[str]]) -> list[dict]:
    dataset, seed, threads, repeats, n_ref, overrides = job
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

    config = load_config(dataset, overrides=["model.head=normalizing_flow", *overrides])
    ev = config.eval
    seed_everything(seed)
    ds, features, split = prepare(config)
    model = build_model(config, ds, features)

    def batcher(w):
        return WindowBatcher(
            w, features, input_size=config.data.input_size, h=config.data.h,
            scaler=config.train.scaler, device="cpu",
        )

    Trainer(model, config.train, device="cpu", verbose=False).fit(
        batcher(split.train), batcher(split.val)
    )
    model.eval()
    with torch.no_grad():
        batch = batcher(split.test).batch([0])
        params = denormalize_params(model(batch.windows), batch.scale)
    t0 = split.test[0].t
    y = ds.values[t0 : t0 + config.data.h].T.astype(np.float64)

    def score(hierarchy):
        samples = hierarchy.squeeze(0).permute(1, 0, 2).cpu().numpy()
        return scrps_by_level(y, samples, ds.level_slices, ev.estimator, ev.quantile_step)

    with torch.no_grad():
        g = torch.Generator().manual_seed(10_000 + seed)
        ref = score(sample_coherent(params, model.S, n_ref, g).hierarchy)
        rows = []
        for name, (stratified, adaptive, n) in VARIANTS.items():
            for r in range(repeats):
                g = torch.Generator().manual_seed(1000 * seed + r)
                start = time.time()
                if adaptive:
                    out = adaptive_sample(
                        params, model.S, ev.adaptive_pilot, ev.adaptive_max,
                        ev.adaptive_tol, ds.level_slices, stratified, g,
                    )
                    hierarchy, n_used = out.hierarchy, out.n_total
                else:
                    hierarchy = sample_coherent(
                        params, model.S, n, g, stratified=stratified
                    ).hierarchy
                    n_used = n
                scores = score(hierarchy)
                rows.append({
                    "dataset": dataset, "seed": seed, "variant": name, "repeat": r,
                    "n_samples": n_used, "seconds": time.time() - start,
                    "scrps": scores, "ref_scrps": ref,
                })
    return rows


def summarize(path: str) -> None:
    import statistics as st
    from collections import defaultdict

    by = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        r = json.loads(line)
        by[(r["dataset"], r["variant"])][r["seed"]].append(r)
    datasets = sorted({d for d, _ in by})
    for d in datasets:
        print(f"\n{d}: sCRPS inflation over reference (x1e-4), Overall / worst level")
        print(f"  {'variant':<13} {'n':>6} {'Overall':>9} {'worst':>9} {'MC sd':>8} {'sec':>6}")
        for v in VARIANTS:
            seeds = by.get((d, v))
            if not seeds:
                continue
            infl, worst, sd, n, sec = [], [], [], [], []
            for rows in seeds.values():
                over = [x["scrps"]["Overall"] - x["ref_scrps"]["Overall"] for x in rows]
                lv = [k for k in rows[0]["scrps"] if k != "Overall"]
                infl.append(st.mean(over))
                worst.append(max(st.mean(x["scrps"][k] - x["ref_scrps"][k] for x in rows) for k in lv))
                sd.append(st.pstdev([x["scrps"]["Overall"] for x in rows]))
                n += [x["n_samples"] for x in rows]
                sec += [x["seconds"] for x in rows]
            print(f"  {v:<13} {st.mean(n):>6.0f} {1e4*st.mean(infl):>9.2f} "
                  f"{1e4*st.mean(worst):>9.2f} {1e4*st.mean(sd):>8.2f} {st.mean(sec):>6.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--ref", type=int, default=20_000)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out", default="bench_adaptive_v2.jsonl")
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    if not a.summary_only:
        jobs = [
            (d, s, a.threads, a.repeats, a.ref, a.override)
            for d in a.dataset or ["labour", "traffic", "wiki2"]
            for s in range(a.seeds)
        ]
        with ProcessPoolExecutor(a.workers) as pool, open(a.out, "a") as f:
            for rows in pool.map(_one, jobs):
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                print(f"done {rows[0]['dataset']} seed {rows[0]['seed']}", flush=True)
    summarize(a.out)


if __name__ == "__main__":
    main()
