from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Predicted vs realized cross-series dependence, by class of bottom pair (the depth of
# the pair's lowest common ancestor). Shows what the factor head believes about how
# series move together, against what their innovations actually do.


def class_means(corr: np.ndarray, cls: np.ndarray) -> dict[int, float]:
    return {int(c): float(corr[cls == c].mean()) for c in np.unique(cls) if c >= 0}


def predicted_correlations(dataset, variant, seed, origins, threads=2, overrides=()):
    import torch

    torch.set_num_threads(threads)
    import pipeline.runner as runner
    from experiments.ric_clover import bench
    from experiments.ric_clover.network import pair_classes
    from model.normalization import denormalize_params
    from pipeline.batching import hist_feature_stats
    from training.trainer import Trainer, seed_everything

    config = bench.configure(dataset, origins, list(overrides))
    bench.install_variant(variant, config)  # patches runner.build_model / WindowBatcher
    seed_everything(seed)
    ds, feats, split = runner.prepare(config)
    model = runner.build_model(config, ds, feats)
    stats = hist_feature_stats(feats, split.train_end)

    def make(windows):
        return runner.WindowBatcher(
            windows,
            feats,
            input_size=config.data.input_size,
            h=config.data.h,
            scaler=config.train.scaler,
            hist_stats=stats,
        )

    Trainer(model, config.train, verbose=False).fit(make(split.train), make(split.val))
    model.eval()
    cls = pair_classes(model.S).numpy()
    test = make(split.test)
    by_h: dict[int, list[dict[int, float]]] = {}
    with torch.no_grad():
        for i in range(test.n_windows):
            batch = test.batch([i])
            p = denormalize_params(model(batch.windows), batch.scale)
            for h in sorted({0, config.data.h // 2, config.data.h - 1}):
                F = p.F[0, h].double().numpy()
                s = p.sigma[0, h].double().numpy()
                cov = F @ F.T + np.diag(s**2)
                sd = np.sqrt(np.diag(cov))
                by_h.setdefault(h + 1, []).append(
                    class_means(cov / np.outer(sd, sd), cls)
                )

    # Realized: innovations over the 36 rows before each test origin, and the test
    # forecast errors themselves are too few to estimate a correlation, so the
    # reference is the recent innovation correlation the generating process implies.
    bottom = ds.bottom
    realized = []
    for w in split.test:
        block = np.diff(bottom[max(0, w.t - 60) : w.t + config.data.h], axis=0)
        realized.append(class_means(np.corrcoef(block.T), cls))
    average = {
        h: {c: float(np.mean([d[c] for d in ds_])) for c in ds_[0]}
        for h, ds_ in by_h.items()
    }
    real = {c: float(np.mean([d[c] for d in realized])) for c in realized[0]}
    return {"predicted": average, "realized_innovations": real}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="labour")
    parser.add_argument("--variants", nargs="+", default=["base", "ric_tree"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--origins", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)
    out = {}
    for variant in args.variants:
        runs = [
            predicted_correlations(
                args.dataset, variant, s, args.origins, overrides=args.overrides
            )
            for s in args.seeds
        ]
        pred = {
            h: {
                c: float(np.mean([r["predicted"][h][c] for r in runs]))
                for c in runs[0]["predicted"][h]
            }
            for h in runs[0]["predicted"]
        }
        out[variant] = {"predicted": pred, "realized": runs[0]["realized_innovations"]}
        print(variant)
        print("  realized innovations by LCA depth:", runs[0]["realized_innovations"])
        for h, v in pred.items():
            print(f"  predicted h={h}:", {k: round(x, 3) for k, x in v.items()})
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
