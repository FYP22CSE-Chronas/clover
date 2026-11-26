from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from multiprocessing import get_context

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

VARIANTS = {
    "base": None,
    "ric": {"source": "diff", "mode": "ric"},
    "ric_level": {"source": "level", "mode": "ric"},
    "independent": {"mode": "independent"},
    "hybrid": {"source": "diff", "mode": "hybrid"},
    "ric_fixed": {"source": "diff", "mode": "ric", "learn_shrink": False},
    "ric_lw": {"source": "diff", "target": "identity", "shrink": "analytic"},
    "ric_tree": {"source": "diff", "target": "tree", "shrink": "analytic"},
    "ric_tree_level": {"source": "level", "target": "tree", "shrink": "analytic"},
    "ric_tree_multi": {"source": "multi", "target": "tree", "shrink": "analytic"},
    "ric_lw_multi": {"source": "multi", "target": "identity", "shrink": "analytic"},
    "ric_lw_s4": {"target": "identity", "shrink": "analytic", "diff_lag": 4},
    "ric_lw_s7": {"target": "identity", "shrink": "analytic", "diff_lag": 7},
    "ric_tree_att": {"target": "tree", "shrink": "analytic", "attenuate": True},
    "ric_lw_att": {"target": "identity", "shrink": "analytic", "attenuate": True},
    "ric_tree_att1": {
        "target": "tree",
        "shrink": "analytic",
        "attenuate": True,
        "attenuate_scale": 1.0,
    },
    "ric_tree_rho50": {
        "target": "tree",
        "shrink": "analytic",
        "attenuate": True,
        "attenuate_init": 0.5,
        "attenuate_scale": 0.0,
    },
    "ric_tree_rho65": {
        "target": "tree",
        "shrink": "analytic",
        "attenuate": True,
        "attenuate_init": 0.65,
        "attenuate_scale": 0.0,
    },
    # `lookback`: rows of history behind the dependence estimate (0 = the window).
    "ric_lw_m60": {"target": "identity", "shrink": "analytic", "lookback": 60},
    "ric_tree_m60": {"target": "tree", "shrink": "analytic", "lookback": 60},
    "ric_lw_m120": {"target": "identity", "shrink": "analytic", "lookback": 120},
    "ric_tree_m120": {"target": "tree", "shrink": "analytic", "lookback": 120},
    "ric_lw_all": {"target": "identity", "shrink": "analytic", "lookback": 100000},
    "ric_tree_all": {"target": "tree", "shrink": "analytic", "lookback": 100000},
    "ric_lw_s4_all": {
        "target": "identity",
        "shrink": "analytic",
        "diff_lag": 4,
        "lookback": 100000,
    },
}


def _original_build_model():
    import pipeline.runner as runner

    return runner.build_model


ORIGINAL_BUILD_MODEL = _original_build_model()


def _original_batcher():
    import pipeline.runner as runner

    return runner.WindowBatcher


ORIGINAL_BATCHER = _original_batcher()


HERE = os.path.dirname(os.path.abspath(__file__))


def _config_root(dataset: str) -> str | None:
    """A config root for a dataset that lives beside this experiment, else None.

    `load_config` wants `default.yaml` beside `datasets/`; the core default is copied
    in fresh on every call so the experiment can never run on a stale copy.
    """
    import shutil

    from config import CONFIG_ROOT

    local = os.path.join(HERE, "configs")
    if not os.path.exists(os.path.join(local, "datasets", f"{dataset}.yaml")):
        return None
    shutil.copyfile(
        os.path.join(CONFIG_ROOT, "default.yaml"), os.path.join(local, "default.yaml")
    )
    return local


def configure(dataset: str, origins: int | None, overrides: list[str], backtest: int = 0):
    """The run config; `backtest=K` drops the final `h` rows (the real test horizon)
    and scores K rolling pseudo-test origins inside what is left, so a choice made on
    it never sees the test window."""
    from config import load_config
    from data import load

    root = _config_root(dataset)
    kwargs = {} if root is None else {"root": root}
    config = load_config(dataset, overrides=overrides, **kwargs)
    if backtest > 0:
        dates = load(config.name, config.data).dates
        end = dates[-config.data.h - 1].strftime("%Y-%m-%d")
        config = replace(config, data=replace(config.data, end_date=end))
        origins = backtest
    if origins is not None:
        split = replace(config.data.split, strategy="rolling", origins=origins)
        config = replace(config, data=replace(config.data, split=split))
    return config


def calibration(samples: np.ndarray, y: np.ndarray, levels) -> dict:
    """Spread/skill ratio and 80% interval coverage per level."""
    m, sd = samples.mean(-1), samples.std(-1)
    lo, hi = np.quantile(samples, 0.1, -1), np.quantile(samples, 0.9, -1)
    out = {}
    for name, (a, b) in levels.items():
        err = m[a:b] - y[a:b]
        out[name] = {
            "spread_rmse": float(np.sqrt((sd[a:b] ** 2).mean() / (err**2).mean())),
            "cov80": float(((y[a:b] >= lo[a:b]) & (y[a:b] <= hi[a:b])).mean()),
        }
    return out


def install_variant(variant: str, config) -> None:
    """Point `pipeline.runner` at the model and batcher this variant needs.

    Always sets both globals, so a patch left by an earlier job in a reused pool
    worker can never leak into this one.
    """
    import pipeline.runner as runner
    from config import ModelDims
    from experiments.ric_clover.batching import RICBatcher
    from experiments.ric_clover.network import RICClover, RICConfig

    spec = VARIANTS[variant]
    runner.WindowBatcher = ORIGINAL_BATCHER
    if spec is not None:
        spec = dict(spec)
        lookback = int(spec.pop("lookback", 0))
        ric = RICConfig(**spec)
        if lookback > 0:
            dataset_obj, _, _ = runner.prepare(config)
            RICBatcher.bottom = dataset_obj.bottom
            RICBatcher.lookback = lookback
            runner.WindowBatcher = RICBatcher

        def build_model(cfg, ds, feats):
            dims = ModelDims(
                h=cfg.data.h,
                input_size=cfg.data.input_size,
                n_stat_features=feats.n_static,
                n_futr_features=feats.n_future,
                n_hist_features=feats.n_hist,
            )
            return RICClover(cfg.model, dims, ds.S, ric, coint_beta=feats.coint_beta)

        runner.build_model = build_model
    else:
        runner.build_model = ORIGINAL_BUILD_MODEL


def job(args: tuple) -> dict:
    dataset, variant, seed, origins, overrides, threads, backtest, joint = args
    import torch

    torch.set_num_threads(threads)
    import pipeline.runner as runner

    config = configure(dataset, origins, overrides, backtest)
    install_variant(variant, config)
    result = runner.run(config, seed=seed, verbose=False)
    extra = {}
    if joint:
        from experiments.ric_clover.joint_metrics import joint_scores

        ds, _, split = runner.prepare(config)
        scale = np.abs(ds.bottom[: split.train_end]).mean(axis=0)
        extra["joint"] = joint_scores(result.samples, result.y_true, ds.n_bottom, scale)
    return {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "origins": origins,
        "backtest": backtest,
        "overrides": overrides,
        "scrps": result.scrps,
        "calibration": calibration(result.samples, result.y_true, result.level_slices),
        "best_val": result.history.best_val,
        "steps_run": result.history.steps_run,
        **extra,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["labour"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--origins", type=int, default=None)
    parser.add_argument("--backtest", type=int, default=0)
    parser.add_argument(
        "--joint", action="store_true", help="also score the joint bottom distribution"
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as handle:
            for line in handle:
                r = json.loads(line)
                done.add((r["dataset"], r["variant"], r["seed"]))
    jobs = [
        (d, v, s, args.origins, args.overrides, args.threads, args.backtest, args.joint)
        for d in args.datasets
        for v in args.variants
        for s in args.seeds
        if (d, v, s) not in done
    ]
    with get_context("spawn").Pool(args.workers) as pool:
        for record in pool.imap_unordered(job, jobs):
            with open(args.out, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                record["dataset"],
                record["variant"],
                record["seed"],
                f"{record['scrps']['Overall']:.5f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
