from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FeatureSpec, load_config  # noqa: E402
from pipeline.runner import run  # noqa: E402

# Variants: how each dataset's cointegration is handed to the model. `level` picks the
# sibling grouping the Johansen problem runs within.
VARIANTS = {
    "base": {},
    "feat": {"encoder": True},
    "ec": {"skip": True},
    "ec_trend": {"skip": True, "trend": 4},
    "trend": {"trend": 4},
    "ec_sep": {"skip": True, "form": "separable"},
    "ec_skiponly": {"skip": True, "encoder": False},
    "ec_sep_skiponly": {"skip": True, "form": "separable", "encoder": False},
    "ec_trend_skiponly": {"skip": True, "trend": 4, "encoder": False},
    "ec_ols_skiponly": {"skip": True, "init": "ols", "encoder": False},
    "ec_fast_skiponly": {"skip": True, "encoder": False, "lr_mult": 10.0},
    "ec_fast_trend_skiponly": {
        "skip": True,
        "encoder": False,
        "lr_mult": 10.0,
        "trend": 4,
    },
    "ec_ols": {"skip": True, "init": "ols"},
    "ec_fast": {"skip": True, "lr_mult": 10.0},
    "ec_ols_trend_skiponly": {"skip": True, "init": "ols", "trend": 4, "encoder": False},
}
LEVELS = {"synthetic_coint": "Group", "labour": "Gender/Geography"}


def configure(
    dataset: str,
    variant: str,
    origins: int | None,
    method: str = "johansen",
    static: bool = True,
    start_date: str | None = None,
):
    config = load_config(dataset)
    if not static:
        # No per-series identity: the shared-weight network cannot learn each
        # series' equilibrium share, so only an explicit error signal can supply it.
        config = replace(config, static_features=())
    if start_date is not None:
        # A shorter training span: fewer windows for the network to learn shares from.
        config = replace(config, data=replace(config.data, start_date=start_date))
    if origins is not None:
        config = replace(
            config,
            data=replace(
                config.data,
                split=replace(config.data.split, strategy="rolling", origins=origins),
            ),
        )
    spec = VARIANTS[variant]
    if not spec:
        return config
    history = (
        FeatureSpec(
            "equilibrium_error",
            {"rank": 3, "level": LEVELS[dataset], "method": method},
        ),
    )
    model = replace(
        config.model,
        error_correction_skip=bool(spec.get("skip", False)),
        error_correction_form=spec.get("form", "full"),
        error_correction_init=spec.get("init", "zero"),
        hist_to_encoder=bool(spec.get("encoder", True)),
        trend_factors=int(spec.get("trend", 0)),
    )
    train = replace(
        config.train, error_correction_lr_mult=float(spec.get("lr_mult", 1.0))
    )
    return replace(config, model=model, train=train, history_features=history)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["synthetic_coint", "labour"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--origins", type=int, default=None)
    parser.add_argument("--method", default="johansen", choices=["johansen", "share"])
    parser.add_argument("--no-static", action="store_true", help="drop static features")
    parser.add_argument("--start-date", default=None, help="truncate rows before this")
    parser.add_argument("--out", default="bench_cointegration.json")
    args = parser.parse_args(argv)

    records = []
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as handle:
            records = json.load(handle)
    done = {(r["dataset"], r["variant"], r["seed"]) for r in records}

    for dataset in args.datasets:
        for variant in args.variants:
            for seed in args.seeds:
                if (dataset, variant, seed) in done:
                    continue
                config = configure(
                    dataset,
                    variant,
                    args.origins,
                    args.method,
                    not args.no_static,
                    args.start_date,
                )
                result = run(config, seed=seed, verbose=False)
                record = {
                    "dataset": dataset,
                    "variant": variant
                    if variant == "base"
                    else f"{variant}:{args.method}",
                    "seed": seed,
                    "scrps": result.scrps,
                    "best_val": result.history.best_val,
                    "steps_run": result.history.steps_run,
                }
                records.append(record)
                with open(args.out, "w", encoding="utf-8") as handle:
                    json.dump(records, handle, indent=1)
                print(
                    f"{dataset:<16} {variant:<9} seed={seed} "
                    f"val={record['best_val']:.5f} test={result.scrps['Overall']:.5f}",
                    flush=True,
                )
    report(records)


def report(records: list[dict]) -> None:
    """Mean test sCRPS per variant, plus the validation-gated choice per seed."""
    by = {}
    for r in records:
        by.setdefault((r["dataset"], r["variant"]), {})[r["seed"]] = r
    datasets = sorted({d for d, _ in by})
    for dataset in datasets:
        print(f"\n{dataset}: Overall sCRPS, mean over seeds (std)")
        variants = sorted(
            {v for d, v in by if d == dataset}, key=lambda v: (v != "base", v)
        )
        seeds = sorted(set.intersection(*(set(by[(dataset, v)]) for v in variants)))
        for v in variants:
            vals = np.array([by[(dataset, v)][s]["scrps"]["Overall"] for s in seeds])
            print(f"  {v:<10} {vals.mean():.5f} ({vals.std():.5f})")
        gated = []
        for s in seeds:
            best = min(variants, key=lambda v: by[(dataset, v)][s]["best_val"])
            gated.append(by[(dataset, best)][s]["scrps"]["Overall"])
        gated = np.array(gated)
        print(f"  {'gated':<10} {gated.mean():.5f} ({gated.std():.5f})  picks by val")


if __name__ == "__main__":
    main()
