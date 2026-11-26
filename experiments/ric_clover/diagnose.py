"""Diagnostics behind the write-up's Section 2 (and the attenuation and mean-side notes).

Each subcommand prints one family of numbers. Pair classes are the depth of a bottom
pair's lowest common ancestor in S (`network.pair_classes`); on Labour, class 0 is
cross-state, 1 is same state but different gender, 2 is full-time/part-time siblings.

    python experiments/ric_clover/diagnose.py eras
    python experiments/ric_clover/diagnose.py oracle   --features none --seeds 0
    python experiments/ric_clover/diagnose.py insample --features none --origins 4
    python experiments/ric_clover/diagnose.py errors   --variant ric_tree --origins 4
    python experiments/ric_clover/diagnose.py shares   --variant ric_tree_att1 --origins 4

`--features none` drops the static and future features (the featureless Labour model
the diagnosis was first run on); `--features paper` keeps the dataset config's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

NO_FEATURES = ["features.static=[]", "features.future=[]"]


def class_means(corr: np.ndarray, cls: np.ndarray) -> dict[int, float]:
    return {
        int(c): round(float(corr[cls == c].mean()), 3) for c in np.unique(cls) if c >= 0
    }


def classes(S: np.ndarray) -> np.ndarray:
    import torch

    from experiments.ric_clover.network import pair_classes

    return pair_classes(torch.as_tensor(S, dtype=torch.float32)).numpy()


def setup(args):
    """Config with the variant installed (patches runner.build_model / WindowBatcher)."""
    import torch

    torch.set_num_threads(args.threads)
    from experiments.ric_clover import bench

    overrides = (NO_FEATURES if args.features == "none" else []) + args.overrides
    config = bench.configure(args.dataset, args.origins, overrides)
    bench.install_variant(args.variant, config)
    return config


# --------------------------------------------------------------------------- eras


def eras(args) -> dict:
    """Realized first-difference correlation by pair class, per era (data only)."""
    from data import load
    from experiments.ric_clover.bench import configure

    config = configure(args.dataset, None, [])  # finds experiment-local configs too
    ds = load(config.name, config.data)
    cls = classes(ds.S)
    diffs = np.diff(ds.bottom, axis=0)  # row k is the change into date k+1
    dates = ds.dates[1:]
    test_start = len(diffs) - config.data.h
    pre = test_start - 36
    bounds = [
        ("to 1994", 0, int(np.searchsorted(dates, np.datetime64("1995-01-01")))),
        ("1995-2007", None, int(np.searchsorted(dates, np.datetime64("2008-01-01")))),
        ("2008 to 36m pre-test", None, pre),
        ("36 months pre-test", None, test_start),
    ]
    out, start = {}, 0
    for name, _, stop in bounds:
        block = diffs[start:stop]
        span = f"{dates[start]:%Y-%m}..{dates[stop - 1]:%Y-%m}"
        out[name] = {
            "span": span,
            "rows": len(block),
            **class_means(np.corrcoef(block.T), cls),
        }
        start = stop
    return out


# --------------------------------------------------------------------------- oracle


def oracle(args) -> dict:
    """sCRPS with every predictive mean replaced by the truth, and the share of squared
    error that is per-origin bias, per level."""
    import pipeline.runner as runner
    from training.metrics import scrps_by_level

    config = setup(args)
    out = {}
    for seed in args.seeds:
        r = runner.run(config, seed=seed, verbose=False)
        x, y, levels = r.samples, r.y_true, r.level_slices
        err = x.mean(-1) - y  # [n_series, origins*h]
        debiased = scrps_by_level(
            y,
            x - err[..., None],
            levels,
            estimator=config.eval.estimator,
            quantile_step=config.eval.quantile_step,
        )
        h = config.data.h
        bias_share = {}
        for name, (a, b) in levels.items():
            blocks = err[a:b].reshape(b - a, -1, h)
            bias_share[name] = round(
                float((blocks.mean(-1) ** 2).mean() / (blocks**2).mean()), 3
            )
        out[seed] = {
            "scrps": {k: round(v, 5) for k, v in r.scrps.items()},
            "oracle_mean_scrps": {k: round(v, 5) for k, v in debiased.items()},
            "bias_share_of_mse": bias_share,
        }
    return out


# --------------------------------------------------------------------------- insample


def _calibration(x: np.ndarray, y: np.ndarray, levels, nb: int) -> dict:
    """Spread/RMSE and 80% coverage per level, and mean bottom forecast-error corr."""
    m, sd = x.mean(-1), x.std(-1)
    lo, hi = np.quantile(x, 0.1, -1), np.quantile(x, 0.9, -1)
    row = {}
    for name, (a, b) in levels.items():
        e = m[a:b] - y[a:b]
        spread = np.sqrt((sd[a:b] ** 2).mean() / (e**2).mean())
        cover = ((y[a:b] >= lo[a:b]) & (y[a:b] <= hi[a:b])).mean()
        row[name] = {
            "spread_rmse": round(float(spread), 2),
            "cov80": round(float(cover), 2),
        }
    C = np.corrcoef(m[-nb:] - y[-nb:])
    row["bottom_error_corr"] = round(float((C.sum() - nb) / (nb * nb - nb)), 3)
    return row


def _insample_seed(config, seed: int) -> dict:
    import pipeline.runner as runner
    from pipeline.batching import hist_feature_stats
    from training.trainer import Trainer, seed_everything

    h = config.data.h
    seed_everything(seed)
    ds, feats, split = runner.prepare(config)
    model = runner.build_model(config, ds, feats)
    stats = hist_feature_stats(feats, split.train_end)

    def make(windows):
        return runner.WindowBatcher(
            windows,
            feats,
            input_size=config.data.input_size,
            h=h,
            scaler=config.train.scaler,
            hist_stats=stats,
        )

    Trainer(model, config.train, verbose=False).fit(make(split.train), make(split.val))

    def report(windows):
        x = runner.forecast_windows(model, make(windows), config.eval.num_samples)
        y = np.concatenate([ds.values[w.t : w.t + h] for w in windows]).T
        return _calibration(x, y, ds.level_slices, ds.n_bottom)

    years = [ds.dates[w.t].year for w in split.train]
    result = {}
    for name, lo, hi in [
        ("train to 1994", 0, 1995),
        ("train 1995-2007", 1995, 2008),
        ("train 2008 on", 2008, 10**4),
    ]:
        wins = [w for w, yr in zip(split.train, years, strict=True) if lo <= yr < hi]
        if wins:
            result[name] = report(wins[::12])
    result["test"] = report(split.test)
    return result


def insample(args) -> dict:
    """Calibration on in-sample training windows by era, against the test windows:
    spread/RMSE and 80% coverage per level, and the mean bottom forecast-error
    correlation. One model per seed; every 12th training window is scored."""
    config = setup(args)
    return {seed: _insample_seed(config, seed) for seed in args.seeds}


# --------------------------------------------------------------------------- errors


def errors(args) -> dict:
    """Test forecast-error correlation by pair class, against the realized innovation
    correlation over the same origins (60 months before each origin plus its horizon),
    and the error split into a per-origin bias part and a within-origin part."""
    import pipeline.runner as runner

    config = setup(args)
    h = config.data.h
    ds, _, split = runner.prepare(config)
    cls = classes(ds.S)
    nb = ds.n_bottom
    errs = []
    for seed in args.seeds:
        r = runner.run(config, seed=seed, verbose=False)
        errs.append(r.samples.mean(-1)[-nb:] - r.y_true[-nb:])
    e = np.concatenate(errs, axis=1)
    blocks = e.reshape(nb, -1, h)
    within = (blocks - blocks.mean(-1, keepdims=True)).reshape(nb, -1)
    realized = [
        np.corrcoef(np.diff(ds.bottom[max(0, w.t - 60) : w.t + h], axis=0).T)
        for w in split.test
    ]
    return {
        "forecast_errors": class_means(np.corrcoef(e), cls),
        "within_origin_errors": class_means(np.corrcoef(within), cls),
        "per_origin_bias": class_means(np.corrcoef(blocks.mean(-1)), cls),
        "realized_innovations": class_means(np.mean(realized, axis=0), cls),
    }


# --------------------------------------------------------------------------- shares


def sibling_groups(S: np.ndarray) -> list[np.ndarray]:
    """Bottom series grouped by their immediate parent (smallest aggregate row)."""
    nb = S.shape[1]
    agg = S[:-nb] > 0
    size = np.where(agg, agg.sum(1, keepdims=True), np.inf)  # [Na, Nb]
    parent = size.argmin(0)
    return [np.where(parent == p)[0] for p in np.unique(parent)]


def shares(args) -> dict:
    """Mean-side check on the bottom level. (1) Split each bottom series' per-origin
    bias into its parent's error allocated by the model's own share, and the rest
    (a within-sibling mis-split). (2) Re-split each sibling group's predicted sum by
    the last observed share or the input-window average share, which leaves every
    aggregate unchanged, and rescore."""
    import pipeline.runner as runner
    from training.metrics import scrps_by_level

    config = setup(args)
    h = config.data.h
    ds, _, split = runner.prepare(config)
    nb, n_agg = ds.n_bottom, ds.S.shape[0] - ds.n_bottom
    groups = sibling_groups(ds.S)
    parts, scores = [], {"model": [], "last_share": [], "window_share": []}
    for seed in args.seeds:
        r = runner.run(config, seed=seed, verbose=False)
        x, y = r.samples, r.y_true
        m = x.mean(-1)[-nb:].reshape(nb, -1, h)
        e = m - y[-nb:].reshape(nb, -1, h)
        bias = e.mean(-1)
        parent_part = np.zeros_like(bias)
        for idx in groups:
            share = m[idx] / m[idx].sum(0, keepdims=True)
            parent_part[idx] = (share * e[idx].sum(0, keepdims=True)).mean(-1)
        within = bias - parent_part
        parts.append(
            [
                (parent_part**2).mean() / (bias**2).mean(),
                (within**2).mean() / (bias**2).mean(),
            ]
        )
        for name in scores:
            xx = x.copy()
            if name != "model":
                for o, w in enumerate(split.test):
                    hist = ds.bottom[: w.t]
                    sl = slice(o * h, (o + 1) * h)
                    for idx in groups:
                        if name == "last_share":
                            sh = hist[-1, idx] / hist[-1, idx].sum()
                        else:
                            win = hist[-config.data.input_size :, idx]
                            sh = (win / win.sum(1, keepdims=True)).mean(0)
                        total = m[idx, o].sum(0)
                        shift = sh[:, None] * total[None, :] - m[idx, o]
                        xx[n_agg + idx, sl, :] += shift[:, :, None]
            sc = scrps_by_level(
                y,
                xx,
                r.level_slices,
                estimator=config.eval.estimator,
                quantile_step=config.eval.quantile_step,
            )
            scores[name].append(sc)
    bottom = list(r.level_slices)[-1]
    out = {
        "bias_share": {
            "parent_error": round(float(np.mean(parts, 0)[0]), 3),
            "within_sibling_split": round(float(np.mean(parts, 0)[1]), 3),
        }
    }
    for name, runs in scores.items():
        rel = [
            s[bottom] / b[bottom] - 1 for s, b in zip(runs, scores["model"], strict=True)
        ]
        out[name] = {
            "overall": round(float(np.mean([s["Overall"] for s in runs])), 5),
            "bottom": round(float(np.mean([s[bottom] for s in runs])), 5),
            "bottom_change_pct": round(100 * float(np.mean(rel)), 2),
        }
    return out


COMMANDS = {
    "eras": eras,
    "oracle": oracle,
    "insample": insample,
    "errors": errors,
    "shares": shares,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--dataset", default="labour")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--origins", type=int, default=None)
    parser.add_argument("--features", choices=["paper", "none"], default="paper")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    result = COMMANDS[args.command](args)
    print(json.dumps(result, indent=1, default=str))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
