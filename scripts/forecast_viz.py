"""Forecast the test window with the Gaussian factor head and plot the result.

    python scripts/forecast_viz.py --dataset tourism_small --seed 0

Trains one seed exactly as `python -m cli` does, then keeps the pieces the CLI
throws away -- the dataset, its dates, and the raw predictive samples -- so the
test window can be drawn rather than only scored. Writes to
`results/figures/<dataset>/`:

    fan_charts.png          history + 50/80/95 predictive bands + actual, per level
    fan_charts_test.png     the same fan chart scaled to the test window alone
    prediction_detail.png   the predictive distribution alone, history dropped
    scrps_by_level.png      sCRPS per level, and per horizon step
    calibration.png         nominal vs empirical central-interval coverage
    pit_histogram.png       probability integral transform of the actuals
    training_curve.png      train loss and validation sCRPS
    forecast_quantiles.csv  per (series, step) mean, median and interval bounds
    forecast_samples.npz    raw samples and actuals -- to replot without refitting
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config  # noqa: E402
from pipeline.batching import WindowBatcher  # noqa: E402
from pipeline.runner import build_model, forecast, prepare  # noqa: E402
from training.metrics import per_location_crps, scrps_by_level  # noqa: E402
from training.trainer import Trainer, seed_everything  # noqa: E402

BANDS = [(0.95, 0.12), (0.80, 0.20), (0.50, 0.32)]  # (central mass, fill alpha)
NOMINAL = np.arange(0.1, 1.0, 0.1)


def fit_and_forecast(dataset_name: str, seed: int, overrides: list[str], device: str):
    """Train one seed and return everything needed to draw its test window."""
    config = load_config(dataset_name, overrides=overrides)
    if config.model.head != "factor_model":
        raise SystemExit(
            f"head is {config.model.head!r}; this script plots the Gaussian factor "
            "model. Drop the override, or pass --set model.head=factor_model."
        )
    seed_everything(seed)
    data, features, split = prepare(config)
    model = build_model(config, data, features)
    print(
        f"{config.name}: T={data.n_steps} Nb={data.n_bottom} N={data.n_series} "
        f"h={config.data.h} L={config.data.input_size} train={len(split.train)} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    def batcher(windows):
        return WindowBatcher(
            windows,
            features,
            input_size=config.data.input_size,
            h=config.data.h,
            scaler=config.train.scaler,
            device=device,
        )

    trainer = Trainer(model, config.train, device=device)
    history = trainer.fit(batcher(split.train), batcher(split.val))

    samples = forecast(model, batcher(split.test), config.eval.num_samples)
    t0 = split.test[0].t
    y_true = data.values[t0 : t0 + config.data.h].T.astype(np.float64)
    scores = scrps_by_level(
        y_true,
        samples,
        data.level_slices,
        estimator=config.eval.estimator,
        quantile_step=config.eval.quantile_step,
    )
    return config, data, history, samples, y_true, t0, scores


def band_quantiles(samples: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bound of the central interval holding `mass` probability."""
    tail = (1.0 - mass) / 2.0
    return np.quantile(samples, tail, axis=-1), np.quantile(samples, 1 - tail, axis=-1)


def coverage(y_true: np.ndarray, samples: np.ndarray, mass: float) -> np.ndarray:
    """Boolean [n_series, h]: did the actual land inside the central interval?"""
    lo, hi = band_quantiles(samples, mass)
    return (y_true >= lo) & (y_true <= hi)


def pit(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Rank of each actual within its own predictive sample, flattened."""
    return (samples < y_true[..., None]).mean(axis=-1).ravel()


def top_series(values: np.ndarray, start: int, stop: int, k: int) -> list[int]:
    """The `k` largest series of a level, by mean level -- the legible ones to draw."""
    means = values[:, start:stop].mean(axis=0)
    return [start + int(i) for i in np.argsort(means)[::-1][:k]]


def step_ticks(steps: np.ndarray) -> np.ndarray:
    """Horizon ticks thinned so they stay legible; police runs 30 steps wide."""
    stride = 1 if steps.size <= 12 else max(1, steps.size // 10)
    return steps[::stride]


def pretty_name(name: str) -> str:
    """Display form of a series name; labour's columns are stringified lists."""
    if name.startswith("[") and name.endswith("]"):
        parts = [part.strip().strip("'\"") for part in name[1:-1].split(",")]
        return " / ".join(part for part in parts if part)
    return name


def draw_bands(ax, future_x, draws):
    """The 50/80/95% bands: a fan over the horizon, a candle when h == 1."""
    for mass, alpha in BANDS:
        lo, hi = band_quantiles(draws, mass)
        label = f"{int(mass * 100)}% interval"
        if future_x.size == 1:
            # fill_between spans nothing across a single x; draw the bar instead.
            ax.vlines(future_x, lo, hi, color="#1f77b4", alpha=alpha, lw=18, label=label)
        else:
            ax.fill_between(
                future_x, lo, hi, color="#1f77b4", alpha=alpha, lw=0, label=label
            )


def plot_fan_charts(
    data, samples, y_true, t0, h, out_path, n_cols=3, context=None, history=True,
    label="Gaussian factor model",
):
    """A fan chart per drawn series: predictive bands, the median, and the actual.

    With `history` the forecast sits against the steps leading up to it, which
    shows whether it continues the series sensibly. Without it the axes scale to
    the test window alone, which is the only way to read how far the actual sits
    from the median when the horizon is short or the bands are tight.
    """
    levels = list(data.level_slices)
    context = context or max(16, 4 * h)
    fig, axes = plt.subplots(
        len(levels), n_cols, figsize=(4.6 * n_cols, 3.1 * len(levels)), squeeze=False
    )
    hist_start = max(0, t0 - context)
    hist_x = np.arange(hist_start - t0, 0)
    # Without history the horizon reads as steps 1..h rather than offsets from t0.
    future_x = np.arange(h) if history else np.arange(1, h + 1)

    for row, level in enumerate(levels):
        start, stop = data.level_slices[level]
        picks = top_series(data.values, start, stop, n_cols)
        for col in range(n_cols):
            ax = axes[row][col]
            if col >= len(picks):
                ax.axis("off")
                continue
            s = picks[col]
            if history:
                ax.plot(
                    hist_x,
                    data.values[hist_start:t0, s],
                    color="#2b2b2b",
                    lw=1.3,
                    label="history",
                )
            draw_bands(ax, future_x, samples[s])
            ax.plot(
                future_x,
                np.median(samples[s], axis=-1),
                color="#1f77b4",
                lw=1.8,
                marker="o",
                ms=4 if not history else 3,
                label="median forecast",
            )
            ax.plot(
                future_x,
                y_true[s],
                color="#d62728",
                lw=1.6,
                ls="--",
                marker="s",
                ms=4 if not history else 3,
                label="actual",
            )
            if history:
                ax.axvline(-0.5, color="#999999", lw=0.8, ls=":")
            else:
                ax.set_xticks(step_ticks(future_x))
                ax.set_xlim(0.5, h + 0.5)
                ax.grid(alpha=0.2)
            ax.set_title(pretty_name(data.series_names[s]), fontsize=9)
            ax.tick_params(labelsize=8)
            if col == 0:
                ax.set_ylabel(level, fontsize=9)
            if row == len(levels) - 1:
                ax.set_xlabel(
                    "steps from forecast origin" if history else "horizon step",
                    fontsize=8,
                )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.012),
    )
    scope = "history and test window" if history else "test window only"
    fig.suptitle(
        f"{label} -- {scope}, "
        f"{data.dates[t0].date()} to {data.dates[t0 + h - 1].date()} "
        f"({data.n_series} series over {len(levels)} levels)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_detail(data, samples, y_true, t0, h, out_path, n_cols=3):
    """The predictive distribution on its own: a violin per horizon step, no history.

    The fan chart puts the forecast in context, which buries it when the horizon
    is short next to a long history -- traffic forecasts one step against 56. This
    view drops the history entirely and scales each panel to the forecast alone.
    """
    levels = list(data.level_slices)
    fig, axes = plt.subplots(
        len(levels), n_cols, figsize=(4.8 * n_cols, 3.4 * len(levels)), squeeze=False
    )
    positions = np.arange(1, h + 1)

    for row, level in enumerate(levels):
        start, stop = data.level_slices[level]
        picks = top_series(data.values, start, stop, n_cols)
        for col in range(n_cols):
            ax = axes[row][col]
            if col >= len(picks):
                ax.axis("off")
                continue
            s = picks[col]
            parts = ax.violinplot(
                [samples[s, k] for k in range(h)],
                positions=positions,
                widths=0.7,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor("#1f77b4")
                body.set_edgecolor("#1f77b4")
                body.set_alpha(0.30)
            for mass, lw in ((0.95, 1.2), (0.80, 3.0), (0.50, 6.0)):
                lo, hi = band_quantiles(samples[s], mass)
                ax.vlines(positions, lo, hi, color="#1f77b4", lw=lw, alpha=0.85)
            ax.plot(
                positions,
                np.median(samples[s], axis=-1),
                ls="none",
                marker="o",
                ms=5,
                mfc="white",
                mec="#1f77b4",
                mew=1.6,
                label="median forecast",
            )
            ax.plot(
                positions,
                y_true[s],
                ls="none",
                marker="D",
                ms=6,
                color="#d62728",
                label="actual",
            )
            if h <= 4:
                # Where the actual fell in its own predictive sample, per step.
                for k in positions:
                    pct = (samples[s, k - 1] < y_true[s, k - 1]).mean()
                    ax.annotate(
                        f"{pct:.0%}",
                        xy=(k, y_true[s, k - 1]),
                        xytext=(9, -3),
                        textcoords="offset points",
                        fontsize=7,
                        color="#d62728",
                    )
            ax.set_title(pretty_name(data.series_names[s]), fontsize=9)
            ax.set_xticks(step_ticks(positions))
            ax.set_xlim(0.4, h + 0.6)
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", alpha=0.2)
            if col == 0:
                ax.set_ylabel(level, fontsize=9)
            if row == len(levels) - 1:
                ax.set_xlabel("horizon step", fontsize=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.012),
    )
    subtitle = (
        "violin is the full predictive sample; bars are the 50/80/95% intervals"
        + (";  % marks where the actual fell in that sample" if h <= 4 else "")
    )
    fig.suptitle(
        f"Predictive distribution -- test window "
        f"{data.dates[t0].date()} to {data.dates[t0 + h - 1].date()}\n{subtitle}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scrps(scores, y_true, samples, level_slices, out_path):
    """sCRPS as a bar per level, and as a curve over the horizon steps."""
    per_loc = per_location_crps(y_true, samples)
    levels = [name for name in scores if name != "Overall"]
    h = y_true.shape[1]
    # With a one-step horizon the second panel would just restate the first.
    if h == 1:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax2 = None
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4))

    names = ["Overall", *levels]
    values = [scores[name] for name in names]
    colors = ["#333333"] + ["#1f77b4"] * len(levels)
    bars = ax1.bar(range(len(names)), values, color=colors)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.replace("/", "/\n") for n in names], fontsize=8)
    ax1.set_ylabel("sCRPS (lower is better)")
    ax1.set_title("sCRPS by level")
    for bar, value in zip(bars, values, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax1.margins(y=0.15)

    if ax2 is None:
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    steps = np.arange(1, h + 1)
    for level in levels:
        start, stop = level_slices[level]
        numerator = per_loc[start:stop].sum(axis=0)
        denominator = np.abs(y_true[start:stop]).sum(axis=0)
        ax2.plot(steps, numerator / denominator, marker="o", lw=1.6, label=level)
    ax2.plot(
        steps,
        per_loc.sum(axis=0) / np.abs(y_true).sum(axis=0),
        marker="s",
        lw=2.2,
        color="#333333",
        label="Overall",
    )
    ax2.set_xticks(step_ticks(steps))
    ax2.set_xlabel("horizon step")
    ax2.set_ylabel("sCRPS")
    ax2.set_title("sCRPS by horizon step")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration(y_true, samples, level_slices, out_path):
    """Empirical coverage against the nominal one: on the diagonal is calibrated."""
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1, label="perfect")
    for level, (start, stop) in level_slices.items():
        empirical = [
            coverage(y_true[start:stop], samples[start:stop], mass).mean()
            for mass in NOMINAL
        ]
        ax.plot(NOMINAL, empirical, marker="o", ms=4, lw=1.5, label=level)
    ax.plot(
        NOMINAL,
        [coverage(y_true, samples, mass).mean() for mass in NOMINAL],
        marker="s",
        ms=5,
        lw=2.2,
        color="#333333",
        label="Overall",
    )
    ax.set_xlabel("nominal central-interval coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Interval calibration on the test window")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pit(y_true, samples, level_slices, out_path):
    """PIT histograms: flat is calibrated, U-shaped means the bands are too narrow."""
    panels = [("Overall", (0, y_true.shape[0]))] + [
        (level, bounds) for level, bounds in level_slices.items()
    ]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.1 * len(panels), 3.1), squeeze=False
    )
    for ax, (name, (start, stop)) in zip(axes[0], panels, strict=True):
        u = pit(y_true[start:stop], samples[start:stop])
        ax.hist(
            u,
            bins=10,
            range=(0, 1),
            color="#1f77b4",
            weights=np.full(u.size, 1.0 / u.size),
            edgecolor="white",
        )
        ax.axhline(0.1, color="#d62728", ls="--", lw=1.2)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("PIT", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, max(0.25, ax.get_ylim()[1]))
    axes[0][0].set_ylabel("frequency", fontsize=8)
    fig.suptitle(
        "PIT of the actuals -- flat at the dashed line is calibrated, "
        "a U-shape means the predictive bands are too narrow",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training(history, out_path):
    """Per-step training CRPS on the left axis, validation sCRPS on the right."""
    fig, ax1 = plt.subplots(figsize=(7.5, 3.8))
    losses = np.asarray(history.train_losses, dtype=float)
    ax1.plot(
        np.arange(1, losses.size + 1),
        losses,
        color="#c6dbef",
        lw=0.8,
        label="train CRPS (step)",
    )
    window = 25
    if losses.size >= window:
        smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax1.plot(
            np.arange(window, losses.size + 1),
            smooth,
            color="#1f77b4",
            lw=1.8,
            label=f"train CRPS ({window}-step mean)",
        )
    ax1.set_xlabel("step")
    ax1.set_ylabel("training loss")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    if history.val_scrps:
        steps, values = zip(*history.val_scrps, strict=True)
        ax2.plot(
            steps,
            values,
            color="#d62728",
            marker="o",
            ms=4,
            lw=1.6,
            label="validation sCRPS",
        )
        ax2.axvline(history.best_step, color="#d62728", ls=":", lw=1.2)
        ax2.annotate(
            f"best {history.best_val:.4f} @ step {history.best_step}",
            xy=(history.best_step, history.best_val),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8,
            color="#d62728",
            bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1.5},
        )
    ax2.set_ylabel("validation sCRPS", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # The axvline is a Line2D too; matplotlib labels it "_child1", so drop it.
    lines = [
        line
        for line in ax1.get_lines() + ax2.get_lines()
        if not line.get_label().startswith("_")
    ]
    ax1.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="best")
    ax1.set_title("Training")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_quantiles(data, samples, y_true, t0, path) -> pd.DataFrame:
    """Tidy one row per (series, horizon step) with the forecast quantiles."""
    level_of = {
        series: level
        for level, (start, stop) in data.level_slices.items()
        for series in range(start, stop)
    }
    rows = []
    for s, name in enumerate(data.series_names):
        for k in range(y_true.shape[1]):
            draws = samples[s, k]
            row = {
                "series": name,
                "level": level_of[s],
                "date": data.dates[t0 + k].date(),
                "step": k + 1,
                "actual": y_true[s, k],
                "mean": float(draws.mean()),
                "median": float(np.median(draws)),
            }
            for q in (0.025, 0.10, 0.25, 0.75, 0.90, 0.975):
                row[f"q{q:.3f}"] = float(np.quantile(draws, q))
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="tourism_small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE"
    )
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        help="history steps drawn in the fan charts; default max(16, 4h). A short "
        "horizon against a long history is unreadable -- shrink it to zoom in.",
    )
    args = parser.parse_args(argv)

    config, data, history, samples, y_true, t0, scores = fit_and_forecast(
        args.dataset, args.seed, args.overrides, args.device
    )
    out_dir = os.path.join(args.out, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    def path(name: str) -> str:
        return os.path.join(out_dir, name)

    h = config.data.h
    plot_fan_charts(
        data, samples, y_true, t0, h, path("fan_charts.png"), context=args.context
    )
    plot_fan_charts(
        data, samples, y_true, t0, h, path("fan_charts_test.png"), history=False
    )
    plot_prediction_detail(data, samples, y_true, t0, h, path("prediction_detail.png"))
    plot_scrps(scores, y_true, samples, data.level_slices, path("scrps_by_level.png"))
    plot_calibration(y_true, samples, data.level_slices, path("calibration.png"))
    plot_pit(y_true, samples, data.level_slices, path("pit_histogram.png"))
    plot_training(history, path("training_curve.png"))
    write_quantiles(data, samples, y_true, t0, path("forecast_quantiles.csv"))
    np.savez_compressed(
        path("forecast_samples.npz"),
        samples=samples.astype(np.float32),
        y_true=y_true,
        series_names=np.array(data.series_names),
        dates=np.array([str(d.date()) for d in data.dates[t0 : t0 + config.data.h]]),
    )

    print(f"\n{args.dataset} seed {args.seed}: Gaussian factor model, test sCRPS")
    for name, value in scores.items():
        print(f"  {name:<32} {value:.4f}")
    print("\ncentral-interval coverage (nominal -> empirical)")
    for mass in (0.5, 0.8, 0.95):
        print(f"  {mass:.0%} -> {coverage(y_true, samples, mass).mean():.1%}")
    print(f"\nwrote {len(os.listdir(out_dir))} files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
