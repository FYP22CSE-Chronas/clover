"""Test the Gaussian marginal assumption against each dataset's own residuals.

    python scripts/marginal_diagnostics.py
    python scripts/marginal_diagnostics.py --datasets labour traffic --out results/

CLOVER's factor model assumes the bottom-level innovation is Gaussian. Nothing in
the repository ever checked that, so this script supplies the model-free evidence
the assumption deserves -- or fails to.

The quantity tested is a *proxy* for the conditional predictive distribution, not
the distribution itself: a seasonal-naive residual `e_t = y_t - y_{t-m}`, divided
by its own series' standard deviation. It removes level and seasonality without
fitting anything, so it cannot smuggle a model's own distributional assumption
into the diagnostic. It is nevertheless a proxy -- a fitted model's conditional
residual would differ, and any remaining trend or heteroskedasticity inflates the
apparent departure from normality. Read the rejection rates as evidence that a
fixed Gaussian marginal is a poor description of these series, not as a measure of
what CLOVER's own residuals look like.

Per dataset it reports, over the bottom-level series and separately over the
aggregate rows:

    skewness, excess kurtosis     sample moments of the standardized residual
    D'Agostino K^2                omnibus normality test (needs n >= 20)
    rejection rate                share of series rejected at alpha, after a
                                  Benjamini-Hochberg correction across series
    zero share                    fraction of zero observations, since sparsity
                                  alone produces non-normality

Writes `marginal_diagnostics.csv` (one row per series) and prints the per-dataset
summary. Datasets with few rows give the test almost no power, so a low rejection
rate on a short panel is not evidence of normality; the printed `n_resid` column
is what makes that readable.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import available_datasets, load_config  # noqa: E402
from data import load  # noqa: E402

# Seasonal period per observation frequency, inferred from the dated index rather
# than hard-coded per dataset.
PERIOD_BY_FREQ = {"D": 7, "W": 52, "M": 12, "Q": 4, "Y": 1}
MIN_RESID = 20  # D'Agostino K^2 is undefined below this


def infer_period(index: pd.DatetimeIndex) -> tuple[int, str]:
    """Seasonal period and a frequency label from the median spacing of dates."""
    gap = np.median(np.diff(index.values).astype("timedelta64[D]").astype(int))
    if gap <= 1:
        return PERIOD_BY_FREQ["D"], "D"
    if gap <= 8:
        return PERIOD_BY_FREQ["W"], "W"
    if gap <= 45:
        return PERIOD_BY_FREQ["M"], "M"
    if gap <= 120:
        return PERIOD_BY_FREQ["Q"], "Q"
    return PERIOD_BY_FREQ["Y"], "Y"


def standardized_residuals(column: np.ndarray, period: int) -> np.ndarray:
    """`(y_t - y_{t-m}) / sd`, dropping the undefined head and any zero-variance case."""
    diff = column[period:] - column[:-period]
    sd = diff.std()
    if not np.isfinite(sd) or sd == 0.0:
        return np.empty(0)
    return diff / sd


def benjamini_hochberg(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Boolean rejections at FDR `alpha`; controls the false-discovery rate."""
    n = pvalues.size
    if n == 0:
        return np.empty(0, dtype=bool)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    threshold = alpha * np.arange(1, n + 1) / n
    passing = np.nonzero(ranked <= threshold)[0]
    rejected = np.zeros(n, dtype=bool)
    if passing.size:
        rejected[order[: passing[-1] + 1]] = True
    return rejected


def per_series_rows(dataset: str) -> list[dict]:
    """One diagnostic row per hierarchy series of one dataset."""
    config = load_config(dataset)
    data = load(config.name, config.data)
    values = np.asarray(data.values, dtype=np.float64)  # [T, Na+Nb]
    index = pd.to_datetime(pd.Index(data.dates))
    period, freq = infer_period(index)
    n_bottom = data.n_bottom
    n_series = data.n_series

    rows = []
    for i in range(n_series):
        column = values[:, i]
        resid = standardized_residuals(column, period)
        row = {
            "dataset": dataset,
            "series": data.series_names[i],
            "block": "bottom" if i >= n_series - n_bottom else "aggregate",
            "freq": freq,
            "period": period,
            "n_obs": column.size,
            "n_resid": resid.size,
            "zero_share": float((column == 0).mean()),
            "skew": float(stats.skew(resid)) if resid.size else np.nan,
            "excess_kurtosis": float(stats.kurtosis(resid)) if resid.size else np.nan,
            "dagostino_p": np.nan,
        }
        if resid.size >= MIN_RESID:
            row["dagostino_p"] = float(stats.normaltest(resid).pvalue)
        rows.append(row)
    return rows


def summarize(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Per (dataset, block) moments and the BH-corrected rejection rate."""
    out = []
    for (dataset, block), group in frame.groupby(["dataset", "block"], sort=False):
        tested = group.dropna(subset=["dagostino_p"])
        rejected = benjamini_hochberg(tested["dagostino_p"].to_numpy(), alpha)
        out.append(
            {
                "dataset": dataset,
                "block": block,
                "n_series": len(group),
                "n_tested": len(tested),
                "n_resid": int(group["n_resid"].median()),
                "median_skew": group["skew"].median(),
                "share_skew_pos": float((group["skew"] > 0).mean()),
                "median_excess_kurtosis": group["excess_kurtosis"].median(),
                "reject_rate": float(rejected.mean()) if len(tested) else np.nan,
                "median_zero_share": group["zero_share"].median(),
            }
        )
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)

    datasets = args.datasets
    if datasets is None:
        # Panels load through a different reader; this script handles the single
        # hierarchies only.
        datasets = [
            name
            for name in available_datasets()
            if os.path.exists(
                os.path.join(PROJECT_ROOT, "data", "raw", name, "data.csv")
            )
        ]

    rows: list[dict] = []
    for dataset in datasets:
        print(f"  {dataset} ...", flush=True)
        rows.extend(per_series_rows(dataset))
    frame = pd.DataFrame(rows)

    out_dir = args.out
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(PROJECT_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "marginal_diagnostics.csv")
    frame.to_csv(csv_path, index=False)

    summary = summarize(frame, args.alpha)
    summary_path = os.path.join(out_dir, "marginal_diagnostics_summary.csv")
    summary.to_csv(summary_path, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(
            "\nseasonal-naive standardized residuals, BH-corrected at "
            f"alpha={args.alpha}"
        )
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {csv_path}\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
