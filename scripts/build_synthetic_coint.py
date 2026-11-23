from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# A hierarchy whose bottom series are cointegrated by construction: each group of
# siblings shares one common stochastic trend, and every sibling's deviation from its
# share of that trend is a mean-reverting AR(1). Within a group of m series the
# cointegrating rank is therefore m - 1. The defaults put most of the variance in the
# deviations (DEVIATION_SIGMA >> TREND_SIGMA) with a half-life of ln(0.5)/ln(PHI) =
# 3.1 steps: a linear predictor given the true deviation cuts 12-step RMSE by about
# 13% over a seasonal-drift one, which is roughly the most an error-correction term
# can be worth on this design. A slower, weaker regime (PHI=0.95, DEVIATION_SIGMA=1,
# TREND_SIGMA=0.8) was tried first and gives no out-of-sample gain at all.

N_GROUPS = 4
GROUP_SIZE = 4
N_STEPS = 360
PHI = 0.8  # deviation persistence
TREND_SIGMA = 0.3  # common-trend innovation
TREND_DRIFT = 0.05
DEVIATION_SIGMA = 4.0
SEASONAL_AMPLITUDE = 2.0
NOISE_SIGMA = 0.3
BASE_LEVEL = 100.0


def generate(
    seed: int = 0,
    phi: float = PHI,
    deviation_sigma: float = DEVIATION_SIGMA,
    trend_sigma: float = TREND_SIGMA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_bottom = N_GROUPS * GROUP_SIZE
    months = np.arange(N_STEPS)
    bottom = np.zeros((N_STEPS, n_bottom))
    for g in range(N_GROUPS):
        trend = np.cumsum(TREND_DRIFT + trend_sigma * rng.normal(size=N_STEPS))
        phase = rng.uniform(0, 2 * np.pi)
        for j in range(GROUP_SIZE):
            n = g * GROUP_SIZE + j
            share = rng.uniform(0.6, 1.4)
            deviation = np.zeros(N_STEPS)
            for t in range(1, N_STEPS):
                deviation[t] = phi * deviation[t - 1] + deviation_sigma * rng.normal()
            seasonal = (
                SEASONAL_AMPLITUDE
                * rng.uniform(0.5, 1.5)
                * np.sin(2 * np.pi * months / 12 + phase)
            )
            bottom[:, n] = (
                BASE_LEVEL
                + 10.0 * j
                + share * trend
                + deviation
                + seasonal
                + NOISE_SIGMA * rng.normal(size=N_STEPS)
            )
    bottom = np.maximum(bottom, 1.0)

    bottom_names = [f"g{g}-s{j}" for g in range(N_GROUPS) for j in range(GROUP_SIZE)]
    group_names = [f"g{g}" for g in range(N_GROUPS)]
    A = np.zeros((1 + N_GROUPS, n_bottom))
    A[0] = 1.0
    for g in range(N_GROUPS):
        A[1 + g, g * GROUP_SIZE : (g + 1) * GROUP_SIZE] = 1.0
    S = np.vstack([A, np.eye(n_bottom)])
    series_names = ["total", *group_names, *bottom_names]
    dates = pd.date_range("1990-01-01", periods=N_STEPS, freq="MS")
    data = pd.DataFrame(bottom @ S.T, index=dates, columns=series_names)
    agg = pd.DataFrame(S, index=series_names, columns=bottom_names)
    return data, agg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the synthetic_coint CSV pair.")
    parser.add_argument("--out", default=os.path.join("data", "raw", "synthetic_coint"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phi", type=float, default=PHI)
    parser.add_argument("--deviation-sigma", type=float, default=DEVIATION_SIGMA)
    parser.add_argument("--trend-sigma", type=float, default=TREND_SIGMA)
    args = parser.parse_args(argv)
    data, agg = generate(args.seed, args.phi, args.deviation_sigma, args.trend_sigma)
    os.makedirs(args.out, exist_ok=True)
    data.to_csv(os.path.join(args.out, "data.csv"), float_format="%.4f")
    agg.to_csv(os.path.join(args.out, "agg_mat.csv"), float_format="%.0f")
    print(f"wrote {data.shape} to {args.out}")


if __name__ == "__main__":
    main()
