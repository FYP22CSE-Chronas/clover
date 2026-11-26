from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# A Labour-shaped hierarchy (Total / Region / Region x Gender / x Status) whose
# dependence is known by construction:
#   * each Region x Gender group total E_g is a log random walk with drift, driven by
#     a national shock whose loading fades from 1 to FADE_TO over the sample, so the
#     cross-group innovation correlation drifts from strong to near zero;
#   * the group splits into full-time / part-time through a share p_g whose logit is a
#     slow AR(1). A share shock moves the two siblings in opposite directions, so
#     sibling innovations are negatively correlated while their sum stays smooth:
#     FT - p/(1-p) PT is the stationary (cointegrating) relation within each group.
# This is the structure measured on Labour: siblings about -0.36, other pairs fading
# from about +0.17 to +0.02.

N_REGIONS = 4
GENDERS = ("Females", "Males")
STATUSES = ("Employed full-time", "Employed part-time")
N_STEPS = 480
DRIFT = 0.0015
COMMON_SIGMA = 0.004
FADE_TO = 0.2
GROUP_SIGMA = 0.002
SHARE_SIGMA = 0.015  # logit scale; times p(1-p) ~ 0.2 gives ~0.003 on the share
SHARE_PHI = 0.97
NOISE_SIGMA = 0.002


def generate(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    t = np.arange(N_STEPS)
    loading = 1.0 + (FADE_TO - 1.0) * t / (N_STEPS - 1)
    common = COMMON_SIGMA * loading * rng.normal(size=N_STEPS)

    names, columns = [], []
    for r in range(N_REGIONS):
        for gender in GENDERS:
            level0 = np.log(rng.uniform(200, 1500))
            shocks = DRIFT + common + GROUP_SIGMA * rng.normal(size=N_STEPS)
            group = np.exp(level0 + np.cumsum(shocks))
            mean_logit = rng.uniform(0.5, 1.2)
            u = np.zeros(N_STEPS)
            for k in range(1, N_STEPS):
                u[k] = SHARE_PHI * u[k - 1] + SHARE_SIGMA * rng.normal()
            share = 1.0 / (1.0 + np.exp(-(mean_logit + u)))
            for status, part in zip(STATUSES, (share, 1.0 - share), strict=True):
                noise = np.exp(NOISE_SIGMA * rng.normal(size=N_STEPS))
                columns.append(group * part * noise)
                names.append(str([status, gender, f"Region{r}"]))
    bottom = np.stack(columns, axis=1)

    n_bottom = bottom.shape[1]
    rows, row_names = [np.ones(n_bottom)], [str(["Total"])]
    per_region = len(GENDERS) * len(STATUSES)
    for r in range(N_REGIONS):
        row = np.zeros(n_bottom)
        row[r * per_region : (r + 1) * per_region] = 1
        rows.append(row)
        row_names.append(str([f"Region{r}"]))
    for g in range(N_REGIONS * len(GENDERS)):
        row = np.zeros(n_bottom)
        row[g * 2 : g * 2 + 2] = 1
        rows.append(row)
        row_names.append(str([GENDERS[g % 2], f"Region{g // 2}"]))
    S = np.vstack([np.stack(rows), np.eye(n_bottom)])
    series = row_names + names
    dates = pd.date_range("1980-01-01", periods=N_STEPS, freq="MS")
    data = pd.DataFrame(bottom @ S.T, index=dates, columns=series)
    agg = pd.DataFrame(S, index=series, columns=names)
    return data, agg


def main(argv: list[str] | None = None) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Write the synthetic_ric CSV pair.")
    parser.add_argument("--out", default=os.path.join(here, "data", "synthetic_ric"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    data, agg = generate(args.seed)
    os.makedirs(args.out, exist_ok=True)
    data.to_csv(os.path.join(args.out, "data.csv"), float_format="%.4f")
    agg.to_csv(os.path.join(args.out, "agg_mat.csv"), float_format="%.0f")
    print(f"wrote {data.shape} to {args.out}")


if __name__ == "__main__":
    main()
