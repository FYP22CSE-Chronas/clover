from __future__ import annotations

import argparse

import numpy as np

from config import load_config
from data import is_panel
from experiments.ec_clover.johansen import estimate_beta_johansen
from pipeline.runner import prepare


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate a Johansen cointegrating beta from TRAINING-split "
        "bottom-level data only, for use as ec.beta_init.values_path."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--det-order", type=int, default=0)
    parser.add_argument("--k-ar-diff", type=int, default=1)
    parser.add_argument(
        "--out", required=True, help="output .npy path for beta [Nb, rank]"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if is_panel(args.dataset):
        raise NotImplementedError(
            f"{args.dataset} is a panel dataset; this script only supports the "
            "single-hierarchy prepare() path used by pipeline.runner"
        )

    config = load_config(args.dataset)
    dataset, _, split = prepare(config)
    # Training windows' targets never reach past this row (see last_h_split /
    # block_split in data/windows.py), so slicing here matches exactly the rows the
    # training objective is ever scored against -- no validation or test leakage.
    train_end = max(w.t + len(w.target) for w in split.train)
    train_bottom = dataset.bottom[:train_end]

    beta = estimate_beta_johansen(
        train_bottom, rank=args.rank, det_order=args.det_order, k_ar_diff=args.k_ar_diff
    )
    np.save(args.out, beta)
    print(
        f"wrote beta {beta.shape} (from training rows 0:{train_end} of "
        f"{dataset.n_steps}) to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
