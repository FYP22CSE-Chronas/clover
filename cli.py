from __future__ import annotations

import argparse
import os

import torch

from config import available_datasets, load_config
from data import is_panel
from pipeline.panel_runner import run_panel
from pipeline.reporting import format_table, results_table, write_results
from pipeline.runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clover", description="Train and score CLOVER on one dataset."
    )
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="typed dotted override, e.g. --set train.learning_rate=1e-3",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="torch intra-op threads; 0 leaves the default. The panel datasets run "
        "fastest well below the core count -- the model is small enough that thread "
        "overhead dominates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run every requested seed and write the aggregated results."""
    args = build_parser().parse_args(argv)
    config = load_config(args.dataset, overrides=args.overrides)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    runner = run_panel if is_panel(args.dataset) else run

    results = []
    for seed in args.seeds:
        checkpoint_dir = (
            os.path.join(args.checkpoint_dir, f"{args.dataset}_seed{seed}")
            if args.checkpoint_dir
            else None
        )
        results.append(
            runner(
                config,
                seed=seed,
                device=args.device,
                checkpoint_dir=checkpoint_dir,
                verbose=not args.quiet,
            )
        )

    csv_path, _ = write_results(results, args.out, dataset=args.dataset)
    print(f"\n{args.dataset}: sCRPS over {len(results)} seed(s)")
    print(format_table(results_table(results)))
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
