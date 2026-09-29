from __future__ import annotations

import argparse

import numpy as np

from config import available_datasets, load_config
from experiments.ec_clover.config import load_ec_config
from experiments.ec_clover.run import run
from pipeline.reporting import format_table, results_table, write_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ec_clover", description="Train and score EC-CLOVER on one dataset."
    )
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument(
        "--ec-config",
        default=None,
        help="path to a YAML file with an 'ec:' section (see experiments/ec_clover/"
        "configs/); omitted means the EC branch is disabled and this reproduces "
        "baseline CLOVER",
    )
    parser.add_argument(
        "--beta-init",
        default=None,
        help="optional .npy override for beta [Nb, rank], taking precedence over "
        "ec.beta_init.values/values_path in the ec config",
    )
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
        help="typed dotted override against the core config, e.g. "
        "--set train.learning_rate=1e-3 (does not accept ec.* -- edit --ec-config)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run every requested seed and write the aggregated results."""
    args = build_parser().parse_args(argv)
    config = load_config(args.dataset, overrides=args.overrides)
    ec_config = load_ec_config(args.ec_config)
    beta_init = np.load(args.beta_init) if args.beta_init else None

    results = []
    for seed in args.seeds:
        checkpoint_dir = (
            f"{args.checkpoint_dir}/{args.dataset}_ec_seed{seed}"
            if args.checkpoint_dir
            else None
        )
        results.append(
            run(
                config,
                ec_config,
                seed=seed,
                device=args.device,
                checkpoint_dir=checkpoint_dir,
                verbose=not args.quiet,
                beta_init=beta_init,
            )
        )

    csv_path, _ = write_results(results, args.out, dataset=f"{args.dataset}_ec")
    print(f"\n{args.dataset} (EC-CLOVER): sCRPS over {len(results)} seed(s)")
    print(format_table(results_table(results)))
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
