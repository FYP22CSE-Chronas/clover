"""Wall-clock cost of RIC against CLOVER at a fixed step budget.

Early stopping is disabled and every variant trains for the same number of steps, so
the ratio is the per-step overhead of RIC's window correlation and eigendecomposition
(training, validation and test sampling included).

    python experiments/ric_clover/timing.py --datasets traffic labour --steps 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["traffic", "labour"])
    parser.add_argument("--variants", nargs="+", default=["base", "ric_tree"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    import pipeline.runner as runner
    from experiments.ric_clover.bench import configure, install_variant

    overrides = [
        f"train.max_steps={args.steps}",
        f"train.early_stop_patience={10**6}",
    ]
    # One untimed warm-up run so import and allocation costs fall outside the timings.
    warm = configure(args.datasets[0], None, [*overrides[:1], "train.max_steps=5"])
    install_variant(args.variants[0], warm)
    runner.run(warm, seed=0, verbose=False)
    for dataset in args.datasets:
        for seed in args.seeds:
            # Alternate the order so warm-up and cache effects don't favour one side.
            order = args.variants if seed % 2 == 0 else args.variants[::-1]
            for variant in order:
                config = configure(dataset, None, overrides)
                install_variant(variant, config)
                start = time.perf_counter()
                result = runner.run(config, seed=seed, verbose=False)
                seconds = time.perf_counter() - start
                print(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "variant": variant,
                            "seed": seed,
                            "steps_run": result.history.steps_run,
                            "seconds": round(seconds, 2),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
