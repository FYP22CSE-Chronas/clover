from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from pipeline.results import RunResult


def results_table(results: Sequence[RunResult]) -> pd.DataFrame:
    """Mean and std of sCRPS per level across seeds."""
    if not results:
        raise ValueError("no results to report")
    levels = list(results[0].scrps)
    values = np.array([[r.scrps[level] for level in levels] for r in results])
    table = pd.DataFrame(
        {"mean": values.mean(axis=0), "std": values.std(axis=0, ddof=0)}, index=levels
    )
    table.index.name = "level"
    return table


def write_results(
    results: Sequence[RunResult], out_dir: str, dataset: str | None = None
) -> tuple[str, list[str]]:
    """Write the seed-aggregated CSV and one JSON per seed; return the paths."""
    name = dataset or results[0].dataset
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}_scrps.csv")
    results_table(results).to_csv(csv_path)

    json_paths = []
    for result in results:
        path = os.path.join(out_dir, f"{name}_seed{result.seed}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(result.summary(), handle, indent=2)
        json_paths.append(path)
    return csv_path, json_paths


def format_table(table: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Render a results table for the terminal."""
    return table.to_string(float_format=lambda v: float_format.format(v))


def combined_table(groups: Iterable[Sequence[RunResult]]) -> pd.DataFrame:
    """Stack several datasets into one frame indexed by (dataset, level)."""
    frames = []
    for results in groups:
        table = results_table(results)
        table.insert(0, "dataset", results[0].dataset)
        frames.append(table.reset_index().set_index(["dataset", "level"]))
    return pd.concat(frames)
