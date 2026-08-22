from __future__ import annotations

from collections.abc import Callable

from config import DataConfig
from data.csv_source import load_csv_hierarchy, verify_coherence
from data.dataset import ExogenousFeatures, HierarchicalDataset
from data.favorita import load_favorita
from data.features import FEATURE_REGISTRY, build_features, check_anchor_lags
from data.hierarchy import level_masks, level_tags, validate_level_slices
from data.panel import PanelDataset, PanelSplit, build_panel_split
from data.windows import SPLITS, Window, WindowSplit, build_split, rolling_windows

HierarchyLoader = Callable[[DataConfig], HierarchicalDataset]
PanelLoader = Callable[[DataConfig], PanelDataset]

# Datasets needing a loader other than the default CSV pair.
HIERARCHY_LOADERS: dict[str, HierarchyLoader] = {}
PANEL_LOADERS: dict[str, PanelLoader] = {"favorita": load_favorita}


def is_panel(name: str) -> bool:
    """True when `name` is a panel of hierarchies rather than a single one."""
    return name in PANEL_LOADERS


def load(name: str, config: DataConfig) -> HierarchicalDataset:
    """Load a single-hierarchy dataset, falling back to the CSV pair loader."""
    if is_panel(name):
        raise KeyError(f"{name!r} is a panel dataset; use load_panel instead")
    return HIERARCHY_LOADERS.get(name, load_csv_hierarchy)(config)


def load_panel(name: str, config: DataConfig) -> PanelDataset:
    """Load a panel dataset by name."""
    try:
        loader = PANEL_LOADERS[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a panel dataset; available: {sorted(PANEL_LOADERS)}"
        ) from None
    return loader(config)


__all__ = [
    "FEATURE_REGISTRY",
    "HIERARCHY_LOADERS",
    "PANEL_LOADERS",
    "SPLITS",
    "ExogenousFeatures",
    "HierarchicalDataset",
    "PanelDataset",
    "PanelSplit",
    "Window",
    "WindowSplit",
    "build_features",
    "build_panel_split",
    "build_split",
    "check_anchor_lags",
    "is_panel",
    "level_masks",
    "level_tags",
    "load",
    "load_csv_hierarchy",
    "load_favorita",
    "load_panel",
    "rolling_windows",
    "validate_level_slices",
    "verify_coherence",
]
