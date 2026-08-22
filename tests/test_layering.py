"""The dependency rule: model/ and data/ never import each other."""

from __future__ import annotations

import ast
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Leaf modules every layer may import.
LEAVES = ("config", "registry", "contracts")

FORBIDDEN = {
    "model": ("data", "pipeline", "training", "pandas"),
    "training": ("data", "pipeline"),
    "data": ("model", "training", "pipeline", "torch"),
}


def _modules(layer: str) -> list[str]:
    return [
        os.path.join(dirpath, name)
        for dirpath, dirnames, files in os.walk(os.path.join(ROOT, layer))
        if "__pycache__" not in dirpath
        for name in files
        if name.endswith(".py")
    ]


def _imported_roots(path: str) -> set[str]:
    """Top-level module name of every import in `path`."""
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_layer_does_not_import_forbidden_modules(layer: str) -> None:
    violations = []
    for path in _modules(layer):
        for name in _imported_roots(path) & set(FORBIDDEN[layer]):
            violations.append(f"{os.path.relpath(path, ROOT)} imports {name}")
    assert not violations, "\n".join(violations)


def test_leaf_modules_import_no_layer() -> None:
    layers = set(FORBIDDEN) | {"pipeline"}
    for leaf in LEAVES:
        path = os.path.join(ROOT, f"{leaf}.py")
        assert not (_imported_roots(path) & layers), f"{leaf}.py imports a layer"


def test_pipeline_may_import_both() -> None:
    from pipeline import runner

    assert runner.CLOVER is not None
    assert runner.load is not None
