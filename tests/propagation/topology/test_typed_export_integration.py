from __future__ import annotations

import ast
from pathlib import Path

from witwin.channel_native.deterministic import solver as deterministic_solver
from witwin.channel_native.path import solver as path_solver
from witwin.channel_native.propagation.topology import export


def _function(module, name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_line(definition: ast.FunctionDef, name: str) -> int:
    return next(
        node.lineno
        for node in ast.walk(definition)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    )


def test_path_and_deterministic_consume_canonical_export_owner_directly():
    assert path_solver.export_evaluated_rows is export.export_evaluated_rows
    assert deterministic_solver.export_evaluated_rows is export.export_evaluated_rows
    assert not hasattr(path_solver, "evaluated_paths_from_topology_batch")
    assert not hasattr(deterministic_solver, "evaluated_paths_from_topology_batch")


def test_canonical_export_has_no_legacy_batch_or_adapter_dependency():
    tree = ast.parse(Path(export.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "export_evaluated_rows"
    )

    assert "witwin.channel_native.core.path_topology" not in imported_modules
    assert "witwin.channel_native.propagation.models.adapters" not in imported_modules
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TopologyBatch"
        for node in ast.walk(definition)
    )


def test_typed_export_remains_after_optional_scattering_append():
    path_solve = _function(path_solver, "_solve_base")
    deterministic_solve = _function(deterministic_solver, "solve")

    for definition in (path_solve, deterministic_solve):
        assert (
            _call_line(definition, "export_topology")
            < _call_line(definition, "append_scattering_paths")
            < _call_line(definition, "export_evaluated_rows")
        )
