from __future__ import annotations

import ast
from pathlib import Path

from witwin.channel_native.core import path_topology as legacy
from witwin.channel_native.deterministic import solver as deterministic_solver
from witwin.channel_native.montecarlo.bdpt import solver as bdpt_solver
from witwin.channel_native.path import solver as path_solver
from witwin.channel_native.propagation.enumerated import engine, scattering


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


def test_path_and_deterministic_consume_typed_engine_and_scattering_directly():
    for solver in (path_solver, deterministic_solver):
        assert solver.evaluate_enumerated_paths is engine.evaluate_enumerated_paths
        assert (
            solver.append_scattering_evaluated_paths
            is scattering.append_scattering_evaluated_paths
        )
        assert not hasattr(solver, "export_topology")
        assert not hasattr(solver, "export_evaluated_rows")


def test_canonical_engine_has_no_legacy_batch_or_adapter_dependency():
    tree = ast.parse(Path(engine.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "evaluate_enumerated_paths"
    )

    assert "witwin.channel_native.core.path_topology" not in imported_modules
    assert "witwin.channel_native.propagation.models.adapters" not in imported_modules
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TopologyBatch"
        for node in ast.walk(definition)
    )


def test_typed_engine_remains_before_optional_scattering_append():
    path_solve = _function(path_solver, "_solve_base")
    deterministic_solve = _function(deterministic_solver, "solve")

    for definition in (path_solve, deterministic_solve):
        assert (
            _call_line(definition, "evaluate_enumerated_paths")
            < _call_line(definition, "append_scattering_evaluated_paths")
        )
        call_names = {
            node.func.id
            for node in ast.walk(definition)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "export_topology" not in call_names
        assert "export_evaluated_rows" not in call_names


def test_bdpt_keeps_the_legacy_compatibility_export_seam():
    assert bdpt_solver.export_topology is legacy.export_topology
    tree = ast.parse(Path(bdpt_solver.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "witwin.channel_native.core.path_topology" in imported_modules
    assert "witwin.channel_native.propagation.enumerated.engine" not in imported_modules
