from __future__ import annotations

import ast
from pathlib import Path

import torch

from tests.path.test_path_evaluated_paths import _evaluated_paths_fixture
from witwin.channel import deterministic as deterministic_module
from witwin.channel.montecarlo.bdpt import pipeline as bdpt_pipeline
from witwin.channel.montecarlo.bdpt import solver as bdpt_solver
from witwin.channel import path as path_module
from witwin.channel.propagation.enumerated import engine, scattering


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
    for solver in (path_module, deterministic_module):
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

    assert "witwin.channel.core.path_topology" not in imported_modules
    assert "witwin.channel.propagation.models.adapters" not in imported_modules
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TopologyBatch"
        for node in ast.walk(definition)
    )


def test_typed_engine_remains_before_optional_scattering_append():
    # ADR-021 budget refactor: the deterministic pipeline now routes the
    # optional scattering append through a single-purpose ``_append_scattering``
    # stage (it wraps ``append_scattering_evaluated_paths`` and the coherent
    # combine gate), while the path pipeline still calls the append inline. In
    # both, the typed engine (``evaluate_enumerated_paths``) still runs before
    # the optional scattering append, and neither uses the legacy exports.
    cases = (
        (
            _function(path_module, "_pipeline_solve_base"),
            "append_scattering_evaluated_paths",
        ),
        (_function(deterministic_module, "_solve_pipeline"), "_append_scattering"),
    )
    for definition, append_call in cases:
        assert _call_line(definition, "evaluate_enumerated_paths") < _call_line(
            definition, append_call
        )
        call_names = {
            node.func.id
            for node in ast.walk(definition)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "export_topology" not in call_names
        assert "export_evaluated_rows" not in call_names


def test_capacity_sanitize_and_terminal_boundaries_follow_outer_result_work():
    path_base = _function(path_module, "_pipeline_solve_base")
    assert _call_line(path_base, "append_scattering_evaluated_paths") < _call_line(
        path_base, "sanitize_enumerated_capacity_transaction"
    )
    assert _call_line(
        path_base, "sanitize_enumerated_capacity_transaction"
    ) < _call_line(path_base, "compact_evaluated_paths")
    assert _call_line(path_base, "compact_evaluated_paths") < _call_line(
        path_base, "pack_evaluated_paths"
    )

    path_solve = _function(path_module, "_pipeline_solve")
    path_calls = sorted(
        [
            node
            for node in ast.walk(path_solve)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "terminal_check"
        ],
        key=lambda node: node.lineno,
    )
    assert len(path_calls) == 2
    assert _call_line(path_solve, "pack_explicit_arrays") < path_calls[0].lineno
    assert _call_line(path_solve, "pack_synthetic_arrays") < path_calls[1].lineno

    deterministic = _function(deterministic_module, "_solve_pipeline")
    deterministic_terminal = _function(
        deterministic_module, "_terminal_check_capacity"
    )
    assert _call_line(deterministic, "_append_scattering") < _call_line(
        deterministic, "sanitize_enumerated_capacity_transaction"
    )
    assert _call_line(
        deterministic, "sanitize_enumerated_capacity_transaction"
    ) < _call_line(deterministic, "accumulate_path_result")
    assert _call_line(deterministic, "Result") < _call_line(
        deterministic, "_terminal_check_capacity"
    )
    assert _call_line(deterministic_terminal, "terminal_check") > 0


def test_bdpt_consumes_the_typed_engine_without_a_mixed_export():
    assert not hasattr(bdpt_solver, "export_topology")
    tree = ast.parse(Path(bdpt_pipeline.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "witwin.channel.propagation" in imported_modules
    assert "witwin.channel.propagation.enumerated.engine" not in imported_modules
    assert "witwin.channel.propagation.topology.export" not in imported_modules


def test_bdpt_connection_samples_read_typed_domains_exactly():
    paths, _ = _evaluated_paths_fixture()
    selected = torch.tensor([1, 4], dtype=torch.int64)

    samples = bdpt_pipeline._evaluated_connection_samples(
        paths, selected, component_out=2
    )

    assert samples is not None
    torch.testing.assert_close(samples["tx_id"], paths.topology.tx_id[selected])
    torch.testing.assert_close(samples["rx_id"], paths.topology.rx_id[selected])
    torch.testing.assert_close(
        samples["contribution"], paths.fields.path_gain[selected]
    )
    torch.testing.assert_close(
        samples["path_length_m"], paths.geometry.path_length_m[selected]
    )
    assert samples["component_id"].tolist() == [2, 2]
