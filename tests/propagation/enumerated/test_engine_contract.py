from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import os
from pathlib import Path
import subprocess
import sys

import torch

from tests.propagation.test_topology_batch_adapter import (
    _GEOMETRY_FIELDS,
    _PATH_FIELDS,
    _TOPOLOGY_FIELDS,
    _assert_exact_tensor,
    _batch,
)
from witwin.channel_native.core import path_topology as legacy
from witwin.channel_native.montecarlo.bdpt import solver as bdpt_solver
from witwin.channel_native.propagation import enumerated
from witwin.channel_native.propagation.enumerated import engine
from witwin.channel_native.propagation.topology import export


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_COMPONENT_STAGES = (
    "_los_topology",
    "_reflection_topology_order1",
    "_reflection_topology_multibounce",
    "_diffraction_topology_order1",
    "_transmission_topology",
    "_coupled_reflection_diffraction_topology_order2",
)


def _function(module, name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _named_call_lines(definition: ast.FunctionDef) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for node in ast.walk(definition):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        result.setdefault(name, []).append(node.lineno)
    return result


def _typed_tensors(paths) -> dict[str, torch.Tensor]:
    return {
        name: getattr(contract, name)
        for contract, names in (
            (paths.topology, _TOPOLOGY_FIELDS),
            (paths.geometry, _GEOMETRY_FIELDS),
            (paths.fields, _PATH_FIELDS),
        )
        for name in names
    }


def test_engine_signature_ownership_and_dependency_boundary():
    signature = inspect.signature(engine.evaluate_enumerated_paths)
    tree = ast.parse(Path(engine.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert list(signature.parameters) == ["scene", "config", "frequency_value"]
    assert signature.parameters["frequency_value"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["frequency_value"].default is None
    assert signature.return_annotation == (
        "tuple[EvaluatedPaths, EvaluatedPathSidecars]"
    )
    assert "witwin.channel_native.core.path_topology" not in imported_modules
    assert "TopologyBatch" not in Path(engine.__file__).read_text(encoding="utf-8")
    assert "evaluate_enumerated_paths" not in enumerated.__all__


def test_engine_preserves_component_order_los_fast_path_and_field_calls():
    definition = _function(engine, "evaluate_enumerated_paths")
    calls = _named_call_lines(definition)
    stage_lines = [calls[name][0] for name in _COMPONENT_STAGES]
    fast_path = next(
        node
        for node in definition.body
        if isinstance(node, ast.If)
        and "components == {'los'}" in ast.unparse(node.test)
    )
    fast_calls = _named_call_lines(fast_path)

    assert stage_lines == sorted(stage_lines)
    assert max(stage_lines) < fast_path.lineno
    assert calls["concatenate_path_blocks"][0] < calls["evaluated_paths_from_block"][0]
    assert calls["evaluated_paths_from_block"][0] < max(
        calls["evaluate_path_fields"]
    )
    assert fast_calls["evaluated_paths_from_result"][0] < fast_calls[
        "evaluate_path_fields"
    ][0]
    assert calls["evaluated_paths_from_result"][0] < calls[
        "concatenate_path_blocks"
    ][0]


def test_typed_result_and_legacy_pack_are_tensor_and_sidecar_exact(monkeypatch):
    source = _batch()
    evaluated, sidecars = export.evaluated_paths_from_result(source)
    sidecars = replace(
        sidecars,
        execution=export.PathExecutionStats(
            launch_count=source.launch_count,
            visibility_rejection_count=source.visibility_rejection_count,
            selected_edge_count=source.selected_edge_count,
            candidate_count=source.candidate_count,
            guardrail_count=source.guardrail_count,
            ad_companion_launches=source.ad_companion_launches,
            ad_tape_bytes=source.ad_tape_bytes,
        ),
        diffraction_vector_field=source.diffraction_vector_field,
    )
    monkeypatch.setattr(
        legacy,
        "evaluate_enumerated_paths",
        lambda *_args, **_kwargs: (evaluated, sidecars),
    )

    packed = legacy.export_topology(object(), object())
    for name, tensor in _typed_tensors(evaluated).items():
        _assert_exact_tensor(getattr(packed, name), tensor)
    assert packed.launch_count == sidecars.execution.launch_count
    assert packed.visibility_rejection_count == (
        sidecars.execution.visibility_rejection_count
    )
    assert packed.selected_edge_count == sidecars.execution.selected_edge_count
    assert packed.candidate_count == sidecars.execution.candidate_count
    assert packed.guardrail_count == sidecars.execution.guardrail_count
    assert packed.ad_companion_launches == sidecars.execution.ad_companion_launches
    assert packed.ad_tape_bytes == sidecars.execution.ad_tape_bytes
    assert packed.diffraction_vector_field is sidecars.diffraction_vector_field


def test_typed_result_constructor_matches_legacy_normalization_exactly():
    source = _batch()
    evaluated, sidecars = export.evaluated_paths_from_result(source)
    packed = legacy._from_path_result(source)

    for name, tensor in _typed_tensors(evaluated).items():
        observed = getattr(packed, name)
        assert torch.equal(observed, tensor)
        assert observed.stride() == tensor.stride()
        assert observed.dtype == tensor.dtype
        assert observed.device == tensor.device
        assert observed.requires_grad == tensor.requires_grad
    assert sidecars.execution.launch_count == packed.launch_count
    assert sidecars.execution.visibility_rejection_count == (
        packed.visibility_rejection_count
    )
    assert sidecars.execution.selected_edge_count == packed.selected_edge_count
    assert sidecars.execution.candidate_count == packed.candidate_count
    assert sidecars.execution.guardrail_count == packed.guardrail_count
    assert sidecars.execution.ad_companion_launches == packed.ad_companion_launches == 0
    assert sidecars.execution.ad_tape_bytes == packed.ad_tape_bytes == 0
    assert sidecars.diffraction_vector_field is packed.diffraction_vector_field is None


def test_typed_block_constructor_preserves_select_gather_result_order():
    definition = _function(export, "evaluated_paths_from_block")
    calls = _named_call_lines(definition)

    assert calls["_canonical_selection_order"][0] < calls[
        "deterministic_gather_topology_block"
    ][0]
    assert calls["deterministic_gather_topology_block"][0] < calls[
        "evaluated_paths_from_result"
    ][0]


def test_core_path_topology_is_a_small_compatibility_facade():
    source = Path(legacy.__file__).read_text(encoding="utf-8")
    export_definition = _function(legacy, "export_topology")
    calls = _named_call_lines(export_definition)

    assert len(source.splitlines()) <= 300
    assert calls["evaluate_enumerated_paths"][0] < calls[
        "_topology_batch_from_evaluated"
    ][0]


def test_bdpt_keeps_legacy_export_identity():
    assert bdpt_solver.export_topology is legacy.export_topology


def test_fresh_import_order_preserves_engine_and_legacy_identity():
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
    )
    for imports in (
        (
            "from witwin.channel_native.propagation.enumerated import engine; "
            "from witwin.channel_native.core import path_topology as legacy"
        ),
        (
            "from witwin.channel_native.core import path_topology as legacy; "
            "from witwin.channel_native.propagation.enumerated import engine"
        ),
    ):
        subprocess.run(
            [
                sys.executable,
                "-c",
                f"{imports}; assert legacy.evaluate_enumerated_paths is "
                "engine.evaluate_enumerated_paths",
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
