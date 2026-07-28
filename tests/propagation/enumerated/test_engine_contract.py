from __future__ import annotations

import ast
import inspect
from pathlib import Path

from witwin.channel.propagation import enumerated
from witwin.channel.propagation import topology as export


_COMPONENT_STAGES = (
    "_los_topology",
    "_reflection_topology_order1",
    "_reflection_topology_multibounce",
    "_diffraction_topology_order1",
    "_transmission_topology",
    # ADR-011 / G3: the engine now dispatches through the public coupled entry,
    # which selects the single-shot or rx-streamed discovery internally.
    "coupled_reflection_diffraction_topology",
)


def _function(module, name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _named_call_lines(definition: ast.AST) -> dict[str, list[int]]:
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


def test_engine_signature_ownership_and_dependency_boundary():
    signature = inspect.signature(enumerated.evaluate_enumerated_paths)
    tree = ast.parse(Path(enumerated.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert list(signature.parameters) == [
        "scene",
        "config",
        "frequency_value",
        "coupled_rx_streaming",
        "defer_capacity_terminal",
        "endpoint_tensors",
    ]
    assert (
        signature.parameters["frequency_value"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert signature.parameters["frequency_value"].default is None
    # ADR-011 / G3: the deterministic grid solver opts into receiver-block
    # streaming of coupled discovery; path and MC keep the single-shot default.
    assert (
        signature.parameters["coupled_rx_streaming"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert signature.parameters["coupled_rx_streaming"].default is False
    assert (
        signature.parameters["defer_capacity_terminal"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert signature.parameters["defer_capacity_terminal"].default is False
    assert signature.return_annotation == "tuple[EvaluatedPaths, EvaluatedPathSidecars]"
    assert not any(
        module.endswith(".core.path_topology") for module in imported_modules
    )
    assert "TopologyBatch" not in Path(enumerated.__file__).read_text(encoding="utf-8")
    assert "evaluate_enumerated_paths" not in enumerated.__all__


def test_engine_preserves_component_order_los_fast_path_and_field_calls():
    definition = _function(enumerated, "evaluate_enumerated_paths")
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
    assert calls["evaluated_paths_from_block"][0] < max(calls["evaluate_path_fields"])
    assert (
        fast_calls["evaluated_paths_from_result"][0]
        < fast_calls["evaluate_path_fields"][0]
    )
    assert calls["evaluated_paths_from_result"][0] < calls["concatenate_path_blocks"][0]


def test_typed_block_constructor_uses_single_canonical_compact_owner():
    definition = _function(export, "evaluated_paths_from_block")
    calls = _named_call_lines(definition)

    assert (
        calls["enumerated_canonical_compact"][0]
        < calls["evaluated_paths_from_result"][0]
    )
    assert "_canonical_selection_order" not in calls
    assert "deterministic_gather_topology_block" not in calls
