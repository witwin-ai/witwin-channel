from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from ci import check_import_graph as graph
from witwin.channel_native.propagation.enumerated import coupled, engine
from witwin.channel_native.propagation.geometry import coupled as geometry_coupled
from witwin.channel_native.propagation.topology.discovery import (
    coupled as discovery_coupled,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"
_COUPLED_DIGEST = "6cd6c11c193f48e09ac4ede98766a2da44c0f3f170f757947f168b492789bbc5"


def _digest(module, name: str) -> str:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return hashlib.sha256(
        ast.dump(definition, include_attributes=False).encode()
    ).hexdigest()


def test_coupled_owner_preserves_function_and_constant_identity():
    owner = coupled._coupled_reflection_diffraction_topology_order2
    assert owner.__module__ == coupled.__name__
    assert _digest(coupled, owner.__name__) == _COUPLED_DIGEST
    assert coupled._COUPLED_CANDIDATE_CHUNK_SIZE == 65_536
    assert coupled._MAX_COUPLED_CANDIDATES == 1_000_000
    assert coupled._COUPLED_CANDIDATE_CHUNK_SIZE is (
        discovery_coupled._COUPLED_CANDIDATE_CHUNK_SIZE
    )
    assert coupled._MAX_COUPLED_CANDIDATES is (
        discovery_coupled._MAX_COUPLED_CANDIDATES
    )
    assert coupled.prepare_coupled_candidate_plan is (
        discovery_coupled.prepare_coupled_candidate_plan
    )
    assert coupled.iter_coupled_candidate_requests is (
        discovery_coupled.iter_coupled_candidate_requests
    )
    assert coupled.CoupledGeometryQuery is geometry_coupled.CoupledGeometryQuery
    assert coupled.query_coupled_geometry is geometry_coupled.query_coupled_geometry


def test_enumerated_coupled_consumes_named_geometry_only():
    tree = ast.parse(Path(coupled.__file__).read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_coupled_reflection_diffraction_topology_order2"
    )

    assert "geometry_bridge" not in coupled.__dict__
    assert not any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "exported"
        for node in ast.walk(definition)
    )
    call_names = {
        node.func.id
        for node in ast.walk(definition)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "prepare_coupled_candidate_plan" in call_names
    assert "iter_coupled_candidate_requests" in call_names
    assert "arange" not in call_names


def test_export_component_stage_order_remains_canonical():
    tree = ast.parse(Path(engine.__file__).read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "evaluate_enumerated_paths"
    )
    call_lines = {
        node.func.id: node.lineno
        for node in ast.walk(definition)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    stages = (
        "_los_topology",
        "_reflection_topology_order1",
        "_reflection_topology_multibounce",
        "_diffraction_topology_order1",
        "_transmission_topology",
        "_coupled_reflection_diffraction_topology_order2",
    )
    assert [call_lines[name] for name in stages] == sorted(
        call_lines[name] for name in stages
    )


def test_coupled_owner_has_no_core_path_dependency_or_scc():
    owner = "witwin.channel_native.propagation.enumerated.coupled"
    core = "witwin.channel_native.core.path_topology"
    edges = graph.collect_import_edges(PACKAGE_ROOT)
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)
    assert core not in adjacency.get(owner, set())
    pending = [owner]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(adjacency.get(current, ()))
    assert core not in seen


def test_enumerated_public_all_is_unchanged():
    import witwin.channel_native.propagation.enumerated as enumerated

    assert enumerated.__all__ == []
