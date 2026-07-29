# Copyright Xingyu Chen.
# Tests coupled owner.

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from ci import check_import_graph as graph
from witwin.channel.propagation import enumerated as engine
from witwin.channel.interactions import coupled
from witwin.channel.interactions import coupled as geometry_coupled
from witwin.channel.interactions import coupled as discovery_coupled


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "witwin" / "channel"
_COUPLED_DIGEST = "c1da3dec6a095f47c6704dc8930122b4f49aeaa588a2a0fa090ae679a7370568"


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
    # D-to-D discovery and geometry symbols share the same owners.
    assert coupled.iter_coupled_dd_candidate_requests is (
        discovery_coupled.iter_coupled_dd_candidate_requests
    )
    assert coupled.CoupledGeometryQuery is geometry_coupled.CoupledGeometryQuery
    assert coupled.query_coupled_geometry is geometry_coupled.query_coupled_geometry
    assert coupled.CoupledDdGeometryQuery is geometry_coupled.CoupledDdGeometryQuery
    assert coupled.query_coupled_dd_geometry is (
        geometry_coupled.query_coupled_dd_geometry
    )


def test_enumerated_coupled_consumes_named_geometry_only():
    tree = ast.parse(Path(coupled.__file__).read_text(encoding="utf-8"))
    # The per-receiver-block worker consumes named discovery and geometry;
    # both order-2 entry points delegate to it.
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_coupled_topology_rx_block"
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
    # The D-to-D stream is consumed in the same block worker.
    assert "iter_coupled_dd_candidate_requests" in call_names
    assert "query_coupled_dd_geometry" in call_names
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
        # The engine dispatches through the public coupled entry.
        "coupled_reflection_diffraction_topology",
    )
    assert [call_lines[name] for name in stages] == sorted(
        call_lines[name] for name in stages
    )


def test_coupled_owner_has_no_core_path_dependency_or_scc():
    owner = "witwin.channel.interactions.coupled"
    core = "witwin.channel.core.path_topology"
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
    import witwin.channel.propagation.enumerated as enumerated

    assert enumerated.__all__ == []
