from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from ci import check_import_graph as graph
from witwin.channel_native.propagation.enumerated import diffraction
from witwin.channel_native.propagation.geometry import (
    diffraction as geometry_diffraction,
)
from witwin.channel_native.propagation.topology.discovery import (
    diffraction as discovery,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"
_DIGESTS = {
    "_tx_visible_diffraction_states": "ea7600e59670741cc8d1d162b3de52f4d2232c27eaec50f45d679970b4e153df",
}


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


def test_diffraction_owner_identity_module_and_constant():
    assert diffraction._deterministic_diffraction_states.__module__ == (
        diffraction.__name__
    )
    assert diffraction._diffraction_topology_order1.__module__ == diffraction.__name__
    assert (
        diffraction._tx_visible_diffraction_states
        is geometry_diffraction._tx_visible_diffraction_states
    )
    assert geometry_diffraction._tx_visible_diffraction_states.__module__ == (
        geometry_diffraction.__name__
    )
    assert (
        _digest(geometry_diffraction, "_tx_visible_diffraction_states")
        == _DIGESTS["_tx_visible_diffraction_states"]
    )
    assert (
        diffraction._DIFFRACTION_PREFILTER_EDGE_FRACTIONS
        is geometry_diffraction._DIFFRACTION_PREFILTER_EDGE_FRACTIONS
    )
    assert geometry_diffraction._DIFFRACTION_PREFILTER_EDGE_FRACTIONS == (
        0.02,
        1.0 / 3.0,
        2.0 / 3.0,
        0.98,
    )
    assert diffraction.prepare_diffraction_order1_plan is (
        discovery.prepare_diffraction_order1_plan
    )
    assert diffraction.iter_diffraction_tx_requests is (
        discovery.iter_diffraction_tx_requests
    )
    assert diffraction.iter_diffraction_rx_chunk_requests is (
        discovery.iter_diffraction_rx_chunk_requests
    )
    assert diffraction.query_diffraction_edges is (
        geometry_diffraction.query_diffraction_edges
    )
    assert diffraction.query_diffraction_order1 is (
        geometry_diffraction.query_diffraction_order1
    )


def test_diffraction_consumers_use_named_geometry_and_canonical_event_order():
    tree = ast.parse(Path(diffraction.__file__).read_text(encoding="utf-8"))
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    state_builder = definitions["_deterministic_diffraction_states"]
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "query_diffraction_edges"
        for node in ast.walk(state_builder)
    )
    assert not any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "edges"
        for node in ast.walk(state_builder)
    )

    owner = definitions["_diffraction_topology_order1"]

    def call_name(node: ast.Call) -> str | None:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    call_lines = {
        name: node.lineno
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and (name := call_name(node)) is not None
    }
    stages = (
        "prepare_diffraction_order1_plan",
        "iter_diffraction_tx_requests",
        "_deterministic_diffraction_states",
        "_tx_visible_diffraction_states",
        "name_diffraction_states",
        "iter_diffraction_rx_chunk_requests",
        "query_diffraction_order1",
        "deterministic_diffraction_order1_compact",
        "index_add_",
        "deterministic_diffraction_vector_field",
        "deterministic_pack_complex",
        "deterministic_delay_to_path_length",
        "deterministic_topology_base_fields",
    )
    assert [call_lines[name] for name in stages] == sorted(
        call_lines[name] for name in stages
    )
    assert not any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "out"
        for node in ast.walk(owner)
    )


def test_diffraction_owner_has_no_core_path_dependency_or_scc():
    core = "witwin.channel_native.core.path_topology"
    edges = graph.collect_import_edges(PACKAGE_ROOT)
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)
    owners = {
        "witwin.channel_native.propagation.enumerated.diffraction",
        "witwin.channel_native.propagation.geometry.diffraction",
        "witwin.channel_native.propagation.topology.discovery.diffraction",
    }
    discovery_owner = (
        "witwin.channel_native.propagation.topology.discovery.diffraction"
    )
    for owner in owners:
        assert core not in adjacency.get(owner, set())
        pending = [owner]
        seen = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(adjacency.get(current, ()))
        assert core not in seen
        if owner == "witwin.channel_native.propagation.geometry.diffraction":
            assert discovery_owner not in seen
    assert not hasattr(diffraction, "_raydn_visibility_mask")
