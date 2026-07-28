from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from ci import check_import_graph as graph
from witwin.channel.propagation.enumerated import diffraction
from witwin.channel.propagation.geometry import (
    diffraction as geometry_diffraction,
)
from witwin.channel.propagation.topology.discovery import (
    diffraction as discovery,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel"
# Re-pinned when the geometry kernel facades moved to
# ``witwin.channel.kernels.geometry``: the only AST difference is the module
# alias the native plan call is spelled through (``geometry_bridge`` ->
# ``geometry_kernels``). Control flow, arguments and evaluation order are
# unchanged.
_DIGESTS = {
    "plan_tx_visible_diffraction_states": "77cc58aea3a87b52b1d039624fc85c5f61e2ad435ab6d66424f578098727a271",
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
    assert diffraction.plan_tx_visible_diffraction_states is (
        geometry_diffraction.plan_tx_visible_diffraction_states
    )
    assert geometry_diffraction.plan_tx_visible_diffraction_states.__module__ == (
        geometry_diffraction.__name__
    )
    assert (
        _digest(geometry_diffraction, "plan_tx_visible_diffraction_states")
        == _DIGESTS["plan_tx_visible_diffraction_states"]
    )
    assert not hasattr(diffraction, "_tx_visible_diffraction_states")
    assert not hasattr(geometry_diffraction, "_tx_visible_diffraction_states")
    assert not hasattr(diffraction, "_DIFFRACTION_PREFILTER_EDGE_FRACTIONS")
    assert not hasattr(geometry_diffraction, "_DIFFRACTION_PREFILTER_EDGE_FRACTIONS")
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
        "plan_tx_visible_diffraction_states",
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
    core = "witwin.channel.core.path_topology"
    edges = graph.collect_import_edges(PACKAGE_ROOT)
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)
    owners = {
        "witwin.channel.propagation.enumerated.diffraction",
        "witwin.channel.propagation.geometry.diffraction",
        "witwin.channel.propagation.topology.discovery.diffraction",
    }
    discovery_owner = (
        "witwin.channel.propagation.topology.discovery.diffraction"
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
        if owner == "witwin.channel.propagation.geometry.diffraction":
            assert discovery_owner not in seen
    assert not hasattr(diffraction, "_rayd_visibility_mask")
