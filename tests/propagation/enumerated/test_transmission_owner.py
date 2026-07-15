from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ci import check_import_graph as graph
from witwin.channel_native.core import path_topology as legacy
from witwin.channel_native.propagation.enumerated import engine, transmission
from witwin.channel_native.propagation.geometry import (
    transmission as geometry_transmission,
)
from witwin.channel_native.propagation.topology.discovery import (
    transmission as discovery_transmission,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"


def test_transmission_typed_boundaries_preserve_canonical_identity():
    owner = transmission._transmission_topology
    assert legacy._transmission_topology is owner
    assert owner.__module__ == transmission.__name__
    assert (
        transmission.TransmissionClosestHitQuery
        is geometry_transmission.TransmissionClosestHitQuery
    )
    assert (
        transmission.query_transmission_closest_hit
        is geometry_transmission.query_transmission_closest_hit
    )
    assert (
        transmission.prepare_transmission_pair_plan
        is discovery_transmission.prepare_transmission_pair_plan
    )
    assert (
        transmission.iter_transmission_active_rows
        is discovery_transmission.iter_transmission_active_rows
    )
    assert (
        transmission.select_transmission_winner_rows
        is discovery_transmission.select_transmission_winner_rows
    )


def test_transmission_owner_consumes_typed_plan_query_and_winners_in_order():
    tree = ast.parse(Path(transmission.__file__).read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_transmission_topology"
    )
    calls: dict[str, list[int]] = {}
    for node in ast.walk(definition):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name is not None:
            calls.setdefault(name, []).append(node.lineno)

    assert (
        calls["prepare_transmission_pair_plan"][0]
        < calls["iter_transmission_active_rows"][0]
        < calls["query_transmission_closest_hit"][0]
        < calls["select_transmission_winner_rows"][0]
        < max(calls["_ensure_topology_fields"])
    )
    assert not any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "hit"
        for node in ast.walk(definition)
    )

    discovery_tree = ast.parse(
        Path(discovery_transmission.__file__).read_text(encoding="utf-8")
    )
    active_rows = next(
        node
        for node in discovery_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "iter_transmission_active_rows"
    )
    assert "range(plan.max_depth + 1)" in ast.unparse(active_rows)


def test_los_only_fast_path_remains_before_general_concat():
    tree = ast.parse(Path(engine.__file__).read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "evaluate_enumerated_paths"
    )
    fast_path = next(
        node
        for node in definition.body
        if isinstance(node, ast.If)
        and "components == {'los'}" in ast.unparse(node.test)
    )
    test_expression = ast.unparse(fast_path.test)
    assert "len(blocks) == 1" in test_expression
    assert "config.max_paths is None" in test_expression
    los_call = next(
        node
        for node in ast.walk(definition)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_los_topology"
    )
    concat_call = next(
        node
        for node in ast.walk(definition)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "concatenate_path_blocks"
    )
    assert los_call.lineno < fast_path.lineno < concat_call.lineno
    assert any(isinstance(node, ast.Return) for node in fast_path.body)


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.propagation.enumerated import transmission; "
            "from witwin.channel_native.propagation.geometry import transmission as geometry_transmission; "
            "from witwin.channel_native.propagation.topology.discovery import transmission as discovery_transmission; "
            "from witwin.channel_native.core import path_topology as legacy"
        ),
        (
            "from witwin.channel_native.core import path_topology as legacy; "
            "from witwin.channel_native.propagation.topology.discovery import transmission as discovery_transmission; "
            "from witwin.channel_native.propagation.geometry import transmission as geometry_transmission; "
            "from witwin.channel_native.propagation.enumerated import transmission"
        ),
    ),
)
def test_fresh_process_import_order_preserves_transmission_identity(imports: str):
    code = (
        f"{imports}; "
        "assert legacy._transmission_topology is transmission._transmission_topology; "
        "assert transmission.query_transmission_closest_hit is geometry_transmission.query_transmission_closest_hit; "
        "assert transmission.prepare_transmission_pair_plan is discovery_transmission.prepare_transmission_pair_plan"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(REPOSITORY_ROOT / "src"),
            environment.get("PYTHONPATH"),
        )
        if value
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_transmission_owner_has_no_core_path_dependency_or_scc():
    owners = {
        "witwin.channel_native.propagation.enumerated.transmission",
        "witwin.channel_native.propagation.geometry.transmission",
        "witwin.channel_native.propagation.topology.discovery.transmission",
    }
    geometry_owner = "witwin.channel_native.propagation.geometry.transmission"
    discovery_owner = (
        "witwin.channel_native.propagation.topology.discovery.transmission"
    )
    core = "witwin.channel_native.core.path_topology"
    edges = graph.collect_import_edges(PACKAGE_ROOT)
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)
    reachable: dict[str, set[str]] = {}
    for owner in owners:
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
        reachable[owner] = seen
    assert discovery_owner not in reachable[geometry_owner]
    assert geometry_owner not in reachable[discovery_owner]
