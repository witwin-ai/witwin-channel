from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import is_dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ci import check_import_graph
from witwin.channel_native.propagation import enumerated
from witwin.channel_native.propagation.enumerated import contracts, scattering


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"
SCATTERING_PATH = PACKAGE_ROOT / "propagation" / "enumerated" / "scattering.py"
_SCATTERING_DEFINITION_DIGEST = (
    "1f99f6030d87ac6c3930d3ac0c64b810bb2581e39fe5d840dc4b3467c2d68f44"
)


def test_enumerated_package_reexports_the_scattering_callable_same_object():
    assert enumerated.__all__ == ["append_scattering_paths"]
    assert enumerated.append_scattering_paths is scattering.append_scattering_paths
    assert scattering.append_scattering_paths.__module__ == scattering.__name__


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.propagation import enumerated; "
            "from witwin.channel_native.propagation.enumerated import scattering"
        ),
        (
            "from witwin.channel_native.propagation.enumerated import scattering; "
            "from witwin.channel_native.propagation import enumerated"
        ),
    ),
)
def test_enumerated_import_order_preserves_callable_identity(imports: str):
    code = (
        f"{imports}; "
        "assert enumerated.append_scattering_paths is "
        "scattering.append_scattering_paths"
    )
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_scattering_definitions_preserve_the_frozen_ast_body_digest():
    tree = ast.parse(SCATTERING_PATH.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    payload = "\n".join(
        ast.dump(node, include_attributes=False) for node in definitions
    )

    assert len(definitions) == 15
    assert hashlib.sha256(payload.encode()).hexdigest() == (
        _SCATTERING_DEFINITION_DIGEST
    )


def test_scattering_contracts_are_minimal_structural_protocols():
    assert contracts.TopologyBatch._is_protocol
    assert contracts.TopologyConfig._is_protocol
    assert not is_dataclass(contracts.TopologyBatch)
    assert set(contracts.TopologyConfig.__annotations__) == {
        "components",
        "max_depth",
        "scattering_samples_per_m2",
        "scattering_power_threshold",
        "scattering_max_paths_per_pair",
    }
    assert set(contracts.TopologyBatch.__annotations__) == {
        "valid",
        "tx_id",
        "rx_id",
        "depth",
        "component_id",
        "primitive_id",
        "edge_id",
        "path_length_m",
        "delay_s",
        "path_gain",
        "path_field",
        "field_xyz",
        "coefficient",
        "field_direction",
        "interaction_position",
        "interaction_normal",
        "material_id",
        "primitive_sequence",
        "material_sequence",
        "interaction_type",
        "interaction_positions",
        "interaction_normals",
        "launch_count",
        "candidate_count",
        "guardrail_count",
    }


def test_scattering_has_no_core_path_topology_dependency_or_scc():
    edges = check_import_graph.collect_import_edges(PACKAGE_ROOT)
    package = "witwin.channel_native"
    scattering_module = f"{package}.propagation.enumerated.scattering"
    contracts_module = f"{package}.propagation.enumerated.contracts"
    legacy_module = f"{package}.core.path_topology"

    assert any(
        edge.source == scattering_module and edge.target == contracts_module
        for edge in edges
    )
    assert not any(
        edge.source == scattering_module and edge.target == legacy_module
        for edge in edges
    )

    graph: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        graph[edge.source].add(edge.target)

    def reachable(source: str, target: str) -> bool:
        queue = deque((source,))
        seen = {source}
        while queue:
            current = queue.popleft()
            if current == target:
                return True
            for child in graph[current]:
                if child.startswith(package) and child not in seen:
                    seen.add(child)
                    queue.append(child)
        return False

    assert not (
        reachable(scattering_module, legacy_module)
        and reachable(legacy_module, scattering_module)
    )


def test_scattering_noop_preserves_topology_row_identity():
    topology = object()
    config = SimpleNamespace(components=frozenset(), max_depth=1)

    exported, info = scattering.append_scattering_paths(object(), config, topology)

    assert exported is topology
    assert info["path_count"] == 0
    assert info["capped_path_count"] == 0
