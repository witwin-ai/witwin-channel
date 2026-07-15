from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ci import check_import_graph as graph
from witwin.channel_native.core import path_topology as legacy
from witwin.channel_native.propagation.enumerated import reflection
from witwin.channel_native.propagation.topology.discovery import (
    reflection as discovery,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"
_DIGESTS = {
    "_reflection_topology_order1": (
        "35cb2a0602c1d2d27dff2207524f1224129c53767858fa1f9ebd366abe2b5564"
    ),
    "_discovered_group_chains": (
        "0ab4299eb5a2c6ece2296ccdf9432de986f1e15fdf127ecd595b61bee6b04fa7"
    ),
    "_reflection_topology_multibounce": (
        "b7a236cf404f607d748f816f00c4f8ac9ea8634bc9e3543b067c7fbe1ed1e0a2"
    ),
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


def test_reflection_owners_and_discovery_constants_preserve_identity():
    assert legacy._reflection_topology_order1 is reflection._reflection_topology_order1
    assert legacy._discovered_group_chains is reflection._discovered_group_chains
    assert legacy._ORDER1_EXHAUSTIVE_GROUP_LIMIT is (
        discovery._ORDER1_EXHAUSTIVE_GROUP_LIMIT
    )
    assert legacy._MULTIBOUNCE_PAIR_CHUNK_SIZE is (
        discovery._MULTIBOUNCE_PAIR_CHUNK_SIZE
    )
    assert legacy._MULTIBOUNCE_DISCOVERY_RAYS is (discovery._MULTIBOUNCE_DISCOVERY_RAYS)
    assert discovery._ORDER1_EXHAUSTIVE_GROUP_LIMIT == 4096
    assert discovery._MULTIBOUNCE_PAIR_CHUNK_SIZE == 4_194_304
    assert discovery._MULTIBOUNCE_DISCOVERY_RAYS == 262_144
    assert reflection.prepare_reflection_order1_plan is (
        discovery.prepare_reflection_order1_plan
    )
    assert reflection.iter_reflection_order1_epc_requests is (
        discovery.iter_reflection_order1_epc_requests
    )


def test_moved_functions_and_untouched_multibounce_have_frozen_ast():
    assert (
        _digest(reflection, "_reflection_topology_order1")
        == _DIGESTS["_reflection_topology_order1"]
    )
    assert (
        _digest(reflection, "_discovered_group_chains")
        == _DIGESTS["_discovered_group_chains"]
    )
    assert (
        _digest(legacy, "_reflection_topology_multibounce")
        == _DIGESTS["_reflection_topology_multibounce"]
    )


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.propagation.enumerated import reflection; "
            "from witwin.channel_native.core import path_topology as legacy"
        ),
        (
            "from witwin.channel_native.core import path_topology as legacy; "
            "from witwin.channel_native.propagation.enumerated import reflection"
        ),
    ),
)
def test_fresh_process_import_order_preserves_owner_identity(imports: str):
    code = (
        f"{imports}; "
        "assert legacy._reflection_topology_order1 is "
        "reflection._reflection_topology_order1; "
        "assert legacy._discovered_group_chains is reflection._discovered_group_chains"
    )
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_reflection_owner_has_no_core_path_dependency_or_scc():
    edges = graph.collect_import_edges(PACKAGE_ROOT)
    owner = "witwin.channel_native.propagation.enumerated.reflection"
    core = "witwin.channel_native.core.path_topology"
    owner_targets = {edge.target for edge in edges if edge.source == owner}
    assert core not in owner_targets

    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)
    pending = [owner]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(adjacency.get(current, ()))
    assert core not in seen


def test_enumerated_public_all_is_unchanged_and_discovery_init_is_empty():
    import witwin.channel_native.propagation.enumerated as enumerated
    import witwin.channel_native.propagation.topology.discovery as package

    assert enumerated.__all__ == ["append_scattering_paths"]
    assert not hasattr(package, "_reflection_topology_order1")
