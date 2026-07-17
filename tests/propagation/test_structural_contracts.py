from __future__ import annotations

import ast
from dataclasses import is_dataclass
import hashlib
from pathlib import Path
import pickle

from ci import check_import_graph as graph
from witwin.channel_native.propagation import models
from witwin.channel_native.propagation.models import contracts


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"

_TOPOLOGY_CONFIG_DIGEST = (
    "a1359261cbd2361e26874a6e951b5877f7a9bcda9642bf1eddbfb5bcec41cbee"
)
_TOPOLOGY_CONFIG_FIELDS = {
    "max_depth",
    "components",
    "max_paths",
    "max_paths_scope",
}
def _definition(module, name: str):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
    )


def _definition_digest(module, name: str) -> str:
    payload = ast.dump(_definition(module, name), include_attributes=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def test_structural_protocols_have_exact_fields():
    assert contracts.TopologyConfig._is_protocol
    assert not is_dataclass(contracts.TopologyConfig)
    assert set(contracts.TopologyConfig.__annotations__) == _TOPOLOGY_CONFIG_FIELDS


def test_topology_config_identity_and_canonical_introspection_owner():
    assert models.TopologyConfig is contracts.TopologyConfig
    assert contracts.TopologyConfig.__module__ == contracts.__name__
    assert "TopologyConfig" in models.__all__


def test_topology_config_class_pickle_uses_canonical_owner():
    payload = pickle.dumps(contracts.TopologyConfig)

    assert b"witwin.channel_native.propagation.models.contracts" in payload
    assert b"witwin.channel_native.core.path_topology" not in payload
    assert pickle.loads(payload) is contracts.TopologyConfig


def test_moved_protocol_bodies_are_exact():
    assert _definition_digest(contracts, "TopologyConfig") == (_TOPOLOGY_CONFIG_DIGEST)


def test_contracts_do_not_depend_on_removed_core_path_topology():
    edges = graph.collect_import_edges(PACKAGE_ROOT)
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)

    def reaches(source: str, target: str) -> bool:
        pending = [source]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            pending.extend(adjacency.get(current, ()))
        return False

    core = "witwin.channel_native.core.path_topology"
    contracts_module = "witwin.channel_native.propagation.models.contracts"
    assert not reaches(contracts_module, core)
