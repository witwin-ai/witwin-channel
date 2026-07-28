from __future__ import annotations

import ast
from dataclasses import is_dataclass
import hashlib
from pathlib import Path
import pickle

from ci import check_import_graph as graph
from witwin.channel.propagation.enumerated import contracts as enumerated_contracts
from witwin.channel.propagation.enumerated import engine


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel"

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
    assert engine.TopologyConfig._is_protocol
    assert not is_dataclass(engine.TopologyConfig)
    assert set(engine.TopologyConfig.__annotations__) == _TOPOLOGY_CONFIG_FIELDS


def test_topology_config_identity_and_canonical_introspection_owner():
    # The protocol lives beside its only consumer and is published from nowhere
    # else, so ``__module__`` names the one module that defines it. The enumerated
    # scattering stages read a same-named but larger protocol; the two must stay
    # distinct objects with distinct field sets.
    assert engine.TopologyConfig.__module__ == engine.__name__
    assert engine.TopologyConfig is not enumerated_contracts.TopologyConfig
    assert set(enumerated_contracts.TopologyConfig.__annotations__) != (
        _TOPOLOGY_CONFIG_FIELDS
    )
    definitions = [
        path
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "TopologyConfig"
    ]
    assert definitions == [
        PACKAGE_ROOT / "propagation" / "enumerated" / "contracts.py",
        PACKAGE_ROOT / "propagation" / "enumerated" / "engine.py",
    ]


def test_topology_config_class_pickle_uses_canonical_owner():
    payload = pickle.dumps(engine.TopologyConfig)

    assert b"witwin.channel.propagation.enumerated.engine" in payload
    assert b"witwin.channel.core.path_topology" not in payload
    assert pickle.loads(payload) is engine.TopologyConfig


def test_moved_protocol_bodies_are_exact():
    assert _definition_digest(engine, "TopologyConfig") == (_TOPOLOGY_CONFIG_DIGEST)


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

    core = "witwin.channel.core.path_topology"
    contracts_module = "witwin.channel.propagation.enumerated.engine"
    assert not reaches(contracts_module, core)
