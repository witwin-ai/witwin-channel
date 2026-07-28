from __future__ import annotations

import ast
from dataclasses import is_dataclass
import hashlib
from pathlib import Path
import pickle

from ci import check_import_graph as graph
from witwin.channel.propagation import enumerated


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "witwin" / "channel"

# Re-frozen when ``propagation/enumerated/`` collapsed into one module. One
# module cannot hold two classes named ``TopologyConfig``, so the engine-side
# protocol - module-private, with no importer outside its own module - was
# renamed to ``EnumeratedPathConfig``. The scattering-side ``TopologyConfig``
# kept its name because ``interactions.scattering`` imports it by name. The
# digest below is the AST of the renamed protocol; the previous value,
# a1359261cbd2361e26874a6e951b5877f7a9bcda9642bf1eddbfb5bcec41cbee, froze the
# byte-identical body under the old name.
_TOPOLOGY_CONFIG_DIGEST = (
    "a1f563fb363359de44f47b4932bc1e342176f7b2c9ed8c9f596d95e0288b899b"
)
_TOPOLOGY_CONFIG_FIELDS = {
    "max_depth",
    "components",
    "max_paths",
    "max_paths_scope",
}
_PROTOCOL_NAMES = ("EnumeratedPathConfig", "TopologyConfig")


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
    assert enumerated.EnumeratedPathConfig._is_protocol
    assert not is_dataclass(enumerated.EnumeratedPathConfig)
    assert set(enumerated.EnumeratedPathConfig.__annotations__) == (
        _TOPOLOGY_CONFIG_FIELDS
    )


def test_topology_config_identity_and_canonical_introspection_owner():
    # The two structural config views now live in one module, so ``__module__``
    # no longer distinguishes them - but a Protocol is exactly its field set,
    # and that is the property this test exists to hold. The engine reads a
    # four-field view; the enumerated scattering stages read a larger, differently
    # shaped one. They must stay two objects, and neither may silently absorb
    # the other by growing into a superset of it.
    engine_config = enumerated.EnumeratedPathConfig
    scattering_config = enumerated.TopologyConfig
    assert engine_config.__module__ == enumerated.__name__
    assert scattering_config.__module__ == enumerated.__name__
    assert engine_config is not scattering_config

    engine_fields = set(engine_config.__annotations__)
    scattering_fields = set(scattering_config.__annotations__)
    assert engine_fields == _TOPOLOGY_CONFIG_FIELDS
    assert scattering_fields != _TOPOLOGY_CONFIG_FIELDS
    # Neither absorbs the other: each names at least one field the other does
    # not, in both directions.
    assert not engine_fields <= scattering_fields
    assert not scattering_fields <= engine_fields

    # Exactly one definition site each, package-wide, and both in the one module
    # that owns them. A third copy anywhere - or a second copy of either name -
    # fails here.
    definitions = [
        (path, node.name)
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name in _PROTOCOL_NAMES
    ]
    owner = PACKAGE_ROOT / "propagation" / "enumerated.py"
    assert definitions == [
        (owner, "TopologyConfig"),
        (owner, "EnumeratedPathConfig"),
    ]


def test_topology_config_class_pickle_uses_canonical_owner():
    payload = pickle.dumps(enumerated.EnumeratedPathConfig)

    assert b"witwin.channel.propagation.enumerated" in payload
    assert b"witwin.channel.propagation.enumerated.engine" not in payload
    assert b"witwin.channel.core.path_topology" not in payload
    assert pickle.loads(payload) is enumerated.EnumeratedPathConfig


def test_moved_protocol_bodies_are_exact():
    assert _definition_digest(enumerated, "EnumeratedPathConfig") == (
        _TOPOLOGY_CONFIG_DIGEST
    )


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
    contracts_module = "witwin.channel.propagation.enumerated"
    assert not reaches(contracts_module, core)
