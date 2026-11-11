# Copyright Xingyu Chen.
# Tests structural contracts.

from __future__ import annotations

import ast
from pathlib import Path
import pickle

from witwin.channel.propagation import enumerated


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "witwin" / "channel"

_TOPOLOGY_CONFIG_FIELDS = {
    "max_depth",
    "components",
    "max_paths",
    "max_paths_scope",
}
_PROTOCOL_NAMES = ("EnumeratedPathConfig", "TopologyConfig")


def _protocol_members(protocol: type) -> set[str]:
    """Field names of a structural config view: annotated members and read-only properties."""

    return set(getattr(protocol, "__annotations__", {})) | {
        name for name, value in vars(protocol).items() if isinstance(value, property)
    }


def test_topology_config_identity_and_canonical_introspection_owner():
    # The two structural config views share a module but must retain distinct
    # Protocol field sets. The engine reads a
    # four-field view; the enumerated scattering stages read a larger, differently
    # shaped one. They must stay two objects, and neither may silently absorb
    # the other by growing into a superset of it.
    engine_config = enumerated.EnumeratedPathConfig
    scattering_config = enumerated.TopologyConfig
    assert engine_config.__module__ == enumerated.__name__
    assert scattering_config.__module__ == enumerated.__name__
    assert engine_config is not scattering_config

    engine_fields = _protocol_members(engine_config)
    scattering_fields = _protocol_members(scattering_config)
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
