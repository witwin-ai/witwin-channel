from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from ci import check_import_graph as graph
from witwin.channel.interactions import reflection
from witwin.channel.propagation import geometry as reevaluate
from witwin.channel.propagation import topology


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "witwin" / "channel"
# ``_face_sequence_chunks`` re-pinned when the topology kernel facades moved to
# ``witwin.channel.kernels.topology``: the only AST difference is the module
# alias the two chunk calls are spelled through. Control flow, arguments and
# evaluation order are unchanged.
# ``_discovered_group_chains`` and ``_reflect_points`` re-pinned for the same
# reason when the geometry kernel facades moved to
# ``witwin.channel.kernels.geometry`` and the
# ``geometry_bridge``/``geometry_primitives`` aliases collapsed into
# ``geometry_kernels``.
# All four digests survive the concept-axis gather unchanged: moving a
# definition between modules does not touch its AST, and ``ast.dump`` here is
# taken without attributes, so position is not part of the pin.
_DIGESTS = {
    "_face_sequence_count": "d7932ea0ae0bb7b781b5113800044489b7b5356129ccd0169d305749a1dbf121",
    "_face_sequence_chunks": "58d450d46c6e15f13176972c3996ec222d7a9b218fdb53ffc9529d0fd4370220",
    "_reflect_points": "ef2b934fe26c236d6c631c842707de838286409485341061add2e8a6a6060811",
    "_discovered_group_chains": (
        "ac3e0339fa6b41914dd6e81465b98d30825498d85aac4c8f3d1df322d92398ff"
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


OWNER = "witwin.channel.interactions.reflection"


def test_reflection_owners_and_discovery_constants_preserve_identity():
    # The concept-axis gather put reflection discovery, EPC geometry and
    # enumerated orchestration in one module. The previous version of this test
    # pinned that the three former modules published the SAME objects, so a
    # copy could not appear in two places. With one module that identity claim
    # is vacuous, so the equivalent pin is stronger: every name must be DEFINED
    # here. A re-export layer or a resurrected second owner shows up as a
    # foreign ``__module__``.
    assert reflection.__name__ == OWNER
    for name in (
        "_face_sequence_count",
        "_face_sequence_chunks",
        "prepare_reflection_order1_plan",
        "iter_reflection_order1_epc_requests",
        "prepare_reflection_multibounce_plan",
        "iter_reflection_multibounce_epc_requests",
        "query_reflection_epc",
        "_discovered_group_chains",
        "_reflection_topology_order1",
        "_reflection_topology_multibounce",
        "ReflectionEpcQuery",
        "ReflectionEpcGeometry",
        "ReflectionOrder1Plan",
        "ReflectionOrder1EpcRequest",
        "ReflectionMultibouncePlan",
        "ReflectionMultibounceEpcRequest",
    ):
        assert getattr(reflection, name).__module__ == OWNER
    globals_ = reflection._reflection_topology_multibounce.__globals__
    assert globals_["_face_sequence_count"] is reflection._face_sequence_count
    assert globals_["_face_sequence_chunks"] is reflection._face_sequence_chunks
    assert reflection._ORDER1_EXHAUSTIVE_GROUP_LIMIT == 4096
    assert reflection._MULTIBOUNCE_PAIR_CHUNK_SIZE == 4_194_304
    assert reflection._MULTIBOUNCE_DISCOVERY_RAYS == 262_144
    assert reevaluate._reflect_points.__module__ == reevaluate.__name__


def test_unmodified_reflection_helpers_keep_frozen_ast():
    assert (
        _digest(reflection, "_discovered_group_chains")
        == _DIGESTS["_discovered_group_chains"]
    )
    assert (
        _digest(reflection, "_face_sequence_count") == _DIGESTS["_face_sequence_count"]
    )
    assert (
        _digest(reflection, "_face_sequence_chunks")
        == _DIGESTS["_face_sequence_chunks"]
    )
    assert _digest(reevaluate, "_reflect_points") == _DIGESTS["_reflect_points"]


def test_reflection_consumers_use_only_named_epc_geometry():
    tree = ast.parse(Path(reflection.__file__).read_text(encoding="utf-8"))
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_reflection_topology_order1",
            "_reflection_topology_multibounce",
        }
    }
    for definition in definitions.values():
        calls = {
            node.func.id
            for node in ast.walk(definition)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert {"ReflectionEpcQuery", "query_reflection_epc"} <= calls
        assert not any(
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "epc"
            for node in ast.walk(definition)
        )
        attributes = {
            node.attr
            for node in ast.walk(definition)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "epc"
        }
        assert attributes == {
            "visible",
            "resolved_prim_ids",
            "hit_positions",
            "normals",
        }


def test_reflection_owner_has_no_core_path_dependency_or_scc():
    edges = graph.collect_import_edges(PACKAGE_ROOT)
    core = "witwin.channel.core.path_topology"
    owner_targets = {edge.target for edge in edges if edge.source == OWNER}
    assert core not in owner_targets

    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)

    def reachable(start: str) -> set[str]:
        pending = [start]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(adjacency.get(current, ()))
        return seen

    assert core not in reachable(OWNER)

    # The three former reflection modules (topology discovery, geometry EPC,
    # enumerated orchestration) are one module now, so the old "geometry must
    # not reach discovery" layering pin has no two endpoints left to name. The
    # cycle it really ruled out is still ruled out, and now for the whole
    # concept: nothing the single owner imports may import the owner back.
    downstream: set[str] = set()
    for target in adjacency.get(OWNER, ()):
        downstream |= reachable(target)
    assert OWNER not in downstream
    assert core not in downstream


def test_enumerated_public_all_is_unchanged_and_discovery_init_is_empty():
    import importlib.util

    import witwin.channel.propagation.enumerated as enumerated

    assert enumerated.__all__ == []
    # The concept axis emptied ``propagation.topology.discovery`` completely, so
    # the package was deleted rather than left as an empty namespace. "Empty"
    # is now "absent": nothing may recreate it as a discovery owner. Collapsing
    # the stage package into ``propagation/topology.py`` makes that structural
    # rather than conventional - a module carries no ``__path__``, so the
    # submodule cannot be created at all and the lookup raises instead of
    # answering ``None``.
    assert not hasattr(topology, "__path__")
    with pytest.raises(ModuleNotFoundError):
        importlib.util.find_spec("witwin.channel.propagation.topology.discovery")
