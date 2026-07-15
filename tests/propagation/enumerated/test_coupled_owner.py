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
from witwin.channel_native.propagation.enumerated import coupled


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"
_COUPLED_DIGEST = "cf85f05efb30d2e21356429316974b5c2dbbfa82476293aa732ece5821972836"


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


def test_coupled_owner_preserves_function_and_constant_identity():
    owner = coupled._coupled_reflection_diffraction_topology_order2
    assert legacy._coupled_reflection_diffraction_topology_order2 is owner
    assert owner.__module__ == coupled.__name__
    assert _digest(coupled, owner.__name__) == _COUPLED_DIGEST
    assert legacy._COUPLED_CANDIDATE_CHUNK_SIZE is (
        coupled._COUPLED_CANDIDATE_CHUNK_SIZE
    )
    assert legacy._MAX_COUPLED_CANDIDATES is coupled._MAX_COUPLED_CANDIDATES
    assert coupled._COUPLED_CANDIDATE_CHUNK_SIZE == 65_536
    assert coupled._MAX_COUPLED_CANDIDATES == 1_000_000


def test_export_component_stage_order_remains_canonical():
    tree = ast.parse(Path(legacy.__file__).read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "export_topology"
    )
    call_lines = {
        node.func.id: node.lineno
        for node in ast.walk(definition)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    stages = (
        "_los_topology",
        "_reflection_topology_order1",
        "_reflection_topology_multibounce",
        "_diffraction_topology_order1",
        "_transmission_topology",
        "_coupled_reflection_diffraction_topology_order2",
    )
    assert [call_lines[name] for name in stages] == sorted(
        call_lines[name] for name in stages
    )


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.propagation.enumerated import coupled; "
            "from witwin.channel_native.core import path_topology as legacy"
        ),
        (
            "from witwin.channel_native.core import path_topology as legacy; "
            "from witwin.channel_native.propagation.enumerated import coupled"
        ),
    ),
)
def test_fresh_process_import_order_preserves_coupled_identity(imports: str):
    code = (
        f"{imports}; "
        "assert legacy._coupled_reflection_diffraction_topology_order2 is "
        "coupled._coupled_reflection_diffraction_topology_order2; "
        "assert legacy._COUPLED_CANDIDATE_CHUNK_SIZE is "
        "coupled._COUPLED_CANDIDATE_CHUNK_SIZE; "
        "assert legacy._MAX_COUPLED_CANDIDATES is coupled._MAX_COUPLED_CANDIDATES"
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


def test_coupled_owner_has_no_core_path_dependency_or_scc():
    owner = "witwin.channel_native.propagation.enumerated.coupled"
    core = "witwin.channel_native.core.path_topology"
    edges = graph.collect_import_edges(PACKAGE_ROOT)
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)
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


def test_enumerated_public_all_is_unchanged():
    import witwin.channel_native.propagation.enumerated as enumerated

    assert enumerated.__all__ == ["append_scattering_paths"]
