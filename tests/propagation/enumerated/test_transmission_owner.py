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
from witwin.channel_native.propagation.enumerated import transmission


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"
_TRANSMISSION_DIGEST = (
    "16a904413e97d78c27271b8b4d3eb89aa4dcc943b4b233a1eadb5efea245c3ce"
)


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


def test_transmission_owner_preserves_function_identity_and_ast():
    owner = transmission._transmission_topology
    assert legacy._transmission_topology is owner
    assert owner.__module__ == transmission.__name__
    assert _digest(transmission, owner.__name__) == _TRANSMISSION_DIGEST


def test_los_only_fast_path_remains_before_general_concat():
    tree = ast.parse(Path(legacy.__file__).read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "export_topology"
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
            "from witwin.channel_native.core import path_topology as legacy"
        ),
        (
            "from witwin.channel_native.core import path_topology as legacy; "
            "from witwin.channel_native.propagation.enumerated import transmission"
        ),
    ),
)
def test_fresh_process_import_order_preserves_transmission_identity(imports: str):
    code = (
        f"{imports}; "
        "assert legacy._transmission_topology is transmission._transmission_topology"
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
    owner = "witwin.channel_native.propagation.enumerated.transmission"
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
