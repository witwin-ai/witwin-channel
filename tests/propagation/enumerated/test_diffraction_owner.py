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
from witwin.channel_native.propagation.enumerated import diffraction


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"
_DIGESTS = {
    "_deterministic_diffraction_states": "b17808b6ab26e59f8066ea434d0266ac54cb4c9a1708e08aa1b92665505f3ebc",
    "_tx_visible_diffraction_states": "ea7600e59670741cc8d1d162b3de52f4d2232c27eaec50f45d679970b4e153df",
    "_diffraction_topology_order1": "6a819f5922fa799f8d5c32734d89d3e1ab9ec9441df9f4d6c9acee89e3fdc70b",
}


def _digest(name: str) -> str:
    tree = ast.parse(Path(diffraction.__file__).read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return hashlib.sha256(
        ast.dump(definition, include_attributes=False).encode()
    ).hexdigest()


def test_diffraction_owner_identity_module_and_constant():
    for name in _DIGESTS:
        owner = getattr(diffraction, name)
        assert getattr(legacy, name) is owner
        assert owner.__module__ == diffraction.__name__
        assert _digest(name) == _DIGESTS[name]
    assert legacy._DIFFRACTION_PREFILTER_EDGE_FRACTIONS is (
        diffraction._DIFFRACTION_PREFILTER_EDGE_FRACTIONS
    )
    assert diffraction._DIFFRACTION_PREFILTER_EDGE_FRACTIONS == (
        0.02,
        1.0 / 3.0,
        2.0 / 3.0,
        0.98,
    )


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.propagation.enumerated import diffraction; "
            "from witwin.channel_native.core import path_topology as legacy"
        ),
        (
            "from witwin.channel_native.core import path_topology as legacy; "
            "from witwin.channel_native.propagation.enumerated import diffraction"
        ),
    ),
)
def test_fresh_process_import_order_preserves_diffraction_identity(imports):
    code = f"{imports}; names={tuple(_DIGESTS)!r}; assert all(getattr(legacy, n) is getattr(diffraction, n) for n in names)"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(REPOSITORY_ROOT / "src"), environment.get("PYTHONPATH"))
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


def test_diffraction_owner_has_no_core_path_dependency_or_scc():
    owner = "witwin.channel_native.propagation.enumerated.diffraction"
    core = "witwin.channel_native.core.path_topology"
    edges = graph.collect_import_edges(PACKAGE_ROOT)
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)
    assert core not in adjacency.get(owner, set())
    pending = [owner]
    seen = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(adjacency.get(current, ()))
    assert core not in seen
    assert not hasattr(diffraction, "_raydn_visibility_mask")
