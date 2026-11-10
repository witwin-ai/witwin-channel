# Copyright Xingyu Chen.
# Tests that the production import graph gains no new import cycle.

from __future__ import annotations

from pathlib import Path

from ci import check_import_graph as graph


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "witwin" / "channel"
# The scene compiler, its resources, endpoint geometry and the material encoder import each other
# lazily. That cycle may shrink or disappear; no other cycle may appear.
KNOWN_CYCLE = frozenset(
    {
        "witwin.channel.materials",
        "witwin.channel.scene.compiler",
        "witwin.channel.scene.endpoints",
        "witwin.channel.scene.resources",
    }
)


def _cycles(edges: list[graph.ImportEdge]) -> list[frozenset[str]]:
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[frozenset[str]] = []

    def visit(node: str) -> None:
        index[node] = low[node] = len(index)
        stack.append(node)
        on_stack.add(node)
        for target in adjacency.get(node, ()):
            if target not in index:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in on_stack:
                low[node] = min(low[node], index[target])
        if low[node] == index[node]:
            component = set()
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.add(member)
                if member == node:
                    break
            if len(component) > 1:
                components.append(frozenset(component))

    for node in sorted(adjacency):
        if node not in index:
            visit(node)
    return components


def test_cycle_detection_finds_a_planted_cycle() -> None:
    edges = [
        graph.ImportEdge("a.py", 1, 0, "pkg.a", "pkg.b", "from", "b", 0, False),
        graph.ImportEdge("b.py", 1, 0, "pkg.b", "pkg.a", "from", "a", 0, False),
        graph.ImportEdge("c.py", 1, 0, "pkg.c", "pkg.a", "from", "a", 0, False),
    ]
    assert _cycles(edges) == [frozenset({"pkg.a", "pkg.b"})]


def test_production_import_graph_has_no_new_cycle() -> None:
    cycles = _cycles(graph.collect_import_edges(PACKAGE_ROOT))
    assert all(cycle <= KNOWN_CYCLE for cycle in cycles), cycles
