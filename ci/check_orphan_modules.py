#!/usr/bin/env python
# Copyright Xingyu Chen.
# Reject a production module that no declared entry point can reach.

"""Reject a production module that no declared entry point can reach."""

from __future__ import annotations

import argparse
import ast
from collections import deque
from pathlib import Path


PACKAGE = "witwin.channel"
DEFAULT_PACKAGE_PATH = Path("witwin/channel")

# The stable public API, per CLAUDE.md: the package root and the four solver
# entry points. The root exports only what Channel owns, so it does not reach a
# solver; each solver is a module a caller imports directly.
ENTRY_POINTS: dict[str, str] = {
    PACKAGE: (
        "the package root and the public surface frozen in "
        "ci/public-api-snapshot.json"
    ),
    f"{PACKAGE}.path": "the Path solver entry point",
    f"{PACKAGE}.deterministic": "the Deterministic solver entry point",
    f"{PACKAGE}.montecarlo.basic": "the Monte Carlo Basic solver entry point",
    f"{PACKAGE}.montecarlo.bdpt": "the Monte Carlo BDPT solver entry point",
}


def module_name(package_root: Path, path: Path) -> str:
    parts = list(path.relative_to(package_root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([PACKAGE, *parts]) if parts else PACKAGE


def _package_of(name: str, is_package: bool) -> str:
    return name if is_package else name.rpartition(".")[0]


def _resolve(base_package: str, level: int, module: str | None) -> str:
    parts = base_package.split(".")
    if level > 1:
        parts = parts[: -(level - 1)]
    base = ".".join(parts)
    return f"{base}.{module}" if module else base


def edges_from(path: Path, name: str, known: set[str]) -> set[str]:
    """The modules that importing `path` also imports."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    base_package = _package_of(name, path.name == "__init__.py")
    out: set[str] = set()

    def note(candidate: str) -> None:
        if candidate in known and candidate != name:
            out.add(candidate)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                note(alias.name)
        elif isinstance(node, ast.ImportFrom):
            target = (
                _resolve(base_package, node.level, node.module)
                if node.level
                else (node.module or "")
            )
            note(target)
            for alias in node.names:
                note(f"{target}.{alias.name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            note(f"{base_package}.{node.value}")

    parent = name.rpartition(".")[0]
    if parent and parent.startswith(PACKAGE):
        note(parent)
    return out


def module_paths(package_root: Path) -> dict[str, Path]:
    return {
        module_name(package_root, path): path
        for path in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def find_orphans(package_root: Path) -> list[str]:
    """Production modules unreachable from every entry point that exists."""

    paths = module_paths(package_root)
    known = set(paths)
    graph = {name: edges_from(path, name, known) for name, path in paths.items()}

    visited: set[str] = set()
    queue = deque(sorted(name for name in ENTRY_POINTS if name in known))
    while queue:
        name = queue.popleft()
        if name in visited:
            continue
        visited.add(name)
        queue.extend(sorted(graph[name] - visited))
    return sorted(known - visited)


def stale_entry_points(package_root: Path) -> list[str]:
    return sorted(set(ENTRY_POINTS) - set(module_paths(package_root)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args(argv)

    repository_root = args.repository_root.resolve()
    package_root = (
        args.package_root or repository_root / DEFAULT_PACKAGE_PATH
    ).resolve()

    stale = stale_entry_points(package_root)
    if stale:
        print(
            "orphan module check failed: ENTRY_POINTS names modules that do "
            "not exist; delete the stale entries:"
        )
        for name in stale:
            print(f"  {name}")
        return 1

    orphans = find_orphans(package_root)
    if orphans:
        paths = module_paths(package_root)
        print(
            "orphan module check failed: unreachable production module(s). "
            "No production module imports these, directly or transitively, "
            "from any entry point. Delete them, or - if a caller imports one "
            "directly - add it to ENTRY_POINTS with the reason:"
        )
        for name in orphans:
            relative = paths[name].resolve()
            try:
                relative = relative.relative_to(repository_root)
            except ValueError:
                pass
            print(f"  {name}  ({relative.as_posix()})")
        return 1

    total = len(module_paths(package_root))
    print(
        f"orphan module check passed for {total} production modules, all "
        f"reachable from {len(ENTRY_POINTS)} declared entry points"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())