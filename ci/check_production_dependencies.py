"""Reject legacy Channel Python stacks from production source trees."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


FORBIDDEN_MODULES = (
    "drjit",
    "mitsuba",
    "raydn",
    "sionna",
    "witwin.channel",
)
LEGACY_CHANNEL_MODULES = ("witwin.channel",)

_NON_PRODUCTION_PARTS = frozenset(
    {
        ".git",
        ".pytest_cache",
        "benchmarks",
        "build",
        "docs",
        "examples",
        "scripts",
        "tests",
        "tutorials",
    }
)


@dataclass(frozen=True, slots=True)
class Violation:
    path: Path
    line: int
    module: str


def _is_forbidden(module: str, forbidden_modules: tuple[str, ...]) -> bool:
    return any(
        module == forbidden or module.startswith(f"{forbidden}.")
        for forbidden in forbidden_modules
    )


def _imported_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level:
        return []
    base = node.module or ""
    if not base:
        return []
    if _is_forbidden(base, FORBIDDEN_MODULES):
        return [base]
    return [f"{base}.{alias.name}" for alias in node.names]


def scan_source(
    source: str,
    path: Path = Path("<memory>"),
    *,
    forbidden_modules: tuple[str, ...] = FORBIDDEN_MODULES,
) -> list[Violation]:
    tree = ast.parse(source.lstrip("\ufeff"), filename=str(path))
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for module in _imported_modules(node):
            if _is_forbidden(module, forbidden_modules):
                violations.append(Violation(path, node.lineno, module))
    return violations


def scan_file(
    path: Path, *, forbidden_modules: tuple[str, ...] = FORBIDDEN_MODULES
) -> list[Violation]:
    return scan_source(
        path.read_text(encoding="utf-8-sig"),
        path,
        forbidden_modules=forbidden_modules,
    )


def production_python_files(root: Path) -> list[Path]:
    root = root.resolve()
    source_root = root / "src" if (root / "src").is_dir() else root
    return sorted(
        path
        for path in source_root.rglob("*.py")
        if not _NON_PRODUCTION_PARTS.intersection(path.relative_to(source_root).parts)
    )


def scan_roots(
    roots: list[Path], *, forbidden_modules: tuple[str, ...] = FORBIDDEN_MODULES
) -> list[Violation]:
    return [
        violation
        for root in roots
        for path in production_python_files(root)
        for violation in scan_file(path, forbidden_modules=forbidden_modules)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path(__file__).resolve().parents[1]],
        help="repository or source roots (defaults to this repository)",
    )
    parser.add_argument(
        "--consumer-roots",
        action="store_true",
        help=(
            "scan sibling products only for witwin.channel; independent stacks "
            "such as Radar's DrJit tracer remain outside Channel migration"
        ),
    )
    args = parser.parse_args(argv)
    forbidden_modules = (
        LEGACY_CHANNEL_MODULES if args.consumer_roots else FORBIDDEN_MODULES
    )
    violations = scan_roots(args.roots, forbidden_modules=forbidden_modules)
    if violations:
        for violation in violations:
            print(f"{violation.path}:{violation.line}: forbidden import {violation.module}")
        return 1
    print(f"production dependency contract passed for {len(args.roots)} root(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
