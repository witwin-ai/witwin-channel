#!/usr/bin/env python
# Copyright Xingyu Chen.
# Reject a second definition site for a protected domain concept.

"""Reject a second definition site for a protected domain concept."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path


PACKAGE = "witwin.channel"
DEFAULT_PACKAGE_PATH = Path("witwin/channel")

_COMPONENT_NAMES = frozenset(
    {"los", "reflection", "diffraction", "transmission", "scattering"}
)


@dataclass(frozen=True, slots=True)
class Signature:
    """A name-independent fingerprint of one concept.

 A function matches when its facts are a superset of every declared field.
 Empty fields are not constraints.
 """

    strings: frozenset[str] = frozenset()
    fragments: frozenset[str] = frozenset()
    numbers: frozenset[float] = frozenset()
    calls: frozenset[str] = frozenset()
    must_raise: bool = False

    def matches(self, facts: FunctionFacts) -> bool:
        if self.must_raise and not facts.raises:
            return False
        if not self.strings <= facts.strings:
            return False
        if not self.numbers <= facts.numbers:
            return False
        if not self.calls <= facts.calls:
            return False
        return all(
            any(fragment in value for value in facts.strings)
            for fragment in self.fragments
        )


@dataclass(frozen=True, slots=True)
class Concept:
    """A domain concept that may have exactly one definition site."""

    name: str
    owner: str
    reason: str
    signature: Signature
    owner_definitions: int = 1
    recorded_duplicates: frozenset[tuple[str, str]] = field(
        default_factory=frozenset
    )
    debt: str = ""


# --- The registry ---------------------------------------------------------
#
# Each entry names the concept, its canonical owner, and why one site is the
# right number. Signatures are vocabulary, never identifiers; see the module
# docstring for why.

CONCEPTS: tuple[Concept, ...] = (
    Concept(
        name="component availability status",
        owner=f"{PACKAGE}.components",
        reason=(
            "the mapping from requested components to enabled/not_requested is "
            "one rule shared by all four solvers; three solver metadata modules "
            "each carried their own copy until Phase 1a"
        ),
        signature=Signature(
            strings=_COMPONENT_NAMES | {"enabled", "not_requested"},
            must_raise=True,
        ),
    ),
    Concept(
        name="component max-depth rule",
        owner=f"{PACKAGE}.components",
        reason=(
            "per-component interaction depth with -1 for 'not requested' is one "
            "rule; it reached a fourth copy in deterministic/pipeline.py while "
            "the plan to remove the first three was being written"
        ),
        signature=Signature(
            strings=_COMPONENT_NAMES,
            numbers=frozenset({0.0, -1.0}),
        ),
    ),
    Concept(
        name="native symbol lookup",
        owner=f"{PACKAGE}.runtime",
        reason=(
            "probing the validated extension for a required symbol is the "
            "runtime's job; a hand-rolled probe is a second place the "
            "fail-loud message and the None-extension case can drift"
        ),
        signature=Signature(
            fragments=frozenset({"_channel."}),
            calls=frozenset({"hasattr"}),
            must_raise=True,
        ),
    ),
)


# --- Fact extraction ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FunctionFacts:
    """What one function scope contains, with every identifier discarded."""

    module: str
    qualname: str
    lineno: int
    strings: frozenset[str]
    numbers: frozenset[float]
    calls: frozenset[str]
    raises: bool


def module_name(package_root: Path, path: Path) -> str:
    parts = list(path.relative_to(package_root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([PACKAGE, *parts]) if parts else PACKAGE


def _callee(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _ScopeCollector(ast.NodeVisitor):
    """Partition a module's nodes by their nearest enclosing function.

 Module-level statements and class bodies land in a synthetic ``<module>``
 scope, so a rule written as a top-level dict is still a definition site.
 """

    def __init__(self, module: str) -> None:
        self.module = module
        self._stack: list[str] = ["<module>"]
        self.scopes: dict[str, dict[str, object]] = {}
        self._open("<module>", 0)

    def _open(self, qualname: str, lineno: int) -> None:
        self.scopes.setdefault(
            qualname,
            {
                "lineno": lineno,
                "strings": set(),
                "numbers": set(),
                "calls": set(),
                "raises": False,
            },
        )

    @property
    def _current(self) -> dict[str, object]:
        return self.scopes[self._stack[-1]]

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = (
            node.name
            if self._stack[-1] == "<module>"
            else f"{self._stack[-1]}.{node.name}"
        )
        self._open(qualname, node.lineno)
        self._stack.append(qualname)
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Constant(self, node: ast.Constant) -> None:
        scope = self._current
        if isinstance(node.value, str):
            scope["strings"].add(node.value)  # type: ignore[union-attr]
        elif isinstance(node.value, bool):
            pass
        elif isinstance(node.value, (int, float)):
            scope["numbers"].add(float(node.value))  # type: ignore[union-attr]

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        # `-1` parses as UnaryOp(USub, Constant(1)). Record the value as
        # written and do not descend, so a function that contains only `-1`
        # does not also claim to contain `1`.
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            value = node.operand.value
            if not isinstance(value, bool) and isinstance(value, (int, float)):
                self._current["numbers"].add(-float(value))  # type: ignore[union-attr]
                return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _callee(node)
        if name is not None:
            self._current["calls"].add(name)  # type: ignore[union-attr]
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self._current["raises"] = True
        self.generic_visit(node)


def function_facts(source: str, module: str, filename: str) -> list[FunctionFacts]:
    collector = _ScopeCollector(module)
    collector.visit(ast.parse(source, filename=filename))
    return [
        FunctionFacts(
            module=module,
            qualname=qualname,
            lineno=int(scope["lineno"]),  # type: ignore[arg-type]
            strings=frozenset(scope["strings"]),  # type: ignore[arg-type]
            numbers=frozenset(scope["numbers"]),  # type: ignore[arg-type]
            calls=frozenset(scope["calls"]),  # type: ignore[arg-type]
            raises=bool(scope["raises"]),
        )
        for qualname, scope in collector.scopes.items()
    ]


def package_facts(package_root: Path) -> list[FunctionFacts]:
    facts: list[FunctionFacts] = []
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        facts.extend(
            function_facts(
                path.read_text(encoding="utf-8"),
                module_name(package_root, path),
                str(path),
            )
        )
    return facts


# --- The rule -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Violation:
    concept: str
    kind: str
    detail: str


def definition_sites(concept: Concept, facts: list[FunctionFacts]) -> list[FunctionFacts]:
    return sorted(
        (fact for fact in facts if concept.signature.matches(fact)),
        key=lambda fact: (fact.module, fact.qualname),
    )


def concept_violations(concept: Concept, facts: list[FunctionFacts]) -> list[Violation]:
    sites = definition_sites(concept, facts)
    # A recorded duplicate is a copy wherever it lives, including inside the
    # owner module, so it never counts towards the owner's definition budget.
    owned = [
        site
        for site in sites
        if site.module == concept.owner
        and (site.module, site.qualname) not in concept.recorded_duplicates
    ]
    matched = {
        (site.module, site.qualname)
        for site in sites
        if (site.module, site.qualname) in concept.recorded_duplicates
    }
    foreign = {
        (site.module, site.qualname) for site in sites if site.module != concept.owner
    } | matched

    violations: list[Violation] = []
    if len(owned) != concept.owner_definitions:
        violations.append(
            Violation(
                concept.name,
                "owner-definition-count",
                f"{concept.owner} defines this concept {len(owned)} time(s), "
                f"expected {concept.owner_definitions}: "
                + (
                    ", ".join(f"{site.qualname}:{site.lineno}" for site in owned)
                    or "no definition found"
                ),
            )
        )
    for module, qualname in sorted(foreign - concept.recorded_duplicates):
        violations.append(
            Violation(
                concept.name,
                "second-definition",
                f"{module}.{qualname} redefines this concept; call "
                f"{concept.owner} instead",
            )
        )
    for module, qualname in sorted(concept.recorded_duplicates - matched):
        violations.append(
            Violation(
                concept.name,
                "stale-recorded-duplicate",
                f"{module}.{qualname} no longer matches; delete it from "
                "recorded_duplicates so the ratchet stays tight",
            )
        )
    return violations


def find_violations(
    package_root: Path, concepts: tuple[Concept, ...] = CONCEPTS,
) -> list[Violation]:
    facts = package_facts(package_root)
    return [
        violation
        for concept in concepts
        for violation in concept_violations(concept, facts)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args(argv)

    package_root = (
        args.package_root or args.repository_root.resolve() / DEFAULT_PACKAGE_PATH
    ).resolve()

    violations = find_violations(package_root)
    if violations:
        print(
            "single-definition check failed: a protected domain concept has "
            "more than one definition site. Detection is by concept "
            "vocabulary, not by function name, so a renamed near-copy still "
            "reports here:"
        )
        for violation in violations:
            print(f"  [{violation.concept}] {violation.kind}: {violation.detail}")
        return 1

    recorded = sum(len(concept.recorded_duplicates) for concept in CONCEPTS)
    print(
        f"single-definition check passed for {len(CONCEPTS)} protected "
        f"concepts ({recorded} recorded pre-existing duplicate(s), which may "
        "only shrink)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())