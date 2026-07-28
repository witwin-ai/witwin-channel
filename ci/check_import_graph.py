"""Enforce the Channel architecture import boundaries with AST analysis."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE = "witwin.channel"
DEFAULT_PACKAGE_PATH = Path("src/witwin/channel")
DEFAULT_ALLOWLIST_PATH = Path("ci/import_graph_allowlist.json")

# This digest freezes the debt universe. Entries may be removed from the active
# allowlist, but relocating or replacing an entry is rejected. The universe was
# extended once to admit the ADR-008 BDPT enumerated-oracle dependency that the
# re-export canonicalization made visible.
#
# Re-baselined 2026-07-27 by owner decision, as a strict reduction. Two things
# had made the frozen universe wrong rather than protective:
#
#   * All eight ``existing_boundary`` entries named a path or a module that no
#     longer exists. Seven of them (boundary-001, -002, -004, -005, -006, -007,
#     -008) reached through ``witwin.channel.core``, the namespace this checker
#     itself rejects via ``_DISSOLVED_PREFIXES``; boundary-003 named
#     ``witwin.channel.runtime.extension``, gone since ``runtime`` collapsed
#     into a single module. Freezing entries that cannot be produced protects
#     nothing, so they were deleted. The group itself is kept with an empty
#     baseline so a future ``existing_boundary`` violation still reports as an
#     unallowlisted violation rather than a missing debt group.
#   * mc-enum-001, the single *active* entry, was keyed on
#     ``montecarlo/bdpt/pipeline.py``. Collapsing that package into
#     ``montecarlo/bdpt.py`` relocates the sanctioned ADR-008 edge without
#     changing it, which the digest rejects by design. The entry was re-keyed
#     ahead of that collapse: only ``path`` and ``source`` moved, and the rule,
#     target, ADR and justification are untouched. Its ``line``/``column``
#     still carried the pre-collapse 23/0; the concept-axis sealing step that
#     landed the collapse corrected ``line`` to the real position in the
#     collapsed module, 105, leaving ``column`` at 0, and recomputed this digest
#     for that single-field correction. Nothing else in the universe moved.
#
# Nothing was added: the new universe is the old one minus those eight entries,
# with mc-enum-001 re-keyed and solver-001 unchanged.
#
# Re-keyed again when ``propagation/enumerated/`` collapsed into
# ``propagation/enumerated.py``. mc-enum-001's ``target`` named the defining
# submodule ``witwin.channel.propagation.enumerated.engine``, which stopped
# existing; the re-export canonicalization now resolves the same BDPT edge to
# ``witwin.channel.propagation.enumerated``. This is the mirror image of the
# bdpt collapse above: the sanctioned ADR-008 edge did not move, its defining
# module did. Only ``target`` changed, and the digest was recomputed for that
# single field. The rule, source, line, column, ADR and justification are
# untouched, and nothing was added to the universe.
FROZEN_BASELINE_DIGEST = (
    "17e3e0010425dca8335cfd513cc578df36e050b91ba1f628c5b699d05efbe91b"
)

_DEBT_GROUP_BY_RULE = {
    "solver_to_solver": "solver_to_solver",
    "public_init_internal": "existing_boundary",
    "relative_cross_domain": "existing_boundary",
    "solver_raw_extension": "existing_boundary",
    "mc_enumerated_dependency": "mc_enumerated_dependency",
}
_SOLVER_PREFIXES = (
    f"{PACKAGE}.path",
    f"{PACKAGE}.deterministic",
    f"{PACKAGE}.montecarlo.basic",
    f"{PACKAGE}.montecarlo.bdpt",
)
# The rule these name is about a public *init*: a package ``__init__.py`` that
# publishes a domain's surface must stay a re-export facade and may not reach
# into kernels, runtime or propagation itself. It is therefore conditioned on
# the source actually being a package init (``ImportEdge.source_is_package``).
# A solver that has collapsed into a single module - ``deterministic.py`` rather
# than ``deterministic/__init__.py`` - is the solver body, not a facade over it,
# so the facade rule cannot apply to it and does not. What that solver may
# import is still bounded by the solver-to-solver, raw-extension and
# propagation-ordering rules below, none of which is relaxed here.
_PUBLIC_INIT_MODULES = frozenset({PACKAGE, *_SOLVER_PREFIXES})
_RAW_EXTENSION_MODULES = frozenset({f"{PACKAGE}._channel"})
# ``witwin.channel.runtime`` is a single module, so the raw accessor it exports
# has no submodule of its own to name. The rule therefore names the symbol.
_RAW_EXTENSION_SYMBOLS = frozenset({"native_extension"})
_COMPILED_SCENE_MODULES = frozenset(
    {f"{PACKAGE}.scene.compiled", f"{PACKAGE}.scene.compiler"}
)
# The `core` grab-bag was dissolved into its real domain owners. Nothing may
# recreate that namespace or import through it.
_DISSOLVED_PREFIXES = (f"{PACKAGE}.core",)


@dataclass(frozen=True, order=True, slots=True)
class ImportEdge:
    path: str
    line: int
    column: int
    source: str
    target: str
    kind: str
    imported_name: str = ""
    relative_level: int = 0
    source_is_package: bool = False


@dataclass(frozen=True, order=True, slots=True)
class Violation:
    path: str
    line: int
    column: int
    rule: str
    source: str
    target: str

    def key(self) -> tuple[str, int, int, str, str, str]:
        return (
            self.path,
            self.line,
            self.column,
            self.rule,
            self.source,
            self.target,
        )


def _matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _solver_owner(module: str) -> str | None:
    return next(
        (prefix for prefix in _SOLVER_PREFIXES if _matches(module, prefix)), None
    )


def _top_domain(module: str) -> str | None:
    if not _matches(module, PACKAGE) or module == PACKAGE:
        return None
    return module[len(PACKAGE) + 1 :].split(".", maxsplit=1)[0]


def _module_name(package_root: Path, path: Path) -> tuple[str, bool]:
    relative = path.relative_to(package_root)
    is_package = relative.name == "__init__.py"
    parts = list(relative.parts[:-1] if is_package else relative.with_suffix("").parts)
    suffix = ".".join(parts)
    return (f"{PACKAGE}.{suffix}" if suffix else PACKAGE), is_package


def _resolve_from_module(
    source: str, *, is_package: bool, level: int, module: str | None
) -> str:
    if level == 0:
        return module or ""
    package_parts = source.split(".") if is_package else source.split(".")[:-1]
    ascend = level - 1
    if ascend > len(package_parts):
        return "<outside-package>"
    base = package_parts[: len(package_parts) - ascend]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def _target_module(base: str, alias: str, known_modules: frozenset[str]) -> str:
    candidate = f"{base}.{alias}" if base else alias
    return candidate if candidate in known_modules else base


def _parse_file(
    package_root: Path,
    path: Path,
    *,
    known_modules: frozenset[str],
) -> list[ImportEdge]:
    source, is_package = _module_name(package_root, path)
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    repository_root = package_root.parents[2]
    display_path = path.relative_to(repository_root).as_posix()
    edges: list[ImportEdge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            edges.extend(
                ImportEdge(
                    display_path,
                    node.lineno,
                    node.col_offset,
                    source,
                    alias.name,
                    "import",
                    alias.name,
                    0,
                    is_package,
                )
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_module(
                source,
                is_package=is_package,
                level=node.level,
                module=node.module,
            )
            edges.extend(
                ImportEdge(
                    display_path,
                    node.lineno,
                    node.col_offset,
                    source,
                    (
                        "*"
                        if alias.name == "*"
                        else _target_module(base, alias.name, known_modules)
                    ),
                    "from",
                    alias.name,
                    node.level,
                    is_package,
                )
                for alias in node.names
            )
    return edges


def _collect_module_info(
    package_root: Path,
) -> tuple[list[Path], frozenset[str], frozenset[str]]:
    files = sorted(package_root.rglob("*.py"))
    named = [_module_name(package_root, path) for path in files]
    known_modules = frozenset(module for module, _ in named)
    known_packages = frozenset(module for module, is_package in named if is_package)
    return files, known_modules, known_packages


def collect_import_edges(package_root: Path) -> list[ImportEdge]:
    package_root = package_root.resolve()
    files, known_modules, _ = _collect_module_info(package_root)
    return sorted(
        edge
        for path in files
        for edge in _parse_file(package_root, path, known_modules=known_modules)
    )


def build_reexport_map(package_root: Path) -> dict[tuple[str, str], str]:
    """Map ``(package module, exposed symbol)`` to the submodule that defines it.

    A package ``__init__`` that re-exports a symbol with ``from .sub import name``
    (or the absolute equivalent) publishes ``name`` from its own namespace while
    the object lives in a submodule. This map lets the classifier resolve such
    re-exports back to the defining module, so a boundary that is bypassed
    through a package facade stays visible instead of hiding behind the package.
    """

    package_root = package_root.resolve()
    files, known_modules, _ = _collect_module_info(package_root)
    reexports: dict[tuple[str, str], str] = {}
    for path in files:
        module, is_package = _module_name(package_root, path)
        if not is_package:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            base = _resolve_from_module(
                module, is_package=True, level=node.level, module=node.module
            )
            if not _matches(base, PACKAGE):
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                exposed = alias.asname or alias.name
                reexports[(module, exposed)] = _target_module(
                    base, alias.name, known_modules
                )
        for function in (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__getattr__"
        ):
            for branch in (node for node in ast.walk(function) if isinstance(node, ast.If)):
                names: set[str] = set()
                if (
                    isinstance(branch.test, ast.Compare)
                    and isinstance(branch.test.left, ast.Name)
                    and branch.test.left.id == "name"
                    and len(branch.test.ops) == 1
                    and len(branch.test.comparators) == 1
                ):
                    comparator = branch.test.comparators[0]
                    if (
                        isinstance(branch.test.ops[0], ast.Eq)
                        and isinstance(comparator, ast.Constant)
                        and isinstance(comparator.value, str)
                    ):
                        names.add(comparator.value)
                    elif isinstance(branch.test.ops[0], ast.In) and isinstance(
                        comparator, (ast.Set, ast.Tuple, ast.List)
                    ):
                        names.update(
                            item.value
                            for item in comparator.elts
                            if isinstance(item, ast.Constant)
                            and isinstance(item.value, str)
                        )
                for assignment in (
                    node
                    for node in branch.body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "value"
                        for target in node.targets
                    )
                ):
                    value = assignment.value
                    imported_module: str | None = None
                    imported_name: str | None = None
                    if (
                        isinstance(value, ast.Attribute)
                        and isinstance(value.value, ast.Call)
                        and isinstance(value.value.func, ast.Name)
                        and value.value.func.id == "import_module"
                        and value.value.args
                        and isinstance(value.value.args[0], ast.Constant)
                        and isinstance(value.value.args[0].value, str)
                    ):
                        imported_module = value.value.args[0].value
                        imported_name = value.attr
                    elif (
                        isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Name)
                        and value.func.id == "getattr"
                        and len(value.args) == 2
                        and isinstance(value.args[0], ast.Call)
                        and isinstance(value.args[0].func, ast.Name)
                        and value.args[0].func.id == "import_module"
                        and value.args[0].args
                        and isinstance(value.args[0].args[0], ast.Constant)
                        and isinstance(value.args[0].args[0].value, str)
                        and isinstance(value.args[1], ast.Name)
                        and value.args[1].id == "name"
                    ):
                        imported_module = value.args[0].args[0].value
                    if imported_module is None or not _matches(
                        imported_module, PACKAGE
                    ):
                        continue
                    for exposed in names:
                        reexports[(module, exposed)] = _target_module(
                            imported_module,
                            imported_name or exposed,
                            known_modules,
                        )
    return reexports


def _canonical_target(
    target: str,
    imported_name: str,
    reexports: dict[tuple[str, str], str],
    known_packages: frozenset[str],
    *,
    max_depth: int = 8,
) -> str:
    """Follow package re-exports until reaching the defining module.

    Resolution stops at a private ``kernels`` module: publishing a kernel symbol
    through a package ``__all__`` is the sanctioned seam for consumers, so those
    edges stay recorded at the public package target and remain governed by the
    domain-kernel boundary rules rather than being rewritten into a private
    cross-domain kernel import.
    """

    seen = {target}
    for _ in range(max_depth):
        if target not in known_packages:
            break
        defining = reexports.get((target, imported_name))
        if defining is None or defining in seen or _is_kernels_module(defining):
            break
        seen.add(defining)
        target = defining
    return target


def _canonicalize_edge(
    edge: ImportEdge,
    reexports: dict[tuple[str, str], str],
    known_packages: frozenset[str],
) -> ImportEdge:
    if edge.kind != "from" or edge.imported_name in {"", "*"}:
        return edge
    target = _canonical_target(
        edge.target, edge.imported_name, reexports, known_packages
    )
    return edge if target == edge.target else replace(edge, target=target)


def _is_solver_result(module: str) -> bool:
    owner = _solver_owner(module)
    return owner is not None and _matches(module, f"{owner}.result")


def _kernel_owner(module: str) -> str | None:
    marker = ".kernels"
    index = module.find(marker, len(PACKAGE))
    if index < 0:
        return None
    return module[:index]


def _is_kernels_module(module: str) -> bool:
    return _kernel_owner(module) is not None


def _violation(edge: ImportEdge, rule: str) -> Violation:
    return Violation(
        edge.path,
        edge.line,
        edge.column,
        rule,
        edge.source,
        edge.target,
    )


def _basic_boundary_violations(edge: ImportEdge) -> list[Violation]:
    violations: list[Violation] = []
    source = edge.source
    target = edge.target
    if (
        edge.relative_level
        and _matches(source, PACKAGE)
        and _matches(target, PACKAGE)
        and _top_domain(source) is not None
        and _top_domain(source) != _top_domain(target)
    ):
        violations.append(_violation(edge, "relative_cross_domain"))

    deleted_modules = {
        f"{PACKAGE}.core.kernels.ops",
        f"{PACKAGE}.core.path_topology",
    }
    if any(_matches(target, prefix) for prefix in _DISSOLVED_PREFIXES):
        violations.append(_violation(edge, "dissolved_module_dependency"))
    imported_target = (
        f"{target}.{edge.imported_name}"
        if edge.kind == "from" and edge.imported_name not in {"", "*"}
        else target
    )
    if target in deleted_modules or imported_target in deleted_modules:
        violations.append(_violation(edge, "deleted_module_dependency"))
    if edge.source_is_package and source in _PUBLIC_INIT_MODULES and (
        _is_kernels_module(target)
        or _matches(target, f"{PACKAGE}.runtime")
        or _matches(target, f"{PACKAGE}.propagation")
    ):
        violations.append(_violation(edge, "public_init_internal"))
    return violations


def _solver_boundary_violations(edge: ImportEdge) -> list[Violation]:
    violations: list[Violation] = []
    source_solver = _solver_owner(edge.source)
    target_solver = _solver_owner(edge.target)

    if source_solver is not None and target_solver not in {None, source_solver}:
        violations.append(_violation(edge, "solver_to_solver"))
    # ``witwin.channel.runtime`` is one module now, so its raw accessor is
    # reachable only as a symbol import. The rule is fail-closed on both
    # spellings: the extension module itself, and the accessor imported out of
    # the collapsed runtime module. No solver needs either - every solver
    # kernel facade goes through ``runtime.required_symbol``, which owns the
    # None-extension case and the fail-loud message.
    if source_solver is not None and (
        edge.target in _RAW_EXTENSION_MODULES
        or (
            edge.target == f"{PACKAGE}.runtime"
            and edge.imported_name in _RAW_EXTENSION_SYMBOLS
        )
    ):
        violations.append(_violation(edge, "solver_raw_extension"))

    if source_solver in {
        f"{PACKAGE}.path",
        f"{PACKAGE}.deterministic",
    } and _matches(edge.target, f"{PACKAGE}.montecarlo"):
        violations.append(_violation(edge, "enumerated_pipeline_mc_internal"))
    if source_solver in {
        f"{PACKAGE}.montecarlo.basic",
        f"{PACKAGE}.montecarlo.bdpt",
    } and _matches(edge.target, f"{PACKAGE}.propagation.enumerated"):
        violations.append(_violation(edge, "mc_enumerated_dependency"))
    return violations


def _propagation_boundary_violations(edge: ImportEdge) -> list[Violation]:
    violations: list[Violation] = []
    source = edge.source
    target = edge.target
    target_solver = _solver_owner(target)

    if _matches(source, f"{PACKAGE}.propagation.enumerated") and (
        target_solver is not None
        or _matches(target, f"{PACKAGE}.montecarlo")
        or _matches(target, f"{PACKAGE}.deterministic.accumulation")
    ):
        violations.append(_violation(edge, "enumerated_forbidden_dependency"))

    if _matches(source, f"{PACKAGE}.propagation.topology") and (
        _matches(target, f"{PACKAGE}.propagation.geometry")
        or _matches(target, f"{PACKAGE}.propagation.fields")
        or target_solver is not None
        or _matches(target, f"{PACKAGE}.scene")
        or target in _COMPILED_SCENE_MODULES
    ):
        violations.append(_violation(edge, "topology_forbidden_dependency"))

    if _matches(source, f"{PACKAGE}.propagation.geometry") and (
        _matches(target, f"{PACKAGE}.propagation.topology.discovery")
        or _matches(target, f"{PACKAGE}.propagation.fields")
        or target_solver is not None
        or (
            _matches(source, f"{PACKAGE}.propagation.geometry.kernels")
            and (
                _matches(target, f"{PACKAGE}.scene")
                or target in _COMPILED_SCENE_MODULES
            )
        )
    ):
        violations.append(_violation(edge, "geometry_forbidden_dependency"))

    if _matches(source, f"{PACKAGE}.propagation.fields") and (
        _matches(target, f"{PACKAGE}.propagation.topology.discovery")
        or _is_solver_result(target)
    ):
        violations.append(_violation(edge, "fields_forbidden_dependency"))
    return violations


def _runtime_oracle_violations(edge: ImportEdge) -> list[Violation]:
    violations: list[Violation] = []
    source = edge.source
    target = edge.target

    if _matches(source, f"{PACKAGE}.runtime") and any(
        _matches(target, prefix)
        for prefix in (
            f"{PACKAGE}.scene",
            f"{PACKAGE}.propagation",
            f"{PACKAGE}.materials",
            f"{PACKAGE}.scattering",
            *_SOLVER_PREFIXES,
        )
    ):
        violations.append(_violation(edge, "runtime_forbidden_dependency"))

    # The NumPy reference oracle now lives in ``tests/reference/em_oracle.py``.
    # Its isolation from production is structural: this checker only walks the
    # shipped package, so a production module cannot reach it at all.
    return violations


def _domain_kernel_violations(edge: ImportEdge) -> list[Violation]:
    violations: list[Violation] = []
    target_solver = _solver_owner(edge.target)
    source_kernel_owner = _kernel_owner(edge.source)
    target_kernel_owner = _kernel_owner(edge.target)

    if source_kernel_owner is not None:
        if target_solver is not None:
            violations.append(_violation(edge, "domain_kernel_solver_dependency"))
        if (
            target_kernel_owner is not None
            and target_kernel_owner != source_kernel_owner
        ):
            violations.append(_violation(edge, "cross_domain_private_kernel"))
        if edge.target == PACKAGE:
            violations.append(_violation(edge, "domain_kernel_public_init"))
    return violations


def classify_edge(edge: ImportEdge) -> list[Violation]:
    if edge.target == "*":
        return [_violation(edge, "wildcard_import")]
    if edge.relative_level and not _matches(edge.target, PACKAGE):
        return [_violation(edge, "relative_outside_package")]

    violations = [
        *_basic_boundary_violations(edge),
        *_solver_boundary_violations(edge),
        *_propagation_boundary_violations(edge),
        *_runtime_oracle_violations(edge),
        *_domain_kernel_violations(edge),
    ]

    return sorted(set(violations))


def scan_package(package_root: Path) -> list[Violation]:
    package_root = package_root.resolve()
    files, known_modules, known_packages = _collect_module_info(package_root)
    reexports = build_reexport_map(package_root)
    edges = (
        _canonicalize_edge(edge, reexports, known_packages)
        for path in files
        for edge in _parse_file(package_root, path, known_modules=known_modules)
    )
    return sorted({violation for edge in edges for violation in classify_edge(edge)})


def _entry_key(entry: dict[str, Any]) -> tuple[str, int, int, str, str, str]:
    return (
        str(entry["path"]),
        int(entry["line"]),
        int(entry["column"]),
        str(entry["rule"]),
        str(entry["source"]),
        str(entry["target"]),
    )


def _baseline_digest(debts: dict[str, Any]) -> str:
    baseline = {name: group["baseline"] for name, group in sorted(debts.items())}
    canonical = json.dumps(
        baseline, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_allowlist(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("import graph allowlist must use schema_version 1")
    debts = data.get("debts")
    if not isinstance(debts, dict):
        raise ValueError("import graph allowlist must contain a debts object")
    return data


def _partition_violations(
    violations: list[Violation], debt_names: set[str]
) -> tuple[
    dict[str, set[tuple[str, int, int, str, str, str]]],
    list[Violation],
    list[str],
]:
    actual_by_group = {name: set() for name in debt_names}
    hard_violations: list[Violation] = []
    issues: list[str] = []
    for violation in violations:
        group = _DEBT_GROUP_BY_RULE.get(violation.rule)
        if group is None:
            hard_violations.append(violation)
        elif group not in actual_by_group:
            issues.append(f"allowlist is missing debt group {group!r}")
        else:
            actual_by_group[group].add(violation.key())
    return actual_by_group, hard_violations, issues


def _check_debt_group(
    name: str,
    group: object,
    actual: set[tuple[str, int, int, str, str, str]],
) -> list[str]:
    if not isinstance(group, dict):
        return [f"debt group {name!r} must be an object"]
    baseline_entries = group.get("baseline")
    allowed_ids = group.get("allowed")
    if not isinstance(baseline_entries, list) or not isinstance(allowed_ids, list):
        return [f"debt group {name!r} needs baseline and allowed lists"]
    if not all(isinstance(entry, dict) for entry in baseline_entries):
        return [f"debt group {name!r} baseline entries must be objects"]
    if not all(isinstance(entry_id, str) for entry_id in allowed_ids):
        return [f"debt group {name!r} allowed entries must be IDs"]

    issues: list[str] = []
    baseline_by_id = {
        str(entry.get("id")): _entry_key(entry) for entry in baseline_entries
    }
    baseline = set(baseline_by_id.values())
    allowed_id_set = set(allowed_ids)
    if len(baseline_by_id) != len(baseline_entries) or "None" in baseline_by_id:
        issues.append(f"debt group {name!r} has duplicate or missing baseline IDs")
    if len(baseline) != len(baseline_entries):
        issues.append(f"debt group {name!r} has duplicate baseline entries")
    if len(allowed_id_set) != len(allowed_ids):
        issues.append(f"debt group {name!r} has duplicate allowed IDs")
    additions = allowed_id_set - set(baseline_by_id)
    if additions:
        issues.append(
            f"debt group {name!r} has unknown allowed IDs: "
            + ", ".join(sorted(additions))
        )
    allowed = {
        baseline_by_id[entry_id]
        for entry_id in allowed_id_set
        if entry_id in baseline_by_id
    }
    issues.extend(
        f"unallowlisted {name} violation: {_format_key(key)}"
        for key in sorted(actual - allowed)
    )
    issues.extend(
        f"stale {name} allowance: {_format_key(key)}"
        for key in sorted(allowed - actual)
    )
    return issues


def check_allowlist(
    violations: list[Violation], allowlist: dict[str, Any]
) -> list[str]:
    debts = allowlist["debts"]
    issues: list[str] = []
    digest = _baseline_digest(debts)
    if digest != FROZEN_BASELINE_DIGEST:
        issues.append(
            "frozen import debt universe changed: "
            f"expected {FROZEN_BASELINE_DIGEST}, got {digest}"
        )
    actual_by_group, hard_violations, partition_issues = _partition_violations(
        violations, set(debts)
    )
    issues.extend(partition_issues)
    for name, group in sorted(debts.items()):
        issues.extend(_check_debt_group(name, group, actual_by_group.get(name, set())))

    issues.extend(
        f"hard import violation: {_format_violation(violation)}"
        for violation in hard_violations
    )
    return issues


def _format_key(key: tuple[str, int, int, str, str, str]) -> str:
    path, line, column, rule, source, target = key
    return f"{path}:{line}:{column}: [{rule}] {source} -> {target}"


def _format_violation(violation: Violation) -> str:
    return _format_key(violation.key())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--allowlist", type=Path)
    parser.add_argument(
        "--print-current",
        action="store_true",
        help="print current violations as canonical JSON without applying allowances",
    )
    args = parser.parse_args(argv)
    repository_root = args.repository_root.resolve()
    package_root = (
        args.package_root or repository_root / DEFAULT_PACKAGE_PATH
    ).resolve()
    violations = scan_package(package_root)
    if args.print_current:
        print(
            json.dumps(
                [asdict(violation) for violation in violations],
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0

    allowlist_path = (
        args.allowlist or repository_root / DEFAULT_ALLOWLIST_PATH
    ).resolve()
    try:
        allowlist = load_allowlist(allowlist_path)
        issues = check_allowlist(violations, allowlist)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"import graph configuration error: {exc}")
        return 2
    if issues:
        for issue in issues:
            print(issue)
        return 1

    counts: dict[str, int] = {}
    for violation in violations:
        group = _DEBT_GROUP_BY_RULE.get(violation.rule, "hard")
        counts[group] = counts.get(group, 0) + 1
    summary = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    print(f"import graph contract passed ({summary or 'no debt'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
