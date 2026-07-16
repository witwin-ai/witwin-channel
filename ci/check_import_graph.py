"""Enforce the Channel Native architecture import boundaries with AST analysis."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE = "witwin.channel_native"
DEFAULT_PACKAGE_PATH = Path("src/witwin/channel_native")
DEFAULT_ALLOWLIST_PATH = Path("ci/import_graph_allowlist.json")

# This digest freezes the initial Phase 3 debt universe. Entries may be removed
# from the active allowlist, but relocating or replacing an entry is rejected.
FROZEN_BASELINE_DIGEST = (
    "daed11abc81eb111186aebc78826c52037b475f7682d68869bf85f0cac6d4e5a"
)

_DEBT_GROUP_BY_RULE = {
    "solver_to_solver": "solver_to_solver",
    "public_init_internal": "existing_boundary",
    "relative_cross_domain": "existing_boundary",
    "solver_raw_extension": "existing_boundary",
}
_SOLVER_PREFIXES = (
    f"{PACKAGE}.path",
    f"{PACKAGE}.deterministic",
    f"{PACKAGE}.montecarlo.basic",
    f"{PACKAGE}.montecarlo.bdpt",
)
_PUBLIC_INIT_MODULES = frozenset({PACKAGE, *_SOLVER_PREFIXES})
_RAW_EXTENSION_MODULES = frozenset(
    {
        f"{PACKAGE}._channel_native",
        f"{PACKAGE}.core.kernels.extension",
        f"{PACKAGE}.runtime.extension",
    }
)
_COMPILED_SCENE_MODULES = frozenset(
    {
        f"{PACKAGE}.scene.compiled",
        f"{PACKAGE}.core.runtime.compiled_scene",
    }
)


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
                )
                for alias in node.names
            )
    return edges


def collect_import_edges(package_root: Path) -> list[ImportEdge]:
    package_root = package_root.resolve()
    files = sorted(package_root.rglob("*.py"))
    modules = frozenset(_module_name(package_root, path)[0] for path in files)
    return sorted(
        edge
        for path in files
        for edge in _parse_file(package_root, path, known_modules=modules)
    )


def _is_solver_result(module: str) -> bool:
    owner = _solver_owner(module)
    return owner is not None and _matches(module, f"{owner}.result")


def _kernel_owner(module: str) -> str | None:
    marker = ".kernels"
    index = module.find(marker, len(PACKAGE))
    if index < 0 or _matches(module, f"{PACKAGE}.core.kernels"):
        return None
    return module[:index]


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
    imported_target = (
        f"{target}.{edge.imported_name}"
        if edge.kind == "from" and edge.imported_name not in {"", "*"}
        else target
    )
    if target in deleted_modules or imported_target in deleted_modules:
        violations.append(_violation(edge, "deleted_module_dependency"))
    if source in _PUBLIC_INIT_MODULES and (
        _matches(target, f"{PACKAGE}.core.kernels")
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
    if source_solver is not None and (
        edge.target in _RAW_EXTENSION_MODULES
        or (
            edge.target in {f"{PACKAGE}.core.kernels", f"{PACKAGE}.runtime"}
            and edge.imported_name == "native_extension"
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
        or _matches(target, f"{PACKAGE}.core.scene")
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
                or _matches(target, f"{PACKAGE}.core.scene")
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

    if source == f"{PACKAGE}.physics.oracle" and any(
        _matches(target, prefix)
        for prefix in (
            "torch",
            f"{PACKAGE}.core.kernels",
            f"{PACKAGE}.runtime",
            f"{PACKAGE}.scene",
            f"{PACKAGE}.materials",
            f"{PACKAGE}.propagation",
            f"{PACKAGE}.scattering",
            *_SOLVER_PREFIXES,
        )
    ):
        violations.append(_violation(edge, "oracle_production_dependency"))
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
    return sorted(
        {
            violation
            for edge in collect_import_edges(package_root)
            for violation in classify_edge(edge)
        }
    )


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
