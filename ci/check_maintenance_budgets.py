"""Enforce deterministic size and complexity budgets for production Python."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any


DEFAULT_BUDGET_PATH = Path("ci/maintenance-budgets.json")


@dataclass(frozen=True, order=True, slots=True)
class FileMetric:
    path: str
    lines: int


@dataclass(frozen=True, order=True, slots=True)
class FunctionMetric:
    path: str
    qualname: str
    line: int
    complexity: int

    @property
    def key(self) -> str:
        return f"{self.path}::{self.qualname}"


@dataclass(frozen=True, slots=True)
class Exemption:
    owner: str
    reason: str
    expires_on: date
    ceiling: int


@dataclass(frozen=True, order=True, slots=True)
class Violation:
    kind: str
    subject: str
    detail: str


class _ComplexityVisitor(ast.NodeVisitor):
    """Count control-flow paths in one function, excluding nested scopes."""

    def __init__(self) -> None:
        self.complexity = 1

    def visit(self, node: ast.AST) -> Any:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
        ):
            return None
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp)):
            self.complexity += 1
        elif isinstance(node, ast.BoolOp):
            self.complexity += len(node.values) - 1
        elif isinstance(node, ast.ExceptHandler):
            self.complexity += 1
        elif isinstance(node, ast.comprehension):
            self.complexity += 1 + len(node.ifs)
        elif isinstance(node, ast.Match):
            self.complexity += len(node.cases)
        return super().visit(node)


def function_complexity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    visitor = _ComplexityVisitor()
    for statement in node.body:
        visitor.visit(statement)
    return visitor.complexity


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.scope: list[str] = []
        self.metrics: list[FunctionMetric] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        qualname = ".".join((*self.scope, node.name))
        self.metrics.append(
            FunctionMetric(
                self.path,
                qualname,
                node.lineno,
                function_complexity(node),
            )
        )
        self.scope.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function


def measure_repository(
    root: Path, source_root: str
) -> tuple[list[FileMetric], list[FunctionMetric]]:
    root = root.resolve()
    source = root / source_root
    files: list[FileMetric] = []
    functions: list[FunctionMetric] = []
    for path in sorted(source.rglob("*.py")):
        text = path.read_text(encoding="utf-8-sig")
        display_path = path.relative_to(root).as_posix()
        files.append(FileMetric(display_path, len(text.splitlines())))
        tree = ast.parse(text, filename=display_path)
        collector = _FunctionCollector(display_path)
        collector.visit(tree)
        functions.extend(collector.metrics)

    duplicate_names = Counter((metric.path, metric.qualname) for metric in functions)
    functions = [
        replace(metric, qualname=f"{metric.qualname}@{metric.line}")
        if duplicate_names[(metric.path, metric.qualname)] > 1
        else metric
        for metric in functions
    ]
    functions.sort(key=lambda metric: metric.key)
    keys = [metric.key for metric in functions]
    if len(keys) != len(set(keys)):
        raise ValueError("function budget keys are not unique")
    return files, functions


def load_budgets(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("maintenance budget root must be an object")
    return data


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _limit(config: dict[str, Any], metric: str, name: str) -> int:
    try:
        value = config["limits"][metric][name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing limits.{metric}.{name}") from error
    return _positive_int(value, f"limits.{metric}.{name}")


def _exemptions(config: dict[str, Any], section: str) -> dict[str, Exemption]:
    raw = config.get(section)
    if not isinstance(raw, dict):
        raise ValueError(f"{section} must be an object")
    exemptions: dict[str, Exemption] = {}
    for key, entry in raw.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            raise ValueError(f"{section} entries must map strings to objects")
        owner = entry.get("owner")
        reason = entry.get("reason")
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError(f"{section}.{key}.owner must be non-empty")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{section}.{key}.reason must be non-empty")
        try:
            expires_on = date.fromisoformat(entry["expires_on"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{section}.{key}.expires_on must be an ISO date"
            ) from error
        exemptions[key] = Exemption(
            owner.strip(),
            reason.strip(),
            expires_on,
            _positive_int(entry.get("ceiling"), f"{section}.{key}.ceiling"),
        )
    return exemptions


def _check_debt(
    *,
    values: dict[str, int],
    recommended: int,
    exemptions: dict[str, Exemption],
    today: date,
    label: str,
) -> list[Violation]:
    violations: list[Violation] = []
    debt = {key: value for key, value in values.items() if value > recommended}
    for key, value in debt.items():
        exemption = exemptions.get(key)
        if exemption is None:
            violations.append(
                Violation(
                    "unbudgeted-debt",
                    key,
                    f"{label} {value} exceeds recommended {recommended}",
                )
            )
            continue
        if exemption.expires_on < today:
            violations.append(
                Violation(
                    "expired-exemption",
                    key,
                    f"expired on {exemption.expires_on.isoformat()}",
                )
            )
        if value > exemption.ceiling:
            violations.append(
                Violation(
                    "budget-growth",
                    key,
                    f"{label} {value} exceeds ceiling {exemption.ceiling}",
                )
            )
    for key in exemptions.keys() - debt.keys():
        violations.append(
            Violation(
                "stale-exemption",
                key,
                f"no longer exceeds recommended {recommended}",
            )
        )
    return violations


def check_budgets(
    root: Path, config: dict[str, Any], *, today: date | None = None
) -> list[Violation]:
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    source_root = config.get("source_root")
    if not isinstance(source_root, str) or not source_root:
        raise ValueError("source_root must be a non-empty string")

    file_recommended = _limit(config, "file_lines", "recommended")
    file_hard = _limit(config, "file_lines", "hard")
    complexity_recommended = _limit(
        config, "function_complexity", "recommended"
    )
    if file_recommended >= file_hard:
        raise ValueError("file line recommended limit must be below hard limit")

    file_exemptions = _exemptions(config, "file_exemptions")
    function_exemptions = _exemptions(config, "function_exemptions")
    if any(item.ceiling > file_hard for item in file_exemptions.values()):
        raise ValueError("file exemption ceilings cannot exceed the hard limit")

    files, functions = measure_repository(root, source_root)
    violations = [
        Violation(
            "hard-limit",
            metric.path,
            f"file lines {metric.lines} exceeds hard limit {file_hard}",
        )
        for metric in files
        if metric.lines > file_hard
    ]
    check_date = today or date.today()
    violations.extend(
        _check_debt(
            values={metric.path: metric.lines for metric in files},
            recommended=file_recommended,
            exemptions=file_exemptions,
            today=check_date,
            label="file lines",
        )
    )
    violations.extend(
        _check_debt(
            values={metric.key: metric.complexity for metric in functions},
            recommended=complexity_recommended,
            exemptions=function_exemptions,
            today=check_date,
            label="complexity",
        )
    )
    return sorted(violations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository to inspect (defaults to this repository)",
    )
    parser.add_argument(
        "--budgets",
        type=Path,
        help="budget JSON (defaults to ci/maintenance-budgets.json under root)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    budget_path = args.budgets or root / DEFAULT_BUDGET_PATH
    try:
        config = load_budgets(budget_path)
        violations = check_budgets(root, config)
    except (OSError, SyntaxError, ValueError, json.JSONDecodeError) as error:
        print(f"maintenance budget configuration error: {error}", file=sys.stderr)
        return 2

    for violation in violations:
        print(
            f"{violation.kind}: {violation.subject}: {violation.detail}",
            file=sys.stderr,
        )
    if violations:
        return 1
    print("maintenance budgets passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
