# Copyright Xingyu Chen.
# Checks coverage.

"""Checks coverage."""

from __future__ import annotations

import argparse
import ast
import json
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "ci" / "coverage-policy.json"
SECTION_FIELDS = frozenset(
    {"path", "section", "members", "owner", "reason", "minimum_statement_percent"}
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_files(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        path.replace("\\", "/"): payload
        for path, payload in report.get("files", {}).items()
    }


def _member_spans(source: str) -> dict[str, tuple[int, int]]:
    """Return the inclusive line span of every top-level definition in ``source``.

 A decorated definition starts at its first decorator, so the decorator
 statements belong to the member they decorate rather than to module level.
 """

    spans: dict[str, tuple[int, int]] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = node.lineno
        for decorator in node.decorator_list:
            start = min(start, decorator.lineno)
        end = node.end_lineno
        assert end is not None
        spans[node.name] = (start, end)
    return spans


def section_line_numbers(path: str, members: list[str]) -> set[int]:
    """Return every source line covered by the named top-level members.

 The section is resolved against the working tree rather than the coverage
 report so that a renamed or deleted member fails loudly instead of silently
 shrinking the guarded statement set.
 """

    spans = _member_spans((ROOT / path).read_text(encoding="utf-8"))
    unknown = sorted(set(members) - set(spans))
    if unknown:
        raise KeyError(f"{path} has no top-level member {unknown}")
    lines: set[int] = set()
    for member in members:
        start, end = spans[member]
        lines.update(range(start, end + 1))
    return lines


def _evaluate_sections(files: dict[str, dict[str, Any]], policy: dict[str, Any]) -> list[str]:
    """Return violations of the per-section statement floors.

 A section floor guards one named group of members inside a consolidated
 module, so well-covered neighbours in the same file cannot lift a section
 that has lost coverage.
 """

    errors: list[str] = []
    for section in policy.get("section_floors", ()):
        missing = SECTION_FIELDS - set(section)
        if missing:
            errors.append(f"section floor entry missing {sorted(missing)}")
            continue
        path = section["path"]
        label = f"{path}::{section['section']}"
        if path not in files:
            errors.append(f"section coverage entry missing: {label}")
            continue
        try:
            lines = section_line_numbers(path, list(section["members"]))
        except (KeyError, OSError, SyntaxError) as error:
            errors.append(f"section {label} cannot be resolved: {error}")
            continue
        payload = files[path]
        if "executed_lines" not in payload or "missing_lines" not in payload:
            errors.append(f"section {label} needs per-line coverage data")
            continue
        executed = lines & set(payload["executed_lines"])
        statements = executed | (lines & set(payload["missing_lines"]))
        if not statements:
            errors.append(f"section {label} measures no statements")
            continue
        percent = 100.0 * len(executed) / len(statements)
        floor = float(section["minimum_statement_percent"])
        if percent < floor:
            errors.append(f"section {label} is {percent:.6f}% < {floor:.6f}%")
    return errors


def evaluate(report: dict[str, Any], policy: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Return deterministic policy violations; an empty list means acceptance."""

    errors: list[str] = []
    totals = report["totals"]
    baseline_totals = baseline["totals"]
    statement = float(totals["percent_statements_covered"])
    branch = float(totals["percent_branches_covered"])
    statement_floor = max(
        float(policy["overall_statement_minimum_percent"]),
        float(baseline_totals["percent_statements_covered"]),
    )
    branch_floor = float(baseline_totals["percent_branches_covered"])
    if statement < statement_floor:
        errors.append(f"overall statement {statement:.6f}% < {statement_floor:.6f}%")
    if branch < branch_floor:
        errors.append(f"overall branch {branch:.6f}% < {branch_floor:.6f}%")

    files = _normalized_files(report)
    core_paths = list(policy["core_files"])
    exemptions = {item["path"]: item for item in policy["file_exemptions"]}
    core_target = float(policy["core_statement_minimum_percent"])
    covered = 0
    statements = 0
    below_target: set[str] = set()
    today = date.today()
    for path in core_paths:
        if path not in files:
            errors.append(f"core coverage entry missing: {path}")
            continue
        summary = files[path]["summary"]
        covered += int(summary["covered_lines"])
        statements += int(summary["num_statements"])
        percent = float(summary["percent_statements_covered"])
        if percent >= core_target:
            continue
        below_target.add(path)
        exemption = exemptions.get(path)
        if exemption is None:
            errors.append(f"core file {path} is {percent:.6f}% without exemption")
            continue
        missing = {"owner", "reason", "expires_on", "minimum_statement_percent"} - set(
            exemption
        )
        if missing:
            errors.append(f"coverage exemption {path} missing {sorted(missing)}")
            continue
        if date.fromisoformat(exemption["expires_on"]) < today:
            errors.append(f"coverage exemption expired: {path}")
        floor = float(exemption["minimum_statement_percent"])
        if percent < floor:
            errors.append(f"core file {path} regressed: {percent:.6f}% < {floor:.6f}%")

    errors.extend(_evaluate_sections(files, policy))

    stale = set(exemptions) - below_target
    for path in sorted(stale):
        errors.append(f"stale or non-core coverage exemption: {path}")
    if statements:
        aggregate = 100.0 * covered / statements
        if aggregate < core_target:
            errors.append(f"core aggregate {aggregate:.6f}% < {core_target:.6f}%")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="Coverage.py JSON report")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args(argv)
    policy = _load(args.policy)
    baseline = _load(ROOT / policy["phase0_baseline"])
    errors = evaluate(_load(args.report), policy, baseline)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("coverage policy passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())