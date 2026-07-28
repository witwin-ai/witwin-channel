from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ci.check_coverage import evaluate, section_line_numbers


ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / "ci/coverage-policy.json").read_text(encoding="utf-8"))
BASELINE = json.loads((ROOT / POLICY["phase0_baseline"]).read_text(encoding="utf-8"))
SECTIONS = {item["path"]: item for item in POLICY["section_floors"]}


def _section_split(path: str, covered_fraction: float) -> tuple[list[int], list[int]]:
    """Split one guarded section into executed and missing lines."""

    lines = sorted(section_line_numbers(path, list(SECTIONS[path]["members"])))
    cut = round(len(lines) * covered_fraction)
    return lines[:cut], lines[cut:]


def _accepted_report() -> dict[str, object]:
    files: dict[str, object] = {}
    exemptions = {item["path"]: item for item in POLICY["file_exemptions"]}
    for path in POLICY["core_files"]:
        percent = float(exemptions.get(path, {}).get("minimum_statement_percent", 95.0))
        executed, missing = ([], [])
        if path in SECTIONS:
            executed, missing = _section_split(path, 1.0)
        files[path] = {
            "summary": {
                "covered_lines": int(percent * 10),
                "num_statements": 1000,
                "percent_statements_covered": percent,
            },
            "executed_lines": executed,
            "missing_lines": missing,
        }
    # Preserve the accepted aggregate independently of rounding in the per-file fixtures.
    files[POLICY["core_files"][0]]["summary"].update(
        {"covered_lines": 3000, "num_statements": 3000}
    )
    return {
        "totals": {
            "percent_statements_covered": 85.5,
            "percent_branches_covered": 65.0,
        },
        "files": files,
    }


def test_coverage_policy_accepts_target_with_explicit_file_exemptions() -> None:
    assert evaluate(_accepted_report(), POLICY, BASELINE) == []


def test_coverage_policy_rejects_overall_and_branch_regression() -> None:
    report = _accepted_report()
    report["totals"]["percent_statements_covered"] = 84.9
    report["totals"]["percent_branches_covered"] = 64.0
    errors = evaluate(report, POLICY, BASELINE)
    assert any("overall statement" in error for error in errors)
    assert any("overall branch" in error for error in errors)


def test_coverage_policy_rejects_core_file_regression() -> None:
    report = copy.deepcopy(_accepted_report())
    path = POLICY["file_exemptions"][0]["path"]
    report["files"][path]["summary"]["percent_statements_covered"] = 10.0
    assert any("regressed" in error for error in evaluate(report, POLICY, BASELINE))


def test_every_section_floor_names_a_former_narrow_module() -> None:
    """The three phase-6 consolidations each keep their pre-merge boundary."""

    assert {item["section"] for item in POLICY["section_floors"]} == {
        "materials/kernels/functional.py",
        "propagation/fields/kernels/functional.py",
        "scattering/kernels/functional.py",
    }
    for path, section in SECTIONS.items():
        assert section_line_numbers(path, list(section["members"]))


@pytest.mark.parametrize("path", sorted(SECTIONS))
def test_section_floor_is_not_masked_by_the_rest_of_the_merged_module(
    path: str,
) -> None:
    """A half-covered former functional section fails even at a 95% file total.

    This is the regression the phase-6 consolidation introduced: the narrow
    floors were re-pointed at far larger modules, so unrelated well-covered
    neighbours could lift the merged percentage past the gate while the guarded
    section rotted.
    """

    report = copy.deepcopy(_accepted_report())
    payload = report["files"][path]
    payload["summary"]["percent_statements_covered"] = 95.0
    payload["summary"]["covered_lines"] = 950
    executed, missing = _section_split(path, 0.5)
    payload["executed_lines"] = executed
    payload["missing_lines"] = missing
    errors = evaluate(report, POLICY, BASELINE)
    assert any(f"section {path}::" in error for error in errors), errors


@pytest.mark.parametrize("path", sorted(SECTIONS))
def test_section_floor_accepts_the_declared_minimum(path: str) -> None:
    """The floor is a floor: exactly meeting it is accepted."""

    report = copy.deepcopy(_accepted_report())
    payload = report["files"][path]
    floor = float(SECTIONS[path]["minimum_statement_percent"])
    executed, missing = _section_split(path, min(1.0, floor / 100.0 + 0.02))
    payload["executed_lines"] = executed
    payload["missing_lines"] = missing
    assert evaluate(report, POLICY, BASELINE) == []


def test_section_floor_fails_loudly_when_a_member_disappears() -> None:
    """Renaming or deleting a guarded member must not silently shrink the gate."""

    policy = copy.deepcopy(POLICY)
    policy["section_floors"][0]["members"].append("__member_that_does_not_exist__")
    errors = evaluate(_accepted_report(), policy, BASELINE)
    assert any("cannot be resolved" in error for error in errors), errors


def test_section_floor_entry_requires_its_documentation_fields() -> None:
    policy = copy.deepcopy(POLICY)
    del policy["section_floors"][0]["reason"]
    errors = evaluate(_accepted_report(), policy, BASELINE)
    assert any("section floor entry missing" in error for error in errors), errors
