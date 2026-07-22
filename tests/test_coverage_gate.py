from __future__ import annotations

import copy
import json
from pathlib import Path

from ci.check_coverage import evaluate


ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / "ci/coverage-policy.json").read_text(encoding="utf-8"))
BASELINE = json.loads((ROOT / POLICY["phase0_baseline"]).read_text(encoding="utf-8"))


def _accepted_report() -> dict[str, object]:
    files: dict[str, object] = {}
    exemptions = {item["path"]: item for item in POLICY["file_exemptions"]}
    for path in POLICY["core_files"]:
        percent = float(exemptions.get(path, {}).get("minimum_statement_percent", 95.0))
        files[path] = {
            "summary": {
                "covered_lines": int(percent * 10),
                "num_statements": 1000,
                "percent_statements_covered": percent,
            }
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
