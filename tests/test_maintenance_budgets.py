from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path

from ci import check_maintenance_budgets as budgets


ROOT = Path(__file__).resolve().parents[1]
BUDGET_PATH = ROOT / "ci" / "maintenance-budgets.json"
BRANCH_SOURCE = """\
def branch(value):
    if value:
        return 1
    if value > 1:
        return 2
    return 0
"""


def _config(
    *,
    file_recommended: int = 100,
    file_hard: int = 200,
    complexity_recommended: int = 2,
) -> dict:
    return {
        "schema_version": 1,
        "source_root": "src/product",
        "limits": {
            "file_lines": {
                "recommended": file_recommended,
                "hard": file_hard,
            },
            "function_complexity": {
                "recommended": complexity_recommended,
            },
        },
        "file_exemptions": {},
        "function_exemptions": {},
    }


def _exemption(ceiling: int, *, expires_on: str = "2099-01-01") -> dict:
    return {
        "owner": "test maintainers",
        "reason": "synthetic existing debt",
        "expires_on": expires_on,
        "ceiling": ceiling,
    }


def _write_source(root: Path, source: str = BRANCH_SOURCE) -> Path:
    path = root / "src" / "product" / "module.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_current_baseline_is_exact_and_passes() -> None:
    config = budgets.load_budgets(BUDGET_PATH)
    files, functions = budgets.measure_repository(ROOT, config["source_root"])
    file_values = {metric.path: metric.lines for metric in files}
    function_values = {metric.key: metric.complexity for metric in functions}
    file_recommended = config["limits"]["file_lines"]["recommended"]
    complexity_recommended = config["limits"]["function_complexity"][
        "recommended"
    ]

    assert budgets.check_budgets(ROOT, config) == []
    assert config["file_exemptions"] == {
        key: entry
        for key, entry in config["file_exemptions"].items()
        if file_values[key] > file_recommended
        and entry["ceiling"] == file_values[key]
    }
    assert config["function_exemptions"] == {
        key: entry
        for key, entry in config["function_exemptions"].items()
        if function_values[key] > complexity_recommended
        and entry["ceiling"] == function_values[key]
    }


def test_native_translation_units_are_within_budget() -> None:
    config = budgets.load_budgets(BUDGET_PATH)
    native_files = budgets.measure_native_files(ROOT, config["native_source_root"])
    recommended = config["limits"]["native_file_lines"]["recommended"]
    hard = config["limits"]["native_file_lines"]["hard"]

    assert native_files, "expected native translation units to be measured"
    largest = max(native_files, key=lambda metric: metric.lines)
    # The largest native unit stays below the recommendation, so no native
    # waiver entry is required; keep the exemption map empty to prove it.
    assert largest.lines < recommended
    assert all(metric.lines <= hard for metric in native_files)
    assert config["native_file_exemptions"] == {}


def test_native_hard_limit_and_waiver_growth_are_enforced(tmp_path: Path) -> None:
    native_dir = tmp_path / "native" / "channel_native"
    native_dir.mkdir(parents=True)
    unit = native_dir / "kernel.cu"
    unit.write_text("\n".join(f"// line {index}" for index in range(9)), encoding="utf-8")

    config = _config()
    config["native_source_root"] = "native/channel_native"
    config["limits"]["native_file_lines"] = {"recommended": 5, "hard": 7}
    config["native_file_exemptions"] = {}

    violations = budgets.check_budgets(tmp_path, config)
    assert {(item.kind, item.subject) for item in violations} == {
        ("unbudgeted-debt", "native/channel_native/kernel.cu"),
        ("hard-limit", "native/channel_native/kernel.cu"),
    }


def test_ast_measurement_is_reproducible() -> None:
    first = budgets.measure_repository(ROOT, "src/witwin/channel")
    second = budgets.measure_repository(ROOT, "src/witwin/channel")

    assert first == second


def test_new_file_and_function_debt_require_exemptions(tmp_path: Path) -> None:
    _write_source(tmp_path)
    config = _config(file_recommended=5)

    violations = budgets.check_budgets(tmp_path, config)

    assert {(item.kind, item.subject) for item in violations} == {
        ("unbudgeted-debt", "src/product/module.py"),
        ("unbudgeted-debt", "src/product/module.py::branch"),
    }


def test_file_hard_limit_and_existing_ceiling_cannot_grow(tmp_path: Path) -> None:
    path = _write_source(tmp_path)
    config = _config(file_recommended=5, file_hard=7, complexity_recommended=10)
    config["file_exemptions"]["src/product/module.py"] = _exemption(6)
    assert budgets.check_budgets(tmp_path, config) == []

    path.write_text(f"{BRANCH_SOURCE}# growth\n", encoding="utf-8")
    assert [item.kind for item in budgets.check_budgets(tmp_path, config)] == [
        "budget-growth"
    ]

    path.write_text(f"{BRANCH_SOURCE}# growth\n# hard limit\n", encoding="utf-8")
    assert {item.kind for item in budgets.check_budgets(tmp_path, config)} == {
        "budget-growth",
        "hard-limit",
    }


def test_function_ceiling_cannot_grow(tmp_path: Path) -> None:
    path = _write_source(tmp_path)
    config = _config()
    function_key = "src/product/module.py::branch"
    config["function_exemptions"][function_key] = _exemption(3)
    assert budgets.check_budgets(tmp_path, config) == []

    path.write_text(
        BRANCH_SOURCE.replace(
            "    return 0\n",
            "    if value > 2:\n        return 3\n    return 0\n",
        ),
        encoding="utf-8",
    )

    assert [item.kind for item in budgets.check_budgets(tmp_path, config)] == [
        "budget-growth"
    ]


def test_expired_and_stale_exemptions_are_rejected(tmp_path: Path) -> None:
    _write_source(tmp_path)
    function_key = "src/product/module.py::branch"
    config = _config()
    config["function_exemptions"][function_key] = _exemption(
        3, expires_on="2026-07-15"
    )
    violations = budgets.check_budgets(
        tmp_path, config, today=date(2026, 7, 16)
    )
    assert [item.kind for item in violations] == ["expired-exemption"]

    stale = deepcopy(config)
    stale["limits"]["function_complexity"]["recommended"] = 3
    violations = budgets.check_budgets(
        tmp_path, stale, today=date(2026, 7, 16)
    )
    assert [item.kind for item in violations] == ["stale-exemption"]
