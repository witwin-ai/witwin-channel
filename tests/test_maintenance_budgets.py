# Copyright Xingyu Chen.
# Tests maintenance budgets.

from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest

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
    *, file_recommended: int = 100, file_hard: int = 200, complexity_recommended: int = 2,
    file_lines: bool = True,
) -> dict:
    """Build a synthetic budget config.

 ``file_lines=False`` produces the retired-gate shape: no
 ``limits.file_lines`` and no ``file_exemptions`` section at all.
 """

    config: dict = {
        "schema_version": 1,
        "source_root": "src/product",
        "limits": {
            "function_complexity": {
                "recommended": complexity_recommended,
            },
        },
        "function_exemptions": {},
    }
    if file_lines:
        config["limits"]["file_lines"] = {
            "recommended": file_recommended,
            "hard": file_hard,
        }
        config["file_exemptions"] = {}
    return config


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
    _, functions = budgets.measure_repository(ROOT, config["source_root"])
    function_values = {metric.key: metric.complexity for metric in functions}
    complexity_recommended = config["limits"]["function_complexity"][
        "recommended"
    ]

    assert budgets.check_budgets(ROOT, config) == []
    assert config["function_exemptions"] == {
        key: entry
        for key, entry in config["function_exemptions"].items()
        if function_values[key] > complexity_recommended
        and entry["ceiling"] == function_values[key]
    }


def test_repository_budget_retires_all_file_size_gates() -> None:
    config = budgets.load_budgets(BUDGET_PATH)

    # Python files have no line-count limit or exemptions.
    assert "file_lines" not in config["limits"]
    assert "file_exemptions" not in config
    # The reason is recorded in the config itself, next to what survived.
    assert "Retired 2026-07-27" in config["limits_policy"]["file_lines"]
    assert "Retired" not in config["limits_policy"]["function_complexity"]
    # Native TU size was retired separately under CUDA source consolidation; complexity remains.
    assert config["limits"]["function_complexity"] == {"recommended": 15}
    assert "native_file_lines" not in config["limits"]
    assert "native_file_exemptions" not in config
    assert "Retired 2026-07-28" in config["limits_policy"]["native_file_lines"]
    # Retiring the file-size gate must not disturb the complexity waivers: the
    # section survives, is still enforced, and every surviving entry is live.
    # Asserting a fixed count instead would be brittle and would say nothing -
    # a refactor that simplifies an exempted function correctly retires its
    # waiver, which is exactly what happened to four of them in source consolidation the initial implementation.
    _, functions = budgets.measure_repository(ROOT, config["source_root"])
    complexity = {metric.key: metric.complexity for metric in functions}
    recommended = config["limits"]["function_complexity"]["recommended"]
    exemptions = config["function_exemptions"]
    assert exemptions
    assert all(
        complexity[key] > recommended and entry["ceiling"] == complexity[key]
        for key, entry in exemptions.items()
    )


def test_absent_file_line_limit_skips_every_file_size_check(tmp_path: Path) -> None:
    # A file far past the retired hard limit of 2000 lines, whose only real
    # violation is control-flow complexity.
    padding = "".join(f"# pad {index}\n" for index in range(5000))
    _write_source(tmp_path, f"{BRANCH_SOURCE}{padding}")

    retired = _config(file_lines=False)
    assert [
        (item.kind, item.subject)
        for item in budgets.check_budgets(tmp_path, retired)
    ] == [("unbudgeted-debt", "src/product/module.py::branch")]

    # The same tree under a declared file_lines limit still reports the file:
    # the gate is skipped because it is absent, not because it stopped working.
    declared = _config(file_recommended=100, file_hard=200)
    assert {
        (item.kind, item.subject)
        for item in budgets.check_budgets(tmp_path, declared)
    } == {
        ("unbudgeted-debt", "src/product/module.py"),
        ("hard-limit", "src/product/module.py"),
        ("unbudgeted-debt", "src/product/module.py::branch"),
    }


def test_absent_file_line_limit_ignores_a_leftover_file_exemption(tmp_path: Path) -> None:
    # A stale file_exemptions section must not resurrect the gate, and must not
    # be reported as a stale exemption either: with no limit there is nothing
    # for it to be stale against.
    _write_source(tmp_path)
    config = _config(file_lines=False)
    config["file_exemptions"] = {"src/product/module.py": _exemption(6)}
    config["limits"]["function_complexity"]["recommended"] = 10

    assert budgets.check_budgets(tmp_path, config) == []


def test_function_complexity_stays_mandatory(tmp_path: Path) -> None:
    _write_source(tmp_path)

    missing_complexity = _config(file_lines=False)
    del missing_complexity["limits"]["function_complexity"]
    with pytest.raises(ValueError, match="limits.function_complexity"):
        budgets.check_budgets(tmp_path, missing_complexity)

    missing_function_exemptions = _config(file_lines=False)
    del missing_function_exemptions["function_exemptions"]
    with pytest.raises(ValueError, match="function_exemptions"):
        budgets.check_budgets(tmp_path, missing_function_exemptions)

    retired_native = _config(file_lines=False)
    retired_native["native_source_root"] = "native/channel"
    assert [
        (item.kind, item.subject)
        for item in budgets.check_budgets(tmp_path, retired_native)
    ] == [("unbudgeted-debt", "src/product/module.py::branch")]


def test_absent_native_line_limit_skips_native_size_checks(tmp_path: Path) -> None:
    _write_source(tmp_path)
    native_dir = tmp_path / "native" / "channel"
    native_dir.mkdir(parents=True)
    unit = native_dir / "kernel.cu"
    unit.write_text(
        "\n".join(f"// line {index}" for index in range(5000)),
        encoding="utf-8",
    )

    config = _config(file_lines=False, complexity_recommended=10)
    config["native_source_root"] = "native/channel"
    config["native_file_exemptions"] = {
        "native/channel/kernel.cu": _exemption(1)
    }

    assert budgets.check_budgets(tmp_path, config) == []


def test_native_hard_limit_and_waiver_growth_are_enforced(tmp_path: Path) -> None:
    native_dir = tmp_path / "native" / "channel"
    native_dir.mkdir(parents=True)
    unit = native_dir / "kernel.cu"
    unit.write_text("\n".join(f"// line {index}" for index in range(9)), encoding="utf-8")

    config = _config()
    config["native_source_root"] = "native/channel"
    config["limits"]["native_file_lines"] = {"recommended": 5, "hard": 7}
    config["native_file_exemptions"] = {}

    violations = budgets.check_budgets(tmp_path, config)
    assert {(item.kind, item.subject) for item in violations} == {
        ("unbudgeted-debt", "native/channel/kernel.cu"),
        ("hard-limit", "native/channel/kernel.cu"),
    }


def test_ast_measurement_is_reproducible() -> None:
    first = budgets.measure_repository(ROOT, "witwin/channel")
    second = budgets.measure_repository(ROOT, "witwin/channel")

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