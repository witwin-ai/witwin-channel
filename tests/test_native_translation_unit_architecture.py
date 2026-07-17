from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPOSITORY_ROOT / "native/channel_native"
INVENTORY_PATH = (
    REPOSITORY_ROOT / "docs/dev/audit/phase9-native-owner-inventory.json"
)


def _translation_unit_line_counts() -> dict[str, int]:
    return {
        path.relative_to(REPOSITORY_ROOT).as_posix(): len(
            path.read_text(encoding="utf-8-sig").splitlines()
        )
        for path in NATIVE_ROOT.rglob("*")
        if path.suffix in {".cpp", ".cu"}
    }


def test_native_translation_unit_hard_limit_debt_is_exact_and_shrinking() -> None:
    policy = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))[
        "translation_unit_policy"
    ]
    counts = _translation_unit_line_counts()
    debt = policy["hard_limit_debt_allowlist"]
    violations = {
        path for path, count in counts.items() if count > policy["hard_limit_lines"]
    }

    assert violations == set(debt)
    assert all(counts[path] <= cap for path, cap in debt.items())


def test_native_translation_unit_recommended_limit_has_only_planned_owners() -> None:
    policy = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))[
        "translation_unit_policy"
    ]
    counts = _translation_unit_line_counts()
    debt = policy["planned_owner_debt"]
    violations = {
        path
        for path, count in counts.items()
        if count > policy["recommended_limit_lines"]
    }

    assert violations == set(debt)
    assert all(counts[path] <= cap for path, cap in debt.items())
