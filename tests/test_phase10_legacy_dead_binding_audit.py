from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

from tools.refactor_baseline import binding_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT / "docs" / "dev" / "audit" / "phase10-legacy-dead-binding.json"
)
OPS_MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "ops_migration_manifest.json"
PYTHON_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"


def _audit() -> dict[str, object]:
    value = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_phase10_zero_reference_inventory_is_complete_and_classified() -> None:
    candidates = _audit()["zero_reference_candidates"]
    assert isinstance(candidates, list)
    names = [candidate["name"] for candidate in candidates]

    assert len(names) == len(set(names)) == 33
    assert Counter(candidate["classification"] for candidate in candidates) == {
        "public": 26,
        "intentional": 5,
        "dead": 2,
    }
    for candidate in candidates:
        assert set(candidate) == {
            "name",
            "definition",
            "classification",
            "decision",
            "evidence",
            "tests",
            "compatibility_cycle",
        }
        assert candidate["evidence"]
        assert candidate["tests"]
        assert candidate["compatibility_cycle"]
        for test_path in candidate["tests"]:
            assert (REPOSITORY_ROOT / test_path).exists(), test_path


def test_phase10_binding_inventory_covers_all_current_symbols_and_python_owners() -> (
    None
):
    audit = _audit()
    ownership = audit["binding_ownership"]
    assert isinstance(ownership, dict)
    audited_names = ownership["symbols"]
    current = binding_manifest(REPOSITORY_ROOT)
    current_names = [symbol["name"] for symbol in current["symbols"]]

    assert ownership["expected_count"] == 174
    assert len(audited_names) == len(set(audited_names)) == 174
    assert audited_names == current_names

    sources = {
        path: path.read_text(encoding="utf-8-sig") for path in PYTHON_ROOT.rglob("*.py")
    }
    missing: list[str] = []
    for name in audited_names:
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
        if not any(pattern.search(source) for source in sources.values()):
            missing.append(name)
    assert missing == []


def test_phase10_dummy_projection_is_exact_and_not_yet_retired() -> None:
    audit = _audit()
    projection = audit["dummy_parameter_projection"]
    assert isinstance(projection, dict)
    parameter_name = projection["parameter_name"]
    current = binding_manifest(REPOSITORY_ROOT)
    current_bindings = {
        symbol["name"]
        for symbol in current["symbols"]
        if any(
            parameter["name"] == parameter_name for parameter in symbol["parameters"]
        )
    }

    assert projection["status"] == "present"
    assert projection["allowed_transition"] == ("remove_from_all_listed_bindings_only")
    assert len(projection["bindings"]) == len(set(projection["bindings"])) == 22
    assert current_bindings == set(projection["bindings"])


def test_phase10_retirement_ledger_is_empty_until_the_definition_is_removed() -> None:
    manifest = json.loads(OPS_MANIFEST_PATH.read_text(encoding="utf-8"))
    audit = _audit()

    assert manifest["retired_ops"] == []
    assert "_raydn_module_handle" in manifest["canonical_owners"]
    assert audit["frozen_ops_contract_digest"] == (
        "ff9c4cd45b2f1091c9ba05e1a311e6e569945e18badc7b7a67a3f8f56ccda3a9"
    )
