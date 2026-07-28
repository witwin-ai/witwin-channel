from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ci import check_contract_coverage as coverage


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "contract-coverage-manifest.json"
OWNER_INVENTORY_PATH = (
    REPOSITORY_ROOT / "docs/dev/audit/phase13-current-native-owner-inventory.json"
)

def _manifest() -> dict[str, object]:
    return coverage.load_manifest(MANIFEST_PATH)


def test_current_contract_matrix_covers_every_public_export_and_native_binding():
    manifest = _manifest()
    binding_manifest = json.loads(
        (REPOSITORY_ROOT / coverage.BINDING_BASELINE_PATH).read_text(encoding="utf-8")
    )

    assert coverage.check_contract_coverage(REPOSITORY_ROOT, manifest) == []
    assert len(manifest["public_exports"]) == coverage.EXPECTED_PUBLIC_EXPORT_COUNT
    assert len(manifest["native_bindings"]) == len(binding_manifest["symbols"])


def test_contract_matrix_rejects_missing_public_and_native_entries():
    manifest = copy.deepcopy(_manifest())
    public_export = manifest["public_exports"].pop()[0]
    native_symbol = manifest["native_bindings"].pop()[0]

    issues = coverage.check_contract_coverage(REPOSITORY_ROOT, manifest)

    assert any(
        f"public export coverage missing: {public_export}" in issue for issue in issues
    )
    assert any(
        f"native binding coverage missing: {native_symbol}" in issue for issue in issues
    )


def test_contract_matrix_rejects_missing_e2e_callers_and_bad_owners():
    manifest = copy.deepcopy(_manifest())
    symbol = manifest["native_bindings"][0][0]
    manifest["native_bindings"][0][1] = "witwin.channel.missing.owner"
    manifest["native_bindings"][0][4] = []

    issues = coverage.check_contract_coverage(REPOSITORY_ROOT, manifest)

    assert f"native binding has no E2E caller: {symbol}" in issues
    assert any(
        f"native binding Python owner does not exist: {symbol}" in issue
        for issue in issues
    )



def test_contract_matrix_has_no_dormant_or_caller_free_native_binding() -> None:
    """Phase-11 acceptance: the dormant ADR-029/030/031 allowlist is empty."""

    manifest = _manifest()
    assert coverage.DORMANT_SYMBOL_FACADES == {}
    assert coverage.DORMANT_EXPERIMENT_SYMBOLS == frozenset()
    assert coverage.DORMANT_FACADE_OWNERS == {}
    assert coverage.DORMANT_ALLOWED_FACADE_CALLERS == {}
    for row in manifest["native_bindings"]:
        assert not row[2].startswith("dormant_"), row[0]
        assert row[4], row[0]

    inventory = json.loads(OWNER_INVENTORY_PATH.read_text(encoding="utf-8"))
    assert not [
        row
        for row in inventory["symbols"]
        if row["liveness"] == "dormant-native-producer"
    ]


def test_contract_matrix_still_rejects_an_unapproved_dormant_binding() -> None:
    manifest = copy.deepcopy(_manifest())
    live = next(row for row in manifest["native_bindings"] if row[0] == "build_info")
    live[2] = "dormant_named_wrapper"
    live[4] = []

    issues = coverage.check_contract_coverage(REPOSITORY_ROOT, manifest)

    assert "unapproved dormant native binding: build_info" in issues


def test_contract_matrix_keeps_the_dormant_production_caller_gate_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A future dormant symbol still needs a named decision and zero callers."""

    symbol = "evaluated_paths_capacity_pack_backward"
    facade = "_evaluated_paths_capacity_pack_backward_native"
    owner = f"witwin.channel.propagation.enumerated.{facade}"
    monkeypatch.setattr(coverage, "DORMANT_SYMBOL_FACADES", {symbol: facade})
    monkeypatch.setattr(
        coverage, "DORMANT_EXPERIMENT_SYMBOLS", frozenset({symbol})
    )
    monkeypatch.setattr(coverage, "DORMANT_FACADE_OWNERS", {facade: owner})
    monkeypatch.setattr(coverage, "DORMANT_ALLOWED_FACADE_CALLERS", {})

    issues = coverage.check_contract_coverage(REPOSITORY_ROOT, _manifest())

    assert any(
        issue.startswith(f"dormant facade has production caller: {facade}:")
        for issue in issues
    )
    assert f"dormant native binding uses live owner kind: {symbol}" in issues



def test_contract_matrix_rejects_nonexistent_contract_nodeids():
    manifest = copy.deepcopy(_manifest())
    manifest["contract_tests"]["public-api-snapshot"] = (
        "tests/test_public_api_snapshot.py::test_missing_contract"
    )

    issues = coverage.check_contract_coverage(REPOSITORY_ROOT, manifest)

    assert any("nodeid does not exist" in issue for issue in issues)


def test_contract_coverage_cli_passes_with_repository_defaults(capsys):
    native_binding_count = len(_manifest()["native_bindings"])
    assert coverage.main(["--repository-root", str(REPOSITORY_ROOT)]) == 0
    assert (
        f"{len(_manifest()['public_exports'])} public exports, "
        f"{native_binding_count} native bindings"
        in capsys.readouterr().out
    )
