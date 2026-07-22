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
_DORMANT_SYMBOLS = {
    "coupled_candidate_capacity_block",
    "deterministic_capacity_finalize",
    "deterministic_diffraction_order1_capacity_block",
    "deterministic_diffraction_pair_reduce",
    "deterministic_diffraction_pair_reduce_backward",
    "deterministic_diffraction_pair_reduce_jvp",
    "deterministic_diffraction_state_capacity_select",
    "deterministic_path_table_capacity_pack",
    "deterministic_path_table_capacity_pack_backward",
    "deterministic_path_table_capacity_pack_jvp",
    "deterministic_reflection_candidate_capacity_block",
    "enumerated_canonical_capacity_select",
    "evaluated_paths_canonical_capacity_gather",
    "evaluated_paths_canonical_capacity_gather_backward",
    "evaluated_paths_canonical_capacity_gather_jvp",
    "evaluated_paths_capacity_pack",
    "evaluated_paths_capacity_pack_backward",
    "evaluated_paths_capacity_pack_jvp",
    "path_result_capacity_pack",
    "path_result_capacity_pack_backward",
    "path_result_capacity_pack_jvp",
}


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
    manifest["native_bindings"][0][1] = "witwin.channel_native.missing.owner"
    manifest["native_bindings"][0][4] = []

    issues = coverage.check_contract_coverage(REPOSITORY_ROOT, manifest)

    assert f"native binding has no E2E caller: {symbol}" in issues
    assert any(
        f"native binding Python owner does not exist: {symbol}" in issue
        for issue in issues
    )


def test_contract_matrix_distinguishes_dormant_execution_from_live_e2e() -> None:
    manifest = copy.deepcopy(_manifest())
    dormant = next(
        row
        for row in manifest["native_bindings"]
        if row[0] == "deterministic_diffraction_pair_reduce"
    )
    assert dormant[2] == "dormant_named_wrapper"
    assert dormant[4] == []

    dormant[4] = ["deterministic-diffraction"]
    issues = coverage.check_contract_coverage(REPOSITORY_ROOT, manifest)
    assert (
        "dormant native binding has E2E caller: "
        "deterministic_diffraction_pair_reduce"
    ) in issues

    dormant[4] = []
    dormant[2] = "named_wrapper"
    issues = coverage.check_contract_coverage(REPOSITORY_ROOT, manifest)
    assert (
        "dormant native binding uses live owner kind: "
        "deterministic_diffraction_pair_reduce"
    ) in issues

    live = next(row for row in manifest["native_bindings"] if row[0] == "build_info")
    live[2] = "dormant_named_wrapper"
    issues = coverage.check_contract_coverage(REPOSITORY_ROOT, manifest)
    assert "unapproved dormant native binding: build_info" in issues


def test_dormant_binding_inventory_has_direct_contracts_but_no_e2e_callers() -> None:
    manifest_rows = {row[0]: row for row in _manifest()["native_bindings"]}
    inventory = json.loads(OWNER_INVENTORY_PATH.read_text(encoding="utf-8"))
    owner_rows = {row["symbol"]: row for row in inventory["symbols"]}

    assert set(coverage.DORMANT_EXPERIMENT_SYMBOLS) == _DORMANT_SYMBOLS
    assert set(coverage.DORMANT_SYMBOL_FACADES) == _DORMANT_SYMBOLS
    assert set(coverage.DORMANT_SYMBOL_FACADES.values()) <= set(
        coverage.DORMANT_FACADE_OWNERS
    )
    assert _DORMANT_SYMBOLS <= set(manifest_rows)
    for symbol in coverage.DORMANT_EXPERIMENT_SYMBOLS:
        assert manifest_rows[symbol][2].startswith("dormant_")
        assert manifest_rows[symbol][3]
        assert manifest_rows[symbol][4] == []
        assert owner_rows[symbol]["liveness"] == "dormant-native-producer"
        assert owner_rows[symbol]["contract_test"]
        assert owner_rows[symbol]["production_callers"] == []
        assert owner_rows[symbol]["e2e_callers"] == []


def test_contract_matrix_rejects_injected_dormant_production_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_call_sites = coverage._python_call_sites

    def injected_call_sites(
        repo: Path, targets: frozenset[str]
    ) -> dict[str, list[tuple[str, str, int]]]:
        rows = real_call_sites(repo, targets)
        rows["evaluated_paths_capacity_pack"].append(
            (
                "witwin.channel_native.path.pipeline._solve_base",
                "src/witwin/channel_native/path/pipeline.py",
                1,
            )
        )
        return rows

    monkeypatch.setattr(coverage, "_python_call_sites", injected_call_sites)
    issues = coverage.check_contract_coverage(REPOSITORY_ROOT, _manifest())
    assert any(
        issue.startswith(
            "dormant facade has production caller: evaluated_paths_capacity_pack:"
        )
        for issue in issues
    )


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
        f"37 public exports, {native_binding_count} native bindings"
        in capsys.readouterr().out
    )
