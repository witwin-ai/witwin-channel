from __future__ import annotations

import copy
from pathlib import Path

from ci import check_contract_coverage as coverage


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "contract-coverage-manifest.json"


def _manifest() -> dict[str, object]:
    return coverage.load_manifest(MANIFEST_PATH)


def test_current_contract_matrix_covers_every_public_export_and_native_binding():
    manifest = _manifest()

    assert coverage.check_contract_coverage(REPOSITORY_ROOT, manifest) == []
    assert len(manifest["public_exports"]) == coverage.EXPECTED_PUBLIC_EXPORT_COUNT
    assert len(manifest["native_bindings"]) == coverage.EXPECTED_NATIVE_BINDING_COUNT


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


def test_contract_matrix_rejects_nonexistent_contract_nodeids():
    manifest = copy.deepcopy(_manifest())
    manifest["contract_tests"]["public-api-snapshot"] = (
        "tests/test_public_api_snapshot.py::test_missing_contract"
    )

    issues = coverage.check_contract_coverage(REPOSITORY_ROOT, manifest)

    assert any("nodeid does not exist" in issue for issue in issues)


def test_contract_coverage_cli_passes_with_repository_defaults(capsys):
    assert coverage.main(["--repository-root", str(REPOSITORY_ROOT)]) == 0
    assert "37 public exports, 211 native bindings" in capsys.readouterr().out
