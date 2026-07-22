"""Immutable historical assertions for the completed Plan 13 Phase 10A cut."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/dev/audit"
RAYD_COMMIT = "4577e744adfe8665f7817e3aff5e8e533ec896e7"
INTEGRATION_SHA256 = (
    "9f95ad9e8e3b790d00f8e762a3e6a09252d46afb65bfc3aba7c42325836cb1fb"
)
TYPED_SHA256 = (
    "66d75a20be16057f03cdfb79e3b9dcc85cacec79b555cd73b019259aa510262a"
)
SHARED_SHA256 = (
    "38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38"
)
PHASE10A = {
    "scattering_table_eval",
    "scattering_table_eval_backward",
    "scattering_table_eval_jvp",
    "scattering_table_sample",
    "scattering_table_pdf",
    "scattering_ensemble_eval",
    "scattering_ensemble_eval_backward",
    "scattering_ensemble_eval_jvp",
    "scattering_patch_integral_eval",
    "scattering_patch_integral_eval_backward",
    "scattering_patch_integral_eval_jvp",
}


def _json(name: str) -> dict[str, object]:
    return json.loads((AUDIT / name).read_text(encoding="utf-8"))


def test_phase10a_activation_record_is_an_immutable_historical_snapshot() -> None:
    evidence = _json("phase13-scattering-phase10a-evidence.json")
    inventory = _json("phase13-current-native-owner-inventory.json")
    migration = _json("phase13-migration-delta.json")
    graph = _json("phase13-shared-rf-dependency-graph.json")
    pin = evidence["activation_pin"]

    assert evidence["phase"] == "10A"
    assert evidence["status"].startswith("active; pushed RayD owner")
    assert pin["rayd_commit"] == RAYD_COMMIT
    assert pin["integration_header_sha256"] == INTEGRATION_SHA256
    assert pin["typed_header_sha256"] == TYPED_SHA256
    assert pin["shared_table_header_sha256"] == SHARED_SHA256
    assert inventory["phase10a_scattering_table_single_bounce"]["rayd_commit"] == RAYD_COMMIT
    assert migration["phase10a_current"]["rayd_commit"] == RAYD_COMMIT
    assert graph["phase10a_activation"]["rayd_commit"] == RAYD_COMMIT


def test_phase10a_owner_launch_codegen_and_deletion_snapshot_remains_complete() -> None:
    evidence = _json("phase13-scattering-phase10a-evidence.json")
    owner = evidence["owner_transfer"]
    switch = evidence["channel_switch"]
    launch = evidence["launch_contract"]
    codegen = evidence["codegen_resource_contract"]

    assert set(owner["symbols"]) == PHASE10A
    assert owner["expected_active_owner_counts"] == {
        "bindings": 202,
        "rayd_numerical": 37,
        "layered": 2,
        "channel_numerical": 163,
    }
    assert len(switch["deleted_numerical_sources"]) == 5
    assert len(switch["retained_chain_sources"]) == 4
    assert launch["zero_row_launch_count"] == 0
    assert launch["explicit_sync_count"] == 0
    assert launch["persistent_tape"] is False
    assert codegen["normalized_sass_equal"] is True
    assert codegen["resource_usage_equal"] is True


def test_phase10a_direct_and_duplication_evidence_remains_frozen() -> None:
    evidence = _json("phase13-scattering-phase10a-evidence.json")
    direct = evidence["rayd_candidate"]["direct_contract_test"]
    duplication = evidence["duplication_contract"]

    assert len(evidence["rayd_candidate"]["numerical_sources"]) == 6
    assert direct["status"] == "passed; full RayD CTest 3/3"
    assert len(direct["sha256"]) == 64
    assert duplication == {
        "combined_duplicate_lines": 9730,
        "combined_total_lines": 81675,
        "coverage_percent": 11.91307,
        "frozen_coverage_percent": 10.211512,
        "region_count": 169,
        "stale_region_count": 0,
        "unclassified_region_count": 0,
        "budget_relaxed": False,
    }
