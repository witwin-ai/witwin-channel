from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.refactor_baseline import binding_manifest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/dev/audit"
EVIDENCE_PATH = AUDIT / "phase13-boundary-dedup-phase11a-evidence.json"
LEDGER_PATH = AUDIT / "duplication-classification.json"
MANIFEST_PATH = ROOT / "ci/native-binding-manifest.json"
BOUNDARY_FILES = {"fields.cpp", "materials.cpp"}
PHASE_P_PRE_SYMBOL_COUNT = 229
PHASE_P_SYMBOL_PATHS = {
    "enumerated_transmission_topology_pack": "native/channel_native/binding/path.cpp",
    "rayd_segment_penetration_backward": "native/channel_native/binding/rayd.cpp",
    "rayd_segment_penetration_forward": "native/channel_native/binding/rayd.cpp",
    "rayd_segment_penetration_forward_tape": "native/channel_native/binding/rayd.cpp",
    "rayd_segment_penetration_jvp": "native/channel_native/binding/rayd.cpp",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase11a_duplication_refresh_is_classified_without_budget_relaxation() -> None:
    evidence = _json(EVIDENCE_PATH)
    ledger = _json(LEDGER_PATH)
    refresh = ledger["phase11a_refresh"]
    current = evidence["duplication"]["current"]

    assert evidence["method"]["comparison"] == "EXACT_TOKEN_MATCH"
    assert evidence["method"]["min_tokens"] == ledger["min_tokens"] == 100
    assert current == {
        "duplicate_lines": refresh["combined_duplicate_lines"],
        "total_lines": refresh["combined_total_lines"],
        "coverage_percent": refresh["coverage_percent"],
        "region_count": refresh["region_count"],
    }
    assert refresh["region_count"] == 155
    assert ledger["phase11b_refresh"]["region_count"] == 143
    assert len(ledger["regions"]) >= ledger["phase11b_refresh"]["region_count"]
    assert refresh["stale_region_count"] == 0
    assert refresh["unclassified_region_count"] == 0
    assert refresh["budget_relaxed"] is False
    assert refresh["coverage_percent"] > refresh["frozen_coverage_percent"]
    assert evidence["duplication"]["frozen_budget"] == {
        "coverage_percent": refresh["frozen_coverage_percent"],
        "relaxed": False,
        "met": False,
        "final_acceptance": "pending Phase 11B",
    }

    stale = set(evidence["ledger_refresh"]["stale_regions_removed"])
    assert len(stale) == 5
    assert stale.isdisjoint(ledger["regions"])
    assert evidence["ledger_refresh"]["new_regions_classified"] == []
    boundary_regions = [
        region
        for region in ledger["regions"].values()
        if BOUNDARY_FILES.intersection(region["files"])
    ]
    assert boundary_regions
    assert all(region["category"] == "other" for region in boundary_regions)
    assert all(region["owner"] == "bindings" for region in boundary_regions)


def test_phase11a_manifest_delta_is_location_only_and_invariants_are_non_numerical() -> (
    None
):
    evidence = _json(EVIDENCE_PATH)
    migration = _json(AUDIT / "phase13-migration-delta.json")
    phase10b = _json(AUDIT / "phase13-scattering-phase10b-evidence.json")
    manifest = _json(MANIFEST_PATH)
    record = evidence["binding_manifest"]

    assert manifest == binding_manifest(ROOT)
    current_symbols = {row["name"]: row["path"] for row in manifest["symbols"]}
    assert len(current_symbols) == 234
    assert len(current_symbols) - PHASE_P_PRE_SYMBOL_COUNT == len(PHASE_P_SYMBOL_PATHS)
    assert {
        name: current_symbols[name] for name in PHASE_P_SYMBOL_PATHS
    } == PHASE_P_SYMBOL_PATHS
    assert record["symbol_count"] == 202
    assert len(record["current_sha256"]) == 64
    assert len(record["current_semantic_sha256"]) == 64
    assert record["pre_semantic_sha256"] == record["current_semantic_sha256"]
    assert record["semantic_changes"] == 0
    assert (
        phase10b["activation"]["binding_manifest_sha256"]
        == record["phase10b_snapshot_sha256"]
    )
    assert migration["phase11a_current"]["evidence"] == str(
        EVIDENCE_PATH.relative_to(ROOT)
    ).replace("\\", "/")
    assert migration["phase11a_current"]["binding_semantic_changes"] == 0
    assert (
        migration["phase8b_device_resident_diffraction_planning"]["native_symbol_added"]
        == "diffraction_tx_visible_state_plan"
    )
    assert all(value is False for value in evidence["invariants"].values())

    for relative in evidence["scope"]:
        source = (ROOT / relative).read_text(encoding="utf-8-sig")
        assert "py::args" not in source
        assert "pybind11::args" not in source
        assert "<<<" not in source
