from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/dev/audit/phase13-adr027-penetration-foundation.json"
OWNER_INVENTORY = ROOT / "docs/dev/audit/phase13-current-native-owner-inventory.json"
SYMBOL_LEDGER = ROOT / "docs/dev/audit/phase13-symbol-delta-ledger.json"
BINDING_MANIFEST = ROOT / "ci/native-binding-manifest.json"
COVERAGE_MANIFEST = ROOT / "ci/contract-coverage-manifest.json"
DUPLICATION_LEDGER = ROOT / "docs/dev/audit/duplication-classification.json"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase_p_binding_owner_and_symbol_ledgers_close_exactly() -> None:
    audit = _json(AUDIT)
    inventory = _json(OWNER_INVENTORY)
    ledger = _json(SYMBOL_LEDGER)
    binding = _json(BINDING_MANIFEST)
    coverage = _json(COVERAGE_MANIFEST)
    expected = {
        "rayd_segment_penetration_forward",
        "rayd_segment_penetration_forward_tape",
        "rayd_segment_penetration_backward",
        "rayd_segment_penetration_jvp",
        "enumerated_transmission_topology_pack",
    }

    recorded = {
        *audit["new_symbols"]["rayd_segment_penetration"],
        *audit["new_symbols"]["channel_topology"],
    }
    binding_names = {entry["name"] for entry in binding["symbols"]}
    coverage_names = {entry[0] for entry in coverage["native_bindings"]}
    owner_rows = {entry["symbol"]: entry for entry in inventory["symbols"]}
    actions = {entry["symbol"]: entry for entry in ledger["actions"]}

    assert recorded == expected
    assert expected <= binding_names == coverage_names == set(owner_rows)
    assert audit["binding_universe"]["after"] == 234
    assert len(binding_names) == inventory["counts"]["bindings"] == 238
    assert (
        sum(
            inventory["counts"][name]
            for name in ("rayd_numerical", "layered", "channel_numerical")
        )
        == 238
    )
    assert ledger["projected_final_count"] == 202
    assert ledger["applied_count_delta"] == -9
    assert ledger["live_binding_count"] == 238
    assert ledger["live_count_delta_from_plan13_baseline"] == 27
    assert ledger["phase6c_phase_p_delta"] == {
        "before": 229,
        "added": 5,
        "after": 234,
    }
    assert all(actions[name]["count_delta"] == 1 for name in expected)
    assert all(owner_rows[name]["production_callers"] for name in expected)


def test_phase_p_freezes_failure_launch_and_memory_contracts() -> None:
    audit = _json(AUDIT)

    assert audit["status"].startswith("implemented as dormant producers")
    assert audit["rayd_lock"] == {
        "commit": "474c122aa3cd6b6d098675e076a73e6f485bd6be",
        "integration_header": "backends/torch/include/rayd/torch/integration.h",
        "integration_header_sha256": (
            "57f83ea460e376166fd5ee22a8243a7c1576a290e1de99c0cbe8e86e93392e14"
        ),
        "integration_header_identity": "rayd.torch.integration",
        "integration_api_version": 6,
        "pushed": True,
    }
    assert audit["typed_contract"]["failure_bit"] == (
        "SEGMENT_PENETRATION_FAILURE = 1 << 7"
    )
    assert audit["typed_contract"]["failure_scope"] == (
        "overflow, request/device-mask contract contradiction, and non-finite "
        "penetration state"
    )
    assert audit["typed_contract"]["host_cardinality_read"] is False
    assert audit["typed_contract"]["partial_result"] is False
    assert audit["launch_budget"]["segment_forward_active"]["optix"] == 1
    assert audit["launch_budget"]["segment_forward_all_inactive"]["optix"] == 0
    assert audit["launch_budget"]["enumerated_transmission_topology_pack_N_gt_0"] == 4
    assert audit["resident_byte_budget"]["segment_primal_plus_tape"] == (
        "23*N + 63*N*D"
    )
    assert audit["resident_byte_budget"]["enumerated_tape_plus_topology"] == (
        "N*(96 + 95*D) + 8"
    )
    assert audit["public_api"]["changed"] is False
    assert audit["public_api"]["generation_suffixed_name_added"] is False


def test_phase_p_live_duplication_refresh_is_closed_without_budget_relaxation() -> None:
    duplication = _json(DUPLICATION_LEDGER)
    refresh = duplication["phase13_phase_p_refresh"]

    assert refresh == {
        "combined_duplicate_lines": 9611,
        "combined_total_lines": 94455,
        "coverage_percent": 10.175216,
        "frozen_coverage_percent": 10.211512,
        "budget_relaxed": False,
        "new_region_count": 14,
        "pruned_region_count": 12,
        "note": (
            "Phase P added the dormant ADR-027 segment-penetration/topology "
            "foundation and adapted existing RayD-owned RF facades to the API 6 "
            "explicit valid-mask contract. All live exact-token regions were "
            "hand-reviewed and stale ids were pruned without changing the frozen "
            "maintenance budget or the immutable Phase 11A/11B evidence."
        ),
        "region_count": 171,
        "stale_region_count": 0,
        "unclassified_region_count": 0,
        "status": (
            "all current regions classified; frozen duplication coverage "
            "acceptance remains met"
        ),
    }
    assert len(duplication["regions"]) == refresh["region_count"]
    assert refresh["coverage_percent"] < refresh["frozen_coverage_percent"]
    assert duplication["phase11a_refresh"]["region_count"] == 155
    assert duplication["phase11b_refresh"]["region_count"] == 143
