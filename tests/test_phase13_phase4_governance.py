from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPOSITORY_ROOT / "docs" / "dev" / "audit"
DELETED_BINDINGS = {
    "bdpt_reflection_accumulation_forward",
    "path_diffraction_paths_order1",
    "bdpt_diffraction_connection_samples_from_tape",
    "bdpt_diffraction_point_connection_samples",
    "bdpt_diffraction_state_pack",
    "bdpt_diffraction_state_wi",
    "bdpt_selected_edge_indices",
    "bdpt_diffraction_edge_geometry",
    "bdpt_surface_group_edge_candidates",
}
PHASE4_RENAMES = {
    "bdpt_intersect_forward": "rayd_intersect_forward",
    "bdpt_visibility_forward": "rayd_visibility_forward",
    "bdpt_diffraction_discover_edges": "mc_diffraction_discover_edges",
    "bdpt_diffraction_discover_edges_counted": (
        "mc_diffraction_discover_edges_counted"
    ),
}
FROZEN_AUDIT_SHA256 = {
    "phase9-native-owner-inventory.json": (
        "7395305a07b617ea018816b7c58ab8a4c4f46cd60341cc7c2eb2efea2c6c2cf6"
    ),
    "phase10-legacy-dead-binding.json": (
        "c4ee73b2cd8274d365b718d40075288e8a7e124f6644842dc1769806215c2db8"
    ),
    "phase13-phase0-audit.md": (
        "9a2de981a7543b32c3dbc8e3857290a4322a0a414402b4213efc7092b546640a"
    ),
}
PHASE4_BINDING_MANIFEST_SHA256 = (
    "283b7ea04fe8eaeeb5c4b8e4316856a099d5a31f7750e38acb2bfb810fd4b205"
)


def _load(name: str) -> dict[str, object]:
    return json.loads((AUDIT_ROOT / name).read_text(encoding="utf-8"))


def test_phase4_reachability_audit_authorizes_exactly_nine_deletions() -> None:
    audit = _load("phase13-phase4-dead-binding-reachability.json")
    records = audit["records"]

    assert audit["phase"] == 4
    assert audit["binding_count_before"] == 211
    assert audit["binding_count_after"] == 202
    assert set(audit["audit_axes"]) == {
        "static_production_caller",
        "dynamic_binding",
        "public_import",
        "real_bdpt_e2e",
    }
    assert {record["symbol"] for record in records} == DELETED_BINDINGS
    assert len(records) == len(DELETED_BINDINGS)
    for record in records:
        assert record["disposition"] == "deleted"
        assert set(record["axes"]) == set(audit["audit_axes"])
        assert all(not result["reachable"] for result in record["axes"].values())


def test_phase4_symbol_ledger_closes_rename_and_count_deltas() -> None:
    ledger = _load("phase13-symbol-delta-ledger.json")
    actions = {entry["symbol"]: entry for entry in ledger["actions"]}

    assert ledger["baseline_binding_count"] == 211
    assert ledger["applied_count_delta"] == -9
    assert ledger["projected_final_count"] == 202
    for source, target in PHASE4_RENAMES.items():
        action = actions[source]
        assert action["action"] == "rename"
        assert action["replacement"] == target
        assert action["count_delta"] == 0
        assert action["status"] == "applied in Phase 4"
    for symbol in DELETED_BINDINGS:
        action = actions[symbol]
        assert action["action"] == "delete"
        assert action["replacement"] is None
        assert action["count_delta"] == -1
        assert action["status"] == "applied in Phase 4"


def test_phase4_current_inventory_counts_and_manifest_hash_are_exact() -> None:
    inventory = _load("phase13-current-native-owner-inventory.json")
    symbols = inventory["symbols"]
    owner_counts = {
        owner: sum(entry["numerical_owner"] == owner for entry in symbols)
        for owner in {
            "RayD",
            "Channel operation / RayD primitives",
            "Channel",
        }
    }
    migration = _load("phase13-migration-delta.json")

    assert inventory["counts"] == {
        "bindings": len(symbols),
        "rayd_numerical": owner_counts["RayD"],
        "layered": owner_counts["Channel operation / RayD primitives"],
        "channel_numerical": owner_counts["Channel"],
    }
    assert inventory["phase4_generic_geometry_and_dead_bridge_cleanup"][
        "binding_manifest_sha256"
    ] == PHASE4_BINDING_MANIFEST_SHA256
    assert (
        migration["phase4_current"]["binding_manifest_sha256"]
        == PHASE4_BINDING_MANIFEST_SHA256
    )


def test_phase4_does_not_rewrite_historical_audits() -> None:
    for name, expected in FROZEN_AUDIT_SHA256.items():
        # Path.read_text normalizes checkout line endings before hashing, so the
        # invariant is stable across Windows and clean-clone CI checkouts.
        payload = (AUDIT_ROOT / name).read_text(encoding="utf-8").encode()
        assert hashlib.sha256(payload).hexdigest() == expected
