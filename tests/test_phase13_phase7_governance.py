from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/dev/audit"
ADR = ROOT / "docs/dev/standards/adr-025-diffraction-operation-family-ownership.md"

FAMILY_IDS = {
    "rayd-order1-export-and-tape",
    "pure-wedge-fixed-winner-field",
    "mc-sionna-fixed-tape-estimator",
    "coupled-rd-field",
    "coupled-dd-field",
    "coupled-rd-stationary-geometry",
    "composed-rd-dd-geometry",
    "solver-propagation-diffraction-ops",
    "bdpt-diffraction-policy-storage",
}
PURE_WEDGE = {
    "field_diffraction_wedge",
    "field_diffraction_wedge_backward",
    "field_diffraction_wedge_jvp",
}
MC_SIONNA = {
    "mc_sionna_diffraction_tape_accumulate",
    "mc_sionna_diffraction_tape_accumulate_backward",
    "mc_sionna_diffraction_tape_accumulate_jvp",
}
COUPLED_RD = {
    "field_coupled_rd",
    "field_coupled_rd_backward",
    "field_coupled_rd_jvp",
}
COUPLED_DD = {
    "field_coupled_dd",
    "field_coupled_dd_backward",
    "field_coupled_dd_jvp",
}
DELETED_BDPT_DIFFRACTION = {
    "bdpt_diffraction_connection_samples_from_tape",
    "bdpt_diffraction_point_connection_samples",
    "bdpt_diffraction_state_pack",
    "bdpt_diffraction_state_wi",
    "bdpt_diffraction_edge_geometry",
}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_symbols() -> set[str]:
    manifest = _json(ROOT / "ci/native-binding-manifest.json")
    return {entry["name"] for entry in manifest["symbols"]}  # type: ignore[index]


def _families() -> dict[str, dict[str, object]]:
    matrix = _json(AUDIT / "phase13-diffraction-family-matrix.json")
    return {
        entry["family_id"]: entry  # type: ignore[misc]
        for entry in matrix["families"]  # type: ignore[index]
    }


def test_phase7_decision_history_is_preserved_after_phase8a_activation() -> None:
    adr = ADR.read_text(encoding="utf-8")
    plan = (
        ROOT / "docs/dev/plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md"
    ).read_text(encoding="utf-8")
    feature = (ROOT / "FEATURE_LIST.md").read_text(encoding="utf-8")
    migration = (
        ROOT / "docs/dev/replacement/channel-native-migration.md"
    ).read_text(encoding="utf-8")
    owner_readme = (
        ROOT / "src/witwin/channel_native/propagation/README.md"
    ).read_text(encoding="utf-8")

    assert "**Status:** Accepted (2026-07-19)" in adr
    assert "ADR-025 已接受；此阶段只接受边界，不执行 Phase 8A/8B" in plan
    assert "2026-07-19" in plan
    for text in (adr, plan, feature, migration, owner_readme):
        normalized = " ".join(text.split())
        assert "Phase 8A" in normalized
        assert "Phase 8B" in normalized
    assert "does not move production code" in migration
    assert "Phase 8A" in feature
    assert "numerical owner to RayD" in " ".join(feature.split())


def test_phase7_family_matrix_freezes_all_nine_complete_owners() -> None:
    matrix = _json(AUDIT / "phase13-diffraction-family-matrix.json")
    families = _families()

    assert matrix["schema_version"] == 2
    assert matrix["decision_phase"] == 7
    assert matrix["family_count"] == len(families) == 9
    assert set(families) == FAMILY_IDS

    pure = families["pure-wedge-fixed-winner-field"]
    assert set(pure["symbols"]) == PURE_WEDGE  # type: ignore[arg-type]
    assert pure["phase7_current_owner"] == "Channel Native"
    assert pure["accepted_authoritative_owner"] == "RayD"
    assert pure["activation_phase"] == "8A atomic pin/switch/delete"

    for family_id, symbols in {
        "mc-sionna-fixed-tape-estimator": MC_SIONNA,
        "coupled-rd-field": COUPLED_RD,
        "coupled-dd-field": COUPLED_DD,
    }.items():
        family = families[family_id]
        assert set(family["symbols"]) == symbols  # type: ignore[arg-type]
        assert family["phase7_current_owner"] == "Channel Native"
        assert family["accepted_authoritative_owner"] == "Channel Native"
        assert family["compile_contract"] == "precise math"


def test_phase7_pre_activation_pure_wedge_snapshot_is_preserved() -> None:
    source = ROOT / "native/channel_native/kernels/field_wedge_ad_diffraction.cu"
    migration = _json(AUDIT / "phase13-migration-delta.json")
    phase8a = migration["phase8a_current"]  # type: ignore[index]
    manifest = _manifest_symbols()

    assert not source.exists()
    assert PURE_WEDGE <= manifest
    assert phase8a["deleted_source_sha256"] == (  # type: ignore[index]
        "68ec3fe180cd900834f0263969ee75d54764ad014e5d22b7c0b57822ea8e975b"
    )
    deletions = phase8a["approved_phase9_body_hash_deletions"]  # type: ignore[index]
    assert len(deletions) == 16
    assert {
        "diffraction_wedge_forward_kernel",
        "diffraction_wedge_backward_kernel",
        "diffraction_wedge_jvp_kernel",
        "cn_field_diffraction_wedge",
        "cn_field_diffraction_wedge_backward",
        "cn_field_diffraction_wedge_jvp",
    } <= {entry["name"] for entry in deletions}


def test_phase8b_legacy_audit_closes_deletions_and_sample_tape_rename() -> None:
    audit = _json(AUDIT / "phase13-diffraction-legacy-audit.json")
    phase4 = _json(AUDIT / "phase13-phase4-dead-binding-reachability.json")
    manifest = _manifest_symbols()
    deleted = {
        record["symbol"]
        for record in audit["closed_deletions"]  # type: ignore[index]
    }
    phase4_deleted = {
        record["symbol"]
        for record in phase4["records"]  # type: ignore[index]
        if record["disposition"] == "deleted"
    }

    assert set(audit["audit_axes"]) == {  # type: ignore[arg-type]
        "static_production_caller",
        "dynamic_binding",
        "public_import",
        "real_bdpt_e2e",
    }
    assert deleted == DELETED_BDPT_DIFFRACTION <= phase4_deleted
    assert deleted.isdisjoint(manifest)
    assert {
        "mc_diffraction_discover_edges",
        "mc_diffraction_discover_edges_counted",
    } <= manifest
    assert {
        "bdpt_diffraction_discover_edges",
        "bdpt_diffraction_discover_edges_counted",
    }.isdisjoint(manifest)

    rename = audit["completed_phase8b_rename"]  # type: ignore[index]
    assert rename == {
        "old_symbol": "bdpt_diffraction_accumulation_forward",
        "new_symbol": "rayd_diffraction_sample_tape_forward",
        "current_numerical_owner": "RayD",
        "real_production_owner": "montecarlo.basic sample-tape production",
        "static_production_caller": (
            "src/witwin/channel_native/montecarlo/basic/rayd_components.py"
        ),
        "rename_only": True,
        "compatibility_alias_allowed": False,
        "trim_output_allowed": False,
        "status": "applied in Phase 8B; no live alias or re-export",
    }
    assert rename["old_symbol"] not in manifest
    assert rename["new_symbol"] in manifest


def test_phase7_freezes_native_tx_visibility_selection_requirements() -> None:
    matrix = _json(AUDIT / "phase13-diffraction-family-matrix.json")
    selection = matrix["tx_visible_state_selection"]  # type: ignore[index]
    source = (
        ROOT / "src/witwin/channel_native/propagation/geometry/diffraction.py"
    ).read_text(encoding="utf-8")

    assert selection["accepted_native_operation"] == (  # type: ignore[index]
        "diffraction_tx_visible_state_plan"
    )
    assert selection["operation_owner"] == (  # type: ignore[index]
        "Channel operation / RayD visibility primitives"
    )
    assert selection["fractions_in_order"] == [  # type: ignore[index]
        "0.02",
        "1/3",
        "2/3",
        "0.98",
    ]
    assert selection["activation_status"] == "active"  # type: ignore[index]
    assert selection["state_capacity"] == 4_194_304  # type: ignore[index]
    assert "def plan_tx_visible_diffraction_states(" in source
    assert "geometry_bridge.diffraction_tx_visible_state_plan(" in source
    for forbidden in (
        "_DIFFRACTION_PREFILTER_EDGE_FRACTIONS",
        "for fraction in",
        "geometry_bridge.rayd_visibility_forward(",
        "bool(visible.all())",
        "tensor[visible]",
    ):
        assert forbidden not in source


def test_phase7_guardrails_are_byte_identical_and_authoritative() -> None:
    agents = (ROOT / "AGENTS.md").read_bytes()
    claude = (ROOT / "CLAUDE.md").read_bytes()

    assert agents == claude
    text = agents.decode("utf-8")
    assert "Under ADR-025" in text
    assert (
        "docs/dev/standards/adr-025-diffraction-operation-family-ownership.md"
        in text
    )


def test_phase7_adr_plan_link_resolves() -> None:
    text = ADR.read_text(encoding="utf-8")

    assert "[Plan 13](../plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md)" in text
    assert (ADR.parent.parent / "plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md").is_file()
