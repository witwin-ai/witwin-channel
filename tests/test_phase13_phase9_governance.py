from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADR = ROOT / "docs/dev/standards/adr-026-rayd-generic-scattering-runtime-ownership.md"
PLAN = ROOT / "docs/dev/plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md"
AUDIT = ROOT / "docs/dev/audit"

FAMILIES = {
    "table evaluation AD": {
        "scattering_table_eval",
        "scattering_table_eval_backward",
        "scattering_table_eval_jvp",
    },
    "table sampling": {
        "scattering_table_sample",
        "scattering_table_pdf",
    },
    "single-bounce ensemble": {
        "scattering_ensemble_eval",
        "scattering_ensemble_eval_backward",
        "scattering_ensemble_eval_jvp",
    },
    "phase-screen patch integral": {
        "scattering_patch_integral_eval",
        "scattering_patch_integral_eval_backward",
        "scattering_patch_integral_eval_jvp",
    },
    "chain ensemble": {
        "scattering_chain_ensemble_eval",
        "scattering_chain_ensemble_eval_backward",
        "scattering_chain_ensemble_eval_jvp",
    },
    "chain realization": {
        "scattering_chain_realization_eval",
        "scattering_chain_realization_eval_backward",
        "scattering_chain_realization_eval_jvp",
    },
}
MOVING = set().union(*FAMILIES.values())
TABLE_HELPERS = {
    "positive_phi",
    "linear_axis",
    "nearest_axis",
    "interp4",
    "eval_te_tm",
    "linear_axis_grad",
    "eval_te_tm_grad",
}
def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase9_accepts_exactly_six_complete_families_and_seventeen_symbols() -> None:
    text = ADR.read_text(encoding="utf-8")
    audit = _json(AUDIT / "phase13-scattering-bindings.json")
    moving_rows = {
        row["symbol"]
        for row in audit["contracts"]  # type: ignore[index]
        if row["target_numerical_owner"] == "RayD after ADR-026"
    }

    assert len(FAMILIES) == 6
    assert len(MOVING) == 17
    assert moving_rows == MOVING
    assert audit["binding_count"] == 18
    assert audit["move_count"] == 17
    assert audit["retain_count"] == 1
    for family, symbols in FAMILIES.items():
        assert family in text
        assert all(f"`{symbol}`" in text for symbol in symbols)
    assert "`scattering_event_probabilities`" in text
    assert audit["phase10a_activation"]["activated_contract_count"] == 11  # type: ignore[index]


def test_phase9_freezes_resource_header_and_channel_policy_owners() -> None:
    text = ADR.read_text(encoding="utf-8")
    decision = _json(AUDIT / "phase13-shared-rf-helper-ownership-decision.json")
    table_group = next(
        group
        for group in decision["groups"]  # type: ignore[index]
        if group["baseline_source"].endswith("scattering_table.cuh")
    )

    assert set(table_group["helpers"]) == TABLE_HELPERS
    assert table_group["accepted_target_owner"] == "RayD after ADR-026"
    assert all(f"`{helper}`" in text for helper in TABLE_HELPERS)
    for owner in (
        "KirchhoffRuntimeResources",
        "KirchhoffTableStack",
        "PhaseScreenRuntime",
        "scattering_event_probabilities",
        "RNG",
        "MIS",
    ):
        assert owner in text
    assert "RayD never\nincludes a Channel-private header" in text
    assert "kirchhoff_table_ad.cu" in text


def test_phase9_freezes_per_tu_flags_and_family_specific_geometry_ad() -> None:
    text = ADR.read_text(encoding="utf-8")
    audit = _json(AUDIT / "phase13-scattering-bindings.json")
    compile_contracts = {
        row["symbol"]: row["compile_contract"]
        for row in audit["contracts"]  # type: ignore[index]
    }

    assert "target-default CUDA flags" in compile_contracts["scattering_table_eval"]
    assert "--fmad=false" in compile_contracts["scattering_table_eval_backward"]
    assert "--fmad=false" in compile_contracts["scattering_ensemble_eval"]
    assert "--fmad=false" in compile_contracts["scattering_patch_integral_eval"]
    assert all(
        "--fmad=false" in compile_contracts[symbol]
        for symbol in FAMILIES["chain ensemble"]
        | FAMILIES["chain realization"]
    )
    assert "table primal/sample/PDF kernels currently in `scattering.cu`" in text
    assert "must not gain `--fmad=false`" in text
    assert "chain-ensemble reverse-mode continuous geometry is unsupported" in text
    assert "chain-realization backward and JVP both support" in text
    assert "supersedes any blanket Plan-13 wording" in text


def test_phase9_snapshot_is_docs_only_and_records_pre_activation_owners() -> None:
    plan = PLAN.read_text(encoding="utf-8")
    snapshot = _json(AUDIT / "phase13-scattering-bindings.json")
    targets = {
        row["symbol"]: row["numerical_owner"]
        for row in _json(AUDIT / "phase13-current-native-owner-inventory.json")[
            "symbols"
        ]  # type: ignore[index]
    }
    phase9_targets = {
        row["symbol"]: row["target_numerical_owner"]
        for row in snapshot["contracts"]  # type: ignore[index]
    }

    assert "Phase 9 moves no source and changes no production" in (
        ROOT / "docs/dev/replacement/channel-migration.md"
    ).read_text(encoding="utf-8")
    assert "此阶段只接受边界，不执行 Phase\n10A/10B" in plan
    assert all(phase9_targets[symbol] == "RayD after ADR-026" for symbol in MOVING)
    assert all(targets[symbol] == "RayD" for symbol in MOVING)


def test_phase9_records_and_repository_guardrails_are_synchronized() -> None:
    plan = PLAN.read_text(encoding="utf-8")
    adr010 = (
        ROOT / "docs/dev/standards/adr-010-native-scattering-kernels.md"
    ).read_text(encoding="utf-8")
    # `scattering` is one module now, so its owner document lives under docs/.
    scattering_readme = (
        ROOT / "docs/dev/scattering/README.md"
    ).read_text(encoding="utf-8")

    assert "ADR-032 已接受" in plan
    assert "ADR-029 已因 Munich E2E/显存/吞吐" in plan
    assert "ADR-030 reducer 保持 Dormant" in plan
    assert "Phase 9 — ADR-026" in plan
    assert "**状态：已完成（2026-07-19）。**" in plan
    assert "implementation ownership is superseded by ADR-026" in adr010
    assert "of all 17 table evaluation/sampling" in scattering_readme
    assert "fused ensemble/realization chain contracts" in scattering_readme
    assert (ROOT / "AGENTS.md").read_bytes() == (ROOT / "CLAUDE.md").read_bytes()
    assert ADR.name in (ROOT / "AGENTS.md").read_text(encoding="utf-8")
