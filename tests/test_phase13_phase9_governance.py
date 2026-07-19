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
    "v2 chain ensemble": {
        "scattering_chain_ensemble_eval",
        "scattering_chain_ensemble_eval_backward",
        "scattering_chain_ensemble_eval_jvp",
    },
    "v2 chain realization": {
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
FMAD_FALSE_TUS = {
    "scattering_table_eval_ad.cu",
    "scattering_ensemble.cu",
    "scattering_ensemble_ad.cu",
    "scattering_patch_integral.cu",
    "scattering_patch_integral_ad.cu",
    "scattering_chain_ensemble.cu",
    "scattering_chain_ensemble_ad.cu",
    "scattering_chain_realization.cu",
    "scattering_chain_realization_ad.cu",
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
    assert "does not activate it" in text


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
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    fmad_block = cmake.split("set_source_files_properties(", 1)[1].split(
        'PROPERTIES COMPILE_OPTIONS "--fmad=false")', 1
    )[0]

    assert "scattering.cu" not in fmad_block
    assert all(tu in fmad_block for tu in FMAD_FALSE_TUS)
    assert "table primal/sample/PDF kernels currently in `scattering.cu`" in text
    assert "must not gain `--fmad=false`" in text
    assert "chain-ensemble reverse-mode continuous geometry is unsupported" in text
    assert "chain-realization backward and JVP both support" in text
    assert "supersedes any blanket Plan-13 wording" in text


def test_phase9_is_docs_only_and_leaves_channel_as_current_numerical_owner() -> None:
    inventory = _json(AUDIT / "phase13-current-native-owner-inventory.json")
    owners = {
        row["symbol"]: row["numerical_owner"]
        for row in inventory["symbols"]  # type: ignore[index]
    }
    materials = (ROOT / "native/channel_native/binding/materials.cpp").read_text(
        encoding="utf-8"
    )

    assert all(owners[symbol] == "Channel Native" for symbol in MOVING)
    assert all(
        (ROOT / path).is_file()
        for path in (
            "native/channel_native/kernels/scattering_table.cuh",
            "native/channel_native/kernels/scattering_table_eval_ad.cu",
            "native/channel_native/kernels/scattering_ensemble.cu",
            "native/channel_native/kernels/scattering_ensemble_ad.cu",
            "native/channel_native/kernels/scattering_patch_integral.cu",
            "native/channel_native/kernels/scattering_patch_integral_ad.cu",
            "native/channel_native/kernels/scattering_chain_ensemble.cu",
            "native/channel_native/kernels/scattering_chain_ensemble_ad.cu",
            "native/channel_native/kernels/scattering_chain_realization.cu",
            "native/channel_native/kernels/scattering_chain_realization_ad.cu",
        )
    )
    assert "rayd::torch::scattering_" not in materials
    assert inventory["current_subphase"] == "8A"


def test_phase9_records_and_repository_guardrails_are_synchronized() -> None:
    plan = PLAN.read_text(encoding="utf-8")
    adr010 = (
        ROOT / "docs/dev/standards/adr-010-native-scattering-kernels.md"
    ).read_text(encoding="utf-8")
    scattering_readme = (
        ROOT / "src/witwin/channel_native/scattering/README.md"
    ).read_text(encoding="utf-8")

    assert "ADR-023/024/025/026 已接受" in plan
    assert "Phase 9 — ADR-026" in plan
    assert "**状态：已完成（2026-07-19）。**" in plan
    assert "implementation ownership is superseded by ADR-026" in adr010
    assert "17 generic resident scattering runtime contracts" in scattering_readme
    assert (ROOT / "AGENTS.md").read_bytes() == (ROOT / "CLAUDE.md").read_bytes()
    assert ADR.name in (ROOT / "AGENTS.md").read_text(encoding="utf-8")
