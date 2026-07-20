from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/dev/audit"
EVIDENCE_PATH = AUDIT / "phase13-boundary-dedup-phase11b-evidence.json"
LEDGER_PATH = AUDIT / "duplication-classification.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _literal_tuple(path: Path, name: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            value = ast.literal_eval(node.value)
            assert isinstance(value, tuple)
            return value
    raise AssertionError(f"missing {name} in {path}")


def test_phase11b_duplication_budget_is_met_without_relaxation() -> None:
    evidence = _json(EVIDENCE_PATH)
    ledger = _json(LEDGER_PATH)
    migration = _json(AUDIT / "phase13-migration-delta.json")
    refresh = ledger["phase11b_refresh"]
    current = evidence["duplication"]["current"]

    assert evidence["method"]["comparison"] == "EXACT_TOKEN_MATCH"
    assert evidence["method"]["min_tokens"] == ledger["min_tokens"] == 100
    assert current == {
        "duplicate_lines": refresh["combined_duplicate_lines"],
        "total_lines": refresh["combined_total_lines"],
        "coverage_percent": refresh["coverage_percent"],
        "region_count": refresh["region_count"],
    }
    assert refresh["region_count"] == len(ledger["regions"]) == 143
    assert refresh["stale_region_count"] == 0
    assert refresh["unclassified_region_count"] == 0
    assert refresh["budget_relaxed"] is False
    assert refresh["coverage_percent"] < refresh["frozen_coverage_percent"]
    assert refresh["frozen_coverage_percent"] == ledger["baseline"]["coverage_percent"]
    assert evidence["duplication"]["frozen_budget"] == {
        "coverage_percent": 10.211512,
        "relaxed": False,
        "met": True,
        "margin_percentage_points": 0.155099,
    }
    assert migration["current_phase"] == 11
    assert migration["current_subphase"] == "11B"
    assert migration["phase11b_current"]["duplication_refresh"]["budget_met"] is True
    assert migration["phase11b_current"]["evidence"] == str(
        EVIDENCE_PATH.relative_to(ROOT)
    ).replace("\\", "/")


def test_phase11b_ledger_refresh_and_source_snapshots_are_current() -> None:
    evidence = _json(EVIDENCE_PATH)
    ledger = _json(LEDGER_PATH)

    stale = set(evidence["ledger_refresh"]["stale_regions_removed"])
    assert len(stale) == 13
    assert stale.isdisjoint(ledger["regions"])
    new = evidence["ledger_refresh"]["new_regions_classified"]
    assert [record["region_id"] for record in new] == ["490234d077127261"]
    assert ledger["regions"]["490234d077127261"]["category"] == "fixture_boilerplate"
    assert evidence["ledger_refresh"]["stale_region_count"] == 0
    assert evidence["ledger_refresh"]["unclassified_region_count"] == 0
    assert all(
        _sha256(ROOT / relative) == digest
        for relative, digest in evidence["source_sha256"].items()
    )

    phase10b = _json(AUDIT / "phase13-scattering-phase10b-evidence.json")
    phase11a = _json(AUDIT / "phase13-boundary-dedup-phase11a-evidence.json")
    historical = evidence["historical_records"]
    assert historical["phase10b_binding_manifest_sha256"] == phase10b["activation"][
        "binding_manifest_sha256"
    ]
    assert historical["phase11a_binding_manifest_sha256"] == phase11a[
        "binding_manifest"
    ]["current_sha256"]
    assert historical["binding_manifest_changed_in_phase11b"] is False


def test_phase11b_explicit_signatures_and_tu_local_macro_contract_are_preserved() -> None:
    functional_path = (
        ROOT / "src/witwin/channel_native/scattering/kernels/functional_chain.py"
    )
    autograd_path = (
        ROOT / "src/witwin/channel_native/scattering/kernels/autograd_chain.py"
    )
    ensemble = _literal_tuple(functional_path, "_CHAIN_ENSEMBLE_PRIMAL_NAMES")
    realization = _literal_tuple(functional_path, "_CHAIN_REALIZATION_PRIMAL_NAMES")
    functional = _module_functions(functional_path)
    autograd = _module_functions(autograd_path)

    for name in (
        "scattering_chain_ensemble_eval",
        "scattering_chain_ensemble_eval_backward",
        "scattering_chain_ensemble_eval_jvp",
    ):
        function = functional[name]
        assert tuple(arg.arg for arg in function.args.args) == ensemble
        assert function.args.vararg is None
    for name in (
        "scattering_chain_realization_eval",
        "scattering_chain_realization_eval_backward",
        "scattering_chain_realization_eval_jvp",
    ):
        function = functional[name]
        assert tuple(arg.arg for arg in function.args.args) == realization
        assert function.args.vararg is None
    assert tuple(
        arg.arg for arg in autograd["scattering_chain_ensemble_eval_ad"].args.args
    ) == ensemble
    assert tuple(
        arg.arg for arg in autograd["scattering_chain_realization_eval_ad"].args.args
    ) == realization
    assert autograd["scattering_chain_ensemble_eval_ad"].args.vararg is None
    assert autograd["scattering_chain_realization_eval_ad"].args.vararg is None

    macros = {
        "native/channel_native/kernels/bdpt_connect_visibility.cu": (
            "CN_BDPT_CHECK_CONNECTION_SAMPLE_TENSORS",
            "CN_BDPT_CHECK_CONNECTION_SAMPLE_ROWS",
            "CN_BDPT_CONNECTION_OUTPUT_POINTERS",
        ),
        "native/channel_native/kernels/diffraction.cu": (
            "CN_DIFFRACTION_CHECK_STATE_PACK_TENSORS",
            "CN_DIFFRACTION_CHECK_STATE_PACK_POWER",
            "CN_DIFFRACTION_CHECK_STATE_PACK_SHAPES",
            "CN_DIFFRACTION_ALLOCATE_STATE_PACK",
            "CN_DIFFRACTION_STATE_PACK_INPUT_POINTERS",
            "CN_DIFFRACTION_STATE_PACK_OUTPUT_POINTERS",
            "CN_DIFFRACTION_STATE_PACK_RESULTS",
        ),
        "native/channel_native/kernels/los.cu": (
            "CN_LOS_CHECK_VISIBILITY_APPLICATION",
            "CN_LOS_VISIBILITY_LAUNCH_ARGUMENTS",
        ),
        "native/channel_native/kernels/reflection.cu": (
            "CN_REFLECTION_PREPARE_LAUNCH_INPUTS",
            "CN_REFLECTION_LAUNCH_INPUT_PREFIX",
        ),
    }
    for relative, names in macros.items():
        source = (ROOT / relative).read_text(encoding="utf-8-sig")
        for name in names:
            assert source.count(f"#define {name}") == 1
            assert source.count(f"#undef {name}") == 1

    invariants = _json(EVIDENCE_PATH)["invariants"]
    assert all(value is False for value in invariants.values())
