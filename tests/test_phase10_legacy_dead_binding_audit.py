# Copyright Xingyu Chen.
# Tests legacy dead binding audit.

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

from tools.refactor_baseline import binding_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT / "docs" / "dev" / "audit" / "phase10-legacy-dead-binding.json"
)
OPS_MANIFEST_PATH = (
    REPOSITORY_ROOT / "docs" / "dev" / "audit" / "phase12-ops-migration-ledger.json"
)
PYTHON_ROOT = REPOSITORY_ROOT / "witwin" / "channel"
PHASE12_RETIRED_EVIDENCE_TESTS = frozenset(
    {"tests/propagation/geometry/test_reevaluate_compat.py"}
)
# Phase-11 cutover: the zero-importer BDPT MIS facade and its only test were
# deleted with the ADR-029/030/031 dormant sweep.
PHASE11_RETIRED_EVIDENCE_TESTS = frozenset(
    {"tests/montecarlo/bdpt/test_mis_weights.py"}
)
RETIRED_EVIDENCE_TESTS = (
    PHASE12_RETIRED_EVIDENCE_TESTS | PHASE11_RETIRED_EVIDENCE_TESTS
)


def _audit() -> dict[str, object]:
    value = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_phase10_zero_reference_inventory_is_complete_and_classified() -> None:
    candidates = _audit()["zero_reference_candidates"]
    assert isinstance(candidates, list)
    names = [candidate["name"] for candidate in candidates]

    assert len(names) == len(set(names)) == 33
    assert Counter(candidate["classification"] for candidate in candidates) == {
        "public": 26,
        "intentional": 5,
        "dead": 2,
    }
    for candidate in candidates:
        required_fields = {
            "name",
            "definition",
            "classification",
            "decision",
            "evidence",
            "tests",
            "compatibility_cycle",
        }
        assert required_fields <= set(candidate)
        assert set(candidate) <= required_fields | {"status"}
        assert candidate["evidence"]
        assert candidate["tests"]
        assert candidate["compatibility_cycle"]
        for test_path in candidate["tests"]:
            if test_path in RETIRED_EVIDENCE_TESTS:
                assert not (REPOSITORY_ROOT / test_path).exists(), test_path
            else:
                assert (REPOSITORY_ROOT / test_path).exists(), test_path


def test_phase10_binding_inventory_remains_a_self_consistent_archive() -> None:
    audit = _audit()
    ownership = audit["binding_ownership"]
    assert isinstance(ownership, dict)
    audited_names = ownership["symbols"]

    # + 2 ADR-013 coupled double-diffraction forward symbols + the two ADR-013
    # AD companions (field_coupled_dd_backward/_jvp) + 2 ADR-017 ISB-taper LoS
    # symbols (los_silhouette_clearance, los_taper_apply) = 185.
    # + 4 ADR-014 scattering JVP/VJP companions
    # + 4 ADR-015 scattering table-eval / table-build JVP/VJP companions = 193.
    # + 6 ADR-021 multi-bounce chain scattering symbols (Op A/Op B forwards plus
    # their _backward/_jvp companions) = 199.
    # + 12 ADR-022 BDPT fixed-topology AD companions (subpath reflect/transmit
    # state, endpoint connection, accumulate, finalize point/maps, each
    # _backward + _jvp) = 211.
    assert ownership["expected_count"] == 211
    assert len(audited_names) == len(set(audited_names)) == 211
    canonical = json.dumps(
        ownership, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == (
        "a8a6244473e2f29f574899a18fbd1a90f4b48b184e8dff7ea3b653e2de1b4618"
    )


def test_phase10_dummy_projection_is_exact_and_removed() -> None:
    audit = _audit()
    projection = audit["dummy_parameter_projection"]
    assert isinstance(projection, dict)
    parameter_name = projection["parameter_name"]
    current = binding_manifest(REPOSITORY_ROOT)
    current_bindings = {
        symbol["name"]
        for symbol in current["symbols"]
        if any(
            parameter["name"] == parameter_name for parameter in symbol["parameters"]
        )
    }

    assert projection["status"] == "removed"
    assert projection["python_symbol_status"] == "retired"
    assert projection["selector_parameter_status"] == "removed"
    assert projection["allowed_transition"] == ("remove_from_all_listed_bindings_only")
    assert len(projection["bindings"]) == len(set(projection["bindings"])) == 22
    assert current_bindings == set()


def test_phase10_python_call_projection_matches_the_ops_ledger() -> None:
    audit = _audit()
    projection = audit["python_call_projection"]
    assert isinstance(projection, dict)
    manifest = json.loads(OPS_MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = projection["entries"]
    ids = [entry["id"] for entry in entries]

    assert projection["status"] == "removed"
    assert isinstance(projection["parameter_expression"], str)
    assert len(ids) == len(set(ids)) == 25
    assert entries == manifest["approved_body_projections"]
    assert all(
        manifest["canonical_owners"][entry["id"]] == entry["owner"] for entry in entries
    )


def test_phase10_module_handle_contract_is_retired_after_definition_removal() -> None:
    manifest = json.loads(OPS_MANIFEST_PATH.read_text(encoding="utf-8"))
    audit = _audit()

    assert len(manifest["retired_ops"]) == 1
    retired_name = manifest["retired_ops"][0]
    assert retired_name not in manifest["canonical_owners"]
    assert not any(
        retired_name in path.read_text(encoding="utf-8-sig")
        for path in PYTHON_ROOT.rglob("*.py")
    )
    assert audit["frozen_ops_contract_digest"] == (
        "ff9c4cd45b2f1091c9ba05e1a311e6e569945e18badc7b7a67a3f8f56ccda3a9"
    )


def test_phase10_rayd_backend_shim_is_removed_without_a_production_replacement() -> (
    None
):
    audit = _audit()
    decision = next(
        value
        for value in audit["legacy_decisions"].values()
        if "backend" in value["definition"]
    )
    shim_path = REPOSITORY_ROOT / decision["definition"]
    production_references = [
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in PYTHON_ROOT.rglob("*.py")
        if "rayd_backend" in path.read_text(encoding="utf-8-sig")
    ]

    assert decision["classification"] == "dead"
    assert decision["decision"] == decision["status"] == "removed"
    assert decision["replacement"] == (
        "witwin.channel_native.runtime.symbols.native_extension"
    )
    assert not shim_path.exists()
    assert production_references == []


def test_phase10_fresnel_scalar_dead_candidate_is_removed_exactly() -> None:
    audit = _audit()
    candidate = next(
        item
        for item in audit["zero_reference_candidates"]
        if item["name"] == "_fresnel_scalar_coefficient"
    )
    field_path = REPOSITORY_ROOT / "witwin/channel/deterministic/field.py"
    production_references = [
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in PYTHON_ROOT.rglob("*.py")
        if "_fresnel_scalar_coefficient"
        in path.read_text(encoding="utf-8-sig")
    ]

    assert candidate["classification"] == "dead"
    assert candidate["decision"] == candidate["status"] == "removed"
    # The deterministic field wrapper module that once held the dead stub is
    # itself removed, so the symbol cannot reappear at top level.
    assert not field_path.exists()
    assert production_references == []


def test_phase10_bdpt_native_diffraction_component_maps_is_removed_exactly() -> None:
    audit = _audit()
    candidate = next(
        item
        for item in audit["zero_reference_candidates"]
        if item["name"] == "_native_diffraction_component_maps"
    )
    pipeline_path = REPOSITORY_ROOT / "witwin/channel/montecarlo/bdpt.py"
    tree = ast.parse(pipeline_path.read_text(encoding="utf-8-sig"))
    top_level_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    production_references = [
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in PYTHON_ROOT.rglob("*.py")
        if "_native_diffraction_component_maps"
        in path.read_text(encoding="utf-8-sig")
    ]

    assert candidate["classification"] == "dead"
    assert candidate["decision"] == candidate["status"] == "removed"
    assert "_native_diffraction_component_maps" not in top_level_names
    assert production_references == []