# Copyright Xingyu Chen.
# Tests evidence performance evidence.

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.phase13_phase12.builds import (
    _RAYD_COMMIT,
    _parse_msvc_environment_stdout,
    parse_cuobjdump_resource_usage,
)
from benchmarks.phase13_phase12.contracts import (
    ADDENDUM_ACCEPTED,
    ADR030_PRODUCTION_CANDIDATE_SUPERSEDED,
    COMPARISON_GROUPS,
    DEFAULT_GATE,
    EvidenceError,
    load_config,
    load_gate,
    require_measured_policy_ready,
)
from benchmarks.phase13_phase12.profilers import load_profile_contract
from benchmarks.phase13_phase12.release import _naming_audit


ROOT = Path(__file__).resolve().parents[2]


def test_formal_builds_pin_the_complete_accepted_rayd_commit() -> None:
    assert _RAYD_COMMIT == "474c122aa3cd6b6d098675e076a73e6f485bd6be"


def test_infrastructure_gate_is_explicitly_non_claiming_until_freeze() -> None:
    gate = load_gate(DEFAULT_GATE)

    assert ADDENDUM_ACCEPTED is False
    assert ADR030_PRODUCTION_CANDIDATE_SUPERSEDED is True
    assert gate["history_policy"]["accepted"] is False
    assert gate["history_policy"]["diffraction_role"] == (
        "adr030_dormant_source_lane_experiment"
    )
    with pytest.raises(EvidenceError, match="superseded by ADR-032"):
        require_measured_policy_ready(gate)
    assert gate["frozen_inputs"]["profile_contract_sha256"] is None
    assert all(
        gate["comparison_groups"][group]["correctness"] is None
        and gate["comparison_groups"][group]["resource_budgets"] is None
        for group in COMPARISON_GROUPS
    )


def test_profile_contract_is_the_only_stable_range_and_scenario_manifest() -> None:
    contract = load_profile_contract()
    groups = contract["groups"]

    assert set(groups) == set(COMPARISON_GROUPS)
    all_names: list[str] = []
    for group in COMPARISON_GROUPS:
        row = groups[group]
        assert isinstance(row, dict)
        all_names.append(str(row["target_timing_range"]))
        for variant in ("baseline", "candidate"):
            variant_row = row["variants"][variant]
            all_names.extend(variant_row["required_ranges"])
            all_names.extend(variant_row["forbidden_ranges"])
            all_names.extend(variant_row["required_markers"])
    assert all(name.startswith("witwin.channel:") for name in all_names)


def test_diffraction_known_live_multiplicity_is_not_historical_compaction() -> None:
    diffraction = load_profile_contract()["groups"]["diffraction"]
    baseline = diffraction["variants"]["baseline"]["known_range_multiplicity_per_solve"]
    candidate = diffraction["variants"]["candidate"]["known_range_multiplicity_per_solve"]

    assert baseline["witwin.channel:diffraction_exporter"] == 13
    assert baseline["witwin.channel:diffraction_topology_packing"] == 13
    assert baseline["witwin.channel:diffraction_total_stage"] == 1
    assert candidate["witwin.channel:diffraction_exporter"] == 2
    assert candidate["witwin.channel:diffraction_total_stage"] == 1


def test_runner_config_rejects_external_prebuilt_extension_field(tmp_path: Path) -> None:
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"not executed")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    row = {
        "checkout": str(checkout.resolve()),
        "python_executable": str(executable.resolve()),
        "external_native_artifact": str((tmp_path / "untrusted.pyd").resolve()),
    }
    config = {
        "schema": {
            "name": "witwin.channel.phase13-phase12-runner-config",
            "version": 3,
        },
        "comparisons": {
            group: {"baseline": copy.deepcopy(row), "candidate": copy.deepcopy(row)}
            for group in COMPARISON_GROUPS
        },
        "rayd_checkout": str((tmp_path / "rayd").resolve()),
        "raw_artifact_parent": str((tmp_path / "raw").resolve()),
        "build_parent": str((tmp_path / "builds").resolve()),
        "output": str((tmp_path / "raw" / "report.json").resolve()),
        "tools": {},
        "datasets": {},
        "runtime_search_paths": [str(tmp_path.resolve())],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(EvidenceError, match="keys differ"):
        load_config(path)


def test_gate_rejects_partial_nested_freeze() -> None:
    gate = copy.deepcopy(load_gate(DEFAULT_GATE))
    gate["comparison_groups"]["diffraction"]["correctness"] = {
        "oracle": {"max_absolute_difference": None}
    }

    with pytest.raises(EvidenceError, match="max_absolute_difference"):
        require_measured_policy_ready(gate)


def test_cuobjdump_resource_parser_accepts_both_function_header_forms() -> None:
    payload = b"""
Function : _Z27segment_penetration_kernelv
 REG: 48 STACK: 16 SHARED: 128 LOCAL: 0
Function deterministic_diffraction_pair_reduce_kernel:
 REG: 64 STACK: 0 SHARED: 256 LOCAL: 8
"""
    assert parse_cuobjdump_resource_usage(payload) == [
        {
            "function": "_Z27segment_penetration_kernelv",
            "registers_per_thread": 48,
            "stack_bytes": 16,
            "shared_bytes": 128,
            "local_bytes": 0,
        },
        {
            "function": "deterministic_diffraction_pair_reduce_kernel",
            "registers_per_thread": 64,
            "stack_bytes": 0,
            "shared_bytes": 256,
            "local_bytes": 8,
        },
    ]


def test_msvc_environment_parser_accepts_only_the_ordered_allowlist() -> None:
    keys = (
        "INCLUDE", "LIB", "LIBPATH", "PATH", "VCToolsInstallDir", "VCINSTALLDIR",
        "WindowsSdkDir", "WindowsSDKVersion", "UCRTVersion", "UniversalCRTSdkDir",
    )
    payload = "\n".join(f"{name}=value-{index}" for index, name in enumerate(keys)).encode()
    assert list(_parse_msvc_environment_stdout(payload)) == list(keys)

    with pytest.raises(EvidenceError, match="missing/duplicate/unknown"):
        _parse_msvc_environment_stdout(payload + b"\nINCLUDE=duplicate")
    with pytest.raises(EvidenceError, match="NUL"):
        _parse_msvc_environment_stdout(payload + b"\x00")


def test_naming_audit_allows_stable_schema_versions_but_rejects_wip_boundary_names(
    tmp_path: Path,
) -> None:
    ci = tmp_path / "ci"
    ci.mkdir()
    (ci / "report.py").write_text(
        'OUTPUT = "artifacts/report.v1.json"\n', encoding="utf-8"
    )
    assert _naming_audit(tmp_path)["passed"] is True
    (ci / "bridge.py").write_text(
        'HEADER = "rayd/torch/integration_v2.h"\n', encoding="utf-8"
    )
    with pytest.raises(EvidenceError, match="provisional generation"):
        _naming_audit(tmp_path)