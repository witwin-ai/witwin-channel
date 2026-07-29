# Copyright Xingyu Chen.
# Tests stable recovery contract.

from __future__ import annotations

import ast
import json
from pathlib import Path
import re

from benchmarks.phase13_phase12.release import _naming_audit
from ci import check_contract_coverage as coverage


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "benchmarks/gates/stable_recovery_munich.sm120.v1.json"

def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8-sig")


def _called_attributes(relative: str) -> set[str]:
    tree = ast.parse(_source(relative), filename=relative)
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def _function_calls(relative: str, function_name: str) -> dict[str, list[int]]:
    tree = ast.parse(_source(relative), filename=relative)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    calls: dict[str, list[int]] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name is not None:
            calls.setdefault(name, []).append(node.lineno)
    return calls


def _evidence() -> dict[str, object]:
    value = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_live_reflection_and_diffraction_producers_remain_compact() -> None:
    reflection_calls = _called_attributes(
        "witwin/channel/interactions/reflection.py"
    )
    diffraction_calls = _called_attributes(
        "witwin/channel/interactions/diffraction.py"
    )

    assert {
        "deterministic_reflection_order1_compact",
        "deterministic_reflection_sequence_compact",
    } <= reflection_calls
    assert "deterministic_diffraction_order1_compact" in diffraction_calls


def test_dormant_adr029_and_adr030_producers_have_no_production_caller() -> None:
    call_sites = coverage._python_call_sites(
        ROOT, frozenset(coverage.DORMANT_FACADE_OWNERS)
    )
    offenders = []
    for facade, rows in sorted(call_sites.items()):
        allowed = coverage.DORMANT_ALLOWED_FACADE_CALLERS.get(facade, frozenset())
        offenders.extend(
            f"{path}:{line}:{facade}:{caller}"
            for caller, path, line in rows
            if caller not in allowed
        )
    assert offenders == []


def test_public_path_exports_retain_actual_row_compaction() -> None:
    path_relative = "witwin/channel/path.py"
    deterministic_pipeline = _source("witwin/channel/deterministic.py")
    path_result = _source("witwin/channel/path.py")
    path_table = ast.parse(_source("witwin/channel/deterministic.py"))

    solve_calls = _function_calls(path_relative, "_pipeline_solve_base")
    assert max(solve_calls["sanitize_enumerated_capacity_transaction"]) < min(
        solve_calls["compact_evaluated_paths"]
    )
    assert max(solve_calls["compact_evaluated_paths"]) < min(
        solve_calls["pack_evaluated_paths"]
    )
    assert "max_paths_per_pair = int(counts.max().item())" in path_result
    assert "num_paths=counts.to(dtype=torch.int32)" in path_result
    assert "build_path_table(\n                evaluated," in deterministic_pipeline

    path_table_class = next(
        node
        for node in path_table.body
        if isinstance(node, ast.ClassDef) and node.name == "PathTable"
    )
    field_names = {
        node.target.id
        for node in path_table_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "valid" in field_names
    assert "num_paths" not in field_names
    assert "path_capacity_per_pair" not in field_names


def test_public_configs_do_not_expose_retired_capacity_controls() -> None:
    retired = {
        "path_capacity_per_pair",
        "diffraction_state_capacity",
        "reflection_candidate_capacity_per_pair",
    }
    for relative in (
        "witwin/channel/path.py",
        "witwin/channel/deterministic.py",
        "witwin/channel/montecarlo/basic.py",
        "witwin/channel/montecarlo/bdpt.py",
    ):
        tree = ast.parse(_source(relative), filename=relative)
        config = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Config"
        )
        fields = {
            node.target.id
            for node in config.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        assert fields.isdisjoint(retired), relative
        assert "_validate_capacity_config" not in _source(relative), relative

    worker = _source("benchmarks/phase13_phase12_worker.py")
    for field in retired:
        assert re.search(rf"\b{field}\s*=", worker) is None


def test_production_naming_has_no_wip_or_version_generation_suffix() -> None:
    assert _naming_audit(ROOT)["passed"] is True


def test_munich_recovery_evidence_has_strict_top_level_schema() -> None:
    evidence = _evidence()
    assert set(evidence) == {
        "schema",
        "decision",
        "environment",
        "scenario",
        "release_thresholds",
        "variants",
        "exactness",
        "raw_artifacts",
    }
    assert evidence["schema"] == {
        "name": "witwin.channel.stable-recovery-munich",
        "version": 1,
    }
    assert evidence["decision"] == {
        "selected_route": "compact_o_k",
        "stable_production_commit": "e7d82d2d1d290bbc106ef68410ebf88aeb1c99e9",
        "capacity_routes": "dormant",
        "retired_public_noop_fields": [
            "path_capacity_per_pair",
            "diffraction_state_capacity",
            "reflection_candidate_capacity_per_pair",
        ],
    }
    assert set(evidence["variants"]) == {
        "compact_o_k",
        "theoretical_capacity_o_n",
        "per_pair_qr20",
        "per_pair_qr128",
    }
    artifacts = evidence["raw_artifacts"]
    assert isinstance(artifacts, list) and len(artifacts) == 12
    assert len({row["name"] for row in artifacts}) == len(artifacts)
    for row in artifacts:
        assert set(row) == {"name", "sha256", "bytes"}
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        assert row["bytes"] > 0


def test_compact_route_is_the_only_variant_meeting_release_thresholds() -> None:
    evidence = _evidence()
    thresholds = evidence["release_thresholds"]
    variants = evidence["variants"]
    compact = variants["compact_o_k"]

    assert compact["wall_median_ms"] <= thresholds["wall_median_ms_max"]
    assert compact["peak_allocated_bytes"] <= thresholds["peak_allocated_bytes_max"]
    assert (
        compact["steady_throughput_solve_per_s"]
        >= thresholds["steady_throughput_solve_per_s_min"]
    )
    assert (
        compact["capacity_active_ratio"]
        <= thresholds["capacity_active_ratio_max"]
    )
    assert compact["d2h_count"] <= thresholds["d2h_count_max"]
    assert compact["d2h_bytes"] <= thresholds["d2h_bytes_max"]
    assert (
        evidence["environment"]["device_total_bytes"]
        - compact["peak_reserved_bytes"]
        >= thresholds["device_headroom_bytes_min"]
    )

    for name in ("theoretical_capacity_o_n", "per_pair_qr20", "per_pair_qr128"):
        candidate = variants[name]
        assert (
            candidate["wall_median_ms"] > thresholds["wall_median_ms_max"]
            or candidate["peak_allocated_bytes"]
            > thresholds["peak_allocated_bytes_max"]
            or candidate["steady_throughput_solve_per_s"]
            < thresholds["steady_throughput_solve_per_s_min"]
            or candidate["capacity_active_ratio"]
            > thresholds["capacity_active_ratio_max"]
        )


def test_recovery_evidence_preserves_result_exactness_claim_boundary() -> None:
    evidence = _evidence()
    exactness = evidence["exactness"]
    thresholds = evidence["release_thresholds"]

    assert exactness["logical_path_fields_compared"] == 24
    assert exactness["logical_path_fields_bitwise_equal"] is True
    assert thresholds["logical_path_fields_bitwise_equal_required"] is True
    assert exactness["whole_map_bitwise_claimed"] is False