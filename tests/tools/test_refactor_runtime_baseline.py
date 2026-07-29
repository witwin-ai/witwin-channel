# Copyright Xingyu Chen.
# Tests refactor runtime baseline.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tools import refactor_runtime_baseline as runtime


@dataclass(frozen=True)
class _NestedPaths:
    valid: torch.Tensor
    primitive_id: torch.Tensor


@dataclass(frozen=True)
class _Result:
    path_gain: torch.Tensor
    paths: _NestedPaths
    metadata: dict[str, object]
    optional: torch.Tensor | None = None


def _result() -> _Result:
    storage = torch.arange(8, dtype=torch.float32)
    return _Result(
        path_gain=storage[1:5].reshape(2, 2),
        paths=_NestedPaths(
            valid=torch.tensor([True, False]),
            primitive_id=storage[:2].view(torch.int32),
        ),
        metadata={
            "kernel": {
                "primitive": "fixture",
                "launch_count": 3,
                "forward_launch_count": 2,
                "backward_launch_count": 1,
                "jvp_launch_count": 0,
                "tape_bytes": 128,
                "peak_memory_bytes": 256,
            },
            "seed": 7,
        },
    )


def test_result_manifest_hashes_every_tensor_and_records_alias_contract():
    result = _result()

    manifest = runtime.result_manifest(result)

    assert manifest["tensor_count"] == 3
    tensors = {row["path"]: row["tensor"] for row in manifest["tensors"]}
    gain = tensors["$.path_gain"]
    primitive = tensors["$.paths.primitive_id"]
    assert gain["shape"] == [2, 2]
    assert gain["dtype"] == "torch.float32"
    assert gain["device"] == "cpu"
    assert gain["stride"] == [2, 1]
    assert gain["storage_offset"] == 1
    assert gain["contiguous"] is True
    assert gain["requires_grad"] is False
    assert gain["alias_group"] == primitive["alias_group"]
    assert {row["path"] for row in manifest["path_identity"]} == {
        "$.paths.valid",
        "$.paths.primitive_id",
    }
    assert manifest["metadata_fingerprint"]
    assert manifest["result_fingerprint"]


def test_tensor_hash_changes_with_value_but_not_with_alias_pointer():
    first = runtime.value_manifest(torch.tensor([1.0, 2.0]))
    same = runtime.value_manifest(torch.tensor([1.0, 2.0]))
    changed = runtime.value_manifest(torch.tensor([1.0, 3.0]))

    assert first["fingerprint"] == same["fingerprint"]
    assert first["fingerprint"] != changed["fingerprint"]


def test_tensor_hash_handles_contiguous_size_one_dimension_with_nonunit_stride():
    cases = (
        torch.arange(2, dtype=torch.float32).as_strided((2, 1), (1, 2)),
        torch.arange(1, dtype=torch.float32).as_strided((1,), (2,)),
    )
    for unusual in cases:
        assert unusual.is_contiguous()
        unusual_manifest = runtime.value_manifest(unusual)
        canonical = torch.tensor(unusual.tolist(), dtype=unusual.dtype)
        canonical_manifest = runtime.value_manifest(canonical)
        assert (
            unusual_manifest["tensors"][0]["tensor"]["sha256"]
            == canonical_manifest["tensors"][0]["tensor"]["sha256"]
        )
        assert unusual_manifest["tensors"][0]["tensor"]["stride"] == list(
            unusual.stride()
        )


def test_launch_ledger_uses_real_aggregate_and_marks_unavailable_detail():
    ledger = runtime.launch_ledger(_result())

    assert ledger["availability"] == "aggregate-only"
    assert ledger["aggregate"] == {
        "primitive": "fixture",
        "launch_count": 3,
        "forward_launch_count": 2,
        "backward_launch_count": 1,
        "jvp_launch_count": 0,
        "tape_bytes": 128,
        "peak_memory_bytes": 256,
    }
    assert ledger["per_launch"] == []
    assert ledger["unavailable_fields"] == [
        "per_launch_name",
        "grid",
        "block",
        "stream",
        "synchronization_points",
    ]


def test_semantic_fingerprint_excludes_only_allowlisted_runtime_metadata():
    first = _result()
    second = _Result(
        path_gain=first.path_gain,
        paths=first.paths,
        metadata={
            **first.metadata,
            "kernel": {
                **first.metadata["kernel"],
                "forward_time_ms": 9.5,
                "peak_memory_bytes": 999,
            },
        },
    )
    third = _Result(
        path_gain=second.path_gain,
        paths=second.paths,
        metadata={**second.metadata, "seed": 8},
    )

    first_manifest = runtime.result_manifest(first)
    second_manifest = runtime.result_manifest(second)
    third_manifest = runtime.result_manifest(third)

    assert (
        first_manifest["semantic_result_fingerprint"]
        == second_manifest["semantic_result_fingerprint"]
    )
    assert (
        first_manifest["semantic_metadata_fingerprint"]
        == second_manifest["semantic_metadata_fingerprint"]
    )
    assert (
        first_manifest["full_result_fingerprint"]
        != second_manifest["full_result_fingerprint"]
    )
    assert {row["path"] for row in second_manifest["volatile_metadata"]} == {
        "$.metadata['kernel']['forward_time_ms']",
        "$.metadata['kernel']['peak_memory_bytes']",
    }
    assert (
        second_manifest["semantic_result_fingerprint"]
        != third_manifest["semantic_result_fingerprint"]
    )


@pytest.mark.parametrize(
    ("processes", "warmup", "repeats", "message"),
    [
        (1, 1, 7, "independent processes"),
        (2, 0, 7, "warmup"),
        (2, 1, 6, "steady repeats"),
    ],
)
def test_measurement_policy_cannot_lower_plan_minimums(
    processes: int, warmup: int, repeats: int, message: str,
):
    with pytest.raises(runtime.RuntimeBaselineError, match=message):
        runtime.validate_measurement_policy(processes, warmup, repeats)


def _row(process_index: int, fingerprint: str = "exact") -> dict[str, object]:
    start = float(process_index * 10)
    return {
        "solver": "path",
        "scenario": "empty-los",
        "config": {"max_depth": 0},
        "scene_input_fingerprint": "scene",
        "result": {
            "result_fingerprint": fingerprint,
            "metadata_fingerprint": "metadata",
        },
        "performance": {
            "steady": [
                {"wall_ms": start + value, "cuda_event_ms": start + value / 2}
                for value in range(1, 8)
            ],
            "memory": {
                "peak_allocated_bytes": 100 + process_index,
                "peak_reserved_bytes": 200 + process_index,
                "peak_temporary_allocated_bytes": 50 + process_index,
            },
        },
    }


def test_aggregate_case_requires_exact_results_and_reports_median_p95():
    aggregate = runtime.aggregate_case([_row(0), _row(1)])

    distribution = aggregate["performance_distribution"]
    assert aggregate["exact_across_processes"] is True
    assert aggregate["result_fingerprint"] == "exact"
    assert distribution["process_count"] == 2
    assert distribution["steady_sample_count"] == 14
    assert distribution["wall_median_ms"] == 9.0
    assert distribution["wall_p95_ms"] == pytest.approx(16.35)
    assert distribution["peak_allocated_bytes_max"] == 101

    with pytest.raises(runtime.RuntimeBaselineError, match="result hash differs"):
        runtime.aggregate_case([_row(0), _row(1, fingerprint="changed")])


def test_parse_child_stdout_ignores_non_json_diagnostics():
    payload = runtime._parse_child_stdout(
        'native diagnostic\n{"solver":"path","result":{"result_fingerprint":"a"}}\n'
    )

    assert payload["solver"] == "path"


def test_runtime_report_is_bound_to_sha_and_immutable(tmp_path: Path):
    sha = "c" * 40
    report = {
        "schema": {"name": runtime.SCHEMA_NAME, "version": runtime.SCHEMA_VERSION},
        "git_sha": sha,
        "profile": "reduced",
        "measurement_policy": {},
        "coverage": {},
        "cases": [],
    }

    destination = runtime.write_immutable_report(report, tmp_path)

    assert destination == tmp_path / sha / "reduced.json"
    assert sorted(path.name for path in destination.parent.iterdir()) == [
        "launch-ledger.json",
        "performance.json",
        "reduced.json",
        "solver-results.json",
    ]
    with pytest.raises(runtime.RuntimeBaselineError, match="immutable"):
        runtime.write_immutable_report(report, tmp_path)


# ---------------------------------------------------------------------------
# Extended backfill profile
# ---------------------------------------------------------------------------


def test_extended_cells_cover_forward_and_path_deterministic_ad_only():
    cells = runtime.extended_cells()
    forward = [cell for cell in cells if cell[2] == "none"]
    ad = [cell for cell in cells if cell[2] != "none"]

    # Forward cells honor the per-scenario solver support table.
    assert ("thin-wall-transmission", "montecarlo-bdpt", "none") in forward
    assert ("coupled-reflection-diffraction", "path", "none") in forward
    assert ("coupled-reflection-diffraction", "deterministic", "none") not in forward
    assert ("coupled-reflection-diffraction", "montecarlo-basic", "none") not in forward
    assert ("rough-scattering-realization", "deterministic", "none") in forward
    assert ("rough-scattering-realization", "path", "none") not in forward

    # AD is path/deterministic + jvp/vjp only; no BDPT/MC-basic, no scattering.
    assert {cell[1] for cell in ad} == {"path", "deterministic"}
    assert {cell[2] for cell in ad} == {"jvp", "vjp"}
    assert ("single-reflection", "path", "vjp") in ad
    assert ("rough-reflection-cr", "deterministic", "jvp") in ad
    assert all("scattering" not in cell[0] for cell in ad)
    # Every (scenario, mode) AD seed is declared.
    for scenario, _solver, mode in ad:
        assert (scenario, mode) in runtime._EXTENDED_AD_SEEDS


def test_extended_builders_use_core_scene_material_and_endpoint_contracts():
    from witwin.core import (
        PhaseScreen,
        PhysicalMaterial,
        ReceiverGrid,
        Scene,
        SurfaceRoughness,
    )

    thin_scene, _, _ = runtime._extended_scene_and_config(
        "path", "thin-wall-transmission"
    )
    grid_scene, _, _ = runtime._extended_scene_and_config(
        "montecarlo-basic", "single-wedge-diffraction"
    )
    realization_scene, _, _ = runtime._extended_scene_and_config(
        "deterministic", "rough-scattering-realization"
    )

    for scene in (thin_scene, grid_scene, realization_scene):
        assert isinstance(scene, Scene)
        assert not hasattr(scene, "frequency")
        assert {endpoint.role for endpoint in scene.endpoints} == {"tx", "rx"}
    assert isinstance(thin_scene.structures[0].material, PhysicalMaterial)
    assert isinstance(grid_scene.endpoints[-1], ReceiverGrid)
    rough_structure = realization_scene.structures[0]
    assert isinstance(rough_structure.material, PhysicalMaterial)
    assert isinstance(rough_structure.material.roughness_front, SurfaceRoughness)
    assert isinstance(rough_structure.phase_screen, PhaseScreen)


def test_reference_frequency_metadata_preserves_and_forwards_tensor_identity():
    from witwin.core import Scene

    frequency = torch.tensor(3.0e9, requires_grad=True)
    scene = Scene(
        metadata={runtime._REFERENCE_FREQUENCY_METADATA_KEY: frequency}
    )
    observed = {}

    def solve(scene_arg, config_arg, *, reference_frequency_hz):
        observed["scene"] = scene_arg
        observed["config"] = config_arg
        observed["frequency"] = reference_frequency_hz
        return "result"

    assert scene.metadata[runtime._REFERENCE_FREQUENCY_METADATA_KEY] is frequency
    assert runtime._reference_frequency(scene) is frequency
    assert runtime._solve_scene(solve, scene, "config") == "result"
    assert observed == {
        "scene": scene,
        "config": "config",
        "frequency": frequency,
    }


def test_extended_frequency_ad_builder_keeps_leaf_in_scene_metadata():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the extended frequency AD leaf")

    scene, _, _, seed = runtime._extended_ad_scene_config(
        "path", "rough-reflection-cr", "vjp"
    )
    leaf = runtime._reference_frequency(scene)

    assert seed == "frequency"
    assert scene.metadata[runtime._REFERENCE_FREQUENCY_METADATA_KEY] is leaf
    assert isinstance(leaf, torch.Tensor)
    assert leaf.requires_grad


def test_gradient_capture_manifest_hashes_gradient_and_excludes_timing():
    capture = runtime._GradientCapture(
        mode="vjp",
        seed="material_eps_r",
        gradient={"material_eps_r": torch.tensor([1.5, -2.0])},
        gradient_is_nonzero=True,
        metadata={
            "kernel": {
                "ad_status": "vjp",
                "forward_launch_count": 1,
                "backward_launch_count": 2,
                "jvp_launch_count": 0,
                "tape_bytes": 256,
                "backward_time_ms": 4.25,
                "peak_memory_bytes": 512,
            }
        },
    )

    first = runtime.result_manifest(capture)
    # The gradient tensor is hashed with the forward tensor scheme.
    grad_paths = {row["path"] for row in first["tensors"]}
    assert "$.gradient['material_eps_r']" in grad_paths
    # Only the allowlisted timing/high-water fields are excluded from the exact
    # fingerprint; launch/tape counters stay exact.
    assert {row["path"] for row in first["volatile_metadata"]} == {
        "$.metadata['kernel']['backward_time_ms']",
        "$.metadata['kernel']['peak_memory_bytes']",
    }
    changed_timing = runtime._GradientCapture(
        mode="vjp",
        seed="material_eps_r",
        gradient={"material_eps_r": torch.tensor([1.5, -2.0])},
        gradient_is_nonzero=True,
        metadata={
            "kernel": {**capture.metadata["kernel"], "backward_time_ms": 9.9}
        },
    )
    second = runtime.result_manifest(changed_timing)
    assert (
        first["semantic_result_fingerprint"]
        == second["semantic_result_fingerprint"]
    )
    changed_grad = runtime._GradientCapture(
        mode="vjp",
        seed="material_eps_r",
        gradient={"material_eps_r": torch.tensor([1.5, -3.0])},
        gradient_is_nonzero=True,
        metadata=capture.metadata,
    )
    assert (
        runtime.result_manifest(changed_grad)["semantic_result_fingerprint"]
        != first["semantic_result_fingerprint"]
    )
    ledger = runtime.launch_ledger(capture)
    assert ledger["aggregate"]["ad_status"] == "vjp"
    assert ledger["aggregate"]["backward_launch_count"] == 2
    assert ledger["aggregate"]["tape_bytes"] == 256


def _extended_case(*, solver="path", scenario="rough-reflection-cr", ad_mode="none", seed=None):
    row = {
        "process_index": 0,
        "scene_input": {"kind": "scene"},
        "result": {"result_fingerprint": "rfp", "metadata_fingerprint": "mfp"},
        "launch_ledger": {"aggregate": {"launch_count": 1}},
        "environment": {"torch": "2.10.0"},
        "performance": {"steady": []},
    }
    return {
        "solver": solver,
        "scenario": scenario,
        "ad_mode": ad_mode,
        "seed": seed,
        "config": {"max_depth": 1},
        "scene_input_fingerprint": "sfp",
        "result_fingerprint": "rfp",
        "metadata_fingerprint": "mfp",
        "exact_across_processes": True,
        "performance_distribution": {"wall_median_ms": 1.0},
        "processes": [row, {**row, "process_index": 1}],
    }


def test_project_extended_report_carries_ad_fields_and_excluded():
    report = {
        "schema": {"name": runtime.SCHEMA_NAME, "version": runtime.SCHEMA_VERSION},
        "git_sha": "a" * 40,
        "profile": runtime.EXTENDED_PROFILE_NAME,
        "measurement_policy": {},
        "coverage": {"status": "extended-g0-backfill"},
        "cases": [_extended_case(ad_mode="vjp", seed="frequency")],
        "excluded": [
            {
                "scenario": "single-wedge-diffraction",
                "solver": "montecarlo-basic",
                "ad_mode": "none",
                "reason": "child_rejected",
                "detail": "boom",
            }
        ],
    }

    projections = runtime._project_extended_report(report)

    assert set(projections) == {
        "extended-solver-results.json",
        "extended-launch-ledger.json",
        "extended-performance.json",
    }
    solver_proj = projections["extended-solver-results.json"]
    assert solver_proj["cases"][0]["ad_mode"] == "vjp"
    assert solver_proj["cases"][0]["seed"] == "frequency"
    assert solver_proj["excluded"][0]["reason"] == "child_rejected"
    perf_proj = projections["extended-performance.json"]
    assert perf_proj["cases"][0]["ad_mode"] == "vjp"


def test_write_extended_report_lands_in_runtime_subdir_and_is_immutable(tmp_path: Path):
    sha = "d" * 40
    report = {
        "schema": {"name": runtime.SCHEMA_NAME, "version": runtime.SCHEMA_VERSION},
        "git_sha": sha,
        "profile": runtime.EXTENDED_PROFILE_NAME,
        "measurement_policy": {},
        "coverage": {},
        "cases": [_extended_case()],
        "excluded": [],
    }

    destination = runtime.write_extended_report(report, tmp_path)

    assert destination == tmp_path / sha / "runtime" / "extended.json"
    assert sorted(path.name for path in destination.parent.iterdir()) == [
        "extended-launch-ledger.json",
        "extended-performance.json",
        "extended-solver-results.json",
        "extended.json",
    ]
    with pytest.raises(runtime.RuntimeBaselineError, match="immutable"):
        runtime.write_extended_report(report, tmp_path)


def test_extended_child_command_selects_profile_and_ad_mode():
    command = runtime._extended_child_command(
        Path("tool.py"),
        solver="path",
        scenario="single-reflection",
        ad_mode="vjp",
        process_index=1,
        warmup=1,
        repeats=7,
    )

    assert "--profile" in command
    assert runtime.EXTENDED_PROFILE_NAME in command
    assert "--ad-mode" in command and "vjp" in command
    assert "--scenario" in command and "single-reflection" in command


def test_extended_cheapest_cell_is_deterministic_across_two_runs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the extended runtime smoke")
    from tests.support.native_ext import inject_native_paths

    if not inject_native_paths():
        pytest.skip("compiled _channel extension was not found")

    scene, config, operation, seed = runtime._load_extended_case(
        "path", "rough-reflection-cr", "none"
    )
    first = runtime.result_manifest(operation())
    second = runtime.result_manifest(operation())
    assert (
        first["semantic_result_fingerprint"]
        == second["semantic_result_fingerprint"]
    )
    assert (
        first["semantic_metadata_fingerprint"]
        == second["semantic_metadata_fingerprint"]
    )