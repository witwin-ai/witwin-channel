from __future__ import annotations

import pytest

from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from tests.support.bin._benchmark_runtime import (
    benchmark_environment_report,
    extract_monitor_performance_timing,
    extract_monitor_runtime_backends,
)
from tests.support.bin.benchmark_multipath_ad import run_single_pass
from witwin.channel import DEFAULT_VARIANT, FieldMonitor, Tracer
import witwin as wt

pytestmark = pytest.mark.gpu


def _build_runtime_trace_case():
    cube = box_drjit_geometry(center=(-2.0, 0.0, 1.5), size=2.0, rotation=None).to_mesh()
    scene = build_test_scene(cube)
    monitor = FieldMonitor(
        "runtime_plane",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-4.0, 4.0)),
        grid_size=24,
    )
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        config={
            "trace": {
                "diffraction_execution": {
                    "suffix_dda": "symbolic",
                }
            }
        },
        reflection_n_rays=64,
        reflection_max_bounces=1,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
    )
    return tracer, monitor


def test_benchmark_environment_report_has_runtime_fields():
    report = benchmark_environment_report()
    assert report["channel_module_file"].endswith("__init__.py")
    assert "native_extension_available" in report
    assert "backend_variant" in report


def test_field_trace_exposes_runtime_backend_and_timing_metadata():
    tracer, monitor = _build_runtime_trace_case()
    payload = tracer.trace((0.0, -5.0, 1.5), monitor=monitor, verbose=False, return_timing=True)

    runtime_backends = extract_monitor_runtime_backends(payload)
    performance_timing = extract_monitor_performance_timing(payload)

    assert set(runtime_backends) == {"reflection", "diffraction", "suffix"}
    assert "implementation" in runtime_backends["reflection"]
    assert "implementation" in runtime_backends["diffraction"]
    assert "implementation" in runtime_backends["suffix"]

    assert "reflection_total_seconds" in performance_timing
    assert "diffraction_total_seconds" in performance_timing
    assert "diffraction_state_preparation_seconds" in performance_timing
    assert "diffraction_utd_accumulation_seconds" in performance_timing
    assert "diffraction_suffix_seconds" in performance_timing
    assert "diffraction_state_preparation_breakdown" in performance_timing
    assert "diffraction_state_preparation_order_reports" in performance_timing
    assert isinstance(performance_timing["diffraction_state_preparation_order_reports"], list)
    assert performance_timing["diffraction_higher_order_candidate_backend"] in {"rayd_edge_bvh", "not_used"}
    for item in performance_timing["diffraction_state_preparation_order_reports"]:
        assert "order" in item
        if item["order"] == 1:
            assert "reflection_prefix_builder" in item
        if item["order"] >= 2:
            assert "higher_order_builder" in item
            assert "inserted_reflection_builder" in item


@pytest.mark.parametrize(
    ("mode", "result_key"),
    (
        ("vjp", "parameter_grad"),
        ("jvp", "loss_jvp"),
    ),
)
def test_ad_benchmark_scalar_loss_workload_matches_full_field(mode, result_key):
    full = run_single_pass(
        parameter="tx_x",
        mode=mode,
        grid_size=24,
        n_rays=64,
        workload="full_field",
        verbose_trace=False,
    )
    scalar = run_single_pass(
        parameter="tx_x",
        mode=mode,
        grid_size=24,
        n_rays=64,
        workload="scalar_loss",
        verbose_trace=False,
    )

    assert full["workload"] == "full_field"
    assert scalar["workload"] == "scalar_loss"
    assert scalar["runtime_backends"] == full["runtime_backends"]

    full_loss = float(full["loss"])
    scalar_loss = float(scalar["loss"])
    loss_tol = max(1e-4, abs(full_loss) * 1e-6)
    assert abs(full_loss - scalar_loss) <= loss_tol

    full_value = float(full["result"][result_key])
    scalar_value = float(scalar["result"][result_key])
    value_tol = max(1e-4, abs(full_value) * 1e-5)
    assert abs(full_value - scalar_value) <= value_tol
