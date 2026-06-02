"""Regression tests for solver modes and mixed-path performance guardrails."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import witwin as wt

from witwin.channel import FieldMonitor, Tracer
from witwin.channel.validation import build_double_wedge_case


def test_fast_approximate_mode_caps_mixed_depth_and_budgets():
    case = build_double_wedge_case()
    tracer = Tracer(
        frequency=1e9,
        scene=case.scene,
        reflection_n_rays=4096,
        reflection_max_bounces=2,
        enable_rd_diffraction=True,
        max_diffractions=4,
        max_inserted_reflections_per_path=3,
        solver_mode="fast_approximate",
    )
    monitor = FieldMonitor(
        "validation_plane",
        axis="z",
        position=case.calculation_height,
        bounds=(case.range_x, case.range_y),
        grid_size=20,
    )

    result = tracer.trace(
        wt.Point3f(*case.tx_pos),
        monitor=monitor,
        verbose=False,
        return_diffraction_audit=True,
    )

    metadata = result.primary.metadata
    mode = metadata["solver_mode"]
    effective = mode["effective"]

    assert mode["selected"] == "fast_approximate"
    assert metadata["execution_intent"]["kind"] == "field"
    assert not metadata["execution_intent"]["path_export_enabled"]
    assert effective["reflection_n_rays"] == 1024
    assert effective["reflection_max_bounces"] == 1
    assert effective["max_diffractions"] == 3
    assert effective["max_inserted_reflections_per_path"] == 1
    assert effective["diffraction_state_budget"] == 768
    assert effective["inserted_reflection_state_budget"] == 256

    guardrails = metadata["performance_guardrails"]
    profiling = guardrails["profiling"]
    assert guardrails["applied"]
    assert len(guardrails["changes"]) >= 4
    assert profiling["packed_state_stride_floats"] > 0
    assert profiling["packed_core_floats"] == 72
    assert profiling["packed_stride_bytes"] == 288
    assert profiling["pre_expansion_policy"]["enabled"]
    assert profiling["pre_expansion_policy"]["policy"] == "fast_approximate_topk_power"
    assert profiling["pre_expansion_policy"]["higher_order_source_budget"] == 192
    assert profiling["pre_expansion_policy"]["inserted_source_budget"] == 192
    assert (
        profiling["peak_higher_order_source_states_after_pre_prune"]
        <= profiling["peak_higher_order_source_states_before_pre_prune"]
    )
    assert (
        profiling["peak_inserted_source_states_after_pre_prune"]
        <= profiling["peak_inserted_source_states_before_pre_prune"]
    )
    assert metadata["path_families"]["Arbitrary alternating mixed chains"]["status"] == "absent"
    assert "S -> D -> R -> D -> R -> D" not in set(
        result.primary.diffraction_detail["state_audit"]["path_sequence"]
    )


def test_accuracy_mode_keeps_requested_alternating_depth_and_profiles_states():
    case = build_double_wedge_case()
    tracer = Tracer(
        frequency=1e9,
        scene=case.scene,
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        enable_rd_diffraction=True,
        max_diffractions=3,
        max_inserted_reflections_per_path=2,
        solver_mode="accuracy",
    )
    monitor = FieldMonitor(
        "validation_plane",
        axis="z",
        position=case.calculation_height,
        bounds=(case.range_x, case.range_y),
        grid_size=20,
    )

    result = tracer.trace(
        wt.Point3f(*case.tx_pos),
        monitor=monitor,
        verbose=False,
        return_diffraction_audit=True,
    )

    metadata = result.primary.metadata
    mode = metadata["solver_mode"]
    guardrails = metadata["performance_guardrails"]

    assert mode["selected"] == "accuracy"
    assert mode["changes"] == []
    assert not guardrails["applied"]
    assert metadata["path_families"]["Arbitrary alternating mixed chains"]["status"] == "approximate"

    sequences = set(result.primary.diffraction_detail["state_audit"]["path_sequence"])
    assert "S -> D -> R -> D -> R -> D" in sequences

    profiling = guardrails["profiling"]
    assert profiling["history_size"] == 3
    assert profiling["packed_core_floats"] == 72
    assert profiling["packed_stride_bytes"] == 288
    assert profiling["bytes_per_state"] > 0
    assert not profiling["pre_expansion_policy"]["enabled"]
    assert profiling["peak_total_states_before_prune"] >= profiling["peak_total_states_after_prune"]
    assert profiling["final_total_states"] == result.primary.diffraction_detail["n_edge_states"]
    assert profiling["risk_level"] in {"low", "medium", "high"}
    assert profiling["max_cartesian_pairs_per_chunk"] >= 0
    for order_report in profiling["per_order"]:
        assert (
            order_report["higher_order_source_states_before_pre_prune"]
            == order_report["higher_order_source_states_after_pre_prune"]
        )
        assert (
            order_report["inserted_source_states_before_pre_prune"]
            == order_report["inserted_source_states_after_pre_prune"]
        )


def test_memory_safe_profile_caps_diffraction_growth_without_fast_mode():
    case = build_double_wedge_case()
    tracer = Tracer(
        frequency=1e9,
        scene=case.scene,
        reflection_n_rays=4096,
        reflection_max_bounces=2,
        enable_rd_diffraction=True,
        max_diffractions=4,
        max_inserted_reflections_per_path=3,
        diffraction_state_budget=8192,
        inserted_reflection_state_budget=2048,
        solver_mode="accuracy",
        memory_profile="memory_safe",
    )
    result = tracer.trace(
        wt.Point3f(*case.tx_pos),
        monitor=FieldMonitor(
            "validation_plane",
            axis="z",
            position=case.calculation_height,
            bounds=(case.range_x, case.range_y),
            grid_size=12,
        ),
        verbose=False,
        return_diffraction_audit=False,
    ).primary
    controls = result.metadata["solver_mode"]
    guardrails = result.metadata["performance_guardrails"]
    profiling = guardrails["profiling"]

    assert controls["selected"] == "accuracy"
    assert controls["effective"]["memory_profile"] == "memory_safe"
    assert controls["effective"]["max_inserted_reflections_per_path"] == 1
    assert controls["effective"]["diffraction_state_budget"] == 2048
    assert controls["effective"]["inserted_reflection_state_budget"] == 512
    assert guardrails["applied"]
    assert profiling["packed_core_floats"] == 72
    assert profiling["packed_stride_bytes"] == 288
    assert profiling["pre_expansion_policy"]["enabled"]
    assert profiling["pre_expansion_policy"]["policy"] == "memory_safe_topk_power"
    assert profiling["pre_expansion_policy"]["higher_order_source_budget"] == 512
    assert profiling["pre_expansion_policy"]["inserted_source_budget"] == 512
    assert (
        profiling["peak_higher_order_source_states_after_pre_prune"]
        <= profiling["peak_higher_order_source_states_before_pre_prune"]
    )
    assert (
        profiling["peak_inserted_source_states_after_pre_prune"]
        <= profiling["peak_inserted_source_states_before_pre_prune"]
    )


