"""Regression tests for mixed-path ownership and budgeting."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import witwin as wt

import drjit as dr

from tests._scene_helpers import box_geometry, build_scene as build_test_scene
from witwin.channel import FieldMonitor, Tracer
def build_scene():
    cube1 = box_geometry(center=(-1.8, -1.2, 1.5), size=2.0)
    cube2 = box_geometry(center=(1.8, 1.2, 1.5), size=2.0)
    return build_test_scene(cube1, cube2)


def field_power(field):
    return float(dr.sum(field.real * field.real + field.imag * field.imag)[0])


def audit_to_numpy(audit, key):
    value = audit[key]
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def test_mixed_diffraction_component_ownership():
    scene = build_scene()
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        enable_rd_diffraction=True,
        max_diffractions=2,
    )
    monitor = FieldMonitor(
        "mixed_plane",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-4.0, 4.0)),
        grid_size=20,
    )

    result = tracer.trace(
        wt.Point3f(0.0, -4.0, 1.5),
        monitor=monitor,
        verbose=False,
        return_diffraction_audit=True,
    )

    payload = result.primary
    dif_reconstruct = wt.Complex2f(
        payload.field.diffraction.real - payload.field.diffraction_direct.real - payload.field.diffraction_mixed.real,
        payload.field.diffraction.imag - payload.field.diffraction_direct.imag - payload.field.diffraction_mixed.imag,
    )
    assert field_power(dif_reconstruct) < 1e-10
    assert "a_rd" not in payload.metadata["field_component_ownership"]

    audit = payload.diffraction_detail["state_audit"]
    prefix_depth = audit_to_numpy(audit, "prefix_reflection_depth")
    intermediate_depth = audit_to_numpy(audit, "intermediate_reflection_depth")
    suffix_depth = audit_to_numpy(audit, "suffix_reflection_depth")
    ownership = np.asarray(audit["ownership"])

    direct_mask = (prefix_depth == 0) & (intermediate_depth == 0) & (suffix_depth == 0)
    mixed_mask = ~direct_mask

    assert np.all(ownership[direct_mask] == "direct_diffraction")
    assert np.all(ownership[mixed_mask] == "mixed_diffraction")


def test_mixed_path_budget_prunes_states_with_visible_metadata():
    scene = build_scene()
    tx = wt.Point3f(0.0, -4.0, 1.5)
    monitor = FieldMonitor(
        "mixed_budget_plane",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-4.0, 4.0)),
        grid_size=20,
    )

    tracer_unbounded = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        enable_rd_diffraction=True,
        max_diffractions=2,
    )
    tracer_bounded = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        enable_rd_diffraction=True,
        max_diffractions=2,
        diffraction_state_budget=10,
        inserted_reflection_state_budget=3,
    )

    res_unbounded = tracer_unbounded.trace(tx, monitor=monitor, verbose=False, return_diffraction_audit=True)
    res_bounded = tracer_bounded.trace(tx, monitor=monitor, verbose=False, return_diffraction_audit=True)

    audit_unbounded = res_unbounded.primary.diffraction_detail["state_audit"]
    audit_bounded = res_bounded.primary.diffraction_detail["state_audit"]

    order_unbounded = audit_to_numpy(audit_unbounded, "order")
    order_bounded = audit_to_numpy(audit_bounded, "order")
    intermediate_unbounded = audit_to_numpy(audit_unbounded, "intermediate_reflection_depth")
    intermediate_bounded = audit_to_numpy(audit_bounded, "intermediate_reflection_depth")

    unbounded_order2 = int(np.sum(order_unbounded == 2))
    bounded_order2 = int(np.sum(order_bounded == 2))
    unbounded_inserted = int(np.sum((order_unbounded == 2) & (intermediate_unbounded > 0)))
    bounded_inserted = int(np.sum((order_bounded == 2) & (intermediate_bounded > 0)))

    assert bounded_order2 <= 10
    assert bounded_inserted <= 3
    assert bounded_order2 < unbounded_order2
    assert bounded_inserted < unbounded_inserted

    budget_policy = res_bounded.primary.metadata["path_budget_policy"]
    assert budget_policy["enabled"] is True
    assert budget_policy["total_state_budget_per_order"] == 10
    assert budget_policy["inserted_state_budget_per_order"] == 3

    per_order = budget_policy["report"]["per_order"]
    assert len(per_order) >= 1
    assert any(item["inserted_budget_applied"] or item["total_budget_applied"] for item in per_order)
    for item in per_order:
        assert item["inserted_states_after_prune"] <= 3
        assert item["total_states_after_prune"] <= 10
        if item["order"] == 1:
            prefix_builder_stats = item["reflection_prefix_builder"]
            assert prefix_builder_stats["bounce_count"] >= 0
            assert prefix_builder_stats["input_paths"] >= 0
            assert prefix_builder_stats["chunk_count"] >= 0
            assert prefix_builder_stats["candidate_pairs"] >= prefix_builder_stats["support_kept_count"]
            assert prefix_builder_stats["support_kept_count"] >= prefix_builder_stats["field_kept_count"]
            assert prefix_builder_stats["output_states"] == item["prefix_states"]
            for value in prefix_builder_stats["timing"].values():
                assert value >= 0.0
        if item["order"] >= 2:
            builder_stats = item["higher_order_builder"]
            inserted_builder_stats = item["inserted_reflection_builder"]
            assert builder_stats["candidate_backend"] == budget_policy["report"]["higher_order_candidate_backend"]
            assert builder_stats["chunk_count"] >= 0
            assert builder_stats["candidate_raw_count"] >= builder_stats["candidate_unique_count"]
            assert builder_stats["visibility_input_count"] >= builder_stats["visibility_kept_count"]
            assert builder_stats["field_input_count"] >= builder_stats["field_kept_count"]
            assert builder_stats["output_states"] == item["direct_states"]
            for value in builder_stats["timing"].values():
                assert value >= 0.0
            assert inserted_builder_stats is not None
            assert inserted_builder_stats["input_states"] >= inserted_builder_stats["eligible_states"]
            assert inserted_builder_stats["chunk_count"] >= 0
            assert inserted_builder_stats["total_rays_cast"] >= 0
            assert inserted_builder_stats["hit_count"] >= 0
            assert inserted_builder_stats["candidate_slot_count"] >= 0
            assert inserted_builder_stats["visibility_kept_count"] >= inserted_builder_stats["field_kept_count"]
            assert inserted_builder_stats["output_states"] == item["inserted_states_before_prune"]
            for value in inserted_builder_stats["timing"].values():
                assert value >= 0.0


