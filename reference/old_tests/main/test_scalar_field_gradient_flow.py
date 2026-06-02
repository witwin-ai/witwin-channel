"""Focused gradient-flow regression tests for scalar field outputs."""

from __future__ import annotations

import drjit as dr
import numpy as np
import pytest
import witwin as wt
from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from witwin.channel import FieldMonitor, Tracer, to_numpy
pytestmark = pytest.mark.gpu


def _monitor():
    return FieldMonitor(
        "grad_plane",
        axis="z",
        position=1.5,
        bounds=((-8.0, 8.0), (-8.0, 8.0)),
        grid_size=16,
    )


def _build_tracer(scene):
    return Tracer(
        frequency=1.0e9,
        scene=scene,
        reflection_n_rays=1024,
        reflection_max_bounces=1,
        reflection_coef=1.0,
    )


def _component_grad_sum(field_component) -> float:
    dr.forward_to(field_component.real, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
    grad = np.asarray(to_numpy(dr.grad(field_component.real)), dtype=np.float64)
    return float(np.sum(np.abs(grad)))


def _scene_with_monitor(center):
    scene = build_test_scene(box_drjit_geometry(center=center, size=4.0))
    scene.add_monitor(_monitor())
    return scene


def test_scene_geometry_grad_flows_to_scalar_reflection_diffraction_and_total_fields():
    tx_pos = wt.Point3f(-5.0, 5.0, 1.5)

    center_reflection = wt.Point3f(0.0, 0.0, 2.0)
    dr.enable_grad(center_reflection)
    dr.set_grad(center_reflection, wt.Vector3f(1.0, 0.0, 0.0))
    reflection_result = _build_tracer(_scene_with_monitor(center_reflection)).trace(tx_pos)
    assert _component_grad_sum(reflection_result.primary.field.reflection) > 0.0
    assert reflection_result.primary.metadata["reflection_backend"]["implementation"] == "native_cuda_custom_op"

    center_diffraction = wt.Point3f(0.0, 0.0, 2.0)
    dr.enable_grad(center_diffraction)
    dr.set_grad(center_diffraction, wt.Vector3f(1.0, 0.0, 0.0))
    diffraction_result = _build_tracer(_scene_with_monitor(center_diffraction)).trace(tx_pos)
    assert _component_grad_sum(diffraction_result.primary.field.diffraction) > 0.0

    center_total = wt.Point3f(0.0, 0.0, 2.0)
    dr.enable_grad(center_total)
    dr.set_grad(center_total, wt.Vector3f(1.0, 0.0, 0.0))
    total_result = _build_tracer(_scene_with_monitor(center_total)).trace(tx_pos)
    assert _component_grad_sum(total_result.primary.field.total) > 0.0


def test_tx_grad_flows_to_scalar_reflection_diffraction_and_total_fields():
    center = wt.Point3f(0.0, 0.0, 2.0)

    tx_reflection = wt.Point3f(-5.0, 5.0, 1.5)
    dr.enable_grad(tx_reflection)
    dr.set_grad(tx_reflection, wt.Vector3f(1.0, 0.0, 0.0))
    reflection_result = _build_tracer(_scene_with_monitor(center)).trace(tx_reflection)
    assert _component_grad_sum(reflection_result.primary.field.reflection) > 0.0

    tx_diffraction = wt.Point3f(-5.0, 5.0, 1.5)
    dr.enable_grad(tx_diffraction)
    dr.set_grad(tx_diffraction, wt.Vector3f(1.0, 0.0, 0.0))
    diffraction_result = _build_tracer(_scene_with_monitor(center)).trace(tx_diffraction)
    assert _component_grad_sum(diffraction_result.primary.field.diffraction) > 0.0

    tx_total = wt.Point3f(-5.0, 5.0, 1.5)
    dr.enable_grad(tx_total)
    dr.set_grad(tx_total, wt.Vector3f(1.0, 0.0, 0.0))
    total_result = _build_tracer(_scene_with_monitor(center)).trace(tx_total)
    assert _component_grad_sum(total_result.primary.field.total) > 0.0
