"""Policy and consistency tests for reflection EPC."""

import math
import sys
from pathlib import Path
TEST_FILE = Path(__file__).resolve()
CHANNEL_ROOT = TEST_FILE.parents[2]
CORE_ROOT = CHANNEL_ROOT.parent / "core"
sys.path.insert(0, str(CORE_ROOT))
sys.path.insert(0, str(CHANNEL_ROOT))

import witwin as wt

import drjit as dr
import pytest

from tests._scene_helpers import box_drjit_geometry, box_geometry, build_scene as build_test_scene
from witwin.channel import Field, native_extension_available
from witwin.channel.trace import compute_reflection_field
FREQUENCY = 1.0e9
WAVELENGTH = 299792458.0 / FREQUENCY
WAVENUMBER = 2.0 * dr.pi / WAVELENGTH


def _build_scene():
    cube1 = box_geometry(center=(-2.0, -2.0, 1.5), size=2.0)
    cube2 = box_geometry(center=(2.0, 1.5, 1.5), size=2.0)
    return build_test_scene(cube1, cube2)


def _build_field():
    return Field(bounds=((-6.0, 6.0), (-6.0, 6.0)), size=(16, 16))


def _field_max_abs_diff(lhs, rhs) -> float:
    real_diff = float(dr.max(dr.abs(lhs.real - rhs.real))[0])
    imag_diff = float(dr.max(dr.abs(lhs.imag - rhs.imag))[0])
    return max(real_diff, imag_diff)


@pytest.mark.gpu
def test_tx_grad_workload_uses_requested_backend_instead_of_exact_replay():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    scene = _build_scene()
    field = _build_field()
    coords = field.get_coordinates()
    tx_x = wt.Float(0.35)
    dr.enable_grad(tx_x)
    tx = wt.Point3f(tx_x, -5.0, 1.5)

    a_ref, _, detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=tx,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=128,
        max_reflections=1,
        mode="2d",
        reflection_coef=0.82,
        reflection_field_backend="native",
        return_per_bounce=False,
        grid_data=coords,
    )

    dda_stats = detail["dda_stats"]
    assert dda_stats["epc_eligible"] is True
    assert dda_stats["policy"] == "fresh_trace_geometry_grad_preserving_requested_backend"
    assert dda_stats["implementation"] == "native_cuda_custom_op"
    assert dda_stats["resolved_backend"] == "native"
    assert dda_stats["discovery_gradients_preserved"] is True
    assert dda_stats["tx_grad_enabled"] is True
    assert dda_stats["scene_geometry_grad_enabled"] is False

    loss = dr.sum(a_ref.real * a_ref.real + a_ref.imag * a_ref.imag)
    dr.backward(loss)
    grad_x = float(dr.grad(tx_x)[0])
    assert math.isfinite(grad_x)


@pytest.mark.gpu
def test_fresh_non_ad_field_trace_uses_exact_replay_for_native_backend():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    scene = _build_scene()
    field = _build_field()
    coords = field.get_coordinates()
    tx = wt.Point3f(0.0, -5.0, 1.5)

    _, _, detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=tx,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=128,
        max_reflections=1,
        mode="2d",
        reflection_coef=0.82,
        reflection_field_backend="native",
        return_per_bounce=False,
        grid_data=coords,
    )

    dda_stats = detail["dda_stats"]
    assert dda_stats["epc_eligible"] is True
    assert dda_stats["policy"] == "fresh_trace_epc"
    assert dda_stats["implementation"] == "epc"
    assert dda_stats["backend"] == "epc"
    assert dda_stats["resolved_backend"] == "native"
    assert dda_stats["discovery_gradients_preserved"] is False
    assert dda_stats["tx_grad_enabled"] is False
    assert dda_stats["scene_geometry_grad_enabled"] is False


@pytest.mark.gpu
def test_fresh_non_ad_field_trace_uses_exact_replay_for_drjit_backend():
    scene = _build_scene()
    field = _build_field()
    coords = field.get_coordinates()
    tx = wt.Point3f(0.0, -5.0, 1.5)

    _, _, detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=tx,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=128,
        max_reflections=1,
        mode="2d",
        reflection_coef=0.82,
        reflection_field_backend="drjit",
        return_per_bounce=False,
        grid_data=coords,
    )

    dda_stats = detail["dda_stats"]
    assert dda_stats["epc_eligible"] is True
    assert dda_stats["policy"] == "fresh_trace_epc"
    assert dda_stats["implementation"] == "epc"
    assert dda_stats["backend"] == "epc"
    assert dda_stats["resolved_backend"] == "drjit"
    assert dda_stats["discovery_gradients_preserved"] is False


@pytest.mark.gpu
def test_scene_geometry_grad_workload_uses_requested_backend_instead_of_exact_replay():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    center = wt.Point3f(0.0, 0.0, 1.5)
    dr.enable_grad(center)
    scene = build_test_scene(box_drjit_geometry(center=center, size=2.0))
    field = _build_field()
    coords = field.get_coordinates()
    tx = wt.Point3f(0.0, -5.0, 1.5)

    a_ref, _, detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=tx,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=128,
        max_reflections=1,
        mode="2d",
        reflection_coef=0.82,
        reflection_field_backend="native",
        return_per_bounce=False,
        grid_data=coords,
    )

    dda_stats = detail["dda_stats"]
    assert dda_stats["epc_eligible"] is True
    assert dda_stats["policy"] == "fresh_trace_geometry_grad_preserving_requested_backend"
    assert dda_stats["implementation"] == "native_cuda_custom_op"
    assert dda_stats["resolved_backend"] == "native"
    assert dda_stats["discovery_gradients_preserved"] is True
    assert dda_stats["tx_grad_enabled"] is False
    assert dda_stats["scene_geometry_grad_enabled"] is True

    loss = dr.sum(a_ref.real * a_ref.real + a_ref.imag * a_ref.imag)
    dr.backward(loss)
    grad_x = float(dr.grad(center.x)[0])
    assert math.isfinite(grad_x)


@pytest.mark.gpu
def test_explicit_reflection_detail_replay_matches_forward_and_reports_frozen_policy():
    scene = _build_scene()
    field = _build_field()
    coords = field.get_coordinates()
    tx = wt.Point3f(0.0, -5.0, 1.5)

    forward_field, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=tx,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=128,
        max_reflections=1,
        mode="2d",
        reflection_coef=0.82,
        return_per_bounce=False,
        grid_data=coords,
    )
    replay_field_native, _, replay_detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=tx,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=128,
        max_reflections=1,
        mode="2d",
        reflection_coef=0.82,
        return_per_bounce=False,
        grid_data=coords,
        reflection_detail=reflection_detail,
        reflection_field_backend="native",
    )
    replay_field_drjit, _, replay_detail_drjit = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=tx,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=128,
        max_reflections=1,
        mode="2d",
        reflection_coef=0.82,
        return_per_bounce=False,
        grid_data=coords,
        reflection_detail=reflection_detail,
        reflection_field_backend="drjit",
    )

    assert math.isfinite(_field_max_abs_diff(forward_field, replay_field_native))
    assert _field_max_abs_diff(replay_field_native, replay_field_drjit) < 1.0e-6

    dda_stats = replay_detail["dda_stats"]
    assert dda_stats["policy"] == "provided_reflection_detail_epc"
    assert dda_stats["implementation"] == "epc"
    assert dda_stats["backend"] == "epc"
    assert dda_stats["reused_discovery"] is True
    assert dda_stats["discovery_gradients_preserved"] is False
    assert replay_detail_drjit["dda_stats"]["implementation"] == "epc"
