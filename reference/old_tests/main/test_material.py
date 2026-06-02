"""Integrated visual test for per-object material response and gradients."""

from __future__ import annotations

import os
from pathlib import Path

import drjit as dr
import numpy as np
import pytest
import witwin as wt
from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from witwin.channel import (
    ChannelConfig,
    DEFAULT_VARIANT,
    DiffractionExecutionConfig,
    Material,
    FieldMonitor,
    TraceConfig,
    Tracer,
    draw_scene,
    to_numpy,
    to_power_db,
)
from witwin.channel.utils.polarization import (
    project_real_polarization_to_ray,
    vector_add,
    vector_eval,
    vector_from_scalar_and_real_direction,
)
from witwin.channel.trace import compute_los_field, compute_reflection_field
from witwin.channel.trace.diffraction import compute_diffraction_field
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

pytestmark = pytest.mark.gpu

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "material.png"
REFLECTION_PHASE_OUTPUT_PATH = OUTPUT_DIR / "material_reflection_phase_derivative.png"
REFLECTION_SPECULAR_OUTPUT_PATH = OUTPUT_DIR / "material_reflection_specular_edge.png"
DEFAULT_MONITOR_NAME = "main_material_grid"
FIELD_EPS_VALUES = (2.0, 8.0, 50.0, 1.0e4)
BASELINE_EPS_R = 1.0e4
GRAD_DIFFRACTION_EXECUTION = DiffractionExecutionConfig(suffix_dda="symbolic")
GRAD_FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad
POSITION_FD_DELTA = 0.01
ROTATION_FD_DELTA = 0.01
MATERIAL_FD_DELTA = 1.0e2


def _scalar_height(value) -> float:
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    try:
        return float(value[0])
    except (TypeError, ValueError, IndexError, KeyError):
        return float(value)


def _monitor_height(tx_pos) -> float:
    if hasattr(tx_pos, "z"):
        return _scalar_height(tx_pos.z)
    return _scalar_height(tx_pos[2])


def _format_eps_r(value: float) -> str:
    value = float(value)
    if abs(value) < 100.0:
        return f"{value:.2f}"
    return f"{value:.0e}"


def _assert_plane_monitor_result(result, monitor: FieldMonitor) -> None:
    payload = result
    if hasattr(result, "monitor"):
        payload = result.monitor(monitor.name)
    sampling = payload.metadata["receiver_sampling"]
    assert sampling["sample_positions"] == "boundary_points"
    assert sampling["index_partitioning"] == "span_over_n_bins"
    assert tuple(payload.grid_shape) == tuple(monitor.grid_shape)
    assert tuple(payload.range_x) == monitor.bounds[0]
    assert tuple(payload.range_y) == monitor.bounds[1]
    assert abs(float(payload.plane_position) - float(monitor.position)) < 1e-6


def _runtime_material_array(value, n_values: int, *, name: str):
    if isinstance(value, wt.Float):
        width = int(dr.width(value))
        if width == n_values:
            return value
        if width == 1:
            return dr.repeat(value, n_values)
        raise ValueError(f"{name} must be scalar or length {n_values}, got DrJit width {width}.")

    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 1:
        return dr.full(wt.Float, float(array[0]), n_values)
    if array.size == n_values:
        return wt.Float(array.tolist())
    raise ValueError(f"{name} must be scalar or length {n_values}, got {array.size}.")


def _override_scene_runtime_material(
    scene,
    *,
    eps_r,
    sigma_e=0.0,
) -> None:
    if scene.tri_data_gpu is None:
        raise ValueError("Runtime material override requires preloaded triangle data.")

    n_triangles = int(scene.tri_data_gpu["n_triangles"])
    eps_r_array = _runtime_material_array(eps_r, n_triangles, name="eps_r")
    sigma_e_array = _runtime_material_array(sigma_e, n_triangles, name="sigma_e")
    material_specified = dr.full(wt.Bool, True, n_triangles)
    structure_idx = scene.tri_data_gpu.get("material_structure_idx")
    if structure_idx is None or int(dr.width(structure_idx)) != n_triangles:
        structure_idx = dr.full(wt.Int32, 0, n_triangles)

    # witwin.core.Material currently stores scalar Python floats, so material AD
    # must be injected into the runtime triangle tables after scene creation.
    scene.tri_data_gpu["material_eps_r"] = eps_r_array
    scene.tri_data_gpu["material_sigma_e"] = sigma_e_array
    scene.tri_data_gpu["material_specified"] = material_specified
    scene.tri_data_gpu["material_structure_idx"] = structure_idx
    scene.tri_data_gpu["material_has_specified_materials"] = True
    scene.tri_data_gpu["material_n_specified_triangles"] = n_triangles
    scene.tri_data_gpu["material_n_default_material_triangles"] = 0
    scene._triangle_material_data = {
        "eps_r": eps_r_array,
        "sigma_e": sigma_e_array,
        "specified": material_specified,
        "structure_idx": structure_idx,
        "has_specified_materials": True,
        "n_specified_triangles": n_triangles,
        "n_default_material_triangles": 0,
    }


def _uniform_override_material_dict(*, eps_r: float, gain: float, sigma_e: float = 0.0) -> dict[str, float]:
    return {
        "relative_permittivity": float(eps_r),
        "conductivity": float(sigma_e),
        "gain": float(gain),
    }


def _trace_scene(
    *,
    center,
    size,
    freq,
    tx_pos,
    range_x,
    range_y,
    grid_size,
    n_rays,
    max_reflections,
    reflection_coef,
    material,
    rotation=None,
    max_diffractions=2,
    enable_rd_diffraction=True,
    runtime_eps_r=None,
    runtime_sigma_e=None,
    reflection_material_override=None,
    diffraction_material_override=None,
    use_scene_materials_for_reflection=True,
    use_scene_materials_for_diffraction=True,
    tx_polarization=(1.0, 0.0, 0.0),
):
    scene = build_test_scene(
        box_drjit_geometry(center=center, size=size, rotation=rotation),
        material=material,
    )
    if runtime_eps_r is not None or runtime_sigma_e is not None:
        _override_scene_runtime_material(
            scene,
            eps_r=scene.tri_data_gpu["material_eps_r"] if runtime_eps_r is None else runtime_eps_r,
            sigma_e=scene.tri_data_gpu["material_sigma_e"] if runtime_sigma_e is None else runtime_sigma_e,
        )
    monitor = FieldMonitor(
        DEFAULT_MONITOR_NAME,
        axis="z",
        position=_monitor_height(tx_pos),
        bounds=(range_x, range_y),
        grid_size=grid_size,
    )
    scene.add_monitor(monitor)
    config = ChannelConfig(
        trace=TraceConfig(
            diffraction_execution=GRAD_DIFFRACTION_EXECUTION,
        )
    )
    tracer = Tracer(
        frequency=freq,
        scene=scene,
        config=config,
        reflection_n_rays=n_rays,
        reflection_max_bounces=max_reflections,
        reflection_coef=reflection_coef,
        reflection_material=reflection_material_override,
        diffraction_material=diffraction_material_override,
        use_scene_materials_for_reflection=use_scene_materials_for_reflection,
        use_scene_materials_for_diffraction=use_scene_materials_for_diffraction,
        enable_rd_diffraction=enable_rd_diffraction,
        max_diffractions=max_diffractions,
        tx_polarization=tx_polarization,
    )
    result = tracer.trace(tx_pos=tx_pos)
    _assert_plane_monitor_result(result, monitor)
    return result, scene


def _multipath_field(result):
    return wt.Complex2f(
        result.primary.field.reflection.real + result.primary.field.diffraction.real,
        result.primary.field.reflection.imag + result.primary.field.diffraction.imag,
    )


def _field_component(result, field_mode: str):
    if field_mode == "multipath":
        return _multipath_field(result)
    if field_mode == "reflection":
        return result.primary.field.reflection
    if field_mode == "diffraction":
        return result.primary.field.diffraction
    if field_mode == "total":
        return result.primary.field.total
    raise ValueError(f"Unsupported field_mode: {field_mode}")


def _field_to_complex_numpy(field) -> np.ndarray:
    return np.asarray(to_numpy(field.real), dtype=np.float64) + 1j * np.asarray(
        to_numpy(field.imag),
        dtype=np.float64,
    )


def _complex_field_gradient(
    result,
    *,
    field_mode: str = "multipath",
) -> np.ndarray:
    field = _field_component(result, field_mode)
    n_values = int(dr.width(field.real))
    dr.forward_to(field.real, field.imag, flags=GRAD_FLAGS)
    grad_re = dr.grad(field.real)
    grad_im = dr.grad(field.imag)
    grad_re_np = (
        np.asarray(to_numpy(grad_re), dtype=np.float64)
        if grad_re is not None
        else np.zeros(n_values, dtype=np.float64)
    )
    grad_im_np = (
        np.asarray(to_numpy(grad_im), dtype=np.float64)
        if grad_im is not None
        else np.zeros(n_values, dtype=np.float64)
    )
    return grad_re_np + 1j * grad_im_np


def _complex_field_gradient_magnitude(
    result,
    *,
    field_mode: str = "multipath",
) -> np.ndarray:
    return np.abs(_complex_field_gradient(result, field_mode=field_mode))


def _complex_field_fd_gradient(
    result_plus,
    result_minus,
    delta: float,
    *,
    field_mode: str = "multipath",
) -> np.ndarray:
    field_plus = _field_component(result_plus, field_mode)
    field_minus = _field_component(result_minus, field_mode)
    grad_re = (
        np.asarray(to_numpy(field_plus.real), dtype=np.float64)
        - np.asarray(to_numpy(field_minus.real), dtype=np.float64)
    ) / (2.0 * delta)
    grad_im = (
        np.asarray(to_numpy(field_plus.imag), dtype=np.float64)
        - np.asarray(to_numpy(field_minus.imag), dtype=np.float64)
    ) / (2.0 * delta)
    return grad_re + 1j * grad_im


def _complex_field_fd_gradient_magnitude(
    result_plus,
    result_minus,
    delta: float,
    *,
    field_mode: str = "multipath",
) -> np.ndarray:
    return np.abs(
        _complex_field_fd_gradient(
            result_plus,
            result_minus,
            delta,
            field_mode=field_mode,
        )
    )


def _relative_l2_error(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_arr = np.asarray(lhs, dtype=np.float64)
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    return float(np.linalg.norm(lhs_arr - rhs_arr) / max(float(np.linalg.norm(rhs_arr)), 1e-12))


def _position_ad_gradient(
    *,
    center_vals,
    size,
    freq,
    tx_vals,
    range_x,
    range_y,
    grid_size,
    n_rays,
    max_reflections,
    reflection_coef,
    baseline_eps_r,
    rotation_val,
    field_mode="multipath",
):
    center = wt.Point3f(*map(float, center_vals))
    tx_pos = wt.Point3f(*map(float, tx_vals))
    dr.enable_grad(center)
    dr.set_grad(center, wt.Vector3f(1.0, 0.0, 0.0))
    result, scene = _trace_scene(
        center=center,
        size=size,
        freq=freq,
        tx_pos=tx_pos,
        range_x=range_x,
        range_y=range_y,
        grid_size=grid_size,
        n_rays=n_rays,
        max_reflections=max_reflections,
        reflection_coef=reflection_coef,
        material=Material(eps_r=baseline_eps_r),
        rotation=wt.Float(float(rotation_val)),
    )
    return result, scene, _complex_field_gradient_magnitude(result, field_mode=field_mode)


def _position_fd_gradient(
    *,
    center_vals,
    size,
    freq,
    tx_vals,
    range_x,
    range_y,
    grid_size,
    n_rays,
    max_reflections,
    reflection_coef,
    baseline_eps_r,
    rotation_val,
    field_mode="multipath",
):
    delta = POSITION_FD_DELTA
    tx_pos = wt.Point3f(*map(float, tx_vals))
    plus_result, _ = _trace_scene(
        center=wt.Point3f(float(center_vals[0]) + delta, float(center_vals[1]), float(center_vals[2])),
        size=size,
        freq=freq,
        tx_pos=tx_pos,
        range_x=range_x,
        range_y=range_y,
        grid_size=grid_size,
        n_rays=n_rays,
        max_reflections=max_reflections,
        reflection_coef=reflection_coef,
        material=Material(eps_r=baseline_eps_r),
        rotation=wt.Float(float(rotation_val)),
    )
    minus_result, _ = _trace_scene(
        center=wt.Point3f(float(center_vals[0]) - delta, float(center_vals[1]), float(center_vals[2])),
        size=size,
        freq=freq,
        tx_pos=tx_pos,
        range_x=range_x,
        range_y=range_y,
        grid_size=grid_size,
        n_rays=n_rays,
        max_reflections=max_reflections,
        reflection_coef=reflection_coef,
        material=Material(eps_r=baseline_eps_r),
        rotation=wt.Float(float(rotation_val)),
    )
    return _complex_field_fd_gradient_magnitude(
        plus_result,
        minus_result,
        delta,
        field_mode=field_mode,
    )


def _rotation_ad_gradient(
    *,
    center_vals,
    size,
    freq,
    tx_vals,
    range_x,
    range_y,
    grid_size,
    n_rays,
    max_reflections,
    reflection_coef,
    baseline_eps_r,
    rotation_val,
    field_mode="multipath",
):
    center = wt.Point3f(*map(float, center_vals))
    tx_pos = wt.Point3f(*map(float, tx_vals))
    rotation = wt.Float(float(rotation_val))
    dr.enable_grad(rotation)
    dr.set_grad(rotation, wt.Float(1.0))
    result, scene = _trace_scene(
        center=center,
        size=size,
        freq=freq,
        tx_pos=tx_pos,
        range_x=range_x,
        range_y=range_y,
        grid_size=grid_size,
        n_rays=n_rays,
        max_reflections=max_reflections,
        reflection_coef=reflection_coef,
        material=Material(eps_r=baseline_eps_r),
        rotation=rotation,
    )
    return result, scene, _complex_field_gradient_magnitude(result, field_mode=field_mode)


def _rotation_fd_gradient(
    *,
    center_vals,
    size,
    freq,
    tx_vals,
    range_x,
    range_y,
    grid_size,
    n_rays,
    max_reflections,
    reflection_coef,
    baseline_eps_r,
    rotation_val,
    field_mode="multipath",
):
    delta = ROTATION_FD_DELTA
    center = wt.Point3f(*map(float, center_vals))
    tx_pos = wt.Point3f(*map(float, tx_vals))
    plus_result, _ = _trace_scene(
        center=center,
        size=size,
        freq=freq,
        tx_pos=tx_pos,
        range_x=range_x,
        range_y=range_y,
        grid_size=grid_size,
        n_rays=n_rays,
        max_reflections=max_reflections,
        reflection_coef=reflection_coef,
        material=Material(eps_r=baseline_eps_r),
        rotation=wt.Float(float(rotation_val) + delta),
    )
    minus_result, _ = _trace_scene(
        center=center,
        size=size,
        freq=freq,
        tx_pos=tx_pos,
        range_x=range_x,
        range_y=range_y,
        grid_size=grid_size,
        n_rays=n_rays,
        max_reflections=max_reflections,
        reflection_coef=reflection_coef,
        material=Material(eps_r=baseline_eps_r),
        rotation=wt.Float(float(rotation_val) - delta),
    )
    return _complex_field_fd_gradient_magnitude(
        plus_result,
        minus_result,
        delta,
        field_mode=field_mode,
    )


def _material_ad_gradient(
    *,
    center_vals,
    size,
    freq,
    tx_vals,
    range_x,
    range_y,
    grid_size,
    n_rays,
    max_reflections,
    reflection_coef,
    baseline_eps_r,
    rotation_val,
    field_mode="multipath",
    return_complex: bool = False,
):
    center = wt.Point3f(*map(float, center_vals))
    tx_pos = wt.Point3f(*map(float, tx_vals))
    eps_r = wt.Float(float(baseline_eps_r))
    dr.enable_grad(eps_r)
    dr.set_grad(eps_r, wt.Float(1.0))
    result, scene = _trace_scene(
        center=center,
        size=size,
        freq=freq,
        tx_pos=tx_pos,
        range_x=range_x,
        range_y=range_y,
        grid_size=grid_size,
        n_rays=n_rays,
        max_reflections=max_reflections,
        reflection_coef=reflection_coef,
        material=Material(eps_r=float(baseline_eps_r), sigma_e=0.0),
        rotation=wt.Float(float(rotation_val)),
        runtime_eps_r=eps_r,
        runtime_sigma_e=0.0,
    )
    gradient_complex = _complex_field_gradient(result, field_mode=field_mode)
    if return_complex:
        return result, scene, gradient_complex
    return result, scene, np.abs(gradient_complex)


def _material_fd_gradients(
    *,
    center_vals,
    size,
    freq,
    tx_vals,
    range_x,
    range_y,
    grid_size,
    n_rays,
    max_reflections,
    reflection_coef,
    baseline_eps_r,
    rotation_val,
    return_complex: bool = False,
):
    delta = MATERIAL_FD_DELTA
    center = wt.Point3f(*map(float, center_vals))
    tx_pos = wt.Point3f(*map(float, tx_vals))
    plus_result, _ = _trace_scene(
        center=center,
        size=size,
        freq=freq,
        tx_pos=tx_pos,
        range_x=range_x,
        range_y=range_y,
        grid_size=grid_size,
        n_rays=n_rays,
        max_reflections=max_reflections,
        reflection_coef=reflection_coef,
        material=Material(eps_r=float(baseline_eps_r), sigma_e=0.0),
        rotation=wt.Float(float(rotation_val)),
        runtime_eps_r=float(baseline_eps_r) + delta,
        runtime_sigma_e=0.0,
    )
    minus_result, _ = _trace_scene(
        center=center,
        size=size,
        freq=freq,
        tx_pos=tx_pos,
        range_x=range_x,
        range_y=range_y,
        grid_size=grid_size,
        n_rays=n_rays,
        max_reflections=max_reflections,
        reflection_coef=reflection_coef,
        material=Material(eps_r=float(baseline_eps_r), sigma_e=0.0),
        rotation=wt.Float(float(rotation_val)),
        runtime_eps_r=float(baseline_eps_r) - delta,
        runtime_sigma_e=0.0,
    )
    reflection_gradient = _complex_field_fd_gradient(
        plus_result,
        minus_result,
        delta,
        field_mode="reflection",
    )
    diffraction_gradient = _complex_field_fd_gradient(
        plus_result,
        minus_result,
        delta,
        field_mode="diffraction",
    )
    multipath_gradient = _complex_field_fd_gradient(
        plus_result,
        minus_result,
        delta,
        field_mode="multipath",
    )
    if return_complex:
        return {
            "reflection": reflection_gradient,
            "diffraction": diffraction_gradient,
            "multipath": multipath_gradient,
        }
    return {
        "reflection": np.abs(reflection_gradient),
        "diffraction": np.abs(diffraction_gradient),
        "multipath": np.abs(multipath_gradient),
    }


def _to_db(array: np.ndarray, *, floor: float = -120.0) -> np.ndarray:
    return np.maximum(20.0 * np.log10(np.asarray(array, dtype=np.float64) + 1e-20), floor)


def _wrapped_phase_difference(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (lhs - rhs)))


def _signed_percentile_limit(*arrays: np.ndarray, percentile: float = 99.5, floor: float = 1e-12) -> float:
    maxima = [
        float(np.percentile(np.abs(np.asarray(array, dtype=np.float64)), percentile))
        for array in arrays
    ]
    return max(max(maxima, default=0.0), floor)


def _zero_crossing_mask(signed_grid: np.ndarray, *, threshold: float | None = None) -> np.ndarray:
    values = np.asarray(signed_grid, dtype=np.float64)
    if threshold is None:
        threshold = max(float(np.percentile(np.abs(values), 99.0)) * 1e-4, 1e-12)

    positive = values > threshold
    negative = values < -threshold
    horizontal = (positive[:, 1:] & negative[:, :-1]) | (negative[:, 1:] & positive[:, :-1])
    vertical = (positive[1:, :] & negative[:-1, :]) | (negative[1:, :] & positive[:-1, :])

    mask = np.zeros_like(values, dtype=bool)
    mask[:, 1:] |= horizontal
    mask[:, :-1] |= horizontal
    mask[1:, :] |= vertical
    mask[:-1, :] |= vertical
    return mask


def _complex_gradient_components(
    field_complex: np.ndarray,
    gradient_complex: np.ndarray,
    *,
    grid_size: int,
) -> dict[str, np.ndarray]:
    field_grid = np.asarray(field_complex, dtype=np.complex128).reshape(grid_size, grid_size)
    gradient_grid = np.asarray(gradient_complex, dtype=np.complex128).reshape(grid_size, grid_size)
    field_mag_sqr = np.maximum(np.abs(field_grid) ** 2, 1e-20)
    complex_response = np.conj(field_grid) * gradient_grid / field_mag_sqr
    return {
        "field": field_grid,
        "gradient": gradient_grid,
        "logamp_derivative": np.real(complex_response),
        "phase_derivative": np.imag(complex_response),
    }


def _binary_boundary_mask(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    boundary = np.zeros_like(values, dtype=bool)
    boundary[:, 1:] |= values[:, 1:] != values[:, :-1]
    boundary[:, :-1] |= values[:, 1:] != values[:, :-1]
    boundary[1:, :] |= values[1:, :] != values[:-1, :]
    boundary[:-1, :] |= values[1:, :] != values[:-1, :]
    return boundary


def _draw_mask_contour(ax, mask: np.ndarray, *, extent, color: str, linewidth: float = 0.8) -> None:
    values = np.asarray(mask, dtype=np.float64)
    if float(np.max(values)) <= 0.0:
        return
    x_min, x_max, y_min, y_max = extent
    x_coords = np.linspace(x_min, x_max, values.shape[1])
    y_coords = np.linspace(y_min, y_max, values.shape[0])
    ax.contour(
        x_coords,
        y_coords,
        values,
        levels=[0.5],
        colors=color,
        linewidths=linewidth,
        origin="lower",
    )


def _draw_zero_contour(ax, signed_grid: np.ndarray, *, extent, color: str, linewidth: float = 0.8) -> None:
    values = np.asarray(signed_grid, dtype=np.float64)
    if not (float(np.min(values)) < 0.0 < float(np.max(values))):
        return
    x_min, x_max, y_min, y_max = extent
    x_coords = np.linspace(x_min, x_max, values.shape[1])
    y_coords = np.linspace(y_min, y_max, values.shape[0])
    ax.contour(
        x_coords,
        y_coords,
        values,
        levels=[0.0],
        colors=color,
        linewidths=linewidth,
        origin="lower",
    )


def _complex_gradient_phase_diagnostics(
    field_complex: np.ndarray,
    gradient_complex: np.ndarray,
    *,
    grid_size: int,
) -> dict[str, float]:
    components = _complex_gradient_components(
        field_complex,
        gradient_complex,
        grid_size=grid_size,
    )
    field_grid = components["field"]
    gradient_grid = components["gradient"]
    gradient_mag = np.abs(gradient_grid)
    gradient_db = _to_db(gradient_mag)
    gradient_phase = np.angle(gradient_grid)

    right_phase_jump = np.abs(_wrapped_phase_difference(gradient_phase[:, 1:], gradient_phase[:, :-1]))
    up_phase_jump = np.abs(_wrapped_phase_difference(gradient_phase[1:, :], gradient_phase[:-1, :]))
    right_db_jump = np.abs(gradient_db[:, 1:] - gradient_db[:, :-1])
    up_db_jump = np.abs(gradient_db[1:, :] - gradient_db[:-1, :])
    right_min_mag = np.minimum(gradient_mag[:, 1:], gradient_mag[:, :-1])
    up_min_mag = np.minimum(gradient_mag[1:, :], gradient_mag[:-1, :])

    phase_jump = np.concatenate([right_phase_jump.ravel(), up_phase_jump.ravel()])
    db_jump = np.concatenate([right_db_jump.ravel(), up_db_jump.ravel()])
    min_mag = np.concatenate([right_min_mag.ravel(), up_min_mag.ravel()])
    top_jump_threshold = float(np.percentile(db_jump, 99.9))
    top_jump_mask = db_jump >= top_jump_threshold
    strong_phase_flip_mask = phase_jump > (0.9 * np.pi)
    phase_derivative = components["phase_derivative"]

    return {
        "gradient_p99": float(np.percentile(gradient_mag, 99.0)),
        "gradient_p999": float(np.percentile(gradient_mag, 99.9)),
        "db_jump_p99": float(np.percentile(db_jump, 99.0)),
        "db_jump_p999": float(np.percentile(db_jump, 99.9)),
        "phase_derivative_p99": float(np.percentile(np.abs(phase_derivative), 99.0)),
        "strong_phase_flip_fraction": float(np.mean(strong_phase_flip_mask)),
        "top_jump_near_zero_fraction": float(np.mean(min_mag[top_jump_mask] < 1e-8)),
        "top_jump_phase_flip_fraction": float(np.mean(strong_phase_flip_mask[top_jump_mask])),
    }


def _field_delta_power(lhs, rhs) -> float:
    delta = wt.Complex2f(lhs.real - rhs.real, lhs.imag - rhs.imag)
    return float(dr.sum(delta.real * delta.real + delta.imag * delta.imag)[0])


def _field_power(field) -> float:
    return float(dr.sum(field.real * field.real + field.imag * field.imag)[0])


def _jones_xy_power(jones_xy) -> float:
    return float(
        dr.sum(
            jones_xy["x"].real * jones_xy["x"].real
            + jones_xy["x"].imag * jones_xy["x"].imag
            + jones_xy["y"].real * jones_xy["y"].real
            + jones_xy["y"].imag * jones_xy["y"].imag
        )[0]
    )


def _jones_xy_power_db_grid(jones_xy, *, grid_size: int, floor: float = -120.0) -> np.ndarray:
    jones_x = _field_to_complex_numpy(jones_xy["x"])
    jones_y = _field_to_complex_numpy(jones_xy["y"])
    power = np.abs(jones_x) ** 2 + np.abs(jones_y) ** 2
    return np.maximum(10.0 * np.log10(power + 1e-20), floor).reshape(grid_size, grid_size)


def _vector_component_power_array(vector_field, component: str) -> np.ndarray:
    return np.abs(_field_to_complex_numpy(vector_field[component])) ** 2


def _power_db_grid(power: np.ndarray, *, grid_size: int, floor: float = -120.0) -> np.ndarray:
    return np.maximum(10.0 * np.log10(np.asarray(power, dtype=np.float64) + 1e-20), floor).reshape(
        grid_size,
        grid_size,
    )


def _field_power_array(field) -> np.ndarray:
    values = _field_to_complex_numpy(field)
    return np.abs(values) ** 2


def _phase_grid(values: np.ndarray, *, grid_size: int) -> np.ndarray:
    return np.angle(np.asarray(values, dtype=np.complex128)).reshape(grid_size, grid_size)


def _rebuild_total_vector_field(
    *,
    scene,
    freq,
    tx_pos,
    range_x,
    range_y,
    grid_size,
    n_rays,
    max_reflections,
    reflection_coef,
    max_diffractions=2,
    enable_rd_diffraction=True,
    reflection_material_override=None,
    diffraction_material_override=None,
    use_scene_materials_for_reflection=True,
    use_scene_materials_for_diffraction=True,
    tx_polarization=(1.0, 0.0, 0.0),
    rx_polarization=None,
):
    # Result currently exposes only the global-XY Jones view, so rebuild the
    # full vector field locally when diagnostics need the z component.
    wavelength = 299792458.0 / float(freq)
    k = 2.0 * np.pi / wavelength
    monitor = FieldMonitor(
        DEFAULT_MONITOR_NAME,
        axis="z",
        position=_monitor_height(tx_pos),
        bounds=(range_x, range_y),
        grid_size=grid_size,
    )
    field = monitor.to_field(wavelength, default_resolution=0.125)
    coords = field.get_coordinates()
    X, Y = coords["X"], coords["Y"]
    rx_z = wt.Float(float(monitor.position))

    a_los = compute_los_field(scene, X, Y, rx_z, tx_pos, wavelength, k)
    los_ray_dir = wt.Vector3f(X - tx_pos.x, Y - tx_pos.y, rx_z - tx_pos.z)
    los_ray_dir = los_ray_dir / (dr.norm(los_ray_dir) + 1e-12)
    los_pol_dir = project_real_polarization_to_ray(tx_polarization, los_ray_dir)
    polarization_los = vector_eval(vector_from_scalar_and_real_direction(a_los, los_pol_dir))

    _, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=float(monitor.position),
        tx_pos=tx_pos,
        scene=scene,
        wavelength=wavelength,
        k=k,
        n_rays=n_rays,
        max_reflections=max_reflections,
        mode=monitor.ray_mode,
        reflection_coef=reflection_coef,
        tx_polarization=tx_polarization,
        rx_polarization=rx_polarization,
        reflection_material=reflection_material_override,
        use_scene_materials=use_scene_materials_for_reflection,
        return_per_bounce=False,
        grid_data=coords,
    )

    _, _, _, diffraction_components = compute_diffraction_field(
        X,
        Y,
        float(monitor.position),
        tx_pos,
        scene,
        wavelength,
        k,
        reflection_detail=reflection_detail if enable_rd_diffraction else None,
        max_diffractions=max_diffractions,
        reflection_n_rays=n_rays if enable_rd_diffraction else 0,
        reflection_max_bounces=max_reflections if enable_rd_diffraction else 0,
        reflection_coef=reflection_coef,
        reflection_mode=monitor.ray_mode,
        grid=field,
        grid_data=coords,
        return_components=True,
        return_per_edge=False,
        return_state_audit=False,
        diffraction_material=diffraction_material_override,
        use_scene_materials=use_scene_materials_for_diffraction,
        tx_polarization=tx_polarization,
        rx_polarization=rx_polarization,
        execution=GRAD_DIFFRACTION_EXECUTION,
    )

    polarization_ref = reflection_detail["polarization_field_total"]
    polarization_dif = vector_add(
        diffraction_components["polarization_direct"],
        diffraction_components["polarization_multi"],
    )
    polarization_tot = vector_eval(
        vector_add(
            vector_add(polarization_los, polarization_ref),
            polarization_dif,
        )
    )
    return {
        "los": polarization_los,
        "reflection": polarization_ref,
        "diffraction": polarization_dif,
        "total": polarization_tot,
    }


def test_material_visual_grid():
    params = {
        "grid_size": 512,
        "freq": 1e9,
        "range_x": (-8.0, 8.0),
        "range_y": (-8.0, 8.0),
        "center_vals": (0.0, 0.0, 2.0),
        "size": 4.0,
        "tx_vals": (-5.0, 5.0, 1.5),
        "rotation_val": float(np.deg2rad(-5.0)),
        "n_rays": 20_000,
        "max_reflections": 1,
        "reflection_coef": 1.0,
        "baseline_eps_r": BASELINE_EPS_R,
    }
    baseline_eps_label = _format_eps_r(params["baseline_eps_r"])

    field_payloads = []
    for eps_r in FIELD_EPS_VALUES:
        result, scene = _trace_scene(
            center=wt.Point3f(*map(float, params["center_vals"])),
            size=params["size"],
            freq=params["freq"],
            tx_pos=wt.Point3f(*map(float, params["tx_vals"])),
            range_x=params["range_x"],
            range_y=params["range_y"],
            grid_size=params["grid_size"],
            n_rays=params["n_rays"],
            max_reflections=params["max_reflections"],
            reflection_coef=params["reflection_coef"],
            material=Material(eps_r=eps_r),
            rotation=wt.Float(float(params["rotation_val"])),
            tx_polarization=(1.0, 0.0, 0.0),
        )
        result_y, _ = _trace_scene(
            center=wt.Point3f(*map(float, params["center_vals"])),
            size=params["size"],
            freq=params["freq"],
            tx_pos=wt.Point3f(*map(float, params["tx_vals"])),
            range_x=params["range_x"],
            range_y=params["range_y"],
            grid_size=params["grid_size"],
            n_rays=params["n_rays"],
            max_reflections=params["max_reflections"],
            reflection_coef=params["reflection_coef"],
            material=Material(eps_r=eps_r),
            rotation=wt.Float(float(params["rotation_val"])),
            tx_polarization=(0.0, 1.0, 0.0),
        )
        vector_fields = _rebuild_total_vector_field(
            scene=scene,
            freq=params["freq"],
            tx_pos=wt.Point3f(*map(float, params["tx_vals"])),
            range_x=params["range_x"],
            range_y=params["range_y"],
            grid_size=params["grid_size"],
            n_rays=params["n_rays"],
            max_reflections=params["max_reflections"],
            reflection_coef=params["reflection_coef"],
        )
        multipath_field = _multipath_field(result)
        multipath_jones = {
            "x": result.primary.jones.reflection["x"] + result.primary.jones.diffraction["x"],
            "y": result.primary.jones.reflection["y"] + result.primary.jones.diffraction["y"],
        }
        los_power = _field_power(result.primary.field.los)
        reflection_power = _field_power(result.primary.field.reflection)
        diffraction_power = _field_power(result.primary.field.diffraction)
        jones_los_power = _jones_xy_power(result.primary.jones.los)
        jones_total_power = _jones_xy_power(result.primary.jones.total)
        jones_diffraction_power = _jones_xy_power(result.primary.jones.diffraction)
        jones_multipath_power = _jones_xy_power(multipath_jones)
        vector_x_power_grid = _vector_component_power_array(vector_fields["total"], "x")
        vector_y_power_grid = _vector_component_power_array(vector_fields["total"], "y")
        vector_z_power_grid = _vector_component_power_array(vector_fields["total"], "z")
        vector_xy_power_grid = vector_x_power_grid + vector_y_power_grid
        vector_total_power_grid = vector_xy_power_grid + vector_z_power_grid
        vector_x_power = float(np.sum(vector_x_power_grid))
        vector_y_power = float(np.sum(vector_y_power_grid))
        vector_z_power = float(np.sum(vector_z_power_grid))
        vector_xy_power = float(np.sum(vector_xy_power_grid))
        vector_total_power = float(np.sum(vector_total_power_grid))
        reflection_complex = _field_to_complex_numpy(result.primary.field.reflection)
        diffraction_complex = _field_to_complex_numpy(result.primary.field.diffraction)
        multipath_phase_complex = reflection_complex + diffraction_complex
        scalar_total_power_grid = _field_power_array(result.primary.field.total)
        scalar_total_power_grid_y = _field_power_array(result_y.primary.field.total)
        dual_pol_scalar_total_power_grid = scalar_total_power_grid + scalar_total_power_grid_y
        dual_pol_scalar_total_power = float(np.sum(dual_pol_scalar_total_power_grid))
        field_payloads.append(
            {
                "eps_r": eps_r,
                "result": result,
                "result_y": result_y,
                "scene": scene,
                "field": result.primary.field.total,
                "total_power": _field_power(result.primary.field.total),
                "total_power_y": _field_power(result_y.primary.field.total),
                "los_power": los_power,
                "los_power_y": _field_power(result_y.primary.field.los),
                "multipath_power": _field_power(multipath_field),
                "reflection_power": reflection_power,
                "diffraction_power": diffraction_power,
                "jones_total_power": jones_total_power,
                "jones_los_power": jones_los_power,
                "jones_multipath_power": jones_multipath_power,
                "jones_diffraction_power": jones_diffraction_power,
                "vector_total_power": vector_total_power,
                "vector_xy_power": vector_xy_power,
                "vector_x_power": vector_x_power,
                "vector_y_power": vector_y_power,
                "vector_z_power": vector_z_power,
                "dual_pol_scalar_total_power": dual_pol_scalar_total_power,
                "vector_xy_rel_error": abs(vector_xy_power - jones_total_power) / max(
                    jones_total_power,
                    1e-12,
                ),
                "dual_pol_scalar_total_power_db": _power_db_grid(
                    dual_pol_scalar_total_power_grid,
                    grid_size=params["grid_size"],
                ),
                "jones_total_power_db": _jones_xy_power_db_grid(
                    result.primary.jones.total,
                    grid_size=params["grid_size"],
                ),
                "vector_total_power_db": _power_db_grid(
                    vector_total_power_grid,
                    grid_size=params["grid_size"],
                ),
                "vector_x_power_db": _power_db_grid(
                    vector_x_power_grid,
                    grid_size=params["grid_size"],
                ),
                "vector_y_power_db": _power_db_grid(
                    vector_y_power_grid,
                    grid_size=params["grid_size"],
                ),
                "jones_diffraction_power_db": _jones_xy_power_db_grid(
                    result.primary.jones.diffraction,
                    grid_size=params["grid_size"],
                ),
                "field_db": to_numpy(to_power_db(result.primary.field.total)).reshape(
                    params["grid_size"],
                    params["grid_size"],
                ),
                "field_db_y": to_numpy(to_power_db(result_y.primary.field.total)).reshape(
                    params["grid_size"],
                    params["grid_size"],
                ),
                "reflection_field_db": to_numpy(to_power_db(result.primary.field.reflection)).reshape(
                    params["grid_size"],
                    params["grid_size"],
                ),
                "diffraction_field_db": to_numpy(to_power_db(result.primary.field.diffraction)).reshape(
                    params["grid_size"],
                    params["grid_size"],
                ),
                "multipath_phase_grid": _phase_grid(
                    multipath_phase_complex,
                    grid_size=params["grid_size"],
                ),
            }
        )

    pos_result, pos_scene, pos_ad = _position_ad_gradient(**params)
    pos_fd = _position_fd_gradient(**params)
    rot_result, rot_scene, rot_ad = _rotation_ad_gradient(**params)
    rot_fd = _rotation_fd_gradient(**params)
    mat_ref_result, _, mat_ref_grad_complex = _material_ad_gradient(
        field_mode="reflection",
        return_complex=True,
        **params,
    )
    mat_ref_ad = np.abs(mat_ref_grad_complex)
    mat_dif_result, _, mat_dif_grad_complex = _material_ad_gradient(
        field_mode="diffraction",
        return_complex=True,
        **params,
    )
    mat_dif_ad = np.abs(mat_dif_grad_complex)
    mat_result, mat_scene, mat_ad = _material_ad_gradient(field_mode="multipath", **params)
    mat_fd_components = _material_fd_gradients(return_complex=True, **params)
    mat_ref_fd_complex = mat_fd_components["reflection"]
    mat_dif_fd_complex = mat_fd_components["diffraction"]
    mat_fd_complex = mat_fd_components["multipath"]
    mat_ref_fd = np.abs(mat_ref_fd_complex)
    mat_dif_fd = np.abs(mat_dif_fd_complex)
    mat_fd = np.abs(mat_fd_complex)
    material_metadata = dict(mat_result.primary.metadata)
    polarization_transport = dict(material_metadata.get("polarization_transport", {}))
    reflection_scalarization = str(
        polarization_transport.get("reflection_scalarization", "unknown")
    )
    diffraction_face_scalarization = str(
        polarization_transport.get("diffraction_face_scalarization", "unknown")
    )
    mat_ref_field = _field_to_complex_numpy(mat_ref_result.primary.field.reflection)
    mat_dif_field = _field_to_complex_numpy(mat_dif_result.primary.field.diffraction)
    mat_ref_components = _complex_gradient_components(
        mat_ref_field,
        mat_ref_grad_complex,
        grid_size=params["grid_size"],
    )
    mat_dif_components = _complex_gradient_components(
        mat_dif_field,
        mat_dif_grad_complex,
        grid_size=params["grid_size"],
    )
    legacy_ref_plus_result, legacy_ref_scene = _trace_scene(
        center=wt.Point3f(*map(float, params["center_vals"])),
        size=params["size"],
        freq=params["freq"],
        tx_pos=wt.Point3f(*map(float, params["tx_vals"])),
        range_x=params["range_x"],
        range_y=params["range_y"],
        grid_size=params["grid_size"],
        n_rays=params["n_rays"],
        max_reflections=params["max_reflections"],
        reflection_coef=params["reflection_coef"],
        material=Material(eps_r=1.0, sigma_e=0.0),
        rotation=wt.Float(float(params["rotation_val"])),
        max_diffractions=0,
        enable_rd_diffraction=False,
        reflection_material_override=_uniform_override_material_dict(
            eps_r=float(params["baseline_eps_r"]) + MATERIAL_FD_DELTA,
            gain=params["reflection_coef"],
        ),
    )
    legacy_ref_minus_result, _ = _trace_scene(
        center=wt.Point3f(*map(float, params["center_vals"])),
        size=params["size"],
        freq=params["freq"],
        tx_pos=wt.Point3f(*map(float, params["tx_vals"])),
        range_x=params["range_x"],
        range_y=params["range_y"],
        grid_size=params["grid_size"],
        n_rays=params["n_rays"],
        max_reflections=params["max_reflections"],
        reflection_coef=params["reflection_coef"],
        material=Material(eps_r=1.0, sigma_e=0.0),
        rotation=wt.Float(float(params["rotation_val"])),
        max_diffractions=0,
        enable_rd_diffraction=False,
        reflection_material_override=_uniform_override_material_dict(
            eps_r=float(params["baseline_eps_r"]) - MATERIAL_FD_DELTA,
            gain=params["reflection_coef"],
        ),
    )
    legacy_ref_fd_complex = _complex_field_fd_gradient(
        legacy_ref_plus_result,
        legacy_ref_minus_result,
        MATERIAL_FD_DELTA,
        field_mode="reflection",
    )
    legacy_ref_fd = np.abs(legacy_ref_fd_complex)

    tx_vals = params["tx_vals"]
    range_x = params["range_x"]
    range_y = params["range_y"]
    grid_size = params["grid_size"]
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]

    field_vmin, field_vmax = -90.0, -30.0
    pos_ad_db = _to_db(pos_ad.reshape(grid_size, grid_size))
    pos_fd_db = _to_db(pos_fd.reshape(grid_size, grid_size))
    rot_ad_db = _to_db(rot_ad.reshape(grid_size, grid_size))
    rot_fd_db = _to_db(rot_fd.reshape(grid_size, grid_size))
    mat_ad_db = _to_db(mat_ad.reshape(grid_size, grid_size))
    mat_fd_db = _to_db(mat_fd.reshape(grid_size, grid_size))
    mat_dif_ad_db = _to_db(mat_dif_ad.reshape(grid_size, grid_size))
    mat_dif_fd_db = _to_db(mat_dif_fd.reshape(grid_size, grid_size))
    mat_ref_ad_db = _to_db(mat_ref_ad.reshape(grid_size, grid_size))
    mat_dif_grad_re = np.real(mat_dif_components["gradient"])
    mat_dif_grad_im = np.imag(mat_dif_components["gradient"])
    mat_dif_zero_cross_re = _zero_crossing_mask(mat_dif_grad_re)
    mat_dif_zero_cross_im = _zero_crossing_mask(mat_dif_grad_im)
    mat_dif_zero_cross_union = mat_dif_zero_cross_re | mat_dif_zero_cross_im
    mat_dif_zero_cross_both = mat_dif_zero_cross_re & mat_dif_zero_cross_im
    mat_dif_signed_limit = _signed_percentile_limit(mat_dif_grad_re, mat_dif_grad_im, percentile=99.5)
    mat_ref_logamp = mat_ref_components["logamp_derivative"]
    mat_ref_phase = mat_ref_components["phase_derivative"]
    mat_ref_logamp_limit = _signed_percentile_limit(mat_ref_logamp, percentile=99.5)
    mat_ref_phase_limit = _signed_percentile_limit(mat_ref_phase, percentile=99.5)

    mat_ad_ref_to_dif_ratio = float(np.sum(mat_ref_ad)) / max(float(np.sum(mat_dif_ad)), 1e-12)
    mat_fd_ref_to_dif_ratio = float(np.sum(mat_ref_fd)) / max(float(np.sum(mat_dif_fd)), 1e-12)
    mat_ad_diff_dominant_fraction = float(np.mean(mat_dif_ad > mat_ref_ad))
    mat_fd_diff_dominant_fraction = float(np.mean(mat_dif_fd > mat_ref_fd))
    mat_rel_l2_error = _relative_l2_error(mat_ad, mat_fd)
    mat_ref_rel_l2_error = _relative_l2_error(mat_ref_ad, mat_ref_fd)
    mat_dif_rel_l2_error = _relative_l2_error(mat_dif_ad, mat_dif_fd)
    mat_ref_phase_stats = _complex_gradient_phase_diagnostics(
        mat_ref_field,
        mat_ref_grad_complex,
        grid_size=grid_size,
    )
    mat_dif_phase_stats = _complex_gradient_phase_diagnostics(
        mat_dif_field,
        mat_dif_grad_complex,
        grid_size=grid_size,
    )
    mat_ref_to_dif_p99_ratio = mat_ref_phase_stats["gradient_p99"] / max(
        mat_dif_phase_stats["gradient_p99"],
        1e-12,
    )
    mat_ref_to_dif_p999_ratio = mat_ref_phase_stats["gradient_p999"] / max(
        mat_dif_phase_stats["gradient_p999"],
        1e-12,
    )
    mat_ref_logamp_p99 = float(np.percentile(np.abs(mat_ref_logamp), 99.0))
    mat_dif_zero_cross_fraction = float(np.mean(mat_dif_zero_cross_union))
    mat_dif_zero_cross_both_fraction = float(np.mean(mat_dif_zero_cross_both))
    legacy_ref_rel_l2_error = _relative_l2_error(mat_ref_fd, legacy_ref_fd)
    legacy_ref_mag_ratio = float(np.sum(mat_ref_fd)) / max(float(np.sum(legacy_ref_fd)), 1e-12)
    legacy_ref_delta = mat_ref_fd.reshape(grid_size, grid_size) - legacy_ref_fd.reshape(grid_size, grid_size)
    legacy_ref_delta_limit = _signed_percentile_limit(legacy_ref_delta, percentile=99.9)
    reflection_support_threshold = max(float(np.percentile(np.abs(mat_ref_field[np.abs(mat_ref_field) > 0.0]), 10.0)) * 1e-3, 1e-16)
    reflection_support_mask = np.abs(mat_ref_field).reshape(grid_size, grid_size) > reflection_support_threshold
    reflection_support_edge = _binary_boundary_mask(reflection_support_mask)
    reflection_specular_target = mat_ref_fd.reshape(grid_size, grid_size) * reflection_support_edge.astype(np.float64)
    reflection_specular_index = int(np.argmax(reflection_specular_target))
    reflection_specular_row, reflection_specular_col = np.unravel_index(reflection_specular_index, reflection_support_edge.shape)
    row_y = np.linspace(range_y[0], range_y[1], grid_size)[reflection_specular_row]
    row_x = np.linspace(range_x[0], range_x[1], grid_size)
    reflection_row_scene = mat_ref_fd.reshape(grid_size, grid_size)[reflection_specular_row, :]
    reflection_row_legacy = legacy_ref_fd.reshape(grid_size, grid_size)[reflection_specular_row, :]
    reflection_row_support = reflection_support_edge[reflection_specular_row, :]
    reflection_row_edge_x = row_x[reflection_row_support]

    diffraction_field_vmin, diffraction_field_vmax = -90.0, -30.0
    phase_vmin, phase_vmax = -np.pi, np.pi
    phase_ticks = [-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi]
    phase_ticklabels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
    n_field_cols = len(field_payloads)
    fig, axes = plt.subplots(10, n_field_cols, figsize=(5.6 * n_field_cols, 37), constrained_layout=False)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.958, bottom=0.03, hspace=0.22, wspace=0.12)

    for col_idx, payload in enumerate(field_payloads):
        ax = axes[0, col_idx]
        image = ax.imshow(
            payload["field_db"],
            extent=extent,
            origin="lower",
            cmap="inferno",
            vmin=field_vmin,
            vmax=field_vmax,
        )
        edges = payload["scene"].get_edge_data(payload["result"].primary.plane_position)["edges_2d"]
        draw_scene(ax, edges, tx_vals, range_x, range_y)
        los_ratio = payload["los_power"] / max(payload["total_power"], 1e-12)
        mp_ratio = payload["multipath_power"] / max(payload["total_power"], 1e-12)
        ax.set_title(
            "Scalar Total Field (Tx=Ex) (dB)\n"
            f"eps_r={_format_eps_r(payload['eps_r'])}",
            fontsize=11,
        )
        ax.text(
            0.02,
            0.02,
            f"P_los/P_tot={los_ratio:.2f}\nP_mp/P_tot={mp_ratio:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            color="white",
            va="bottom",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
        )
        plt.colorbar(image, ax=ax, shrink=0.84)

    for col_idx, payload in enumerate(field_payloads):
        ax = axes[1, col_idx]
        image = ax.imshow(
            payload["field_db_y"],
            extent=extent,
            origin="lower",
            cmap="inferno",
            vmin=field_vmin,
            vmax=field_vmax,
        )
        edges = payload["scene"].get_edge_data(payload["result"].primary.plane_position)["edges_2d"]
        draw_scene(ax, edges, tx_vals, range_x, range_y)
        y_los_ratio = payload["los_power_y"] / max(payload["total_power_y"], 1e-12)
        ax.set_title(
            "Scalar Total Field (Tx=Ey) (dB)\n"
            f"eps_r={_format_eps_r(payload['eps_r'])}",
            fontsize=11,
        )
        ax.text(
            0.02,
            0.02,
            f"P_los,Ey/P_tot,Ey={y_los_ratio:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            color="white",
            va="bottom",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
        )
        plt.colorbar(image, ax=ax, shrink=0.84)

    for col_idx, payload in enumerate(field_payloads):
        ax = axes[2, col_idx]
        image = ax.imshow(
            payload["dual_pol_scalar_total_power_db"],
            extent=extent,
            origin="lower",
            cmap="inferno",
            vmin=field_vmin,
            vmax=field_vmax,
        )
        edges = payload["scene"].get_edge_data(payload["result"].primary.plane_position)["edges_2d"]
        draw_scene(ax, edges, tx_vals, range_x, range_y)
        x_ratio = payload["total_power"] / max(payload["dual_pol_scalar_total_power"], 1e-12)
        y_ratio = payload["total_power_y"] / max(payload["dual_pol_scalar_total_power"], 1e-12)
        ax.set_title(
            "Dual-Pol Scalar Total Power (dB)\n"
            f"|E_tot,Tx=Ex|^2 + |E_tot,Tx=Ey|^2, eps_r={_format_eps_r(payload['eps_r'])}",
            fontsize=11,
        )
        ax.text(
            0.02,
            0.02,
            f"Ex share={x_ratio:.2f}\nEy share={y_ratio:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            color="white",
            va="bottom",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
        )
        plt.colorbar(image, ax=ax, shrink=0.84)

    for col_idx, payload in enumerate(field_payloads):
        ax = axes[3, col_idx]
        image = ax.imshow(
            payload["reflection_field_db"],
            extent=extent,
            origin="lower",
            cmap="inferno",
            vmin=field_vmin,
            vmax=field_vmax,
        )
        edges = payload["scene"].get_edge_data(payload["result"].primary.plane_position)["edges_2d"]
        draw_scene(ax, edges, tx_vals, range_x, range_y)
        ratio = payload["reflection_power"] / max(payload["diffraction_power"], 1e-12)
        ref_ratio = payload["reflection_power"] / max(payload["total_power"], 1e-12)
        ax.set_title(
            "Scalar Reflection Field (Tx=Ex) (dB)\n"
            f"eps_r={_format_eps_r(payload['eps_r'])}",
            fontsize=11,
        )
        ax.text(
            0.02,
            0.02,
            f"P_ref/P_tot={ref_ratio:.2f}\nP_ref/P_dif={ratio:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            color="white",
            va="bottom",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
        )
        plt.colorbar(image, ax=ax, shrink=0.84)

    for col_idx, payload in enumerate(field_payloads):
        ax = axes[4, col_idx]
        image = ax.imshow(
            payload["diffraction_field_db"],
            extent=extent,
            origin="lower",
            cmap="inferno",
            vmin=diffraction_field_vmin,
            vmax=diffraction_field_vmax,
        )
        edges = payload["scene"].get_edge_data(payload["result"].primary.plane_position)["edges_2d"]
        draw_scene(ax, edges, tx_vals, range_x, range_y)
        dif_ratio = payload["diffraction_power"] / max(payload["total_power"], 1e-12)
        dif_mp_ratio = payload["diffraction_power"] / max(payload["multipath_power"], 1e-12)
        ax.set_title(
            "Scalar Diffraction Field (Tx=Ex) (dB)\n"
            f"eps_r={_format_eps_r(payload['eps_r'])}",
            fontsize=11,
        )
        ax.text(
            0.02,
            0.02,
            f"P_dif/P_tot={dif_ratio:.2f}\nP_dif/P_mp={dif_mp_ratio:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            color="white",
            va="bottom",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
        )
        plt.colorbar(image, ax=ax, shrink=0.84)

    for col_idx, payload in enumerate(field_payloads):
        ax = axes[5, col_idx]
        image = ax.imshow(
            payload["multipath_phase_grid"],
            extent=extent,
            origin="lower",
            cmap="twilight",
            vmin=phase_vmin,
            vmax=phase_vmax,
        )
        edges = payload["scene"].get_edge_data(payload["result"].primary.plane_position)["edges_2d"]
        draw_scene(ax, edges, tx_vals, range_x, range_y)
        ax.set_title(
            "Phase(Reflection + Diffraction) (Tx=Ex)\n"
            f"eps_r={_format_eps_r(payload['eps_r'])}",
            fontsize=11,
        )
        ax.text(
            0.02,
            0.02,
            "Wrapped phase of scalar multipath field",
            transform=ax.transAxes,
            fontsize=8,
            color="white",
            va="bottom",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
        )
        colorbar = plt.colorbar(image, ax=ax, shrink=0.84)
        colorbar.set_ticks(phase_ticks)
        colorbar.set_ticklabels(phase_ticklabels)

    gradient_panels = (
        {
            "scene": pos_scene,
            "result": pos_result,
            "ad_db": pos_ad_db,
            "fd_db": pos_fd_db,
            "ad_title": f"Position Gradient AD (dB)\n|dE_mp / d center_x|, baseline eps_r={baseline_eps_label}",
            "fd_title": f"Position Gradient FD (dB)\n|dE_mp / d center_x|, baseline eps_r={baseline_eps_label}",
        },
        {
            "scene": rot_scene,
            "result": rot_result,
            "ad_db": rot_ad_db,
            "fd_db": rot_fd_db,
            "ad_title": f"Rotation Gradient AD (dB)\n|dE_mp / d rotation_z|, baseline eps_r={baseline_eps_label}",
            "fd_title": f"Rotation Gradient FD (dB)\n|dE_mp / d rotation_z|, baseline eps_r={baseline_eps_label}",
        },
        {
            "scene": mat_scene,
            "result": mat_result,
            "ad_db": mat_ad_db,
            "fd_db": mat_fd_db,
            "ad_title": (
                "Material Gradient AD (dB)\n"
                f"|dE_mp / d eps_r|, baseline eps_r={baseline_eps_label}\n"
                "Default scalar Fresnel field"
            ),
            "fd_title": (
                "Material Gradient FD (dB)\n"
                f"|dE_mp / d eps_r|, baseline eps_r={baseline_eps_label}\n"
                "Default scalar Fresnel field"
            ),
        },
    )
    for panel_idx, panel in enumerate(gradient_panels):
        panel_vmax = max(
            float(np.percentile(panel["ad_db"], 99.5)),
            float(np.percentile(panel["fd_db"], 99.5)),
        )
        panel_vmin = -90.0
        edges = panel["scene"].get_edge_data(panel["result"].primary.plane_position)["edges_2d"]

        ad_ax = axes[6, panel_idx]
        ad_image = ad_ax.imshow(
            panel["ad_db"],
            extent=extent,
            origin="lower",
            cmap="RdBu_r",
            vmin=panel_vmin,
            vmax=panel_vmax,
        )
        draw_scene(ad_ax, edges, tx_vals, range_x, range_y)
        ad_ax.set_title(panel["ad_title"], fontsize=11)
        if panel_idx == 2:
            ad_ax.text(
                0.02,
                0.02,
                f"sum|dE_ref| / sum|dE_dif|: AD {mat_ad_ref_to_dif_ratio:.2f}, FD {mat_fd_ref_to_dif_ratio:.2f}\n"
                f"rel-L2 total={mat_rel_l2_error:.2e}, ref={mat_ref_rel_l2_error:.2e}, dif={mat_dif_rel_l2_error:.2e}\n"
                f"reflection scalarization={reflection_scalarization}",
                transform=ad_ax.transAxes,
                fontsize=8,
                color="white",
                va="bottom",
                ha="left",
                bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
            )
        plt.colorbar(ad_image, ax=ad_ax, shrink=0.84)

        fd_ax = axes[7, panel_idx]
        fd_image = fd_ax.imshow(
            panel["fd_db"],
            extent=extent,
            origin="lower",
            cmap="RdBu_r",
            vmin=panel_vmin,
            vmax=panel_vmax,
        )
        draw_scene(fd_ax, edges, tx_vals, range_x, range_y)
        fd_ax.set_title(panel["fd_title"], fontsize=11)
        if panel_idx == 2:
            fd_ax.text(
                0.02,
                0.02,
                f"pixels(|dE_dif|>|dE_ref|): AD {mat_ad_diff_dominant_fraction:.0%}, FD {mat_fd_diff_dominant_fraction:.0%}\n"
                f"FD ground truth uses central difference with delta={MATERIAL_FD_DELTA:.2e}\n"
                f"diffraction scalarization={diffraction_face_scalarization}",
                transform=fd_ax.transAxes,
                fontsize=8,
                color="white",
                va="bottom",
                ha="left",
                bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
            )
        plt.colorbar(fd_image, ax=fd_ax, shrink=0.84)

    diffraction_vmax = max(
        float(np.percentile(mat_dif_ad_db, 99.5)),
        float(np.percentile(mat_dif_fd_db, 99.5)),
    )
    diffraction_vmin = -90.0
    diffraction_edges = mat_scene.get_edge_data(mat_result.primary.plane_position)["edges_2d"]
    diffraction_panels = (
        {
            "ax": axes[8, 0],
            "image_db": mat_dif_ad_db,
            "title": f"Diffraction Gradient AD (dB)\n|dE_dif / d eps_r|, baseline eps_r={baseline_eps_label}",
            "note": f"rel-L2(AD, FD)={mat_dif_rel_l2_error:.2e}",
        },
        {
            "ax": axes[8, 1],
            "image_db": mat_dif_fd_db,
            "title": f"Diffraction Gradient FD (dB)\n|dE_dif / d eps_r|, baseline eps_r={baseline_eps_label}",
            "note": f"central difference delta={MATERIAL_FD_DELTA:.2e}",
        },
    )
    for panel in diffraction_panels:
        image = panel["ax"].imshow(
            panel["image_db"],
            extent=extent,
            origin="lower",
            cmap="RdBu_r",
            vmin=diffraction_vmin,
            vmax=diffraction_vmax,
        )
        draw_scene(panel["ax"], diffraction_edges, tx_vals, range_x, range_y)
        panel["ax"].set_title(panel["title"], fontsize=11)
        panel["ax"].text(
            0.02,
            0.02,
            panel["note"],
            transform=panel["ax"].transAxes,
            fontsize=8,
            color="white",
            va="bottom",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
        )
        plt.colorbar(image, ax=panel["ax"], shrink=0.84)

    summary_ax = axes[8, 2]
    summary_ax.axis("off")
    summary_ax.set_title("Diffraction Gradient Summary", fontsize=11)
    summary_ax.text(
        0.05,
        0.95,
        "Integrated into material.png for direct comparison with the multipath material panels.\n"
        f"reflection scalarization={reflection_scalarization}\n"
        f"diffraction scalarization={diffraction_face_scalarization}\n"
        f"rel-L2(AD, FD)={mat_dif_rel_l2_error:.2e}\n"
        f"central difference delta={MATERIAL_FD_DELTA:.2e}\n"
        f"p99 |dE_ref| / |dE_dif| = {mat_ref_to_dif_p99_ratio:.2f}\n"
        f"p99.9 |dE_ref| / |dE_dif| = {mat_ref_to_dif_p999_ratio:.2f}\n"
        f"phase-flip neighbors (>0.9pi): ref {mat_ref_phase_stats['strong_phase_flip_fraction']:.02%}, "
        f"dif {mat_dif_phase_stats['strong_phase_flip_fraction']:.02%}\n"
        f"top 0.1% dB jumps near zero: ref {mat_ref_phase_stats['top_jump_near_zero_fraction']:.0%}, "
        f"dif {mat_dif_phase_stats['top_jump_near_zero_fraction']:.0%}\n"
        f"zero-cross pixels (Re or Im): {mat_dif_zero_cross_fraction:.0%}, both: {mat_dif_zero_cross_both_fraction:.0%}\n"
        f"p99 |d phase / d eps_r|: ref {mat_ref_phase_stats['phase_derivative_p99']:.2e}, "
        f"dif {mat_dif_phase_stats['phase_derivative_p99']:.2e}",
        transform=summary_ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "black", "pad": 6.0},
    )

    signed_diffraction_panels = (
        {
            "ax": axes[9, 0],
            "values": mat_dif_grad_re,
            "title": "Signed Re(dE_dif / d eps_r) AD",
            "contour_color": "black",
            "note": "black contour = Re zero crossing",
        },
        {
            "ax": axes[9, 1],
            "values": mat_dif_grad_im,
            "title": "Signed Im(dE_dif / d eps_r) AD",
            "contour_color": "black",
            "note": "black contour = Im zero crossing",
        },
    )
    for panel in signed_diffraction_panels:
        image = panel["ax"].imshow(
            panel["values"],
            extent=extent,
            origin="lower",
            cmap="coolwarm",
            vmin=-mat_dif_signed_limit,
            vmax=mat_dif_signed_limit,
        )
        draw_scene(panel["ax"], diffraction_edges, tx_vals, range_x, range_y)
        _draw_zero_contour(
            panel["ax"],
            panel["values"],
            extent=extent,
            color=panel["contour_color"],
        )
        panel["ax"].set_title(panel["title"], fontsize=11)
        panel["ax"].text(
            0.02,
            0.02,
            panel["note"],
            transform=panel["ax"].transAxes,
            fontsize=8,
            color="white",
            va="bottom",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
        )
        plt.colorbar(image, ax=panel["ax"], shrink=0.84)

    zero_cross_ax = axes[9, 2]
    zero_cross_image = zero_cross_ax.imshow(
        mat_dif_zero_cross_union.astype(np.float64),
        extent=extent,
        origin="lower",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
    )
    draw_scene(zero_cross_ax, diffraction_edges, tx_vals, range_x, range_y)
    _draw_zero_contour(zero_cross_ax, mat_dif_grad_re, extent=extent, color="red")
    _draw_zero_contour(zero_cross_ax, mat_dif_grad_im, extent=extent, color="cyan")
    zero_cross_ax.set_title("Diffraction Zero-Crossing Lines", fontsize=11)
    zero_cross_ax.text(
        0.02,
        0.02,
        f"union={mat_dif_zero_cross_fraction:.0%}, both={mat_dif_zero_cross_both_fraction:.0%}\n"
        "red=Re zero, cyan=Im zero",
        transform=zero_cross_ax.transAxes,
        fontsize=8,
        color="white",
        va="bottom",
        ha="left",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
    )
    plt.colorbar(zero_cross_image, ax=zero_cross_ax, shrink=0.84)

    for row_idx, used_cols in ((6, 3), (7, 3), (8, 3), (9, 3)):
        for col_idx in range(used_cols, n_field_cols):
            axes[row_idx, col_idx].axis("off")

    fig.suptitle(
        "Material Visual Test: Top Rows Compare The Default Tx=Ex Scalar Total Field, A Separate Tx=Ey Scalar Total Field, An Incoherent Ex-plus-Ey Dual-Pol Scalar Total Power, Plus Scalar Reflection, Scalar Diffraction, And The Wrapped Phase Of Reflection Plus Diffraction For Tx=Ex; Lower Rows Keep AD/FD Gradient Comparisons And Diffraction Diagnostics\n"
        "Material gradients are injected through runtime triangle material buffers because witwin.core.Material stores scalar values",
        fontsize=14,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)

    reflection_diag_fig, reflection_diag_axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.5),
        constrained_layout=True,
    )
    reflection_panels = (
        {
            "ax": reflection_diag_axes[0],
            "values": mat_ref_ad_db,
            "cmap": "inferno",
            "vmin": -90.0,
            "vmax": float(np.percentile(mat_ref_ad_db, 99.5)),
            "title": "Reflection Gradient Magnitude (dB)\n|dE_ref / d eps_r|",
            "note": f"p99 |dE_ref| / |dE_dif|={mat_ref_to_dif_p99_ratio:.2f}",
        },
        {
            "ax": reflection_diag_axes[1],
            "values": mat_ref_logamp,
            "cmap": "coolwarm",
            "vmin": -mat_ref_logamp_limit,
            "vmax": mat_ref_logamp_limit,
            "title": "Reflection Signed d log|E_ref| / d eps_r",
            "note": f"p99 |d log|E||={mat_ref_logamp_p99:.2e}",
        },
        {
            "ax": reflection_diag_axes[2],
            "values": mat_ref_phase,
            "cmap": "coolwarm",
            "vmin": -mat_ref_phase_limit,
            "vmax": mat_ref_phase_limit,
            "title": "Reflection Signed d phase(E_ref) / d eps_r",
            "note": f"p99 |d phase|={mat_ref_phase_stats['phase_derivative_p99']:.2e}",
        },
    )
    reflection_edges = mat_scene.get_edge_data(mat_result.primary.plane_position)["edges_2d"]
    for panel in reflection_panels:
        image = panel["ax"].imshow(
            panel["values"],
            extent=extent,
            origin="lower",
            cmap=panel["cmap"],
            vmin=panel["vmin"],
            vmax=panel["vmax"],
        )
        draw_scene(panel["ax"], reflection_edges, tx_vals, range_x, range_y)
        panel["ax"].set_title(panel["title"], fontsize=11)
        panel["ax"].text(
            0.02,
            0.02,
            panel["note"],
            transform=panel["ax"].transAxes,
            fontsize=8,
            color="white",
            va="bottom",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
        )
        plt.colorbar(image, ax=panel["ax"], shrink=0.84)
    reflection_diag_fig.suptitle(
        "Reflection-Only Material Diagnostics: The Specular Ridge Is Amplitude-Dominated, Not Phase-Dominated",
        fontsize=14,
    )
    reflection_diag_fig.savefig(REFLECTION_PHASE_OUTPUT_PATH, dpi=180)
    plt.close(reflection_diag_fig)

    reflection_specular_fig, reflection_specular_axes = plt.subplots(
        2,
        2,
        figsize=(14, 11),
        constrained_layout=True,
    )
    reflection_specular_vmax = max(
        float(np.percentile(mat_ref_ad_db, 99.5)),
        float(np.percentile(_to_db(legacy_ref_fd.reshape(grid_size, grid_size)), 99.5)),
    )
    reflection_specular_vmin = -90.0
    legacy_ref_fd_db = _to_db(legacy_ref_fd.reshape(grid_size, grid_size))

    current_ax = reflection_specular_axes[0, 0]
    current_image = current_ax.imshow(
        mat_ref_ad_db,
        extent=extent,
        origin="lower",
        cmap="inferno",
        vmin=reflection_specular_vmin,
        vmax=reflection_specular_vmax,
    )
    draw_scene(current_ax, reflection_edges, tx_vals, range_x, range_y)
    _draw_mask_contour(current_ax, reflection_support_edge, extent=extent, color="cyan")
    current_ax.axhline(row_y, color="white", linewidth=0.8, linestyle="--")
    current_ax.set_title("Current Scene Material Table\nReflection |dE_ref / d eps_r|", fontsize=11)
    current_ax.text(
        0.02,
        0.02,
        "cyan contour = support edge\nwhite dashed line = sampled row",
        transform=current_ax.transAxes,
        fontsize=8,
        color="white",
        va="bottom",
        ha="left",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
    )
    plt.colorbar(current_image, ax=current_ax, shrink=0.84)

    legacy_ax = reflection_specular_axes[0, 1]
    legacy_image = legacy_ax.imshow(
        legacy_ref_fd_db,
        extent=extent,
        origin="lower",
        cmap="inferno",
        vmin=reflection_specular_vmin,
        vmax=reflection_specular_vmax,
    )
    draw_scene(legacy_ax, legacy_ref_scene.get_edge_data(legacy_ref_plus_result.primary.plane_position)["edges_2d"], tx_vals, range_x, range_y)
    _draw_mask_contour(legacy_ax, reflection_support_edge, extent=extent, color="cyan")
    legacy_ax.axhline(row_y, color="white", linewidth=0.8, linestyle="--")
    legacy_ax.set_title("Legacy Explicit Override (pre-abfb8fb path)\nReflection |dE_ref / d eps_r|", fontsize=11)
    legacy_ax.text(
        0.02,
        0.02,
        f"scene-vs-legacy rel-L2={legacy_ref_rel_l2_error:.2e}\n"
        f"sum ratio={legacy_ref_mag_ratio:.6f}",
        transform=legacy_ax.transAxes,
        fontsize=8,
        color="white",
        va="bottom",
        ha="left",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
    )
    plt.colorbar(legacy_image, ax=legacy_ax, shrink=0.84)

    diff_ax = reflection_specular_axes[1, 0]
    diff_image = diff_ax.imshow(
        legacy_ref_delta,
        extent=extent,
        origin="lower",
        cmap="coolwarm",
        vmin=-legacy_ref_delta_limit,
        vmax=legacy_ref_delta_limit,
    )
    draw_scene(diff_ax, reflection_edges, tx_vals, range_x, range_y)
    _draw_mask_contour(diff_ax, reflection_support_edge, extent=extent, color="black")
    diff_ax.axhline(row_y, color="black", linewidth=0.8, linestyle="--")
    diff_ax.set_title("Current Minus Legacy\nReflection |dE_ref / d eps_r|", fontsize=11)
    diff_ax.text(
        0.02,
        0.02,
        "Difference stays tiny even on the support edge.",
        transform=diff_ax.transAxes,
        fontsize=8,
        color="white",
        va="bottom",
        ha="left",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
    )
    plt.colorbar(diff_image, ax=diff_ax, shrink=0.84)

    profile_ax = reflection_specular_axes[1, 1]
    profile_ax.plot(row_x, reflection_row_scene, label="Current scene material", linewidth=1.6)
    profile_ax.plot(row_x, reflection_row_legacy, label="Legacy override", linewidth=1.2, linestyle="--")
    for edge_x in reflection_row_edge_x:
        profile_ax.axvline(edge_x, color="cyan", linewidth=0.6, alpha=0.5)
    profile_ax.set_title(f"Specular-Edge Row Profile at y={row_y:.2f}", fontsize=11)
    profile_ax.set_xlabel("x")
    profile_ax.set_ylabel("|dE_ref / d eps_r|")
    profile_ax.grid(True, alpha=0.25)
    profile_ax.legend(loc="upper right", fontsize=8)
    profile_ax.text(
        0.02,
        0.02,
        "cyan lines = support-edge crossings",
        transform=profile_ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "black", "pad": 3.0},
    )

    reflection_specular_fig.suptitle(
        "Specular-Edge Comparison: Current Scene Material vs Legacy Explicit Override",
        fontsize=14,
    )
    reflection_specular_fig.savefig(REFLECTION_SPECULAR_OUTPUT_PATH, dpi=180)
    plt.close(reflection_specular_fig)

    assert OUTPUT_PATH.exists()
    assert OUTPUT_PATH.stat().st_size > 0
    assert REFLECTION_PHASE_OUTPUT_PATH.exists()
    assert REFLECTION_PHASE_OUTPUT_PATH.stat().st_size > 0
    assert REFLECTION_SPECULAR_OUTPUT_PATH.exists()
    assert REFLECTION_SPECULAR_OUTPUT_PATH.stat().st_size > 0
    for lhs, rhs in zip(field_payloads, field_payloads[1:]):
        assert _field_delta_power(lhs["field"], rhs["field"]) > 1e-6
    assert all(payload["reflection_power"] > 0.0 for payload in field_payloads)
    assert all(payload["diffraction_power"] > 0.0 for payload in field_payloads)
    assert all(payload["dual_pol_scalar_total_power"] >= payload["total_power"] for payload in field_payloads)
    assert all(payload["vector_total_power"] >= payload["vector_xy_power"] for payload in field_payloads)
    assert all(payload["vector_xy_power"] >= payload["vector_x_power"] for payload in field_payloads)
    assert all(payload["vector_xy_power"] >= payload["vector_y_power"] for payload in field_payloads)
    assert all(payload["vector_z_power"] >= 0.0 for payload in field_payloads)
    assert all(payload["vector_xy_rel_error"] < 5e-4 for payload in field_payloads)
    assert float(np.sum(pos_ad)) > 0.0
    assert float(np.sum(pos_fd)) > 0.0
    assert float(np.sum(rot_ad)) > 0.0
    assert float(np.sum(rot_fd)) > 0.0
    assert float(np.sum(mat_ad)) > 0.0
    assert float(np.sum(mat_fd)) > 0.0
    assert float(np.sum(mat_dif_ad)) > 0.0
    assert float(np.sum(mat_ref_ad)) > 0.0
    assert float(np.sum(mat_dif_fd)) > 0.0
    assert float(np.sum(mat_ref_fd)) > 0.0
    assert reflection_scalarization == "default_receiver_projection_from_jones"
    assert diffraction_face_scalarization == "default_receiver_projection_from_jones_face_operator"
    assert mat_rel_l2_error < 2e-2
    assert mat_ref_rel_l2_error < 2e-2
    assert mat_dif_rel_l2_error < 2e-2
    if float(params["baseline_eps_r"]) < 100.0:
        assert mat_ref_phase_stats["gradient_p99"] > mat_dif_phase_stats["gradient_p99"]
        assert mat_ref_phase_stats["gradient_p999"] > mat_dif_phase_stats["gradient_p999"]
        assert mat_dif_phase_stats["db_jump_p99"] > mat_ref_phase_stats["db_jump_p99"]
        assert (
            mat_dif_phase_stats["strong_phase_flip_fraction"]
            > mat_ref_phase_stats["strong_phase_flip_fraction"]
        )
        assert mat_ref_phase_stats["top_jump_near_zero_fraction"] > 0.95
        assert mat_dif_phase_stats["top_jump_near_zero_fraction"] > 0.60
        assert mat_ref_phase_stats["top_jump_phase_flip_fraction"] < 0.25
        assert mat_dif_phase_stats["top_jump_phase_flip_fraction"] < 0.25
        assert mat_ref_phase_stats["phase_derivative_p99"] < 1e-4
        assert mat_dif_phase_stats["phase_derivative_p99"] > 1e-2
        assert mat_dif_zero_cross_fraction > 0.05
    else:
        for value in (
            mat_ref_phase_stats["gradient_p99"],
            mat_ref_phase_stats["gradient_p999"],
            mat_dif_phase_stats["gradient_p99"],
            mat_dif_phase_stats["gradient_p999"],
            mat_ref_phase_stats["db_jump_p99"],
            mat_dif_phase_stats["db_jump_p99"],
            mat_ref_phase_stats["phase_derivative_p99"],
            mat_dif_phase_stats["phase_derivative_p99"],
            mat_dif_zero_cross_fraction,
            mat_dif_zero_cross_both_fraction,
        ):
            assert np.isfinite(value)
            assert value >= 0.0
        for value in (
            mat_ref_phase_stats["strong_phase_flip_fraction"],
            mat_dif_phase_stats["strong_phase_flip_fraction"],
            mat_ref_phase_stats["top_jump_near_zero_fraction"],
            mat_dif_phase_stats["top_jump_near_zero_fraction"],
            mat_ref_phase_stats["top_jump_phase_flip_fraction"],
            mat_dif_phase_stats["top_jump_phase_flip_fraction"],
        ):
            assert 0.0 <= value <= 1.0
    assert mat_ref_logamp_p99 > 1e3 * max(mat_ref_phase_stats["phase_derivative_p99"], 1e-12)
    assert legacy_ref_rel_l2_error < 2e-3
    assert abs(legacy_ref_mag_ratio - 1.0) < 5e-4
    assert float(np.max(np.abs(legacy_ref_delta))) < 1e-6
