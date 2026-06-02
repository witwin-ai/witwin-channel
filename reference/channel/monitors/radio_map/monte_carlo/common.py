from __future__ import annotations

import math
from typing import Mapping

import drjit as dr
import witwin as wt

from ....utils import scalar
from ....utils.constants import EPS, RAY_ORIGIN_BIAS
from ....utils.plane_axes import point_on_axis_aligned_plane
from ....utils.polarization import (
    implicit_basis_vector,
    vector_from_scalar_and_real_direction,
)


_MC_BATCH_ALIGN = 256
_MC_MIN_BATCH_MEMORY_BYTES = 16 * 1024 * 1024
_MC_RAY_BATCH_BUDGET_RATIO = 0.5
_MC_DIFFRACTION_BATCH_BUDGET_RATIO = 0.5
_MC_ESTIMATED_RAY_WORKING_SET_BYTES = 768
_MC_ESTIMATED_DIFFRACTION_SAMPLE_BYTES = 3072
_MC_UINT32_MASK = 0xFFFFFFFF
_MC_MIN_RR_PROBABILITY = 1.0e-8
_MC_DIFFRACTION_OFFSET = 5.0e-2


def _align_monte_carlo_batch_size(target: int, *, upper_bound: int) -> int:
    if upper_bound <= 0:
        return 0
    resolved_target = max(1, int(target))
    if resolved_target >= int(upper_bound):
        return int(upper_bound)
    if resolved_target < _MC_BATCH_ALIGN:
        return resolved_target
    return int(min(upper_bound, (resolved_target // _MC_BATCH_ALIGN) * _MC_BATCH_ALIGN))


def _phase_target_batch_size(
    *,
    samples_per_tx: int,
    cuda_memory_report: Mapping[str, object],
    budget_ratio: float,
    estimated_bytes_per_sample: int,
):
    if samples_per_tx <= 0:
        return 0, "disabled", 0
    if not bool(cuda_memory_report.get("available", False)):
        return int(samples_per_tx), "full_batch_no_cuda_memory_report", 0
    free_bytes = max(0, int(cuda_memory_report.get("free_bytes", 0)))
    if free_bytes <= 0:
        return int(samples_per_tx), "full_batch_no_free_memory_report", free_bytes
    budget_bytes = max(
        _MC_MIN_BATCH_MEMORY_BYTES,
        int(float(budget_ratio) * free_bytes),
    )
    if estimated_bytes_per_sample <= 0:
        return int(samples_per_tx), "full_batch_no_estimate", free_bytes
    target = max(1, budget_bytes // int(estimated_bytes_per_sample))
    if target >= int(samples_per_tx):
        return int(samples_per_tx), "full_batch_with_memory_guardrail", free_bytes
    return int(target), "cuda_free_memory_guardrail", free_bytes


def _resolve_monte_carlo_batch_plan(
    *,
    samples_per_tx: int,
    diffraction_state_count: int,
    cuda_memory_report: Mapping[str, object],
):
    ray_target, ray_reason, free_bytes = _phase_target_batch_size(
        samples_per_tx=samples_per_tx,
        cuda_memory_report=cuda_memory_report,
        budget_ratio=_MC_RAY_BATCH_BUDGET_RATIO,
        estimated_bytes_per_sample=_MC_ESTIMATED_RAY_WORKING_SET_BYTES,
    )
    ray_batch_size = _align_monte_carlo_batch_size(ray_target, upper_bound=samples_per_tx)
    ray_batch_count = 0 if ray_batch_size <= 0 else int(math.ceil(samples_per_tx / ray_batch_size))

    diffraction_enabled = int(diffraction_state_count) > 0
    if diffraction_enabled:
        diff_target, diff_reason, _ = _phase_target_batch_size(
            samples_per_tx=samples_per_tx,
            cuda_memory_report=cuda_memory_report,
            budget_ratio=_MC_DIFFRACTION_BATCH_BUDGET_RATIO,
            estimated_bytes_per_sample=_MC_ESTIMATED_DIFFRACTION_SAMPLE_BYTES,
        )
        diffraction_batch_size = _align_monte_carlo_batch_size(
            diff_target,
            upper_bound=samples_per_tx,
        )
        diffraction_batch_count = (
            0
            if diffraction_batch_size <= 0
            else int(math.ceil(samples_per_tx / diffraction_batch_size))
        )
    else:
        diff_reason = "disabled"
        diffraction_batch_size = 0
        diffraction_batch_count = 0

    return {
        "ray_batch_size": int(ray_batch_size),
        "ray_batch_count": int(ray_batch_count),
        "ray_policy": str(ray_reason),
        "diffraction_batch_size": int(diffraction_batch_size),
        "diffraction_batch_count": int(diffraction_batch_count),
        "diffraction_policy": str(diff_reason),
        "free_cuda_bytes": int(free_bytes),
        "scatter_safe_batch_cap": int(samples_per_tx),
        "ray_estimated_bytes_per_sample": int(_MC_ESTIMATED_RAY_WORKING_SET_BYTES),
        "diffraction_estimated_bytes_per_sample": int(_MC_ESTIMATED_DIFFRACTION_SAMPLE_BYTES),
    }


def _hash_uniform_uint32(index, *, stream: int, seed: int):
    resolved_seed = int(seed) & _MC_UINT32_MASK
    stream_value = wt.UInt32(stream) + wt.UInt32(1)
    value = (
        index * wt.UInt32(747796405)
        + wt.UInt32(resolved_seed + 1) * wt.UInt32(2891336453)
        + stream_value * wt.UInt32(277803737)
    )
    value = (value ^ (value >> 16)) * wt.UInt32(2246822519)
    value = (value ^ (value >> 13)) * wt.UInt32(3266489917)
    value = value ^ (value >> 16)
    mantissa = value & wt.UInt32(0x00FFFFFF)
    return wt.Float(mantissa) / wt.Float(16777216.0)


def _stop_threshold_linear(stop_threshold: float | None) -> float:
    if stop_threshold is None:
        return 0.0
    resolved = float(stop_threshold)
    if resolved <= 0.0:
        return 0.0
    return float(math.pow(10.0, resolved / 10.0))


def _axis_unit_normal(axis: str):
    if axis == "x":
        return wt.Vector3f(1.0, 0.0, 0.0)
    if axis == "y":
        return wt.Vector3f(0.0, 1.0, 0.0)
    return wt.Vector3f(0.0, 0.0, 1.0)


def _solid_angle_per_ray(ray_sampling_metadata: Mapping[str, object], samples_per_tx: int) -> float:
    if samples_per_tx <= 0:
        return 0.0
    selected = str(ray_sampling_metadata.get("selected_ray_sampling", "full_sphere"))
    if selected == "full_sphere":
        return float(4.0 * math.pi / samples_per_tx)
    if selected in {"hemisphere_facing_monitor", "circle_2d"}:
        return float(2.0 * math.pi / samples_per_tx)
    raise RuntimeError(f"Unsupported Monte Carlo ray distribution: {selected!r}")


def _generate_sionna_full_sphere_directions(n_rays: int, *, ray_index=None):
    if ray_index is None:
        if n_rays <= 0:
            zero = dr.zeros(wt.Float, 0)
            return wt.Vector3f(zero, zero, zero)
        width = int(n_rays)
    else:
        width = int(dr.width(ray_index))
        if n_rays <= 0 or width <= 0:
            zero = dr.zeros(wt.Float, width)
            return wt.Vector3f(zero, zero, zero)
    float64_t = dr.float64_array_t(wt.Float)
    if ray_index is None:
        indices = dr.arange(float64_t, 0, n_rays)
    else:
        indices = float64_t(ray_index)
    golden_ratio = float64_t((1.0 + math.sqrt(5.0)) / 2.0)
    azimuth_u = indices / golden_ratio
    azimuth_u = azimuth_u - dr.floor(azimuth_u)
    if n_rays == 1:
        elevation_v = dr.zeros(float64_t, width)
    else:
        elevation_v = indices / float64_t(n_rays - 1)
    phi = wt.Float(float(2.0 * math.pi)) * wt.Float(azimuth_u)
    z = wt.Float(1.0) - wt.Float(2.0) * wt.Float(elevation_v)
    radial = dr.sqrt(dr.maximum(wt.Float(0.0), wt.Float(1.0) - z * z))
    sin_phi, cos_phi = dr.sincos(phi)
    return wt.Vector3f(radial * cos_phi, radial * sin_phi, z)


def _sionna_full_sphere_sampling_metadata(*, axis: str, plane_position: float, tx_pos):
    plane_distance = abs(float(plane_position) - float(scalar(getattr(tx_pos, str(axis)))))
    return {
        "requested_ray_sampling": "full_sphere",
        "selected_ray_sampling": "full_sphere",
        "sampling_sequence": "sionna_fibonacci_square_to_uniform_sphere",
        "monitor_plane_distance_to_tx": plane_distance,
        "near_plane_sampling_threshold": 0.0,
    }


def _monte_carlo_source_field_vector(ray_dir):
    return vector_from_scalar_and_real_direction(
        wt.Complex2f(1.0, 0.0),
        implicit_basis_vector(ray_dir),
    )


def _plane_hit_from_segment(*, ray_origin, ray_dir, blocker_dist, grid, active):
    axis = str(grid.axis)
    axis_dir = getattr(ray_dir, axis)
    safe_axis_dir = axis_dir + dr.select(axis_dir >= 0.0, wt.Float(EPS), -wt.Float(EPS))
    t_plane = (wt.Float(grid.position) - getattr(ray_origin, axis)) / safe_axis_dir
    hit_point = ray_origin + ray_dir * t_plane
    if axis == "x":
        coord_0 = hit_point.y
        coord_1 = hit_point.z
    elif axis == "y":
        coord_0 = hit_point.x
        coord_1 = hit_point.z
    else:
        coord_0 = hit_point.x
        coord_1 = hit_point.y
    within_bounds = (
        (coord_0 >= wt.Float(grid.bounds[0][0]))
        & (coord_0 < wt.Float(grid.bounds[0][1]))
        & (coord_1 >= wt.Float(grid.bounds[1][0]))
        & (coord_1 < wt.Float(grid.bounds[1][1]))
    )
    valid = (
        active
        & (dr.abs(axis_dir) > wt.Float(EPS))
        & (t_plane > wt.Float(RAY_ORIGIN_BIAS))
        & (t_plane < blocker_dist)
        & within_bounds
    )
    target_pos = point_on_axis_aligned_plane(
        axis=axis,
        position=grid.position,
        tangential_0=coord_0,
        tangential_1=coord_1,
    )
    return {
        "valid": valid,
        "coord_0": coord_0,
        "coord_1": coord_1,
        "target_pos": target_pos,
        "distance": t_plane,
        "cos_theta": dr.abs(axis_dir),
    }


def _spawn_offset_ray_origin(point_pos, ray_dir, normal_dir):
    normal_hat = normal_dir / (dr.norm(normal_dir) + wt.Float(EPS))
    direction_sign = dr.sign(dr.dot(ray_dir, normal_hat))
    offset_scale = wt.Float(1.0e-5) * (
        wt.Float(1.0) + dr.max(dr.abs(point_pos), axis=0)
    )
    signed_offset = dr.detach(dr.mulsign(offset_scale, direction_sign))
    return point_pos + signed_offset * normal_hat


def _axis_aligned_cell_index(*, grid, coord_0, coord_1):
    (coord_0_min, _), (coord_1_min, _) = grid.bounds
    nx, ny = grid.grid_shape
    ix = dr.clip(wt.Int32((coord_0 - coord_0_min) / grid.cell_size[0]), 0, nx - 1)
    iy = dr.clip(wt.Int32((coord_1 - coord_1_min) / grid.cell_size[1]), 0, ny - 1)
    return wt.UInt32(iy * nx + ix)


def _scatter_component(*, grid, weighted_diagnostics, component: str, coord_0, coord_1, power, active):
    active_mask = active & (power != wt.Float(0.0))
    if int(dr.width(power)) <= 0:
        return
    cell_idx = _axis_aligned_cell_index(
        grid=grid,
        coord_0=coord_0,
        coord_1=coord_1,
    )
    dr.scatter_reduce(
        dr.ReduceOp.Add,
        weighted_diagnostics["incoherent"][str(component)],
        dr.select(active_mask, power, wt.Float(0.0)),
        cell_idx,
        active_mask,
    )


