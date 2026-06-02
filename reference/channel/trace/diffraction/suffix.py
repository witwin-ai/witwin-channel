"""Reflected diffraction suffix tracing with monitor-neutral grid accumulation."""

from __future__ import annotations

import math

import drjit as dr
import witwin as wt

from ...config import coerce_diffraction_execution
from ...kernels.trace.packed_state import gather_field_evaluation_state_fields
from ...kernels.monitors.common.suffix_grid import drjit_impl as suffix_grid_drjit
from ...kernels.monitors.common.suffix_grid import native_impl as suffix_grid_native
from ...trace.materials import coerce_reflection_material_context
from ...trace.diffraction.state import (
    gather_path_export_field_state_fields,
    is_path_export_reduced_state_arrays,
)
from ...utils.constants import EPS, RAY_ORIGIN_BIAS
from ...utils.drjit_ops import ArrayInit, complex_abs_sqr, eval_complex
from ...utils.polarization import (
    reflect_field_vector,
    vector_eval,
    vector_scale,
    vector_select,
    vector_zero,
)
from ...utils.raygen import generate_circle_directions, generate_sphere_directions
from .constants import SPEED_OF_LIGHT
from .field import _edge_state_field_to_targets
from .geometry import (
    _intersect_rays_ad_with_prim,
    _surface_reflection_coefficient,
)


def _suffix_roulette_random(state_idx, dir_idx, bounce):
    seed = (
        state_idx * wt.UInt32(747796405)
        + dir_idx * wt.UInt32(2891336453)
        + wt.UInt32(bounce + 1) * wt.UInt32(277803737)
    )
    seed = (seed ^ (seed >> 16)) * wt.UInt32(2246822519)
    seed = (seed ^ (seed >> 13)) * wt.UInt32(3266489917)
    seed = seed ^ (seed >> 16)
    mantissa = seed & wt.UInt32(0x00FFFFFF)
    return wt.Float(mantissa) / wt.Float(16777216.0)


def _suffix_roulette_survival_probability(reflected_field, hit):
    reflected_power = complex_abs_sqr(reflected_field)
    active_power = dr.select(hit, reflected_power, wt.Float(0.0))
    max_power = dr.max(active_power)
    safe_max_power = dr.maximum(max_power, wt.Float(1e-20))
    normalized_power = active_power / safe_max_power
    survival_probability = dr.sqrt(normalized_power)
    survival_probability = dr.maximum(survival_probability, wt.Float(0.1))
    survival_probability = dr.minimum(survival_probability, wt.Float(1.0))
    return dr.select(hit, survival_probability, wt.Float(0.0))


def _suffix_accumulator(execution):
    if execution.suffix_backend == "native":
        return suffix_grid_native.accumulate_reflected_segment_fields_batched
    return suffix_grid_drjit.accumulate_reflected_segment_fields_batched


def _gather_suffix_field_state_fields(state_arrays, indices):
    if is_path_export_reduced_state_arrays(state_arrays):
        return gather_path_export_field_state_fields(state_arrays, indices)
    return gather_field_evaluation_state_fields(state_arrays, indices)


def trace_reflected_suffix_from_edge_states(
    state_arrays,
    suffix,
    scene,
    wavelength,
    k,
    tx_polarization=(1.0, 0.0, 0.0),
    execution=None,
    receiver_tiles=None,
):
    execution = coerce_diffraction_execution(execution)
    n_rx = suffix.grid.n_cells
    n_states = 0 if state_arrays is None else state_arrays["n_states"]
    if scene is None or suffix.n_rays <= 0 or suffix.max_bounces <= 0 or n_states <= 0:
        return ArrayInit.complex_zero(n_rx), vector_zero(n_rx)
    reflection_context = coerce_reflection_material_context(
        suffix.detail,
        default_gain=suffix.coef,
    )
    accumulate_segment_fields = _suffix_accumulator(execution)

    rays_per_state = max(128, int(suffix.n_rays / max(1, n_states)))
    if suffix.mode == "2d":
        base_ray_dir = generate_circle_directions(rays_per_state)
    else:
        base_ray_dir = generate_sphere_directions(rays_per_state)

    n_total_rays = n_states * rays_per_state
    ray_idx = dr.arange(wt.UInt32, n_total_rays)
    state_idx = ray_idx // rays_per_state
    dir_idx = ray_idx % rays_per_state
    batch_states = _gather_suffix_field_state_fields(state_arrays, state_idx)

    ray_origin = batch_states["edge_pos"]
    ray_dir = wt.Vector3f(
        dr.gather(wt.Float, base_ray_dir.x, dir_idx),
        dr.gather(wt.Float, base_ray_dir.y, dir_idx),
        dr.gather(wt.Float, base_ray_dir.z, dir_idx),
    )
    active = dr.full(wt.Bool, True, n_total_rays)
    total = ArrayInit.complex_zero(n_rx)
    total_vector = vector_zero(n_rx)
    tri_data = scene.tri_data_gpu
    current_field = None
    current_vector = None
    omega = wt.Float(2.0 * math.pi * SPEED_OF_LIGHT / wavelength)

    for bounce in range(suffix.max_bounces):
        hit, _, hit_p, hit_n, hit_prim_idx = _intersect_rays_ad_with_prim(
            ray_origin,
            ray_dir,
            active,
            scene,
            tri_data,
        )
        if not dr.any(hit):
            break

        if bounce == 0:
            field_at_hit, field_at_hit_vector = _edge_state_field_to_targets(
                batch_states,
                hit_p,
                k,
                return_vector=True,
                wavelength=wavelength,
                material_detail=suffix.detail,
            )
        else:
            seg_len = dr.norm(hit_p - ray_origin) + EPS
            phase = dr.exp(wt.Complex2f(0, -wt.Float(k) * seg_len))
            fspl = wt.Float(wavelength / (4.0 * math.pi)) / seg_len
            field_at_hit = current_field * fspl * phase
            field_at_hit_vector = vector_scale(current_vector, fspl * phase)

        field_at_hit = dr.select(hit, field_at_hit, ArrayInit.complex_zero(n_total_rays))
        field_at_hit_vector = vector_select(hit, field_at_hit_vector, vector_zero(n_total_rays))
        safe_ray_dir = dr.select(hit, ray_dir, wt.Vector3f(1.0, 0.0, 0.0))
        safe_hit_n = dr.select(hit, hit_n, wt.Vector3f(0.0, 0.0, 1.0))
        reflection_weight, material_inputs = _surface_reflection_coefficient(
            incident_dir=ray_dir,
            normal=hit_n,
            scene=scene,
            prim_idx=wt.Int32(hit_prim_idx),
            material_override=reflection_context.reflection_material,
            reflection_coef=reflection_context.reflection_gain,
            wavelength=wavelength,
            tx_polarization=tx_polarization,
            valid_mask=hit,
            use_scene_materials=reflection_context.use_scene_materials,
        )
        reflected_field = field_at_hit * reflection_weight
        reflected_vector = reflect_field_vector(
            field_at_hit_vector,
            safe_ray_dir,
            safe_hit_n,
            eta_r=material_inputs["eta_r"],
            sigma=material_inputs["sigma"],
            omega=omega,
            gain=material_inputs["gain"],
        )
        reflected_field = dr.select(hit, reflected_field, ArrayInit.complex_zero(n_total_rays))
        reflected_vector = vector_select(hit, reflected_vector, vector_zero(n_total_rays))

        dot_dn = dr.dot(ray_dir, hit_n)
        reflected_dir = ray_dir - 2.0 * dot_dn * hit_n
        next_origin = hit_p + reflected_dir * RAY_ORIGIN_BIAS

        next_hit, next_blocker, _, _, _ = _intersect_rays_ad_with_prim(
            next_origin,
            reflected_dir,
            hit,
            scene,
            tri_data,
        )
        accumulate_kwargs = dict(
            grid=suffix.grid,
            grid_data=suffix.grid_data,
            seg_origin=hit_p,
            seg_dir=reflected_dir,
            blocker_dist=next_blocker,
            seg_field=reflected_field,
            seg_vector=reflected_vector,
            state_idx=state_idx,
            n_states=n_states,
            wavelength=wavelength,
            k=k,
            active=hit,
            execution=execution,
        )
        if execution.suffix_backend == "native":
            accumulate_kwargs["receiver_tiles"] = receiver_tiles
        segment_field, segment_vector = accumulate_segment_fields(**accumulate_kwargs)
        segment_field = eval_complex(segment_field)
        segment_vector = vector_eval(segment_vector)
        total = eval_complex(total + segment_field)
        total_vector = vector_eval(
            {
                "x": total_vector["x"] + segment_vector["x"],
                "y": total_vector["y"] + segment_vector["y"],
                "z": total_vector["z"] + segment_vector["z"],
            }
        )

        if execution.suffix_russian_roulette and bounce > 0 and bounce + 1 < suffix.max_bounces:
            survival_probability = _suffix_roulette_survival_probability(reflected_field, next_hit)
            survive = next_hit & (
                _suffix_roulette_random(state_idx, dir_idx, bounce) < survival_probability
            )
            survival_scale = dr.rcp(dr.maximum(survival_probability, wt.Float(1e-6)))
            reflected_field = dr.select(
                survive,
                reflected_field * wt.Complex2f(survival_scale, wt.Float(0.0)),
                ArrayInit.complex_zero(n_total_rays),
            )
            reflected_vector = vector_select(
                survive,
                vector_scale(reflected_vector, survival_scale),
                vector_zero(n_total_rays),
            )
            next_hit = survive

        ray_origin = next_origin
        ray_dir = reflected_dir
        current_field = eval_complex(reflected_field)
        current_vector = vector_eval(reflected_vector)
        active = next_hit

    return total, total_vector

__all__ = [
    "trace_reflected_suffix_from_edge_states",
]
