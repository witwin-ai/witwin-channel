from __future__ import annotations

import time

import drjit as dr

from ....config import ReflectionSuffixConfig
from ....kernels.trace.utd import utd_accumulate_forward
from ....trace.diffraction.suffix import trace_reflected_suffix_from_edge_states
from ....trace.los import compute_los_field
from ....trace.reflection import compute_reflection_field
from ....utils.drjit_ops import ArrayInit, EvalSync
from ....utils.polarization import vector_zero
from .scheduler import (
    resolve_radio_map_receiver_tiles,
    select_radio_map_diffraction_receiver_tiles,
)


def accumulate_radio_map_los_coherent(*, scene, rx_pos, tx_pos, wavelength: float, k: float):
    return compute_los_field(
        scene,
        rx_pos,
        tx_pos,
        wavelength,
        k,
    )


def accumulate_radio_map_reflection_coherent(
    *,
    sample_grid,
    tx_pos,
    scene,
    wavelength: float,
    k: float,
    reflection_n_rays: int,
    reflection_max_bounces: int,
    ray_mode: str,
    reflection_coef: float,
    min_ray_contribution_threshold: float,
    reflection_field_backend: str,
    tx_polarization,
    rx_polarization,
    reflection_relative_permittivity: float,
    reflection_conductivity: float,
    reflection_material,
    use_scene_materials: bool,
    reflection_detail,
    return_timing: bool,
    return_vector: bool = False,
):
    timing_seconds = 0.0
    if return_timing:
        EvalSync.sync()
        t0 = time.perf_counter()
    reflection_coherent, _, reflection_detail = compute_reflection_field(
        grid=sample_grid,
        rx_z=sample_grid.position,
        tx_pos=tx_pos,
        scene=scene,
        wavelength=wavelength,
        k=k,
        n_rays=reflection_n_rays,
        max_reflections=reflection_max_bounces,
        mode=ray_mode,
        ray_sampling="full_sphere" if ray_mode == "3d" else "circle",
        reflection_coef=reflection_coef,
        min_ray_contribution_threshold=min_ray_contribution_threshold,
        reflection_field_backend=reflection_field_backend,
        tx_polarization=tx_polarization,
        rx_polarization=rx_polarization,
        reflection_relative_permittivity=reflection_relative_permittivity,
        reflection_conductivity=reflection_conductivity,
        reflection_material=reflection_material,
        use_scene_materials=use_scene_materials,
        return_per_bounce=False,
        grid_data=sample_grid.get_coordinates(),
        reflection_detail=reflection_detail,
        include_field_payload=return_vector,
        prefer_epc=False,
    )
    reflection_vector = None
    if return_vector:
        reflection_vector = reflection_detail["polarization_field_total"]
    if return_timing:
        timing_seconds = time.perf_counter() - t0
    return reflection_coherent, reflection_vector, reflection_detail, float(timing_seconds)


def accumulate_radio_map_diffraction_coherent(
    *,
    state_arrays,
    edge_data,
    sample_grid=None,
    rx_pos,
    scene,
    wavelength: float,
    k: float,
    material_detail,
    suffix: ReflectionSuffixConfig,
    tx_polarization,
    rx_polarization,
    execution,
    return_timing: bool,
    return_vector: bool = False,
    receiver_axis: str | None = None,
):
    n_rx = int(dr.width(rx_pos.x))
    receiver_tiles = resolve_radio_map_receiver_tiles(
        grid=sample_grid,
        receiver_positions=rx_pos,
    )
    scheduler_decision = select_radio_map_diffraction_receiver_tiles(
        state_arrays=state_arrays,
        receiver_tiles=receiver_tiles,
        receiver_count=n_rx,
    )

    zero = ArrayInit.complex_zero(n_rx)
    zero_vector = vector_zero(n_rx) if return_vector else None
    if state_arrays is None or int(state_arrays["n_states"]) <= 0:
        if return_timing:
            return zero, zero_vector, scheduler_decision, {
                "utd_accumulation_seconds": 0.0,
                "suffix_seconds": 0.0,
                "postprocess_seconds": 0.0,
            }
        return zero, zero_vector, scheduler_decision, None

    timing = None
    if return_timing:
        timing = {
            "utd_accumulation_seconds": 0.0,
            "suffix_seconds": 0.0,
            "postprocess_seconds": 0.0,
        }
        EvalSync.sync()
        t0 = time.perf_counter()
    direct_total, multi_total, direct_vector_total, multi_vector_total, _ = utd_accumulate_forward(
        state_arrays,
        rx_pos,
        k,
        edge_data["n_edges"],
        False,
        scene=scene,
        wavelength=wavelength,
        material_detail=material_detail,
        rx_polarization=rx_polarization,
        receiver_axis=(
            str(receiver_axis)
            if receiver_axis is not None
            else str(sample_grid.axis)
        ),
        execution=execution,
        receiver_tiles=scheduler_decision.receiver_tiles,
    )
    if return_timing:
        EvalSync.barrier(direct_total, multi_total, multi_vector_total)
        timing["utd_accumulation_seconds"] = time.perf_counter() - t0

    if suffix.enabled:
        if return_timing:
            EvalSync.sync()
            t0 = time.perf_counter()
        reflected_suffix, reflected_suffix_vector = trace_reflected_suffix_from_edge_states(
            state_arrays=state_arrays,
            suffix=suffix,
            scene=scene,
            wavelength=wavelength,
            k=k,
            tx_polarization=tx_polarization,
            execution=execution,
            receiver_tiles=scheduler_decision.receiver_tiles,
        )
        multi_total = multi_total + reflected_suffix
        multi_vector_total = {
            "x": multi_vector_total["x"] + reflected_suffix_vector["x"],
            "y": multi_vector_total["y"] + reflected_suffix_vector["y"],
            "z": multi_vector_total["z"] + reflected_suffix_vector["z"],
        }
        if return_timing:
            EvalSync.barrier(reflected_suffix, reflected_suffix_vector, multi_total)
            timing["suffix_seconds"] = time.perf_counter() - t0

    if return_timing:
        EvalSync.sync()
        t0 = time.perf_counter()
    total = direct_total + multi_total
    total_vector = (
        None
        if not return_vector
        else {
            "x": direct_vector_total["x"] + multi_vector_total["x"],
            "y": direct_vector_total["y"] + multi_vector_total["y"],
            "z": direct_vector_total["z"] + multi_vector_total["z"],
        }
    )
    if return_timing:
        if total_vector is None:
            EvalSync.barrier(total)
        else:
            EvalSync.barrier(
                total,
                total_vector["x"],
                total_vector["y"],
                total_vector["z"],
            )
        timing["postprocess_seconds"] = time.perf_counter() - t0
        return total, total_vector, scheduler_decision, timing
    return total, total_vector, scheduler_decision, None


__all__ = [
    "accumulate_radio_map_diffraction_coherent",
    "accumulate_radio_map_los_coherent",
    "accumulate_radio_map_reflection_coherent",
]
