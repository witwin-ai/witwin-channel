"""Reflection field assembly + receiver-power accumulation."""

from __future__ import annotations

import drjit as dr
from witwin.channel.deterministic import types as wt

from ..kernels.radio_map_accumulate.native_impl import (
    accumulate_vector_power_pairs,
)
from witwin.channel.core.runtime import Material, Rx, Tx, Wave
from witwin.channel.core.numerics.arrays import gather
from witwin.channel.core.physics.polarization import (
    jones_tangential,
    scalarize_tangential_jones,
    vector_eval,
    vector_scale,
    vector_zero,
)
from ..diffraction.state import Geo
from . import epc, paths
from . import common
from .detail import build_trace_detail, coerce_trace_detail


# ============================================================================
# Per-bounce field assembly (build polarization vectors + TraceDetail)
# ============================================================================

def assemble_outputs(
    *,
    n_rx: int,
    grid_axis: str,
    polarization_per_bounce,
    source_paths_per_bounce,
    reflection_model: str,
    reflection_model_source: str,
    reflection_gain: float,
    active_rx_polarization,
    return_per_bounce: bool = False,
    polarization_total=None,
    reflection_transition_mode: str = "hard",
    reflection_f_weight_boundary_radius_wavelengths: float = 2.0,
    reflection_f_weight_max_edges_per_slot: int = 1,
    reflection_secondary_visibility_mode: str = "hard",
):
    a_ref_list = []
    if return_per_bounce:
        a_ref_list = [
            scalarize_tangential_jones(
                jones_tangential(pf, axis=grid_axis),
                active_rx_polarization,
                axis=grid_axis,
            )
            for pf in polarization_per_bounce
        ]

    if polarization_total is None:
        polarization_total = {
            axis: common.sum_complex_fields(
                (pf[axis] for pf in polarization_per_bounce), n_rx=n_rx,
            )
            for axis in ("x", "y", "z")
        }
    a_ref_total = scalarize_tangential_jones(
        jones_tangential(polarization_total, axis=grid_axis),
        active_rx_polarization,
        axis=grid_axis,
    )

    rd_detail = build_trace_detail(
        reflection_model=reflection_model,
        reflection_model_source=reflection_model_source,
        reflection_gain=float(reflection_gain),
        source_paths_per_bounce=source_paths_per_bounce,
        reflection_transition_mode=reflection_transition_mode,
        reflection_f_weight_boundary_radius_wavelengths=reflection_f_weight_boundary_radius_wavelengths,
        reflection_f_weight_max_edges_per_slot=reflection_f_weight_max_edges_per_slot,
        reflection_secondary_visibility_mode=reflection_secondary_visibility_mode,
    )
    return a_ref_total, a_ref_list, rd_detail, polarization_total


def compute_field_impl(
    grid,
    rx_z,
    tx: Tx,
    scene,
    wave: Wave,
    n_rays,
    max_reflections,
    mode,
    material: Material,
    ray_sampling,
    return_per_bounce,
    tri_data,
    rx: Rx | None = None,
):
    n_rx = grid.n_cells
    grid_axis, plane_position = common.resolve_plane(grid, rx_z)
    if tri_data is None:
        raise RuntimeError("Reflection field tracing requires RayD-backed triangle runtime.")

    if rx is None:
        rx = Rx(positions=grid.receiver_positions_3d(position=plane_position), polarization=None)
    active_rx_polarization = rx.effective_polarization(tx)
    trace_data = paths.trace_paths(
        tx=tx, scene=scene, wave=wave, n_rays=n_rays,
        max_reflections=max_reflections, mode=mode, material=material,
        ray_sampling=ray_sampling, sampling_axis=grid_axis,
        sampling_bounds=grid.bounds, sampling_plane_position=plane_position,
        tri_data=tri_data,
    )
    source_paths_per_bounce = list(trace_data["source_paths_per_bounce"])
    reflection_chain_detail = build_trace_detail(
        reflection_model=trace_data["reflection_model"],
        reflection_model_source=trace_data["reflection_model_source"],
        reflection_gain=material.gain_scalar,
        source_paths_per_bounce=source_paths_per_bounce,
    )
    polarization_per_bounce = paths.accumulate_paths_exact(
        rx=rx, tx=tx, scene=scene, wave=wave,
        source_paths_per_bounce=source_paths_per_bounce,
        reflection_detail=reflection_chain_detail,
    )
    polarization_total = {
        axis: common.sum_complex_fields(
            (pf[axis] for pf in polarization_per_bounce), n_rx=n_rx,
        )
        for axis in ("x", "y", "z")
    }
    if not return_per_bounce:
        polarization_per_bounce = []

    a_ref_total, a_ref_list, rd_detail, polarization_total = assemble_outputs(
        n_rx=n_rx, grid_axis=grid_axis,
        polarization_per_bounce=polarization_per_bounce,
        source_paths_per_bounce=tuple(source_paths_per_bounce),
        reflection_model=trace_data["reflection_model"],
        reflection_model_source=trace_data["reflection_model_source"],
        reflection_gain=material.gain_scalar,
        active_rx_polarization=active_rx_polarization,
        return_per_bounce=return_per_bounce,
        polarization_total=polarization_total,
        reflection_transition_mode=reflection_chain_detail.reflection_transition_mode,
        reflection_f_weight_boundary_radius_wavelengths=(
            reflection_chain_detail.reflection_f_weight_boundary_radius_wavelengths
        ),
        reflection_f_weight_max_edges_per_slot=reflection_chain_detail.reflection_f_weight_max_edges_per_slot,
        reflection_secondary_visibility_mode=reflection_chain_detail.reflection_secondary_visibility_mode,
    )
    if return_per_bounce:
        return a_ref_total, a_ref_list, rd_detail, polarization_total
    return a_ref_total, [], rd_detail, polarization_total


# ============================================================================
# Receiver-power accumulation (matched-isotropic baseline)
# ============================================================================

def any_grad_enabled(*arrays) -> bool:
    for arr in arrays:
        if arr is None:
            continue
        try:
            if bool(dr.grad_enabled(arr)):
                return True
        except Exception:
            pass
    return False


def accumulate_chunk_vector(
    *,
    paths_set,
    chunk_path_idx,
    receiver_idx,
    rx: Rx,
    tx: Tx,
    scene,
    wave: Wave,
    reflection_detail,
    active_rx_polarization,
    vector_coherent,
    epc_descriptor=None,
):
    rx_pos = rx.positions
    chunk_n_paths = int(dr.width(chunk_path_idx))
    local_n_rx = int(dr.width(receiver_idx))
    if chunk_n_paths <= 0 or local_n_rx <= 0:
        return

    descriptor_full_paths = (
        epc_descriptor is not None
        and int(getattr(epc_descriptor, "n_paths", 0)) == int(paths_set.n_paths)
    )
    if epc_descriptor is None:
        epc_descriptor = epc.build_descriptor(
            paths=paths_set, path_idx=chunk_path_idx,
            scene=scene, reflection_detail=reflection_detail,
        )
    n_pairs = chunk_n_paths * local_n_rx
    pair_idx = dr.arange(wt.UInt32, n_pairs)
    local_path_idx = pair_idx // local_n_rx
    local_rx_slot = pair_idx % local_n_rx
    rx_idx = dr.gather(type(receiver_idx), receiver_idx, local_rx_slot)
    target_pos = gather(rx_pos, rx_idx)
    descriptor_path_idx = (
        dr.gather(type(chunk_path_idx), chunk_path_idx, local_path_idx)
        if descriptor_full_paths else local_path_idx
    )
    image_source = gather(epc_descriptor.image_source, descriptor_path_idx)
    valid, chain_vector, geometry = epc.chain_to_target(
        paths=paths_set, path_idx=descriptor_path_idx, target_pos=target_pos,
        scene=scene, target_adjacent_faces=(),
        reflection_detail=reflection_detail, wave=wave, tx=tx,
        return_endpoints=True, epc_descriptor=epc_descriptor,
    )
    keep_idx = dr.compress(valid)
    if dr.width(keep_idx) == 0:
        return

    rx_idx_keep = dr.gather(type(rx_idx), rx_idx, keep_idx)
    target_pos_keep = gather(target_pos, keep_idx)
    image_source_keep = gather(image_source, keep_idx)
    last_hit_keep = gather(geometry["last_hit"], keep_idx)
    field_vector = {
        axis: dr.gather(wt.Complex2f, chain_vector[axis], keep_idx)
        for axis in ("x", "y", "z")
    }
    unit_field = Geo.source_field(image_source_keep, wt.Complex2f(1.0, 0.0), target_pos_keep, wave)
    field_vector = vector_scale(field_vector, unit_field)
    arrival_dir = target_pos_keep - last_hit_keep
    if not any_grad_enabled(
        arrival_dir.x, arrival_dir.y, arrival_dir.z,
        *(field_vector[a].real for a in "xyz"),
        *(field_vector[a].imag for a in "xyz"),
    ):
        _, _, total_vector, _ = accumulate_vector_power_pairs(
            rx_idx_keep, field_vector, arrival_dir,
            n_output_rx=int(dr.width(rx_pos.x)),
            rx_polarization=active_rx_polarization,
        )
        for axis in ("x", "y", "z"):
            vector_coherent[axis].real = vector_coherent[axis].real + total_vector[axis].real
            vector_coherent[axis].imag = vector_coherent[axis].imag + total_vector[axis].imag
        return

    for axis in ("x", "y", "z"):
        dr.scatter_reduce(dr.ReduceOp.Add, vector_coherent[axis].real, field_vector[axis].real, rx_idx_keep)
        dr.scatter_reduce(dr.ReduceOp.Add, vector_coherent[axis].imag, field_vector[axis].imag, rx_idx_keep)


def accumulate_vector_field(
    *,
    rx: Rx,
    tx: Tx,
    scene,
    wave: Wave,
    reflection_detail,
):
    rx_pos = rx.positions
    n_rx = int(dr.width(rx_pos.x))
    vector_coherent = vector_zero(n_rx)
    if reflection_detail is None:
        return vector_eval(vector_coherent)
    detail = coerce_trace_detail(reflection_detail)
    active_rx_polarization = rx.effective_polarization(tx)
    receiver_idx = dr.arange(wt.UInt32, n_rx)
    for paths_set in detail.source_paths_per_bounce:
        chain_depth = 0 if paths_set is None else int(paths_set.chain_depth)
        n_paths = 0 if paths_set is None else int(paths_set.n_paths)
        if chain_depth <= 0 or n_paths <= 0:
            continue
        chunk_size = Geo.cart_chunk(n_paths, int(dr.width(receiver_idx)))
        for path_start in range(0, n_paths, chunk_size):
            chunk_n = min(chunk_size, n_paths - path_start)
            chunk_path_idx = dr.arange(wt.UInt32, chunk_n) + wt.UInt32(path_start)
            accumulate_chunk_vector(
                paths_set=paths_set,
                chunk_path_idx=chunk_path_idx,
                receiver_idx=receiver_idx,
                rx=rx, tx=tx, scene=scene, wave=wave,
                reflection_detail=reflection_detail,
                active_rx_polarization=active_rx_polarization,
                vector_coherent=vector_coherent,
            )
    return vector_eval(vector_coherent)


__all__ = [
    "assemble_outputs",
    "compute_field_impl",
    "accumulate_vector_field",
]
