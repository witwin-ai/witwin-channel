from __future__ import annotations

import time
from typing import Mapping

import drjit as dr
import witwin as wt

from ...trace.diffraction.constants import _cartesian_chunk_size
from ...trace.diffraction.field import _edge_state_field_to_targets
from ...trace.diffraction.geometry import (
    _edge_owner_structure_idx,
    _point_source_field,
    _segment_visibility_mask,
)
from ...trace.diffraction.state import (
    gather_path_export_eval_state_fields,
    gather_path_export_replay_state_fields,
    is_path_export_reduced_state_arrays,
    path_export_state_layout,
)
from ...kernels.trace.packed_state import (
    build_diffraction_path_slots,
    gather_inserted_reflection_state_fields,
    gather_state_arrays,
)
from ...trace.los import compute_los_field, los_blocked
from ...trace.materials import coerce_reflection_trace_detail
from ...trace.reflection import discover_reflection_paths
from ...trace.reflection.epc import (
    build_reflection_epc_descriptor,
    epc_reflection_chain_to_target,
)
from ...scene.runtime_queries import gather_structure_indices
from ...utils.angles import spherical_angles
from ...utils.drjit_ops import ArrayInit, Concat, Gather
from ...utils import scalar
from ...utils.polarization import (
    effective_rx_polarization,
    project_real_polarization_to_ray,
    scalarize_vector_to_polarization,
    vector_from_scalar_and_real_direction,
    vector_scale,
)
from ...types import InteractionType

_MATERIALIZED_PATH_PAYLOAD = "materialized_path_payload_v1"
_DIFFRACTION_STATE_REFS_PAYLOAD = "diffraction_state_refs_v1"
_REFLECTION_PATH_REFS_PAYLOAD = "reflection_path_refs_v1"


def empty_raw_paths(
    *,
    depth: int = 1,
    return_geometry: bool = False,
    payload_kind: str = _MATERIALIZED_PATH_PAYLOAD,
) -> dict[str, object]:
    resolved_depth = max(1, int(depth))
    raw = {
        "payload_kind": str(payload_kind),
        "rx_index": dr.zeros(wt.UInt32, 0),
        "a": ArrayInit.empty_complex(),
        "tau": dr.zeros(wt.Float, 0),
        "theta_t": dr.zeros(wt.Float, 0),
        "phi_t": dr.zeros(wt.Float, 0),
        "theta_r": dr.zeros(wt.Float, 0),
        "phi_r": dr.zeros(wt.Float, 0),
        "type_slots": tuple(dr.zeros(wt.Int32, 0) for _ in range(resolved_depth)),
        "vertex_slots": None,
        "normal_slots": None,
        "object_slots": None,
        "metadata": {},
    }
    if return_geometry:
        raw["vertex_slots"] = tuple(ArrayInit.empty_point3() for _ in range(resolved_depth))
        raw["normal_slots"] = tuple(ArrayInit.empty_vector3() for _ in range(resolved_depth))
        raw["object_slots"] = tuple(dr.full(wt.Int32, -1, 0) for _ in range(resolved_depth))
    return raw


def finalize_raw_paths(
    *,
    rx_index,
    a,
    tau,
    theta_t,
    phi_t,
    theta_r,
    phi_r,
    type_slots,
    vertex_slots,
    normal_slots,
    object_slots,
    payload_kind: str = _MATERIALIZED_PATH_PAYLOAD,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "payload_kind": str(payload_kind),
        "rx_index": rx_index,
        "a": a,
        "tau": tau,
        "theta_t": theta_t,
        "phi_t": phi_t,
        "theta_r": theta_r,
        "phi_r": phi_r,
        "type_slots": tuple(type_slots),
        "vertex_slots": None if vertex_slots is None else tuple(vertex_slots),
        "normal_slots": None if normal_slots is None else tuple(normal_slots),
        "object_slots": None if object_slots is None else tuple(object_slots),
        "metadata": dict(metadata or {}),
    }


def _finalize_diffraction_state_ref_paths(
    *,
    rx_index,
    local_rx_index,
    state_idx,
    a,
    tau,
    tx_pos,
    rx_positions,
    state_arrays,
    edge_data,
    edge_object_idx,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "payload_kind": _DIFFRACTION_STATE_REFS_PAYLOAD,
        "rx_index": rx_index,
        "local_rx_index": local_rx_index,
        "state_idx": state_idx,
        "a": a,
        "tau": tau,
        "tx_pos": tx_pos,
        "rx_positions": rx_positions,
        "state_arrays": state_arrays,
        "edge_data": edge_data,
        "edge_object_idx": edge_object_idx,
        "metadata": dict(metadata or {}),
    }


def _finalize_reflection_path_refs(
    *,
    rx_index,
    path_group_index,
    path_idx,
    a,
    tau,
    tx_pos,
    rx_positions,
    scene,
    reflection_detail,
    wavelength,
    tx_polarization,
    theta_t=None,
    phi_t=None,
    theta_r=None,
    phi_r=None,
    path_depth=None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    detail = coerce_reflection_trace_detail(reflection_detail)
    max_depth_hint = max(
        1,
        max(
            (
                int(paths.chain_depth)
                for paths in detail.source_paths_per_bounce
                if paths is not None
            ),
            default=1,
        ),
    )
    payload = {
        "payload_kind": _REFLECTION_PATH_REFS_PAYLOAD,
        "rx_index": rx_index,
        "path_group_index": path_group_index,
        "path_idx": path_idx,
        "a": a,
        "tau": tau,
        "tx_pos": tx_pos,
        "rx_positions": rx_positions,
        "scene": scene,
        "reflection_detail": reflection_detail,
        "wavelength": float(wavelength),
        "tx_polarization": tuple(float(value) for value in tx_polarization),
        "max_depth_hint": int(max_depth_hint),
        "metadata": dict(metadata or {}),
    }
    if theta_t is not None:
        payload["theta_t"] = theta_t
    if phi_t is not None:
        payload["phi_t"] = phi_t
    if theta_r is not None:
        payload["theta_r"] = theta_r
    if phi_r is not None:
        payload["phi_r"] = phi_r
    if path_depth is not None:
        payload["path_depth"] = path_depth
    return payload


def _take_raw_path_value(value, indices):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_take_raw_path_value(item, indices) for item in value)
    if isinstance(value, list):
        return [_take_raw_path_value(item, indices) for item in value]
    return dr.gather(type(value), value, indices)


def _take_diffraction_state_path_refs(
    raw: Mapping[str, object],
    indices,
) -> dict[str, object]:
    if raw.get("payload_kind") != _DIFFRACTION_STATE_REFS_PAYLOAD:
        return dict(raw)
    local_rx_index = raw.get("local_rx_index", raw["rx_index"])
    return _finalize_diffraction_state_ref_paths(
        rx_index=_take_raw_path_value(raw["rx_index"], indices),
        local_rx_index=_take_raw_path_value(local_rx_index, indices),
        state_idx=_take_raw_path_value(raw["state_idx"], indices),
        a=_take_raw_path_value(raw["a"], indices),
        tau=_take_raw_path_value(raw["tau"], indices),
        tx_pos=raw["tx_pos"],
        rx_positions=raw["rx_positions"],
        state_arrays=raw["state_arrays"],
        edge_data=raw.get("edge_data"),
        edge_object_idx=raw.get("edge_object_idx"),
        metadata=dict(raw.get("metadata", {})),
    )


def _take_reflection_path_refs(
    raw: Mapping[str, object],
    indices,
) -> dict[str, object]:
    if raw.get("payload_kind") != _REFLECTION_PATH_REFS_PAYLOAD:
        return dict(raw)
    return _finalize_reflection_path_refs(
        rx_index=_take_raw_path_value(raw["rx_index"], indices),
        path_group_index=_take_raw_path_value(raw["path_group_index"], indices),
        path_idx=_take_raw_path_value(raw["path_idx"], indices),
        a=_take_raw_path_value(raw["a"], indices),
        tau=_take_raw_path_value(raw["tau"], indices),
        tx_pos=raw["tx_pos"],
        rx_positions=raw["rx_positions"],
        scene=raw["scene"],
        reflection_detail=raw["reflection_detail"],
        wavelength=raw["wavelength"],
        tx_polarization=raw["tx_polarization"],
        theta_t=None if raw.get("theta_t") is None else _take_raw_path_value(raw["theta_t"], indices),
        phi_t=None if raw.get("phi_t") is None else _take_raw_path_value(raw["phi_t"], indices),
        theta_r=None if raw.get("theta_r") is None else _take_raw_path_value(raw["theta_r"], indices),
        phi_r=None if raw.get("phi_r") is None else _take_raw_path_value(raw["phi_r"], indices),
        path_depth=(
            None
            if raw.get("path_depth") is None
            else _take_raw_path_value(raw["path_depth"], indices)
        ),
        metadata=dict(raw.get("metadata", {})),
    )


def _reflection_path_refs_have_cached_materialization(raw: Mapping[str, object]) -> bool:
    return (
        raw.get("theta_t") is not None
        and raw.get("phi_t") is not None
        and raw.get("theta_r") is not None
        and raw.get("phi_r") is not None
        and raw.get("path_depth") is not None
    )


def _materialize_cached_reflection_path_refs(
    raw: Mapping[str, object],
    *,
    path_indices=None,
) -> dict[str, object]:
    if path_indices is None:
        rx_index = raw["rx_index"]
        a = raw["a"]
        tau = raw["tau"]
        theta_t = raw["theta_t"]
        phi_t = raw["phi_t"]
        theta_r = raw["theta_r"]
        phi_r = raw["phi_r"]
        path_depth = raw["path_depth"]
    else:
        rx_index = _take_raw_path_value(raw["rx_index"], path_indices)
        a = _take_raw_path_value(raw["a"], path_indices)
        tau = _take_raw_path_value(raw["tau"], path_indices)
        theta_t = _take_raw_path_value(raw["theta_t"], path_indices)
        phi_t = _take_raw_path_value(raw["phi_t"], path_indices)
        theta_r = _take_raw_path_value(raw["theta_r"], path_indices)
        phi_r = _take_raw_path_value(raw["phi_r"], path_indices)
        path_depth = _take_raw_path_value(raw["path_depth"], path_indices)

    count = int(dr.width(rx_index))
    if count == 0:
        return empty_raw_paths(depth=1, return_geometry=False)

    max_depth = max(1, int(scalar(dr.max(path_depth))))
    reflection_code = dr.full(wt.Int32, InteractionType.REFLECTION, count)
    none_code = dr.full(wt.Int32, InteractionType.NONE, count)
    type_slots = tuple(
        dr.select(path_depth > wt.UInt32(slot), reflection_code, none_code)
        for slot in range(max_depth)
    )
    return finalize_raw_paths(
        rx_index=rx_index,
        a=a,
        tau=tau,
        theta_t=theta_t,
        phi_t=phi_t,
        theta_r=theta_r,
        phi_r=phi_r,
        type_slots=type_slots,
        vertex_slots=None,
        normal_slots=None,
        object_slots=None,
        metadata=dict(raw.get("metadata", {})),
    )


def _materialize_diffraction_state_path_refs(
    raw: Mapping[str, object],
    *,
    return_geometry: bool,
    path_indices=None,
) -> dict[str, object]:
    if raw.get("payload_kind") != _DIFFRACTION_STATE_REFS_PAYLOAD:
        return dict(raw)
    if path_indices is not None:
        raw = _take_diffraction_state_path_refs(raw, path_indices)

    rx_index = raw["rx_index"]
    count = int(dr.width(rx_index))
    if count == 0:
        return empty_raw_paths(depth=1, return_geometry=return_geometry)

    local_rx_index = raw.get("local_rx_index", rx_index)
    state_idx = raw["state_idx"]
    state_arrays = raw["state_arrays"]
    keep_states = _gather_path_replay_state_fields(state_arrays, state_idx)
    keep_rx = dr.gather(wt.Point3f, raw["rx_positions"], local_rx_index)
    arrival_dir = keep_rx - keep_states["edge_pos"]
    departure_dir = keep_states["first_interaction_pos"] - raw["tx_pos"]
    theta_t, phi_t = spherical_angles(departure_dir)
    theta_r, phi_r = spherical_angles(arrival_dir)
    type_slots, vertex_slots, normal_slots, object_slots, _ = _build_type_and_geometry_slots(
        keep_states=keep_states,
        edge_data=raw.get("edge_data"),
        edge_object_idx=raw.get("edge_object_idx"),
        return_geometry=return_geometry,
    )
    return finalize_raw_paths(
        rx_index=rx_index,
        a=raw["a"],
        tau=raw["tau"],
        theta_t=theta_t,
        phi_t=phi_t,
        theta_r=theta_r,
        phi_r=phi_r,
        type_slots=type_slots,
        vertex_slots=vertex_slots,
        normal_slots=normal_slots,
        object_slots=object_slots,
        metadata=dict(raw.get("metadata", {})),
    )


def _materialize_reflection_path_refs(
    raw: Mapping[str, object],
    *,
    return_geometry: bool,
    path_indices=None,
) -> dict[str, object]:
    if raw.get("payload_kind") != _REFLECTION_PATH_REFS_PAYLOAD:
        return dict(raw)
    if not return_geometry and _reflection_path_refs_have_cached_materialization(raw):
        return _materialize_cached_reflection_path_refs(raw, path_indices=path_indices)
    if path_indices is not None:
        raw = _take_reflection_path_refs(raw, path_indices)

    rx_index = raw["rx_index"]
    count = int(dr.width(rx_index))
    if count == 0:
        return empty_raw_paths(depth=1, return_geometry=return_geometry)

    detail = coerce_reflection_trace_detail(raw["reflection_detail"])
    target_pos_all = dr.gather(wt.Point3f, raw["rx_positions"], rx_index)
    rx_index_parts = []
    a_parts = []
    tau_parts = []
    theta_t_parts = []
    phi_t_parts = []
    theta_r_parts = []
    phi_r_parts = []
    type_slot_parts: list[list[object]] = []
    vertex_slot_parts: list[list[object]] | None = [] if return_geometry else None
    normal_slot_parts: list[list[object]] | None = [] if return_geometry else None
    object_slot_parts: list[list[object]] | None = [] if return_geometry else None
    total_paths = 0
    max_depth = 1

    group_count = len(detail.source_paths_per_bounce)
    for group_idx_scalar in range(group_count):
        paths = detail.source_paths_per_bounce[group_idx_scalar]
        chain_depth = 0 if paths is None else int(paths.chain_depth)
        if chain_depth <= 0:
            continue
        group_mask = raw["path_group_index"] == wt.UInt32(group_idx_scalar)
        group_keep = dr.compress(group_mask)
        if dr.width(group_keep) == 0:
            continue

        while len(type_slot_parts) < chain_depth:
            backfill = [dr.zeros(wt.Int32, total_paths)] if total_paths > 0 else []
            type_slot_parts.append(backfill)
            if return_geometry:
                vertex_backfill = [ArrayInit.zeros_point3(total_paths)] if total_paths > 0 else []
                normal_backfill = [ArrayInit.zeros_vector3(total_paths)] if total_paths > 0 else []
                object_backfill = [dr.full(wt.Int32, -1, total_paths)] if total_paths > 0 else []
                vertex_slot_parts.append(vertex_backfill)
                normal_slot_parts.append(normal_backfill)
                object_slot_parts.append(object_backfill)
        max_depth = max(max_depth, chain_depth)

        local_path_idx = dr.gather(wt.UInt32, raw["path_idx"], group_keep)
        target_pos = dr.gather(wt.Point3f, target_pos_all, group_keep)
        replay_valid, _, geometry = epc_reflection_chain_to_target(
            paths=paths,
            path_idx=local_path_idx,
            target_pos=target_pos,
            scene=raw["scene"],
            target_adjacent_faces=(),
            reflection_detail=raw["reflection_detail"],
            wavelength=raw["wavelength"],
            tx_polarization=raw["tx_polarization"],
            return_geometry=return_geometry,
            return_endpoints=not return_geometry,
        )
        valid_keep = dr.compress(replay_valid)
        if dr.width(valid_keep) == 0:
            continue

        group_rx_index = dr.gather(wt.UInt32, raw["rx_index"], group_keep)
        group_a = dr.gather(wt.Complex2f, raw["a"], group_keep)
        group_tau = dr.gather(wt.Float, raw["tau"], group_keep)
        group_rx_index = dr.gather(wt.UInt32, group_rx_index, valid_keep)
        group_a = dr.gather(wt.Complex2f, group_a, valid_keep)
        group_tau = dr.gather(wt.Float, group_tau, valid_keep)
        target_pos = dr.gather(wt.Point3f, target_pos, valid_keep)
        tx_pos = dr.gather(wt.Point3f, geometry["tx_pos"], valid_keep)
        if return_geometry:
            first_hit = dr.gather(wt.Point3f, geometry["hit_points"][0], valid_keep)
            last_hit = dr.gather(wt.Point3f, geometry["hit_points"][-1], valid_keep)
        else:
            first_hit = dr.gather(wt.Point3f, geometry["first_hit"], valid_keep)
            last_hit = dr.gather(wt.Point3f, geometry["last_hit"], valid_keep)

        departure_dir = first_hit - tx_pos
        arrival_dir = target_pos - last_hit
        theta_t, phi_t = spherical_angles(departure_dir)
        theta_r, phi_r = spherical_angles(arrival_dir)
        keep_count = int(dr.width(group_rx_index))
        rx_index_parts.append(group_rx_index)
        a_parts.append(group_a)
        tau_parts.append(group_tau)
        theta_t_parts.append(theta_t)
        phi_t_parts.append(phi_t)
        theta_r_parts.append(theta_r)
        phi_r_parts.append(phi_r)

        for slot in range(len(type_slot_parts)):
            if slot < chain_depth:
                type_slot_parts[slot].append(
                    dr.full(wt.Int32, InteractionType.REFLECTION, keep_count)
                )
                if return_geometry:
                    hit_slot = dr.gather(wt.Point3f, geometry["hit_points"][slot], valid_keep)
                    normal_slot = dr.gather(wt.Vector3f, geometry["normals"][slot], valid_keep)
                    prim_slot = dr.gather(wt.Int32, geometry["prim_indices"][slot], valid_keep)
                    vertex_slot_parts[slot].append(hit_slot)
                    normal_slot_parts[slot].append(normal_slot)
                    object_slot_parts[slot].append(
                        gather_structure_indices(raw["scene"], prim_slot)
                    )
            else:
                type_slot_parts[slot].append(dr.zeros(wt.Int32, keep_count))
                if return_geometry:
                    vertex_slot_parts[slot].append(ArrayInit.zeros_point3(keep_count))
                    normal_slot_parts[slot].append(ArrayInit.zeros_vector3(keep_count))
                    object_slot_parts[slot].append(dr.full(wt.Int32, -1, keep_count))
        total_paths += keep_count

    if total_paths == 0:
        return empty_raw_paths(depth=1, return_geometry=return_geometry)

    type_slots = []
    vertex_slots = [] if return_geometry else None
    normal_slots = [] if return_geometry else None
    object_slots = [] if return_geometry else None
    for slot in range(max_depth):
        if slot < len(type_slot_parts):
            type_slots.append(Concat.ints(type_slot_parts[slot]))
            if return_geometry:
                vertex_slots.append(Concat.points(vertex_slot_parts[slot]))
                normal_slots.append(Concat.vectors(normal_slot_parts[slot]))
                object_slots.append(Concat.ints(object_slot_parts[slot]))
        else:
            type_slots.append(dr.zeros(wt.Int32, total_paths))
            if return_geometry:
                vertex_slots.append(ArrayInit.zeros_point3(total_paths))
                normal_slots.append(ArrayInit.zeros_vector3(total_paths))
                object_slots.append(dr.full(wt.Int32, -1, total_paths))

    return finalize_raw_paths(
        rx_index=Concat.uints(rx_index_parts),
        a=Concat.complex(a_parts),
        tau=Concat.floats(tau_parts),
        theta_t=Concat.floats(theta_t_parts),
        phi_t=Concat.floats(phi_t_parts),
        theta_r=Concat.floats(theta_r_parts),
        phi_r=Concat.floats(phi_r_parts),
        type_slots=type_slots,
        vertex_slots=vertex_slots,
        normal_slots=normal_slots,
        object_slots=object_slots,
        metadata=dict(raw.get("metadata", {})),
    )


def collect_los_paths(
    *,
    scene,
    rx_positions,
    tx_pos,
    wavelength,
    k,
    tx_polarization,
    rx_polarization,
):
    blocked = los_blocked(scene, tx_pos, rx_positions)
    ray_dir = rx_positions - tx_pos
    distance = dr.norm(ray_dir)
    active_rx_polarization = effective_rx_polarization(rx_polarization, tx_polarization)
    coeff = compute_los_field(scene, rx_positions, tx_pos, wavelength, k)
    tx_pol_dir = project_real_polarization_to_ray(tx_polarization, ray_dir)
    field_vec = vector_from_scalar_and_real_direction(coeff, tx_pol_dir)
    scalar_coeff = scalarize_vector_to_polarization(field_vec, ray_dir, active_rx_polarization)
    keep = dr.compress(~blocked)
    if dr.width(keep) == 0:
        return empty_raw_paths(depth=1, return_geometry=False)

    kept_ray_dir = dr.gather(wt.Vector3f, ray_dir, keep)
    theta, phi = spherical_angles(kept_ray_dir)
    keep_count = dr.width(keep)
    return finalize_raw_paths(
        rx_index=keep,
        a=dr.gather(wt.Complex2f, scalar_coeff, keep),
        tau=dr.gather(wt.Float, distance, keep) / 299792458.0,
        theta_t=theta,
        phi_t=phi,
        theta_r=theta,
        phi_r=phi,
        type_slots=(dr.full(wt.Int32, InteractionType.NONE, keep_count),),
        vertex_slots=None,
        normal_slots=None,
        object_slots=None,
        metadata={"n_paths": int(keep_count)},
    )


def _sampling_frame_from_positions(rx_positions):
    coord_min = (
        float(scalar(dr.min(rx_positions.x))),
        float(scalar(dr.min(rx_positions.y))),
        float(scalar(dr.min(rx_positions.z))),
    )
    coord_max = (
        float(scalar(dr.max(rx_positions.x))),
        float(scalar(dr.max(rx_positions.y))),
        float(scalar(dr.max(rx_positions.z))),
    )
    spans = [upper - lower for lower, upper in zip(coord_min, coord_max)]
    axis_idx = int(min(range(3), key=spans.__getitem__))
    axis = ("x", "y", "z")[axis_idx]
    tangential_indices = tuple(index for index in range(3) if index != axis_idx)
    bounds = tuple((coord_min[t_idx], coord_max[t_idx]) for t_idx in tangential_indices)
    plane_position = float((coord_min[axis_idx] + coord_max[axis_idx]) * 0.5)
    return axis, plane_position, bounds


def collect_reflection_paths(
    *,
    scene,
    rx_positions,
    tx_pos,
    wavelength,
    k,
    n_rays,
    max_reflections,
    mode,
    tx_polarization,
    rx_polarization,
    reflection_coef,
    min_ray_contribution_threshold,
    reflection_relative_permittivity,
    reflection_conductivity,
    reflection_material,
    use_scene_materials,
    return_geometry: bool,
    reflection_detail=None,
):
    del min_ray_contribution_threshold
    if n_rays <= 0 or max_reflections <= 0:
        return empty_raw_paths(depth=1, return_geometry=return_geometry), reflection_detail

    detail_payload = reflection_detail
    if detail_payload is None:
        sampling_axis, sampling_plane_position, sampling_bounds = _sampling_frame_from_positions(
            rx_positions
        )
        detail_payload = discover_reflection_paths(
            tx_pos=tx_pos,
            scene=scene,
            wavelength=wavelength,
            k=k,
            n_rays=n_rays,
            max_reflections=max_reflections,
            mode=mode,
            reflection_coef=reflection_coef,
            ray_sampling="full_sphere",
            tx_polarization=tx_polarization,
            reflection_relative_permittivity=reflection_relative_permittivity,
            reflection_conductivity=reflection_conductivity,
            reflection_material=reflection_material,
            use_scene_materials=use_scene_materials,
            sampling_axis=sampling_axis,
            sampling_plane_position=sampling_plane_position,
            sampling_bounds=sampling_bounds,
        )
    detail = coerce_reflection_trace_detail(detail_payload)
    active_rx_polarization = effective_rx_polarization(rx_polarization, tx_polarization)

    rx_index_parts = []
    path_group_index_parts = []
    path_idx_parts = []
    a_parts = []
    tau_parts = []
    theta_t_parts = []
    phi_t_parts = []
    theta_r_parts = []
    phi_r_parts = []
    path_depth_parts = []
    total_paths = 0

    num_rx = int(dr.width(rx_positions.x))
    for bounce_idx, paths in enumerate(detail.source_paths_per_bounce):
        chain_depth = 0 if paths is None else int(paths.chain_depth)
        n_paths = 0 if paths is None else int(paths.n_paths)
        if chain_depth <= 0 or n_paths <= 0:
            continue
        path_chunk_size = _cartesian_chunk_size(n_paths, num_rx)
        for path_start in range(0, n_paths, path_chunk_size):
            chunk_n_paths = min(path_chunk_size, n_paths - path_start)
            chunk_path_idx = dr.arange(wt.UInt32, chunk_n_paths) + wt.UInt32(path_start)
            epc_descriptor = build_reflection_epc_descriptor(
                paths=paths,
                path_idx=chunk_path_idx,
                scene=scene,
                reflection_detail=detail_payload,
            )
            n_pairs = chunk_n_paths * num_rx
            pair_idx = dr.arange(wt.UInt32, n_pairs)
            local_path_idx = pair_idx // num_rx
            path_idx = local_path_idx + wt.UInt32(path_start)
            rx_idx = pair_idx % num_rx
            image_source = Gather.point3(paths.image_source, path_idx)
            target_pos = wt.Point3f(
                dr.gather(wt.Float, rx_positions.x, rx_idx),
                dr.gather(wt.Float, rx_positions.y, rx_idx),
                dr.gather(wt.Float, rx_positions.z, rx_idx),
            )
            valid, chain_vector, geometry = epc_reflection_chain_to_target(
                paths=paths,
                path_idx=local_path_idx,
                target_pos=target_pos,
                scene=scene,
                target_adjacent_faces=(),
                reflection_detail=detail_payload,
                wavelength=wavelength,
                tx_polarization=tx_polarization,
                return_geometry=False,
                return_endpoints=True,
                epc_descriptor=epc_descriptor,
            )
            keep_idx = dr.compress(valid)
            if dr.width(keep_idx) == 0:
                continue

            rx_idx_keep = dr.gather(wt.UInt32, rx_idx, keep_idx)
            path_idx_keep = dr.gather(wt.UInt32, path_idx, keep_idx)
            image_source_keep = Gather.point3(image_source, keep_idx)
            target_pos_keep = dr.gather(wt.Point3f, target_pos, keep_idx)
            tx_pos_keep = dr.gather(wt.Point3f, geometry["tx_pos"], keep_idx)
            first_hit_keep = dr.gather(wt.Point3f, geometry["first_hit"], keep_idx)
            last_hit_keep = dr.gather(wt.Point3f, geometry["last_hit"], keep_idx)

            field_vector = {
                axis: dr.gather(wt.Complex2f, chain_vector[axis], keep_idx)
                for axis in ("x", "y", "z")
            }
            unit_field = _point_source_field(
                image_source_keep,
                wt.Complex2f(1.0, 0.0),
                target_pos_keep,
                wavelength,
                k,
            )
            field_vector = vector_scale(field_vector, unit_field)
            arrival_dir = target_pos_keep - last_hit_keep
            departure_dir = first_hit_keep - tx_pos_keep
            scalar_coeff = scalarize_vector_to_polarization(
                field_vector,
                arrival_dir,
                active_rx_polarization,
            )

            keep_count = int(dr.width(rx_idx_keep))
            theta_t, phi_t = spherical_angles(departure_dir)
            theta_r, phi_r = spherical_angles(arrival_dir)
            rx_index_parts.append(rx_idx_keep)
            path_group_index_parts.append(dr.full(wt.UInt32, bounce_idx, keep_count))
            path_idx_parts.append(path_idx_keep)
            a_parts.append(scalar_coeff)
            tau_parts.append(dr.norm(target_pos_keep - image_source_keep) / 299792458.0)
            theta_t_parts.append(theta_t)
            phi_t_parts.append(phi_t)
            theta_r_parts.append(theta_r)
            phi_r_parts.append(phi_r)
            path_depth_parts.append(dr.full(wt.UInt32, chain_depth, keep_count))
            total_paths += keep_count

    if total_paths == 0:
        return empty_raw_paths(depth=1, return_geometry=return_geometry), detail_payload

    return (
        _finalize_reflection_path_refs(
            rx_index=Concat.uints(rx_index_parts),
            path_group_index=Concat.uints(path_group_index_parts),
            path_idx=Concat.uints(path_idx_parts),
            a=Concat.complex(a_parts),
            tau=Concat.floats(tau_parts),
            tx_pos=tx_pos,
            rx_positions=rx_positions,
            scene=scene,
            reflection_detail=detail_payload,
            wavelength=wavelength,
            tx_polarization=tx_polarization,
            theta_t=Concat.floats(theta_t_parts),
            phi_t=Concat.floats(phi_t_parts),
            theta_r=Concat.floats(theta_r_parts),
            phi_r=Concat.floats(phi_r_parts),
            path_depth=Concat.uints(path_depth_parts),
            metadata={
                "n_paths": total_paths,
                "per_bounce_counts": tuple(
            int(paths.n_paths) if paths is not None else 0
                    for paths in detail.source_paths_per_bounce
                ),
            },
        ),
        detail_payload,
    )


def _edge_object_indices(scene, edge_data):
    if edge_data is None or edge_data["n_edges"] == 0:
        return dr.zeros(wt.Int32, 0)
    return _edge_owner_structure_idx(
        scene,
        edge_data["adjacent_face0"],
        edge_data["adjacent_face1"],
    )


def _build_type_and_geometry_slots(
    *,
    keep_states,
    edge_data,
    edge_object_idx,
    return_geometry: bool,
):
    return build_diffraction_path_slots(
        keep_states=keep_states,
        edge_data=edge_data,
        edge_object_idx=edge_object_idx,
        return_geometry=return_geometry,
    )


def _gather_path_eval_state_fields(state_arrays, indices):
    if is_path_export_reduced_state_arrays(state_arrays):
        return gather_path_export_eval_state_fields(state_arrays, indices)
    return gather_state_arrays(state_arrays, indices)


def _gather_path_replay_state_fields(state_arrays, indices):
    if is_path_export_reduced_state_arrays(state_arrays):
        return gather_path_export_replay_state_fields(state_arrays, indices)
    return gather_inserted_reflection_state_fields(state_arrays, indices)


def collect_diffraction_state_paths(
    *,
    state_arrays,
    edge_data,
    scene,
    rx_positions,
    tx_pos,
    wavelength,
    k,
    tx_polarization,
    rx_polarization,
    material_detail=None,
    return_geometry: bool,
    ignore_emitter_structure_visibility: bool = False,
    stats: dict[str, object] | None = None,
):
    total_start = time.perf_counter()
    if stats is None:
        stats = {}
    stats.clear()
    stats.update(
        {
            "input_states": 0 if state_arrays is None else int(state_arrays["n_states"]),
            "receiver_count": int(dr.width(rx_positions.x)) if rx_positions is not None else 0,
            "chunk_count": 0,
            "candidate_pairs": 0,
            "max_candidate_pairs_per_chunk": 0,
            "visibility_kept_count": 0,
            "field_kept_count": 0,
            "output_paths": 0,
            "payload_kind": _DIFFRACTION_STATE_REFS_PAYLOAD,
            "materialization_deferred": True,
            "sparse_reference_count": 0,
            "state_layout": path_export_state_layout(state_arrays) or "full",
            "timing": {
                "visibility_seconds": 0.0,
                "field_eval_seconds": 0.0,
                "slot_assembly_seconds": 0.0,
                "concat_seconds": 0.0,
                "total_seconds": 0.0,
            },
        }
    )
    if (
        state_arrays is None
        or state_arrays["n_states"] == 0
        or dr.width(rx_positions.x) == 0
    ):
        stats["timing"]["total_seconds"] = time.perf_counter() - total_start
        return empty_raw_paths(depth=1, return_geometry=return_geometry)

    active_rx_polarization = effective_rx_polarization(rx_polarization, tx_polarization)
    edge_object_idx = _edge_object_indices(scene, edge_data) if return_geometry else None

    rx_index_parts = []
    local_rx_index_parts = []
    state_idx_parts = []
    a_parts = []
    tau_parts = []
    total_paths = 0

    n_states = state_arrays["n_states"]
    n_rx = int(dr.width(rx_positions.x))
    state_chunk_size = _cartesian_chunk_size(n_states, n_rx)
    for state_start in range(0, n_states, state_chunk_size):
        stats["chunk_count"] += 1
        chunk_n_states = min(state_chunk_size, n_states - state_start)
        n_pairs = chunk_n_states * n_rx
        stats["candidate_pairs"] += int(n_pairs)
        stats["max_candidate_pairs_per_chunk"] = max(
            int(stats["max_candidate_pairs_per_chunk"]),
            int(n_pairs),
        )
        pair_idx = dr.arange(wt.UInt32, n_pairs)
        state_idx = pair_idx // n_rx + wt.UInt32(state_start)
        rx_idx = pair_idx % n_rx

        state_edge_pos = dr.gather(wt.Point3f, state_arrays["edge_pos"], state_idx)
        adjacent_face0 = dr.gather(wt.Int32, state_arrays["adjacent_face0"], state_idx)
        adjacent_face1 = dr.gather(wt.Int32, state_arrays["adjacent_face1"], state_idx)
        owner_structure_idx = (
            _edge_owner_structure_idx(scene, adjacent_face0, adjacent_face1)
            if ignore_emitter_structure_visibility
            else None
        )
        batch_rx_all = wt.Point3f(
            dr.gather(wt.Float, rx_positions.x, rx_idx),
            dr.gather(wt.Float, rx_positions.y, rx_idx),
            dr.gather(wt.Float, rx_positions.z, rx_idx),
        )
        visibility_start = time.perf_counter()
        visible = _segment_visibility_mask(
            state_edge_pos,
            batch_rx_all,
            scene,
            ignore_prim_idx=(adjacent_face0, adjacent_face1),
            ignore_structure_idx=owner_structure_idx,
        )
        stats["timing"]["visibility_seconds"] += time.perf_counter() - visibility_start
        visible_idx = dr.compress(visible)
        stats["visibility_kept_count"] += int(dr.width(visible_idx))
        if dr.width(visible_idx) == 0:
            continue

        state_idx = dr.gather(wt.UInt32, state_idx, visible_idx)
        rx_idx = dr.gather(wt.UInt32, rx_idx, visible_idx)
        batch_states = _gather_path_eval_state_fields(state_arrays, state_idx)
        batch_rx = dr.gather(wt.Point3f, batch_rx_all, visible_idx)
        field_start = time.perf_counter()
        _, pair_vector, pair_valid = _edge_state_field_to_targets(
            batch_states,
            batch_rx,
            k,
            return_vector=True,
            return_valid=True,
            wavelength=wavelength,
            material_detail=material_detail,
            scene=scene,
            smooth_exterior_shadow=True,
        )
        stats["timing"]["field_eval_seconds"] += time.perf_counter() - field_start
        keep_idx = dr.compress(pair_valid)
        stats["field_kept_count"] += int(dr.width(keep_idx))
        if dr.width(keep_idx) == 0:
            continue

        keep_state_idx = dr.gather(wt.UInt32, state_idx, keep_idx)
        keep_rx = dr.gather(wt.Point3f, batch_rx, keep_idx)
        keep_rx_idx = dr.gather(wt.UInt32, rx_idx, keep_idx)
        keep_edge_pos = dr.gather(wt.Point3f, batch_states["edge_pos"], keep_idx)
        keep_path_length_prefix = dr.gather(
            wt.Float,
            state_arrays["path_length_prefix"],
            keep_state_idx,
        )
        keep_vector = {
            axis: dr.gather(wt.Complex2f, pair_vector[axis], keep_idx)
            for axis in ("x", "y", "z")
        }
        arrival_dir = keep_rx - keep_edge_pos
        scalar_coeff = scalarize_vector_to_polarization(
            keep_vector,
            arrival_dir,
            active_rx_polarization,
        )
        tau = (keep_path_length_prefix + dr.norm(arrival_dir)) / 299792458.0
        keep_count = int(dr.width(keep_rx_idx))
        rx_index_parts.append(keep_rx_idx)
        local_rx_index_parts.append(keep_rx_idx)
        state_idx_parts.append(keep_state_idx)
        a_parts.append(scalar_coeff)
        tau_parts.append(tau)
        total_paths += keep_count
        stats["sparse_reference_count"] += keep_count

    if total_paths == 0:
        stats["timing"]["total_seconds"] = time.perf_counter() - total_start
        return empty_raw_paths(depth=1, return_geometry=return_geometry)

    concat_start = time.perf_counter()
    rx_index = Concat.uints(rx_index_parts)
    local_rx_index = Concat.uints(local_rx_index_parts)
    state_idx = Concat.uints(state_idx_parts)
    a = Concat.complex(a_parts)
    tau = Concat.floats(tau_parts)
    stats["timing"]["concat_seconds"] = time.perf_counter() - concat_start
    stats["output_paths"] = int(total_paths)
    stats["timing"]["total_seconds"] = time.perf_counter() - total_start

    return _finalize_diffraction_state_ref_paths(
        rx_index=rx_index,
        local_rx_index=local_rx_index,
        state_idx=state_idx,
        a=a,
        tau=tau,
        tx_pos=tx_pos,
        rx_positions=rx_positions,
        state_arrays=state_arrays,
        edge_data=edge_data,
        edge_object_idx=edge_object_idx,
        metadata={"n_paths": total_paths},
    )


__all__ = [
    "collect_diffraction_state_paths",
    "collect_los_paths",
    "collect_reflection_paths",
    "epc_reflection_chain_to_target",
]
