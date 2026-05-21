from __future__ import annotations

import time
from typing import Mapping

import drjit as dr
import rayd

from witwin.channel.core.numerics import arrays
from witwin.channel.core.numerics.arrays import scalar
from witwin.channel.core.numerics.constants import EPS, RAY_EPS
from witwin.channel.core.physics.polarization import (
    effective_rx_polarization,
    project_real_polarization_to_ray,
    scalarize_vector_to_polarization,
    vector_from_scalar,
    vector_scale,
)
from witwin.channel.deterministic.kernels.packed_state import (
    build_diffraction_path_slots,
    gather_inserted_reflection_state_fields,
    gather_state_arrays,
)
from witwin.channel.deterministic.diffraction.forward import ForwardEval
from witwin.channel.deterministic.diffraction.state import (
    PATH_EXPORT_REDUCED_STATE_LAYOUT,
    State,
)
from witwin.channel.deterministic.trace.reflection import discover_paths as discover_reflection_paths
from witwin.channel.deterministic.reflection.detail import (
    SourcePathSet,
    TraceDetail,
    coerce_trace_detail as coerce_reflection_trace_detail,
)
from witwin.channel.deterministic.reflection.epc import (
    build_descriptor as build_reflection_epc_descriptor,
    chain_to_target as epc_reflection_chain_to_target,
)
from witwin.channel.core.runtime import Material, Tx, Wave

from witwin.channel.core.numerics.tensors import drjit_to_torch_view
from witwin.channel.deterministic import types as wt
from witwin.channel.deterministic.types import InteractionType

_MATERIALIZED_PATH_PAYLOAD = "materialized_path_payload_v1"
_DIFFRACTION_STATE_REFS_PAYLOAD = "diffraction_state_refs_v1"
_REFLECTION_PATH_REFS_PAYLOAD = "reflection_path_refs_v1"
_CARTESIAN_PAIR_CHUNK_BUDGET = 1 << 25
_PATH_RESULT_REPLAY_CHUNK_SIZE = 4096


def _cartesian_chunk_size(left_size, right_size) -> int:
    left = max(0, int(left_size))
    right = max(0, int(right_size))
    if left == 0 or right == 0:
        return 0
    return max(1, min(left, _CARTESIAN_PAIR_CHUNK_BUDGET // right))


def spherical_angles(vectors) -> tuple[wt.Float, wt.Float]:
    radius = dr.maximum(dr.norm(vectors), wt.Float(1e-12))
    theta = dr.acos(dr.clip(vectors.z / radius, -1.0, 1.0))
    phi = dr.atan2(vectors.y, vectors.x)
    phi = dr.select(phi < 0.0, phi + wt.Float(2.0 * dr.pi), phi)
    return theta, phi


def _los_blocked(scene, tx_pos, rx_positions):
    width = dr.width(rx_positions.x)
    if scene is None or len(getattr(scene, "structures", ())) == 0:
        return dr.zeros(wt.Bool, width)
    tri_data = getattr(scene, "tri_data_gpu", None)
    if tri_data is not None and int(tri_data.get("n_triangles", 0)) <= 0:
        return dr.zeros(wt.Bool, width)
    ray_dir = rx_positions - tx_pos
    ray_length = dr.norm(ray_dir)
    ray_dir_normalized = ray_dir / (ray_length + wt.Float(EPS))
    rays = rayd.Ray(tx_pos, ray_dir_normalized)
    rays.tmax = ray_length - wt.Float(RAY_EPS)
    with dr.suspend_grad():
        return scene.ray_test(rays)


def _compute_los_field(scene, rx_positions, tx_pos, wavelength, k):
    blocked = _los_blocked(scene, tx_pos, rx_positions)
    ray_dir = rx_positions - tx_pos
    distance = dr.norm(ray_dir)
    coeff = wt.Float(wavelength) / (wt.Float(4.0) * dr.pi * distance)
    phase = wt.Complex2f(0.0, -wt.Float(k) * distance)
    field = coeff * dr.exp(phase)
    return dr.select(blocked, wt.Complex2f(0.0, 0.0), field), blocked


def _point_source_field(source_pos, source_weight, target_pos, wavelength, k):
    width = dr.width(target_pos.x)
    source_pos_b = arrays.broadcast_point(source_pos, width)
    source_weight_b = arrays.broadcast_complex(source_weight, width)
    distance = dr.norm(target_pos - source_pos_b)
    coeff = wt.Float(wavelength) / (wt.Float(4.0) * dr.pi * (distance + wt.Float(EPS)))
    phase = dr.exp(wt.Complex2f(0.0, -wt.Float(k) * distance))
    return source_weight_b * coeff * phase


def _segment_visibility_mask(
    start_pos,
    end_pos,
    scene,
    *,
    ignore_prim_idx=None,
):
    del start_pos, scene, ignore_prim_idx
    return dr.full(wt.Bool, True, dr.width(end_pos.x))


def is_path_export_reduced_state_arrays(state_arrays) -> bool:
    return (
        state_arrays is not None
        and state_arrays.get("__path_export_state_layout__") == PATH_EXPORT_REDUCED_STATE_LAYOUT
    )


def path_export_state_layout(state_arrays) -> str | None:
    if is_path_export_reduced_state_arrays(state_arrays):
        return PATH_EXPORT_REDUCED_STATE_LAYOUT
    return None


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
        "tx_index": dr.zeros(wt.UInt32, 0),
        "a": arrays.empty_complex(),
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
        raw["vertex_slots"] = tuple(arrays.empty_point3() for _ in range(resolved_depth))
        raw["normal_slots"] = tuple(arrays.empty_vector3() for _ in range(resolved_depth))
        raw["object_slots"] = tuple(dr.full(wt.Int32, -1, 0) for _ in range(resolved_depth))
    return raw


def finalize_raw_paths(
    *,
    rx_index,
    tx_index=None,
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
        "tx_index": dr.zeros(wt.UInt32, dr.width(rx_index)) if tx_index is None else tx_index,
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
    tx_index=None,
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
        "tx_index": dr.zeros(wt.UInt32, dr.width(rx_index)) if tx_index is None else tx_index,
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
    tx_index=None,
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
    type_slots=None,
    vertex_slots=None,
    normal_slots=None,
    object_slots=None,
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
        "tx_index": dr.zeros(wt.UInt32, dr.width(rx_index)) if tx_index is None else tx_index,
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
    if type_slots is not None:
        payload["type_slots"] = tuple(type_slots)
    if vertex_slots is not None:
        payload["vertex_slots"] = tuple(vertex_slots)
    if normal_slots is not None:
        payload["normal_slots"] = tuple(normal_slots)
    if object_slots is not None:
        payload["object_slots"] = tuple(object_slots)
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
    raw_tx_index = raw.get("tx_index", dr.zeros(wt.UInt32, dr.width(raw["rx_index"])))
    return _finalize_diffraction_state_ref_paths(
        rx_index=_take_raw_path_value(raw["rx_index"], indices),
        tx_index=_take_raw_path_value(raw_tx_index, indices),
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
    raw_tx_index = raw.get("tx_index", dr.zeros(wt.UInt32, dr.width(raw["rx_index"])))
    return _finalize_reflection_path_refs(
        rx_index=_take_raw_path_value(raw["rx_index"], indices),
        tx_index=_take_raw_path_value(raw_tx_index, indices),
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
        type_slots=None if raw.get("type_slots") is None else _take_raw_path_value(raw["type_slots"], indices),
        vertex_slots=(
            None
            if raw.get("vertex_slots") is None
            else _take_raw_path_value(raw["vertex_slots"], indices)
        ),
        normal_slots=(
            None
            if raw.get("normal_slots") is None
            else _take_raw_path_value(raw["normal_slots"], indices)
        ),
        object_slots=(
            None
            if raw.get("object_slots") is None
            else _take_raw_path_value(raw["object_slots"], indices)
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


def _reflection_path_refs_have_cached_geometry(raw: Mapping[str, object]) -> bool:
    return (
        _reflection_path_refs_have_cached_materialization(raw)
        and raw.get("type_slots") is not None
        and raw.get("vertex_slots") is not None
        and raw.get("normal_slots") is not None
        and raw.get("object_slots") is not None
    )


def _materialize_cached_reflection_path_refs(
    raw: Mapping[str, object],
    *,
    return_geometry: bool,
    path_indices=None,
) -> dict[str, object]:
    if path_indices is None:
        rx_index = raw["rx_index"]
        tx_index = raw.get("tx_index", dr.zeros(wt.UInt32, dr.width(rx_index)))
        a = raw["a"]
        tau = raw["tau"]
        theta_t = raw["theta_t"]
        phi_t = raw["phi_t"]
        theta_r = raw["theta_r"]
        phi_r = raw["phi_r"]
        path_depth = raw["path_depth"]
        cached_type_slots = raw.get("type_slots")
        cached_vertex_slots = raw.get("vertex_slots")
        cached_normal_slots = raw.get("normal_slots")
        cached_object_slots = raw.get("object_slots")
    else:
        rx_index = _take_raw_path_value(raw["rx_index"], path_indices)
        raw_tx_index = raw.get("tx_index", dr.zeros(wt.UInt32, dr.width(raw["rx_index"])))
        tx_index = _take_raw_path_value(raw_tx_index, path_indices)
        a = _take_raw_path_value(raw["a"], path_indices)
        tau = _take_raw_path_value(raw["tau"], path_indices)
        theta_t = _take_raw_path_value(raw["theta_t"], path_indices)
        phi_t = _take_raw_path_value(raw["phi_t"], path_indices)
        theta_r = _take_raw_path_value(raw["theta_r"], path_indices)
        phi_r = _take_raw_path_value(raw["phi_r"], path_indices)
        path_depth = _take_raw_path_value(raw["path_depth"], path_indices)
        cached_type_slots = (
            None if raw.get("type_slots") is None else _take_raw_path_value(raw["type_slots"], path_indices)
        )
        cached_vertex_slots = (
            None if raw.get("vertex_slots") is None else _take_raw_path_value(raw["vertex_slots"], path_indices)
        )
        cached_normal_slots = (
            None if raw.get("normal_slots") is None else _take_raw_path_value(raw["normal_slots"], path_indices)
        )
        cached_object_slots = (
            None if raw.get("object_slots") is None else _take_raw_path_value(raw["object_slots"], path_indices)
        )

    count = int(dr.width(rx_index))
    if count == 0:
        return empty_raw_paths(depth=1, return_geometry=return_geometry)

    if cached_type_slots is None:
        max_depth = max(1, int(scalar(dr.max(path_depth))))
        reflection_code = dr.full(wt.Int32, InteractionType.REFLECTION, count)
        none_code = dr.full(wt.Int32, InteractionType.NONE, count)
        type_slots = tuple(
            dr.select(path_depth > wt.UInt32(slot), reflection_code, none_code)
            for slot in range(max_depth)
        )
    else:
        type_slots = tuple(cached_type_slots)
    return finalize_raw_paths(
        rx_index=rx_index,
        tx_index=tx_index,
        a=a,
        tau=tau,
        theta_t=theta_t,
        phi_t=phi_t,
        theta_r=theta_r,
        phi_r=phi_r,
        type_slots=type_slots,
        vertex_slots=cached_vertex_slots if return_geometry else None,
        normal_slots=cached_normal_slots if return_geometry else None,
        object_slots=cached_object_slots if return_geometry else None,
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
    tx_index = raw.get("tx_index", dr.zeros(wt.UInt32, dr.width(rx_index)))
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
        tx_index=tx_index,
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
    if _reflection_path_refs_have_cached_materialization(raw) and (
        not return_geometry or _reflection_path_refs_have_cached_geometry(raw)
    ):
        return _materialize_cached_reflection_path_refs(
            raw,
            return_geometry=return_geometry,
            path_indices=path_indices,
        )
    if path_indices is not None:
        raw = _take_reflection_path_refs(raw, path_indices)

    rx_index = raw["rx_index"]
    tx_index = raw.get("tx_index", dr.zeros(wt.UInt32, dr.width(rx_index)))
    count = int(dr.width(rx_index))
    if count == 0:
        return empty_raw_paths(depth=1, return_geometry=return_geometry)

    detail = coerce_reflection_trace_detail(raw["reflection_detail"])
    target_pos_all = dr.gather(wt.Point3f, raw["rx_positions"], rx_index)
    rx_index_parts = []
    tx_index_parts = []
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
                vertex_backfill = [arrays.zeros_point3(total_paths)] if total_paths > 0 else []
                normal_backfill = [arrays.zeros_vector3(total_paths)] if total_paths > 0 else []
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
            wave=Wave(wavelength=raw["wavelength"]),
            tx=Tx(position=raw["tx_pos"], polarization=raw["tx_polarization"]),
            return_geometry=return_geometry,
            return_endpoints=not return_geometry,
        )
        valid_keep = dr.compress(replay_valid)
        if dr.width(valid_keep) == 0:
            continue

        group_rx_index = dr.gather(wt.UInt32, raw["rx_index"], group_keep)
        group_tx_index = dr.gather(wt.UInt32, tx_index, group_keep)
        group_a = dr.gather(wt.Complex2f, raw["a"], group_keep)
        group_tau = dr.gather(wt.Float, raw["tau"], group_keep)
        group_rx_index = dr.gather(wt.UInt32, group_rx_index, valid_keep)
        group_tx_index = dr.gather(wt.UInt32, group_tx_index, valid_keep)
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
        tx_index_parts.append(group_tx_index)
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
                        raw["scene"].gather_structure_indices(prim_slot)
                    )
            else:
                type_slot_parts[slot].append(dr.zeros(wt.Int32, keep_count))
                if return_geometry:
                    vertex_slot_parts[slot].append(arrays.zeros_point3(keep_count))
                    normal_slot_parts[slot].append(arrays.zeros_vector3(keep_count))
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
            type_slots.append(arrays.concat_ints(type_slot_parts[slot]))
            if return_geometry:
                vertex_slots.append(arrays.concat_points(vertex_slot_parts[slot]))
                normal_slots.append(arrays.concat_vectors(normal_slot_parts[slot]))
                object_slots.append(arrays.concat_ints(object_slot_parts[slot]))
        else:
            type_slots.append(dr.zeros(wt.Int32, total_paths))
            if return_geometry:
                vertex_slots.append(arrays.zeros_point3(total_paths))
                normal_slots.append(arrays.zeros_vector3(total_paths))
                object_slots.append(dr.full(wt.Int32, -1, total_paths))

    return finalize_raw_paths(
        rx_index=arrays.concat_uints(rx_index_parts),
        tx_index=arrays.concat_uints(tx_index_parts),
        a=arrays.concat_complex(a_parts),
        tau=arrays.concat_floats(tau_parts),
        theta_t=arrays.concat_floats(theta_t_parts),
        phi_t=arrays.concat_floats(phi_t_parts),
        theta_r=arrays.concat_floats(theta_r_parts),
        phi_r=arrays.concat_floats(phi_r_parts),
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
    num_rx = int(dr.width(rx_positions.x))
    num_tx = int(dr.width(tx_pos.x))
    if num_rx == 0 or num_tx == 0:
        return empty_raw_paths(depth=1, return_geometry=False)

    pair_count = num_rx * num_tx
    pair_idx = dr.arange(wt.UInt32, pair_count)
    rx_idx = pair_idx // wt.UInt32(num_tx)
    tx_idx = pair_idx % wt.UInt32(num_tx)
    pair_rx = wt.Point3f(
        dr.gather(wt.Float, rx_positions.x, rx_idx),
        dr.gather(wt.Float, rx_positions.y, rx_idx),
        dr.gather(wt.Float, rx_positions.z, rx_idx),
    )
    pair_tx = wt.Point3f(
        dr.gather(wt.Float, tx_pos.x, tx_idx),
        dr.gather(wt.Float, tx_pos.y, tx_idx),
        dr.gather(wt.Float, tx_pos.z, tx_idx),
    )

    ray_dir = pair_rx - pair_tx
    distance = dr.norm(ray_dir)
    active_rx_polarization = effective_rx_polarization(rx_polarization, tx_polarization)
    coeff, blocked = _compute_los_field(scene, pair_rx, pair_tx, wavelength, k)
    tx_pol_dir = project_real_polarization_to_ray(tx_polarization, ray_dir)
    field_vec = vector_from_scalar(coeff, tx_pol_dir)
    scalar_coeff = scalarize_vector_to_polarization(field_vec, ray_dir, active_rx_polarization)
    keep = dr.compress(~blocked)
    if dr.width(keep) == 0:
        return empty_raw_paths(depth=1, return_geometry=False)

    kept_ray_dir = dr.gather(wt.Vector3f, ray_dir, keep)
    theta, phi = spherical_angles(kept_ray_dir)
    keep_count = dr.width(keep)
    return finalize_raw_paths(
        rx_index=dr.gather(wt.UInt32, rx_idx, keep),
        tx_index=dr.gather(wt.UInt32, tx_idx, keep),
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


def _gather_tx_position(tx_positions, tx_index: int) -> wt.Point3f:
    index = dr.full(wt.UInt32, int(tx_index), 1)
    return wt.Point3f(
        dr.gather(wt.Float, tx_positions.x, index),
        dr.gather(wt.Float, tx_positions.y, index),
        dr.gather(wt.Float, tx_positions.z, index),
    )


def _discover_reflection_detail(
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
):
    sampling_axis, sampling_plane_position, sampling_bounds = _sampling_frame_from_positions(
        rx_positions
    )
    return discover_reflection_paths(
        tx=Tx(position=tx_pos, polarization=tx_polarization),
        scene=scene,
        wave=Wave(wavelength=wavelength, k=k),
        n_rays=n_rays,
        max_reflections=max_reflections,
        mode=mode,
        material=Material(reflection_coef=1.0),
        ray_sampling="full_sphere",
        sampling_axis=sampling_axis,
        sampling_plane_position=sampling_plane_position,
        sampling_bounds=sampling_bounds,
    )


def _raw_with_tx_index(raw: Mapping[str, object], tx_index: int) -> dict[str, object]:
    stamped = dict(raw)
    stamped["tx_index"] = dr.full(wt.UInt32, int(tx_index), int(dr.width(stamped["rx_index"])))
    return stamped


def _empty_source_path_set(chain_depth: int) -> SourcePathSet:
    depth = max(1, int(chain_depth))
    return SourcePathSet(
        image_source=arrays.empty_point3(),
        discovery_count=dr.zeros(wt.UInt32, 0),
        chain_depth=depth,
        n_paths=0,
        path_prim_idx=tuple(dr.zeros(wt.Int32, 0) for _ in range(depth)),
        path_plane_point=tuple(arrays.empty_point3() for _ in range(depth)),
        path_plane_normal=tuple(arrays.empty_vector3() for _ in range(depth)),
        path_hit_point=tuple(arrays.empty_point3() for _ in range(depth)),
    )


def _merge_source_path_sets_for_bounce(
    details,
    bounce_idx: int,
) -> tuple[SourcePathSet, wt.UInt32]:
    depth = int(bounce_idx) + 1
    path_sets: list[SourcePathSet] = []
    owner_parts = []
    for tx_index, detail_payload in enumerate(details):
        if detail_payload is None:
            continue
        detail = coerce_reflection_trace_detail(detail_payload)
        if bounce_idx >= len(detail.source_paths_per_bounce):
            continue
        paths = detail.source_paths_per_bounce[bounce_idx]
        if paths is None or int(paths.n_paths) <= 0:
            continue
        if int(paths.chain_depth) != depth:
            raise ValueError("Reflection source path depths must match their bounce slot.")
        path_sets.append(paths)
        owner_parts.append(dr.full(wt.UInt32, int(tx_index), int(paths.n_paths)))

    if not path_sets:
        return _empty_source_path_set(depth), dr.zeros(wt.UInt32, 0)

    return (
        SourcePathSet(
            image_source=arrays.concat_points([paths.image_source for paths in path_sets]),
            discovery_count=arrays.concat_uints([paths.discovery_count for paths in path_sets]),
            chain_depth=depth,
            n_paths=sum(int(paths.n_paths) for paths in path_sets),
            path_prim_idx=tuple(
                arrays.concat_ints([paths.prim_idx(slot) for paths in path_sets])
                for slot in range(depth)
            ),
            path_plane_point=tuple(
                arrays.concat_points([paths.plane_point(slot) for paths in path_sets])
                for slot in range(depth)
            ),
            path_plane_normal=tuple(
                arrays.concat_vectors([paths.plane_normal(slot) for paths in path_sets])
                for slot in range(depth)
            ),
            path_hit_point=tuple(
                arrays.concat_points([paths.hit_point(slot) for paths in path_sets])
                for slot in range(depth)
            ),
        ),
        arrays.concat_uints(owner_parts),
    )


def _merged_trace_detail(details, source_paths_per_bounce) -> TraceDetail:
    first = next(
        (coerce_reflection_trace_detail(detail) for detail in details if detail is not None),
        None,
    )
    if first is None:
        return TraceDetail(
            reflection_model="materialized",
            reflection_model_source="default",
            reflection_gain=1.0,
            source_paths_per_bounce=tuple(source_paths_per_bounce),
        )
    return TraceDetail(
        reflection_model=first.reflection_model,
        reflection_model_source=first.reflection_model_source,
        reflection_gain=first.reflection_gain,
        source_paths_per_bounce=tuple(source_paths_per_bounce),
        reflection_transition_mode=first.reflection_transition_mode,
        reflection_f_weight_boundary_radius_wavelengths=first.reflection_f_weight_boundary_radius_wavelengths,
        reflection_f_weight_max_edges_per_slot=first.reflection_f_weight_max_edges_per_slot,
        reflection_secondary_visibility_mode=first.reflection_secondary_visibility_mode,
    )


def _details_use_hard_reflection(details) -> bool:
    for detail_payload in details:
        if detail_payload is None:
            continue
        detail = coerce_reflection_trace_detail(detail_payload)
        if detail.reflection_transition_mode != "hard":
            return False
        if detail.reflection_secondary_visibility_mode != "hard":
            return False
    return True


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
    min_ray_contribution_threshold,
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
            tx=Tx(position=tx_pos, polarization=tx_polarization),
            scene=scene,
            wave=Wave(wavelength=wavelength, k=k),
            n_rays=n_rays,
            max_reflections=max_reflections,
            mode=mode,
            material=Material(reflection_coef=1.0),
            ray_sampling="full_sphere",
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
    type_slot_parts: list[list[object]] = []
    vertex_slot_parts: list[list[object]] | None = [] if return_geometry else None
    normal_slot_parts: list[list[object]] | None = [] if return_geometry else None
    object_slot_parts: list[list[object]] | None = [] if return_geometry else None
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
            image_source = arrays.gather_point3(paths.image_source, path_idx)
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
                wave=Wave(wavelength=wavelength, k=k),
                tx=Tx(position=tx_pos, polarization=tx_polarization),
                return_geometry=return_geometry,
                return_endpoints=not return_geometry,
                epc_descriptor=epc_descriptor,
            )
            keep_idx = dr.compress(valid)
            if dr.width(keep_idx) == 0:
                continue

            rx_idx_keep = dr.gather(wt.UInt32, rx_idx, keep_idx)
            path_idx_keep = dr.gather(wt.UInt32, path_idx, keep_idx)
            image_source_keep = arrays.gather_point3(image_source, keep_idx)
            target_pos_keep = dr.gather(wt.Point3f, target_pos, keep_idx)
            tx_pos_keep = dr.gather(wt.Point3f, geometry["tx_pos"], keep_idx)
            if return_geometry:
                first_hit_keep = dr.gather(wt.Point3f, geometry["hit_points"][0], keep_idx)
                last_hit_keep = dr.gather(wt.Point3f, geometry["hit_points"][-1], keep_idx)
            else:
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
            if return_geometry:
                while len(type_slot_parts) < chain_depth:
                    backfill = [dr.zeros(wt.Int32, total_paths)] if total_paths > 0 else []
                    type_slot_parts.append(backfill)
                    vertex_backfill = [arrays.zeros_point3(total_paths)] if total_paths > 0 else []
                    normal_backfill = [arrays.zeros_vector3(total_paths)] if total_paths > 0 else []
                    object_backfill = [dr.full(wt.Int32, -1, total_paths)] if total_paths > 0 else []
                    vertex_slot_parts.append(vertex_backfill)
                    normal_slot_parts.append(normal_backfill)
                    object_slot_parts.append(object_backfill)
                for slot in range(len(type_slot_parts)):
                    if slot < chain_depth:
                        type_slot_parts[slot].append(
                            dr.full(wt.Int32, InteractionType.REFLECTION, keep_count)
                        )
                        hit_slot = dr.gather(wt.Point3f, geometry["hit_points"][slot], keep_idx)
                        normal_slot = dr.gather(wt.Vector3f, geometry["normals"][slot], keep_idx)
                        prim_slot = dr.gather(wt.Int32, geometry["prim_indices"][slot], keep_idx)
                        vertex_slot_parts[slot].append(hit_slot)
                        normal_slot_parts[slot].append(normal_slot)
                        object_slot_parts[slot].append(scene.gather_structure_indices(prim_slot))
                    else:
                        type_slot_parts[slot].append(dr.zeros(wt.Int32, keep_count))
                        vertex_slot_parts[slot].append(arrays.zeros_point3(keep_count))
                        normal_slot_parts[slot].append(arrays.zeros_vector3(keep_count))
                        object_slot_parts[slot].append(dr.full(wt.Int32, -1, keep_count))
            total_paths += keep_count

    if total_paths == 0:
        return empty_raw_paths(depth=1, return_geometry=return_geometry), detail_payload

    type_slots = None
    vertex_slots = None
    normal_slots = None
    object_slots = None
    if return_geometry:
        type_slots = tuple(arrays.concat_ints(parts) for parts in type_slot_parts)
        vertex_slots = tuple(arrays.concat_points(parts) for parts in vertex_slot_parts)
        normal_slots = tuple(arrays.concat_vectors(parts) for parts in normal_slot_parts)
        object_slots = tuple(arrays.concat_ints(parts) for parts in object_slot_parts)

    return (
        _finalize_reflection_path_refs(
            rx_index=arrays.concat_uints(rx_index_parts),
            path_group_index=arrays.concat_uints(path_group_index_parts),
            path_idx=arrays.concat_uints(path_idx_parts),
            a=arrays.concat_complex(a_parts),
            tau=arrays.concat_floats(tau_parts),
            tx_pos=tx_pos,
            rx_positions=rx_positions,
            scene=scene,
            reflection_detail=detail_payload,
            wavelength=wavelength,
            tx_polarization=tx_polarization,
            theta_t=arrays.concat_floats(theta_t_parts),
            phi_t=arrays.concat_floats(phi_t_parts),
            theta_r=arrays.concat_floats(theta_r_parts),
            phi_r=arrays.concat_floats(phi_r_parts),
            path_depth=arrays.concat_uints(path_depth_parts),
            type_slots=type_slots,
            vertex_slots=vertex_slots,
            normal_slots=normal_slots,
            object_slots=object_slots,
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


def collect_reflection_paths_for_transmitters(
    *,
    scene,
    rx_positions,
    tx_positions,
    wavelength,
    k,
    n_rays,
    max_reflections,
    mode,
    tx_polarization,
    rx_polarization,
    min_ray_contribution_threshold,
    use_scene_materials,
    return_geometry: bool,
    reflection_details=None,
):
    del min_ray_contribution_threshold, use_scene_materials
    tx_count = int(dr.width(tx_positions.x))
    if tx_count <= 0:
        return (empty_raw_paths(depth=1, return_geometry=return_geometry),), tuple()
    if n_rays <= 0 or max_reflections <= 0:
        return (
            (empty_raw_paths(depth=1, return_geometry=return_geometry),),
            tuple(None for _ in range(tx_count)),
        )

    details = []
    for tx_index in range(tx_count):
        if reflection_details is not None and tx_index < len(reflection_details):
            detail_payload = reflection_details[tx_index]
        else:
            detail_payload = None
        if detail_payload is None:
            detail_payload = _discover_reflection_detail(
                scene=scene,
                rx_positions=rx_positions,
                tx_pos=_gather_tx_position(tx_positions, tx_index),
                wavelength=wavelength,
                k=k,
                n_rays=n_rays,
                max_reflections=max_reflections,
                mode=mode,
                tx_polarization=tx_polarization,
            )
        details.append(detail_payload)

    if not _details_use_hard_reflection(details):
        raw_collections = []
        for tx_index, detail_payload in enumerate(details):
            raw, _ = collect_reflection_paths(
                scene=scene,
                rx_positions=rx_positions,
                tx_pos=_gather_tx_position(tx_positions, tx_index),
                wavelength=wavelength,
                k=k,
                n_rays=n_rays,
                max_reflections=max_reflections,
                mode=mode,
                tx_polarization=tx_polarization,
                rx_polarization=rx_polarization,
                min_ray_contribution_threshold=0.0,
                use_scene_materials=True,
                return_geometry=return_geometry,
                reflection_detail=detail_payload,
            )
            raw_collections.append(_raw_with_tx_index(raw, tx_index))
        return tuple(raw_collections), tuple(details)

    source_paths_per_bounce = []
    path_tx_index_per_bounce = []
    for bounce_idx in range(int(max_reflections)):
        paths, path_tx_index = _merge_source_path_sets_for_bounce(details, bounce_idx)
        source_paths_per_bounce.append(paths)
        path_tx_index_per_bounce.append(path_tx_index)

    detail_payload = _merged_trace_detail(details, source_paths_per_bounce)
    active_rx_polarization = effective_rx_polarization(rx_polarization, tx_polarization)

    rx_index_parts = []
    tx_index_parts = []
    path_group_index_parts = []
    path_idx_parts = []
    a_parts = []
    tau_parts = []
    theta_t_parts = []
    phi_t_parts = []
    theta_r_parts = []
    phi_r_parts = []
    path_depth_parts = []
    type_slot_parts: list[list[object]] = []
    vertex_slot_parts: list[list[object]] | None = [] if return_geometry else None
    normal_slot_parts: list[list[object]] | None = [] if return_geometry else None
    object_slot_parts: list[list[object]] | None = [] if return_geometry else None
    total_paths = 0

    num_rx = int(dr.width(rx_positions.x))
    for bounce_idx, paths in enumerate(source_paths_per_bounce):
        chain_depth = int(paths.chain_depth)
        n_paths = int(paths.n_paths)
        if chain_depth <= 0 or n_paths <= 0:
            continue
        path_tx_index_lookup = path_tx_index_per_bounce[bounce_idx]
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
            tx_idx = dr.gather(wt.UInt32, path_tx_index_lookup, path_idx)
            image_source = arrays.gather_point3(paths.image_source, path_idx)
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
                wave=Wave(wavelength=wavelength, k=k),
                tx=Tx(position=_gather_tx_position(tx_positions, 0), polarization=tx_polarization),
                return_geometry=return_geometry,
                return_endpoints=not return_geometry,
                epc_descriptor=epc_descriptor,
            )
            keep_idx = dr.compress(valid)
            if dr.width(keep_idx) == 0:
                continue

            rx_idx_keep = dr.gather(wt.UInt32, rx_idx, keep_idx)
            tx_idx_keep = dr.gather(wt.UInt32, tx_idx, keep_idx)
            path_idx_keep = dr.gather(wt.UInt32, path_idx, keep_idx)
            image_source_keep = arrays.gather_point3(image_source, keep_idx)
            target_pos_keep = dr.gather(wt.Point3f, target_pos, keep_idx)
            tx_pos_keep = dr.gather(wt.Point3f, geometry["tx_pos"], keep_idx)
            if return_geometry:
                first_hit_keep = dr.gather(wt.Point3f, geometry["hit_points"][0], keep_idx)
                last_hit_keep = dr.gather(wt.Point3f, geometry["hit_points"][-1], keep_idx)
            else:
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
            tx_index_parts.append(tx_idx_keep)
            path_group_index_parts.append(dr.full(wt.UInt32, bounce_idx, keep_count))
            path_idx_parts.append(path_idx_keep)
            a_parts.append(scalar_coeff)
            tau_parts.append(dr.norm(target_pos_keep - image_source_keep) / 299792458.0)
            theta_t_parts.append(theta_t)
            phi_t_parts.append(phi_t)
            theta_r_parts.append(theta_r)
            phi_r_parts.append(phi_r)
            path_depth_parts.append(dr.full(wt.UInt32, chain_depth, keep_count))
            if return_geometry:
                while len(type_slot_parts) < chain_depth:
                    backfill = [dr.zeros(wt.Int32, total_paths)] if total_paths > 0 else []
                    type_slot_parts.append(backfill)
                    vertex_backfill = [arrays.zeros_point3(total_paths)] if total_paths > 0 else []
                    normal_backfill = [arrays.zeros_vector3(total_paths)] if total_paths > 0 else []
                    object_backfill = [dr.full(wt.Int32, -1, total_paths)] if total_paths > 0 else []
                    vertex_slot_parts.append(vertex_backfill)
                    normal_slot_parts.append(normal_backfill)
                    object_slot_parts.append(object_backfill)
                for slot in range(len(type_slot_parts)):
                    if slot < chain_depth:
                        type_slot_parts[slot].append(
                            dr.full(wt.Int32, InteractionType.REFLECTION, keep_count)
                        )
                        hit_slot = dr.gather(wt.Point3f, geometry["hit_points"][slot], keep_idx)
                        normal_slot = dr.gather(wt.Vector3f, geometry["normals"][slot], keep_idx)
                        prim_slot = dr.gather(wt.Int32, geometry["prim_indices"][slot], keep_idx)
                        vertex_slot_parts[slot].append(hit_slot)
                        normal_slot_parts[slot].append(normal_slot)
                        object_slot_parts[slot].append(scene.gather_structure_indices(prim_slot))
                    else:
                        type_slot_parts[slot].append(dr.zeros(wt.Int32, keep_count))
                        vertex_slot_parts[slot].append(arrays.zeros_point3(keep_count))
                        normal_slot_parts[slot].append(arrays.zeros_vector3(keep_count))
                        object_slot_parts[slot].append(dr.full(wt.Int32, -1, keep_count))
            total_paths += keep_count

    if total_paths == 0:
        return (empty_raw_paths(depth=1, return_geometry=return_geometry),), tuple(details)

    type_slots = None
    vertex_slots = None
    normal_slots = None
    object_slots = None
    if return_geometry:
        type_slots = tuple(arrays.concat_ints(parts) for parts in type_slot_parts)
        vertex_slots = tuple(arrays.concat_points(parts) for parts in vertex_slot_parts)
        normal_slots = tuple(arrays.concat_vectors(parts) for parts in normal_slot_parts)
        object_slots = tuple(arrays.concat_ints(parts) for parts in object_slot_parts)

    return (
        (
            _finalize_reflection_path_refs(
                rx_index=arrays.concat_uints(rx_index_parts),
                tx_index=arrays.concat_uints(tx_index_parts),
                path_group_index=arrays.concat_uints(path_group_index_parts),
                path_idx=arrays.concat_uints(path_idx_parts),
                a=arrays.concat_complex(a_parts),
                tau=arrays.concat_floats(tau_parts),
                tx_pos=_gather_tx_position(tx_positions, 0),
                rx_positions=rx_positions,
                scene=scene,
                reflection_detail=detail_payload,
                wavelength=wavelength,
                tx_polarization=tx_polarization,
                theta_t=arrays.concat_floats(theta_t_parts),
                phi_t=arrays.concat_floats(phi_t_parts),
                theta_r=arrays.concat_floats(theta_r_parts),
                phi_r=arrays.concat_floats(phi_r_parts),
                path_depth=arrays.concat_uints(path_depth_parts),
                type_slots=type_slots,
                vertex_slots=vertex_slots,
                normal_slots=normal_slots,
                object_slots=object_slots,
                metadata={
                    "n_paths": total_paths,
                    "per_bounce_counts": tuple(int(paths.n_paths) for paths in source_paths_per_bounce),
                    "batched_transmitter_count": tx_count,
                },
            ),
        ),
        tuple(details),
    )


def _edge_object_indices(scene, edge_data):
    if edge_data is None or edge_data["n_edges"] == 0:
        return dr.zeros(wt.Int32, 0)
    return _edge_owner_structure_idx(
        scene,
        edge_data["adjacent_face0"],
        edge_data["adjacent_face1"],
    )


def _edge_owner_structure_idx(scene, adjacent_face0, adjacent_face1):
    face0 = wt.Int32(adjacent_face0)
    face1 = wt.Int32(adjacent_face1)
    use_face0 = face0 >= wt.Int32(0)
    selected_face = dr.select(use_face0, face0, face1)
    valid = selected_face >= wt.Int32(0)
    return scene.gather_structure_indices(wt.UInt32(dr.select(valid, selected_face, wt.Int32(0))), valid_mask=valid)


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
        return State.gather_path_export_eval(state_arrays, indices)
    return gather_state_arrays(state_arrays, indices)


def _gather_path_replay_state_fields(state_arrays, indices):
    if is_path_export_reduced_state_arrays(state_arrays):
        return State.gather_path_export_replay(state_arrays, indices)
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
    del ignore_emitter_structure_visibility
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
        _, pair_vector, pair_valid = ForwardEval.to_targets(
            batch_states,
            batch_rx,
            Wave(wavelength=wavelength, k=k),
            return_vector=True,
            return_valid=True,
            material=Material(reflection_coef=1.0),
            scene=scene,
            smooth_exterior_shadow=True,
            tx=Tx(position=tx_pos, polarization=tx_polarization),
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
    rx_index = arrays.concat_uints(rx_index_parts)
    local_rx_index = arrays.concat_uints(local_rx_index_parts)
    state_idx = arrays.concat_uints(state_idx_parts)
    a = arrays.concat_complex(a_parts)
    tau = arrays.concat_floats(tau_parts)
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
    "collect_reflection_paths_for_transmitters",
    "epc_reflection_chain_to_target",
]
