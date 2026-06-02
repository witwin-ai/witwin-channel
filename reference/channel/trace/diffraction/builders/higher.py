"""Higher-order and inserted-reflection diffraction state construction."""

import math
import time

import drjit as dr
import rayd
import witwin as wt

from ....utils.polarization import (
    reflect_field_vector,
    vector_scale,
    vector_select,
    vector_zero,
)
from ....utils.raygen import generate_circle_directions, generate_sphere_directions
from ....utils.drjit_ops import ArrayInit
from ..constants import (
    APPROX_MODE_RECURSIVE_DIFFRACTION,
    APPROX_MODE_SAMPLED_INSERTED_REFLECTION,
    APPROX_MODE_SAMPLED_INSERTED_REFLECTION_CHAIN,
    APPROX_MODE_SAMPLED_REFLECTION_PREFIX_CHAIN,
    SPEED_OF_LIGHT,
    _cartesian_chunk_size,
    _state_history_size,
)
from ..finite_wedge import require_edge_data_line_bounds
from ..field import _edge_state_field_to_targets
from ..geometry import (
    _intersect_rays_ad_with_prim,
    _point_source_field,
    _point_source_field_normal_derivative,
    _segment_visibility_mask,
    _surface_reflection_coefficient,
)
from ..state import (
    _empty_state_arrays,
    _finalize_state_lineage,
    _make_state_arrays,
    _state_has_cold_metadata,
    _state_ids,
    _state_lineage_store,
)
from witwin.channel.kernels.trace.packed_state import (
    concat_state_arrays,
    gather_field_evaluation_state_fields,
    gather_inserted_reflection_state_fields,
    subset_state_arrays,
)
from witwin.channel.kernels.trace.cartesian_filter.native_impl import (
    compact_index_pairs,
    deduplicate_cartesian_pairs,
)
from . import _state_edge_face_material_response, _vector_field_power

__all__ = [
    "_build_higher_order_state_arrays",
    "_build_inserted_reflection_state_arrays",
    "_resolve_higher_order_candidate_backend",
]


_HIGHER_ORDER_CANDIDATE_BACKENDS = {"auto", "rayd_edge_bvh", "bruteforce"}
_HIGHER_ORDER_EDGE_BVH_PROBE_COUNT = 18
_HIGHER_ORDER_EDGE_BVH_PROBE_RADIUS_SCALE = 0.6
_HIGHER_ORDER_EDGE_BVH_PROBE_RADIUS_MIN = 0.5
_HIGHER_ORDER_EDGE_BVH_PROBE_RADIUS_MAX = 4.0
_HIGHER_ORDER_EDGE_BVH_EXACT_COMPLETION_PAIR_LIMIT = 1 << 12


def _complex_select(mask, true_value, false_value):
    return wt.Complex2f(
        dr.select(mask, true_value.real, false_value.real),
        dr.select(mask, true_value.imag, false_value.imag),
    )


def _ensure_finalized_source_lineage(state_arrays):
    if not _state_has_cold_metadata(state_arrays):
        return state_arrays
    state_ids = _state_ids(state_arrays)
    lineage_store = _state_lineage_store(state_arrays)
    if (
        state_ids is not None
        and lineage_store is not None
        and not bool(dr.any(wt.Int32(state_ids) < 0))
    ):
        return state_arrays
    next_state_id = 0 if lineage_store is None else int(lineage_store.get("n_states", 0))
    finalized, _, _ = _finalize_state_lineage(
        state_arrays,
        lineage_store=lineage_store,
        next_state_id=next_state_id,
    )
    return finalized


def _rayd_edge_query_handle(scene):
    if scene is None:
        return None
    handle = getattr(scene, "_rayd_scene", None)
    if handle is None and not hasattr(scene, "_rayd_scene") and all(
        hasattr(scene, name) for name in ("nearest_edge", "set_edge_mask", "edge_mask")
    ):
        handle = scene
    if handle is None:
        return None
    required = ("nearest_edge", "set_edge_mask", "edge_mask")
    if not all(hasattr(handle, name) for name in required):
        return None
    return handle


def _resolve_higher_order_candidate_backend(scene, edge_data, global_to_local_idx, candidate_backend="auto"):
    if candidate_backend not in _HIGHER_ORDER_CANDIDATE_BACKENDS:
        raise ValueError(
            f"Unsupported higher-order candidate backend '{candidate_backend}'. "
            "Supported values are 'auto', 'rayd_edge_bvh', and 'bruteforce'."
        )

    rayd_handle = _rayd_edge_query_handle(scene)
    rayd_ready = (
        rayd_handle is not None
        and edge_data is not None
        and edge_data["n_edges"] > 0
        and edge_data.get("global_idx") is not None
        and global_to_local_idx is not None
    )
    if candidate_backend == "auto":
        candidate_backend = "rayd_edge_bvh"
    if candidate_backend == "rayd_edge_bvh" and not rayd_ready:
        raise RuntimeError(
            "Higher-order diffraction now requires RayD edge BVH candidate construction. "
            "Provide an active RayD scene plus selected-edge global/local indexing, or use "
            "candidate_backend='bruteforce' explicitly in tests."
        )
    return candidate_backend


def _selected_edge_bvh_mask(edge_data, current_mask):
    current_mask = current_mask if hasattr(current_mask, "index") else wt.Bool(current_mask)
    n_edges_total = dr.width(current_mask)
    if (
        edge_data is None
        or edge_data.get("global_idx") is None
        or edge_data["n_edges"] <= 0
        or n_edges_total == 0
    ):
        return dr.full(wt.Bool, False, n_edges_total)

    selected_flags = dr.zeros(wt.UInt32, n_edges_total)
    global_idx = wt.Int32(edge_data["global_idx"])
    valid = global_idx >= 0
    safe_global_idx = wt.UInt32(dr.select(valid, global_idx, wt.Int32(0)))
    dr.scatter(
        selected_flags,
        dr.full(wt.UInt32, 1, dr.width(global_idx)),
        safe_global_idx,
        valid,
    )
    return current_mask & (selected_flags != wt.UInt32(0))


def _deduplicate_candidate_pairs(prev_idx, edge_idx, n_edges):
    if dr.width(prev_idx) == 0:
        return dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0)
    return deduplicate_cartesian_pairs(prev_idx, edge_idx, n_edges)


def _build_bruteforce_candidate_pairs(prev_state_arrays, prev_start, chunk_n_prev, n_edges):
    if chunk_n_prev <= 0 or n_edges <= 0:
        return dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0), 0
    n_pairs = chunk_n_prev * n_edges
    pair_idx = dr.arange(wt.UInt32, n_pairs)
    prev_idx_all = pair_idx // n_edges + wt.UInt32(prev_start)
    edge_idx_all = pair_idx % n_edges
    prev_edge_idx_all = dr.gather(wt.UInt32, prev_state_arrays["edge_idx"], prev_idx_all)
    distinct_edge = edge_idx_all != prev_edge_idx_all
    prev_idx, edge_idx = compact_index_pairs(prev_idx_all, edge_idx_all, distinct_edge)
    return prev_idx, edge_idx, int(n_pairs)


def _build_higher_order_bvh_candidate_pairs(
    prev_state_arrays,
    edge_data,
    prev_start,
    chunk_n_prev,
    scene,
    global_to_local_idx,
):
    if chunk_n_prev <= 0:
        return dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0)

    n_probes = chunk_n_prev * _HIGHER_ORDER_EDGE_BVH_PROBE_COUNT
    probe_idx = dr.arange(wt.UInt32, n_probes)
    state_offset = probe_idx // _HIGHER_ORDER_EDGE_BVH_PROBE_COUNT
    prev_idx_all = state_offset + wt.UInt32(prev_start)
    probe_slot = probe_idx % _HIGHER_ORDER_EDGE_BVH_PROBE_COUNT
    probe_grid_slot = probe_slot % wt.UInt32(_HIGHER_ORDER_EDGE_BVH_PROBE_COUNT // 2)
    probe_u = wt.Float(probe_grid_slot // wt.UInt32(3)) - 1.0
    probe_v = wt.Float(probe_grid_slot % wt.UInt32(3)) - 1.0
    probe_sign = dr.select(
        probe_slot < wt.UInt32(_HIGHER_ORDER_EDGE_BVH_PROBE_COUNT // 2),
        wt.Float(1.0),
        wt.Float(-1.0),
    )

    edge_pos = dr.gather(wt.Point3f, prev_state_arrays["edge_pos"], prev_idx_all)
    source_pos = dr.gather(wt.Point3f, prev_state_arrays["source_pos"], prev_idx_all)
    basis_u = dr.gather(wt.Vector3f, prev_state_arrays["incident_basis_u"], prev_idx_all)
    basis_v = dr.gather(wt.Vector3f, prev_state_arrays["incident_basis_v"], prev_idx_all)
    basis_k = dr.gather(wt.Vector3f, prev_state_arrays["incident_basis_k"], prev_idx_all)
    prev_edge_idx = dr.gather(wt.UInt32, prev_state_arrays["edge_idx"], prev_idx_all)

    source_distance = dr.norm(edge_pos - source_pos)
    probe_radius = dr.clip(
        source_distance * _HIGHER_ORDER_EDGE_BVH_PROBE_RADIUS_SCALE,
        wt.Float(_HIGHER_ORDER_EDGE_BVH_PROBE_RADIUS_MIN),
        wt.Float(_HIGHER_ORDER_EDGE_BVH_PROBE_RADIUS_MAX),
    )
    origin_offset = (
        basis_u * (probe_radius * probe_u)
        + basis_v * (probe_radius * probe_v)
    )
    ray_origin = edge_pos + origin_offset
    ray_dir = basis_k * probe_sign

    nearest = scene.nearest_edge(rayd.Ray(ray_origin, ray_dir))
    valid = nearest.is_valid()
    global_edge_idx = wt.Int32(nearest.global_edge_id)
    valid = valid & (global_edge_idx >= 0)

    safe_global_idx = wt.UInt32(dr.select(valid, global_edge_idx, wt.Int32(0)))
    local_edge_idx_i32 = dr.gather(wt.Int32, global_to_local_idx, safe_global_idx)
    valid = valid & (local_edge_idx_i32 >= 0)
    valid = valid & (wt.Int32(prev_edge_idx) != local_edge_idx_i32)
    candidate_idx = dr.compress(valid)
    if dr.width(candidate_idx) == 0:
        return dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0)

    prev_idx = dr.gather(wt.UInt32, prev_idx_all, candidate_idx)
    edge_idx = dr.gather(
        wt.UInt32,
        wt.UInt32(dr.select(valid, local_edge_idx_i32, wt.Int32(0))),
        candidate_idx,
    )
    return _deduplicate_candidate_pairs(prev_idx, edge_idx, edge_data["n_edges"])


def _build_higher_order_state_arrays(
    prev_state_arrays,
    edge_data,
    k,
    scene=None,
    wavelength=None,
    material_detail=None,
    reflection_coef=1.0,
    use_scene_materials=False,
    tx_polarization=(1.0, 0.0, 0.0),
    global_to_local_idx=None,
    candidate_backend="auto",
    retain_cold_metadata=True,
    preserve_candidate_topology: bool = False,
    stats=None,
):
    prev_state_arrays = _ensure_finalized_source_lineage(prev_state_arrays)
    n_prev = prev_state_arrays["n_states"]
    n_edges = edge_data["n_edges"]
    history_size = _state_history_size(prev_state_arrays)
    total_start = time.perf_counter()
    if stats is None:
        stats = {}
    stats.clear()
    stats.update(
        {
            "candidate_backend": None,
            "topology_preserved": bool(preserve_candidate_topology),
            "input_states": int(n_prev),
            "edge_count": int(n_edges),
            "chunk_count": 0,
            "candidate_raw_count": 0,
            "max_candidate_pairs_per_chunk": 0,
            "candidate_unique_count": 0,
            "visibility_input_count": 0,
            "visibility_kept_count": 0,
            "field_input_count": 0,
            "field_kept_count": 0,
            "output_states": 0,
            "timing": {
                "candidate_seconds": 0.0,
                "visibility_seconds": 0.0,
                "incident_field_seconds": 0.0,
                "material_response_seconds": 0.0,
                "state_pack_seconds": 0.0,
                "state_array_assembly_seconds": 0.0,
                "concat_seconds": 0.0,
                "total_seconds": 0.0,
            },
        }
    )
    if n_prev == 0 or n_edges == 0:
        stats["timing"]["total_seconds"] = time.perf_counter() - total_start
        return _empty_state_arrays(history_size=history_size)
    line_min_all, line_max_all = require_edge_data_line_bounds(
        edge_data,
        context="_build_higher_order_state_arrays",
    )

    resolved_candidate_backend = _resolve_higher_order_candidate_backend(
        scene,
        edge_data,
        global_to_local_idx,
        candidate_backend=candidate_backend,
    )
    stats["candidate_backend"] = resolved_candidate_backend
    chunk_states = []
    rayd_handle = _rayd_edge_query_handle(scene) if resolved_candidate_backend == "rayd_edge_bvh" else None
    prev_chunk_size = _cartesian_chunk_size(
        n_prev,
        n_edges if resolved_candidate_backend == "bruteforce" else _HIGHER_ORDER_EDGE_BVH_PROBE_COUNT,
    )
    saved_edge_mask = None
    if rayd_handle is not None:
        saved_edge_mask = rayd_handle.edge_mask()
        rayd_handle.set_edge_mask(_selected_edge_bvh_mask(edge_data, saved_edge_mask))
        if hasattr(rayd_handle, "sync"):
            rayd_handle.sync()

    try:
        for prev_start in range(0, n_prev, prev_chunk_size):
            chunk_n_prev = min(prev_chunk_size, n_prev - prev_start)
            stats["chunk_count"] += 1
            candidate_start = time.perf_counter()
            chunk_candidate_pairs = 0
            if resolved_candidate_backend == "bruteforce":
                prev_idx, edge_idx, chunk_candidate_pairs = _build_bruteforce_candidate_pairs(
                    prev_state_arrays,
                    prev_start,
                    chunk_n_prev,
                    n_edges,
                )
                stats["candidate_raw_count"] += int(chunk_candidate_pairs)
                if dr.width(prev_idx) == 0:
                    continue
            else:
                if chunk_n_prev * n_edges <= _HIGHER_ORDER_EDGE_BVH_EXACT_COMPLETION_PAIR_LIMIT:
                    prev_idx, edge_idx, chunk_candidate_pairs = _build_bruteforce_candidate_pairs(
                        prev_state_arrays,
                        prev_start,
                        chunk_n_prev,
                        n_edges,
                    )
                else:
                    chunk_candidate_pairs = int(chunk_n_prev * _HIGHER_ORDER_EDGE_BVH_PROBE_COUNT)
                    prev_idx, edge_idx = _build_higher_order_bvh_candidate_pairs(
                        prev_state_arrays,
                        edge_data,
                        prev_start,
                        chunk_n_prev,
                        scene,
                        global_to_local_idx,
                    )
                if dr.width(prev_idx) == 0:
                    stats["candidate_raw_count"] += chunk_candidate_pairs
                    stats["max_candidate_pairs_per_chunk"] = max(
                        int(stats["max_candidate_pairs_per_chunk"]),
                        int(chunk_candidate_pairs),
                    )
                    stats["timing"]["candidate_seconds"] += time.perf_counter() - candidate_start
                    continue
                stats["candidate_raw_count"] += chunk_candidate_pairs
            stats["max_candidate_pairs_per_chunk"] = max(
                int(stats["max_candidate_pairs_per_chunk"]),
                int(chunk_candidate_pairs),
            )
            stats["candidate_unique_count"] += int(dr.width(prev_idx))
            stats["timing"]["candidate_seconds"] += time.perf_counter() - candidate_start

            visibility_mask = None
            if scene is not None:
                stats["visibility_input_count"] += int(dr.width(prev_idx))
                visibility_start = time.perf_counter()
                if preserve_candidate_topology and candidate_backend == "rayd_edge_bvh":
                    visible = dr.full(wt.Bool, True, dr.width(prev_idx))
                else:
                    prev_edge_pos = dr.gather(wt.Point3f, prev_state_arrays["edge_pos"], prev_idx)
                    edge_pos = dr.gather(wt.Point3f, edge_data["pos"], edge_idx)
                    prev_adjacent_face0 = dr.gather(wt.Int32, prev_state_arrays["adjacent_face0"], prev_idx)
                    prev_adjacent_face1 = dr.gather(wt.Int32, prev_state_arrays["adjacent_face1"], prev_idx)
                    next_adjacent_face0 = dr.gather(wt.Int32, edge_data["adjacent_face0"], edge_idx)
                    next_adjacent_face1 = dr.gather(wt.Int32, edge_data["adjacent_face1"], edge_idx)
                    visible = _segment_visibility_mask(
                        prev_edge_pos,
                        edge_pos,
                        scene,
                        ignore_prim_idx=(
                            prev_adjacent_face0,
                            prev_adjacent_face1,
                            next_adjacent_face0,
                            next_adjacent_face1,
                        ),
                    )
                visible_count = int(dr.width(dr.compress(visible)))
                stats["visibility_kept_count"] += visible_count
                stats["timing"]["visibility_seconds"] += time.perf_counter() - visibility_start
                if preserve_candidate_topology:
                    visibility_mask = visible
                else:
                    prev_idx, edge_idx = compact_index_pairs(prev_idx, edge_idx, visible)
                if not preserve_candidate_topology and visible_count == 0:
                    continue
            else:
                visibility_mask = dr.full(wt.Bool, True, dr.width(prev_idx))

            stats["field_input_count"] += int(dr.width(prev_idx))
            field_start = time.perf_counter()
            prev_states = gather_field_evaluation_state_fields(prev_state_arrays, prev_idx)
            candidate_edge_pos = dr.gather(wt.Point3f, edge_data["pos"], edge_idx)
            candidate_adjacent_face0 = dr.gather(wt.Int32, edge_data["adjacent_face0"], edge_idx)
            candidate_adjacent_face1 = dr.gather(wt.Int32, edge_data["adjacent_face1"], edge_idx)
            (
                incident_field,
                incident_normal_derivative,
                incident_vector,
                incident_normal_derivative_vector,
            ) = _edge_state_field_to_targets(
                prev_states,
                candidate_edge_pos,
                k,
                return_normal_derivative=True,
                return_vector=True,
                wavelength=wavelength,
                material_detail=material_detail,
                smooth_exterior_shadow=bool(preserve_candidate_topology),
            )
            stats["timing"]["incident_field_seconds"] += time.perf_counter() - field_start
            field_power = _vector_field_power(incident_vector)
            field_valid = field_power > wt.Float(1e-20)
            active_mask = (
                field_valid
                if visibility_mask is None
                else (visibility_mask & field_valid)
            )
            keep_count = int(dr.width(dr.compress(active_mask)))
            stats["field_kept_count"] += keep_count
            if not preserve_candidate_topology and keep_count == 0:
                continue

            pack_start = time.perf_counter()
            if preserve_candidate_topology:
                keep_prev_idx = prev_idx
                keep_edge_idx = edge_idx
            else:
                visible_local_idx = dr.arange(wt.UInt32, dr.width(prev_idx))
                keep_idx, keep_edge_idx = compact_index_pairs(
                    visible_local_idx,
                    edge_idx,
                    active_mask,
                )
                keep_prev_idx = dr.gather(wt.UInt32, prev_idx, keep_idx)
            kept_prev_states = gather_inserted_reflection_state_fields(
                prev_state_arrays,
                keep_prev_idx,
            )
            if preserve_candidate_topology:
                keep_edge_pos = candidate_edge_pos
                keep_edge_dir = dr.gather(wt.Vector3f, edge_data["edge_dir"], keep_edge_idx)
                keep_n0 = dr.gather(wt.Vector3f, edge_data["n0"], keep_edge_idx)
                keep_nn = dr.gather(wt.Vector3f, edge_data["n_face_n"], keep_edge_idx)
                keep_wedge_n = dr.gather(wt.Float, edge_data["wedge_n"], keep_edge_idx)
                keep_edge_line_min = dr.gather(wt.Float, line_min_all, keep_edge_idx)
                keep_edge_line_max = dr.gather(wt.Float, line_max_all, keep_edge_idx)
                keep_adjacent_face0 = candidate_adjacent_face0
                keep_adjacent_face1 = candidate_adjacent_face1
            else:
                keep_edge_pos = dr.gather(wt.Point3f, candidate_edge_pos, keep_idx)
                keep_edge_dir = dr.gather(wt.Vector3f, edge_data["edge_dir"], keep_edge_idx)
                keep_n0 = dr.gather(wt.Vector3f, edge_data["n0"], keep_edge_idx)
                keep_nn = dr.gather(wt.Vector3f, edge_data["n_face_n"], keep_edge_idx)
                keep_wedge_n = dr.gather(wt.Float, edge_data["wedge_n"], keep_edge_idx)
                keep_edge_line_min = dr.gather(wt.Float, line_min_all, keep_edge_idx)
                keep_edge_line_max = dr.gather(wt.Float, line_max_all, keep_edge_idx)
                keep_adjacent_face0 = dr.gather(wt.Int32, candidate_adjacent_face0, keep_idx)
                keep_adjacent_face1 = dr.gather(wt.Int32, candidate_adjacent_face1, keep_idx)
            keep_source_pos = kept_prev_states["edge_pos"]
            keep_path_length_prefix = None
            keep_first_interaction_pos = None
            if retain_cold_metadata:
                keep_prev_path_length = kept_prev_states["path_length_prefix"]
                keep_path_length_prefix = keep_prev_path_length + dr.norm(
                    keep_edge_pos - keep_source_pos
                )
                keep_first_interaction_pos = kept_prev_states["first_interaction_pos"]
            if preserve_candidate_topology:
                zero_field = ArrayInit.complex_zero(dr.width(keep_prev_idx))
                zero_vector = vector_zero(dr.width(keep_prev_idx))
                keep_incident_field = _complex_select(active_mask, incident_field, zero_field)
                keep_incident_normal_derivative = _complex_select(
                    active_mask,
                    incident_normal_derivative,
                    zero_field,
                )
                keep_incident_vector = vector_select(
                    active_mask,
                    incident_vector,
                    zero_vector,
                )
                keep_incident_normal_derivative_vector = vector_select(
                    active_mask,
                    incident_normal_derivative_vector,
                    zero_vector,
                )
            else:
                keep_incident_field = dr.gather(wt.Complex2f, incident_field, keep_idx)
                keep_incident_normal_derivative = dr.gather(
                    wt.Complex2f,
                    incident_normal_derivative,
                    keep_idx,
                )
                keep_incident_vector = {
                    "x": dr.gather(wt.Complex2f, incident_vector["x"], keep_idx),
                    "y": dr.gather(wt.Complex2f, incident_vector["y"], keep_idx),
                    "z": dr.gather(wt.Complex2f, incident_vector["z"], keep_idx),
                }
                keep_incident_normal_derivative_vector = {
                    "x": dr.gather(wt.Complex2f, incident_normal_derivative_vector["x"], keep_idx),
                    "y": dr.gather(wt.Complex2f, incident_normal_derivative_vector["y"], keep_idx),
                    "z": dr.gather(wt.Complex2f, incident_normal_derivative_vector["z"], keep_idx),
                }
            stats["timing"]["state_pack_seconds"] += time.perf_counter() - pack_start
            material_start = time.perf_counter()
            face0_operator, face1_operator, face0_material, face1_material = _state_edge_face_material_response(
                edge_pos=keep_edge_pos,
                edge_dir=keep_edge_dir,
                n0=keep_n0,
                nn=keep_nn,
                source_pos=keep_source_pos,
                adjacent_face0=keep_adjacent_face0,
                adjacent_face1=keep_adjacent_face1,
                wavelength=wavelength,
                scene=scene,
                material_detail=material_detail,
                reflection_coef=reflection_coef,
                use_scene_materials=use_scene_materials,
                tx_polarization=tx_polarization,
            )
            stats["timing"]["material_response_seconds"] += time.perf_counter() - material_start
            keep_prev_order = kept_prev_states["order"]
            keep_order = keep_prev_order + 1
            keep_parent_state_id = _state_ids(kept_prev_states)
            keep_lineage_store = _state_lineage_store(kept_prev_states)
            keep_source_type_code = (
                kept_prev_states["source_type_code"] if retain_cold_metadata else None
            )
            keep_prefix_reflection_depth = kept_prev_states["prefix_reflection_depth"]
            keep_intermediate_reflection_depth = kept_prev_states["intermediate_reflection_depth"]
            keep_suffix_reflection_depth = kept_prev_states["suffix_reflection_depth"]
            keep_approximation_mode_code = dr.select(
                keep_intermediate_reflection_depth > wt.UInt32(0),
                wt.UInt32(APPROX_MODE_SAMPLED_INSERTED_REFLECTION_CHAIN),
                dr.select(
                    keep_prefix_reflection_depth > wt.UInt32(0),
                    wt.UInt32(APPROX_MODE_SAMPLED_REFLECTION_PREFIX_CHAIN),
                    wt.UInt32(APPROX_MODE_RECURSIVE_DIFFRACTION),
                ),
            )
            state_array_start = time.perf_counter()
            chunk_states.append(_make_state_arrays(
                edge_idx=keep_edge_idx,
                edge_pos=keep_edge_pos,
                edge_dir=keep_edge_dir,
                n0=keep_n0,
                nn=keep_nn,
                wedge_n=keep_wedge_n,
                adjacent_face0=keep_adjacent_face0,
                adjacent_face1=keep_adjacent_face1,
                source_pos=keep_source_pos,
                path_length_prefix=keep_path_length_prefix,
                first_interaction_pos=keep_first_interaction_pos,
                edge_line_min=keep_edge_line_min,
                edge_line_max=keep_edge_line_max,
                incident_field=keep_incident_field,
                incident_normal_derivative=keep_incident_normal_derivative,
                incident_vector=keep_incident_vector,
                incident_normal_derivative_vector=keep_incident_normal_derivative_vector,
                is_direct_tx=(
                    dr.full(wt.Bool, False, dr.width(keep_prev_idx))
                    if retain_cold_metadata
                    else None
                ),
                face0_operator=face0_operator,
                face1_operator=face1_operator,
                face0_material=face0_material,
                face1_material=face1_material,
                source_type_code=keep_source_type_code,
                prefix_reflection_depth=keep_prefix_reflection_depth,
                intermediate_reflection_depth=keep_intermediate_reflection_depth,
                suffix_reflection_depth=keep_suffix_reflection_depth,
                approximation_mode_code=keep_approximation_mode_code,
                order=keep_order,
                lineage_parent_state_id=keep_parent_state_id,
                lineage_last_edge_idx=wt.Int32(keep_edge_idx),
                lineage_last_reflection_depth_delta=dr.zeros(
                    wt.UInt32,
                    dr.width(keep_prev_idx),
                ),
                lineage_store=keep_lineage_store,
                retain_cold_metadata=retain_cold_metadata,
            ))
            stats["timing"]["state_array_assembly_seconds"] += time.perf_counter() - state_array_start
    finally:
        if rayd_handle is not None and saved_edge_mask is not None:
            rayd_handle.set_edge_mask(saved_edge_mask)
            if hasattr(rayd_handle, "sync"):
                rayd_handle.sync()

    concat_start = time.perf_counter()
    result = concat_state_arrays(chunk_states)
    stats["timing"]["concat_seconds"] = time.perf_counter() - concat_start
    stats["output_states"] = int(result["n_states"])
    stats["timing"]["total_seconds"] = time.perf_counter() - total_start
    return result


def _build_inserted_reflection_state_arrays(
    prev_state_arrays,
    scene,
    edge_data,
    global_to_local_idx,
    wavelength,
    k,
    n_rays,
    reflection_gain,
    reflection_material_override,
    reflection_use_scene_materials=False,
    material_detail=None,
    use_scene_materials=False,
    reflection_mode="2d",
    max_inserted_reflections_per_path=None,
    tx_polarization=(1.0, 0.0, 0.0),
    retain_cold_metadata=True,
    stats=None,
):
    prev_state_arrays = _ensure_finalized_source_lineage(prev_state_arrays)
    history_size = _state_history_size(prev_state_arrays)
    total_start = time.perf_counter()
    if stats is None:
        stats = {}
    stats.clear()
    stats.update(
        {
            "input_states": int(prev_state_arrays["n_states"]),
            "eligible_states": 0,
            "rays_per_state": 0,
            "chunk_count": 0,
            "total_rays_cast": 0,
            "hit_count": 0,
            "candidate_slot_count": 0,
            "visibility_kept_count": 0,
            "field_kept_count": 0,
            "output_states": 0,
            "timing": {
                "eligibility_seconds": 0.0,
                "ray_setup_seconds": 0.0,
                "intersection_seconds": 0.0,
                "field_to_hit_seconds": 0.0,
                "surface_reflection_seconds": 0.0,
                "visibility_seconds": 0.0,
                "incident_field_seconds": 0.0,
                "state_pack_seconds": 0.0,
                "material_response_seconds": 0.0,
                "state_array_assembly_seconds": 0.0,
                "chunk_concat_seconds": 0.0,
                "concat_seconds": 0.0,
                "total_seconds": 0.0,
            },
        }
    )
    if (
        scene is None or scene.tri_data_gpu is None or edge_data is None or global_to_local_idx is None
        or prev_state_arrays["n_states"] == 0 or n_rays <= 0
    ):
        stats["timing"]["total_seconds"] = time.perf_counter() - total_start
        return _empty_state_arrays(history_size=history_size)
    line_min_all, line_max_all = require_edge_data_line_bounds(
        edge_data,
        context="_build_inserted_reflection_state_arrays",
    )

    eligibility_start = time.perf_counter()
    if max_inserted_reflections_per_path is None:
        eligible = dr.full(wt.Bool, True, prev_state_arrays["n_states"])
    else:
        eligible = prev_state_arrays["intermediate_reflection_depth"] < wt.UInt32(
            max(0, int(max_inserted_reflections_per_path))
        )
    prev_state_arrays = subset_state_arrays(prev_state_arrays, eligible)
    stats["timing"]["eligibility_seconds"] = time.perf_counter() - eligibility_start
    stats["eligible_states"] = int(prev_state_arrays["n_states"])
    if prev_state_arrays["n_states"] == 0:
        stats["timing"]["total_seconds"] = time.perf_counter() - total_start
        return _empty_state_arrays(history_size=history_size)

    n_states = prev_state_arrays["n_states"]
    rays_per_state = max(8, int(math.ceil(n_rays / max(1, n_states))))
    stats["rays_per_state"] = int(rays_per_state)
    ray_setup_start = time.perf_counter()
    if reflection_mode == "2d":
        base_ray_dir = generate_circle_directions(rays_per_state)
    else:
        base_ray_dir = generate_sphere_directions(rays_per_state)
    stats["timing"]["ray_setup_seconds"] += time.perf_counter() - ray_setup_start

    chunk_states = []
    state_chunk_size = _cartesian_chunk_size(n_states, rays_per_state)
    for state_start in range(0, n_states, state_chunk_size):
        stats["chunk_count"] += 1
        chunk_n_states = min(state_chunk_size, n_states - state_start)
        n_total_rays = chunk_n_states * rays_per_state
        stats["total_rays_cast"] += int(n_total_rays)
        ray_setup_start = time.perf_counter()
        ray_idx = dr.arange(wt.UInt32, n_total_rays)
        state_idx = ray_idx // rays_per_state + wt.UInt32(state_start)
        dir_idx = ray_idx % rays_per_state
        ray_origin = dr.gather(wt.Point3f, prev_state_arrays["edge_pos"], state_idx)
        ray_dir = wt.Vector3f(
            dr.gather(wt.Float, base_ray_dir.x, dir_idx),
            dr.gather(wt.Float, base_ray_dir.y, dir_idx),
            dr.gather(wt.Float, base_ray_dir.z, dir_idx),
        )
        active = dr.full(wt.Bool, True, n_total_rays)
        stats["timing"]["ray_setup_seconds"] += time.perf_counter() - ray_setup_start
        intersection_start = time.perf_counter()
        hit, _, hit_p, hit_n, prim_idx = _intersect_rays_ad_with_prim(
            ray_origin, ray_dir, active, scene, scene.tri_data_gpu
        )
        stats["timing"]["intersection_seconds"] += time.perf_counter() - intersection_start
        hit_idx = dr.compress(hit)
        hit_count = int(dr.width(hit_idx))
        stats["hit_count"] += hit_count
        if hit_count == 0:
            continue

        hit_state_idx = dr.gather(wt.UInt32, state_idx, hit_idx)
        batch_states = gather_field_evaluation_state_fields(prev_state_arrays, hit_state_idx)
        batch_state_edge_idx = dr.gather(wt.UInt32, prev_state_arrays["edge_idx"], hit_state_idx)
        hit_p = dr.gather(wt.Point3f, hit_p, hit_idx)
        hit_n = dr.gather(wt.Vector3f, hit_n, hit_idx)
        prim_idx_i32 = dr.gather(wt.Int32, wt.Int32(prim_idx), hit_idx)
        ray_dir = dr.gather(wt.Vector3f, ray_dir, hit_idx)
        n_hit = dr.width(hit_idx)

        field_start = time.perf_counter()
        field_at_hit, vector_at_hit = _edge_state_field_to_targets(
            batch_states,
            hit_p,
            k,
            return_vector=True,
            wavelength=wavelength,
            material_detail=material_detail,
        )
        stats["timing"]["field_to_hit_seconds"] += time.perf_counter() - field_start
        reflection_start = time.perf_counter()
        reflection_weight, material_inputs = _surface_reflection_coefficient(
            incident_dir=ray_dir,
            normal=hit_n,
            scene=scene,
            prim_idx=prim_idx_i32,
            material_override=reflection_material_override,
            reflection_coef=reflection_gain,
            wavelength=wavelength,
            tx_polarization=tx_polarization,
            valid_mask=dr.full(wt.Bool, True, n_hit),
            use_scene_materials=reflection_use_scene_materials,
        )
        reflected_field = field_at_hit * reflection_weight
        reflected_vector = reflect_field_vector(
            vector_at_hit,
            ray_dir,
            hit_n,
            eta_r=material_inputs["eta_r"],
            sigma=material_inputs["sigma"],
            omega=wt.Float(2.0 * math.pi * SPEED_OF_LIGHT / wavelength),
            gain=material_inputs["gain"],
        )
        stats["timing"]["surface_reflection_seconds"] += time.perf_counter() - reflection_start

        surface_edges = scene.get_triangle_surface_edge_candidates(prim_idx_i32)
        candidate_globals = list(surface_edges["slots"])
        candidate_valids = [edge_idx >= 0 for edge_idx in candidate_globals]

        per_candidate_states = []
        for slot, global_edge_idx in enumerate(candidate_globals):
            slot_valid = candidate_valids[slot]
            for prev_slot in range(slot):
                slot_valid = slot_valid & (
                    (global_edge_idx != candidate_globals[prev_slot]) | ~candidate_valids[prev_slot]
                )

            safe_global_idx = wt.UInt32(dr.select(slot_valid, global_edge_idx, wt.Int32(0)))
            local_edge_idx_i32 = dr.gather(wt.Int32, global_to_local_idx, safe_global_idx)
            slot_valid = slot_valid & (local_edge_idx_i32 >= 0)
            slot_valid = slot_valid & (local_edge_idx_i32 != wt.Int32(batch_state_edge_idx))
            if not dr.any(slot_valid):
                continue

            stats["candidate_slot_count"] += 1
            local_edge_idx = wt.UInt32(dr.select(slot_valid, local_edge_idx_i32, wt.Int32(0)))
            edge_pos = dr.gather(wt.Point3f, edge_data["pos"], local_edge_idx)
            n0 = dr.gather(wt.Vector3f, edge_data["n0"], local_edge_idx)
            adjacent_face0 = dr.gather(wt.Int32, edge_data["adjacent_face0"], local_edge_idx)
            adjacent_face1 = dr.gather(wt.Int32, edge_data["adjacent_face1"], local_edge_idx)

            visibility_start = time.perf_counter()
            visible = _segment_visibility_mask(
                hit_p,
                edge_pos,
                scene,
                ignore_prim_idx=(prim_idx_i32, adjacent_face0, adjacent_face1),
            )
            slot_valid = slot_valid & visible
            stats["timing"]["visibility_seconds"] += time.perf_counter() - visibility_start
            visible_count = int(dr.width(dr.compress(slot_valid)))
            stats["visibility_kept_count"] += visible_count
            if visible_count == 0:
                continue

            incident_start = time.perf_counter()
            incident_field = _point_source_field(
                hit_p,
                reflected_field,
                edge_pos,
                wavelength,
                k,
            )
            incident_normal_derivative = _point_source_field_normal_derivative(
                hit_p,
                reflected_field,
                edge_pos,
                n0,
                wavelength,
                k,
            )
            unit_incident_field = _point_source_field(
                hit_p,
                wt.Complex2f(1.0, 0.0),
                edge_pos,
                wavelength,
                k,
            )
            unit_incident_normal_derivative = _point_source_field_normal_derivative(
                hit_p,
                wt.Complex2f(1.0, 0.0),
                edge_pos,
                n0,
                wavelength,
                k,
            )
            incident_vector = vector_scale(reflected_vector, unit_incident_field)
            incident_normal_derivative_vector = vector_scale(reflected_vector, unit_incident_normal_derivative)
            stats["timing"]["incident_field_seconds"] += time.perf_counter() - incident_start
            field_power = _vector_field_power(incident_vector)
            keep_idx = dr.compress(slot_valid & (field_power > wt.Float(1e-20)))
            stats["field_kept_count"] += int(dr.width(keep_idx))
            if dr.width(keep_idx) == 0:
                continue

            pack_start = time.perf_counter()
            keep_state_idx = dr.gather(wt.UInt32, hit_state_idx, keep_idx)
            kept_batch_states = gather_inserted_reflection_state_fields(
                prev_state_arrays,
                keep_state_idx,
            )
            keep_edge_idx = dr.gather(wt.UInt32, local_edge_idx, keep_idx)
            keep_edge_pos = dr.gather(wt.Point3f, edge_pos, keep_idx)
            keep_edge_dir = dr.gather(wt.Vector3f, edge_data["edge_dir"], keep_edge_idx)
            keep_n0 = dr.gather(wt.Vector3f, n0, keep_idx)
            keep_nn = dr.gather(wt.Vector3f, edge_data["n_face_n"], keep_edge_idx)
            keep_wedge_n = dr.gather(wt.Float, edge_data["wedge_n"], keep_edge_idx)
            keep_edge_line_min = dr.gather(wt.Float, line_min_all, keep_edge_idx)
            keep_edge_line_max = dr.gather(wt.Float, line_max_all, keep_edge_idx)
            keep_adjacent_face0 = dr.gather(wt.Int32, adjacent_face0, keep_idx)
            keep_adjacent_face1 = dr.gather(wt.Int32, adjacent_face1, keep_idx)
            keep_source_pos = wt.Point3f(
                dr.gather(wt.Float, hit_p.x, keep_idx),
                dr.gather(wt.Float, hit_p.y, keep_idx),
                dr.gather(wt.Float, hit_p.z, keep_idx),
            )
            keep_prev_edge_pos = kept_batch_states["edge_pos"]
            keep_path_length_prefix = None
            keep_first_interaction_pos = None
            if retain_cold_metadata:
                keep_prev_path_length = kept_batch_states["path_length_prefix"]
                keep_path_length_prefix = (
                    keep_prev_path_length
                    + dr.norm(keep_source_pos - keep_prev_edge_pos)
                    + dr.norm(keep_edge_pos - keep_source_pos)
                )
                keep_first_interaction_pos = kept_batch_states["first_interaction_pos"]
            keep_incident_field = wt.Complex2f(
                dr.gather(wt.Float, incident_field.real, keep_idx),
                dr.gather(wt.Float, incident_field.imag, keep_idx),
            )
            keep_incident_normal_derivative = wt.Complex2f(
                dr.gather(wt.Float, incident_normal_derivative.real, keep_idx),
                dr.gather(wt.Float, incident_normal_derivative.imag, keep_idx),
            )
            keep_incident_vector = {
                "x": dr.gather(wt.Complex2f, incident_vector["x"], keep_idx),
                "y": dr.gather(wt.Complex2f, incident_vector["y"], keep_idx),
                "z": dr.gather(wt.Complex2f, incident_vector["z"], keep_idx),
            }
            keep_incident_normal_derivative_vector = {
                "x": dr.gather(wt.Complex2f, incident_normal_derivative_vector["x"], keep_idx),
                "y": dr.gather(wt.Complex2f, incident_normal_derivative_vector["y"], keep_idx),
                "z": dr.gather(wt.Complex2f, incident_normal_derivative_vector["z"], keep_idx),
            }
            stats["timing"]["state_pack_seconds"] += time.perf_counter() - pack_start
            material_start = time.perf_counter()
            face0_operator, face1_operator, face0_material, face1_material = _state_edge_face_material_response(
                edge_pos=keep_edge_pos,
                edge_dir=keep_edge_dir,
                n0=keep_n0,
                nn=keep_nn,
                source_pos=keep_source_pos,
                adjacent_face0=keep_adjacent_face0,
                adjacent_face1=keep_adjacent_face1,
                wavelength=wavelength,
                scene=scene,
                material_detail=material_detail,
                reflection_coef=reflection_gain,
                use_scene_materials=use_scene_materials,
                tx_polarization=tx_polarization,
            )
            stats["timing"]["material_response_seconds"] += time.perf_counter() - material_start
            keep_prev_order = kept_batch_states["order"]
            keep_order = keep_prev_order + 1
            keep_parent_state_id = _state_ids(kept_batch_states)
            keep_lineage_store = _state_lineage_store(kept_batch_states)
            keep_source_type_code = (
                kept_batch_states["source_type_code"] if retain_cold_metadata else None
            )
            keep_prefix_reflection_depth = kept_batch_states["prefix_reflection_depth"]
            keep_intermediate_reflection_depth = kept_batch_states["intermediate_reflection_depth"] + wt.UInt32(1)
            keep_suffix_reflection_depth = kept_batch_states["suffix_reflection_depth"]

            assembly_start = time.perf_counter()
            per_candidate_states.append(
                _make_state_arrays(
                    edge_idx=keep_edge_idx,
                    edge_pos=keep_edge_pos,
                    edge_dir=keep_edge_dir,
                    n0=keep_n0,
                    nn=keep_nn,
                    wedge_n=keep_wedge_n,
                    adjacent_face0=keep_adjacent_face0,
                    adjacent_face1=keep_adjacent_face1,
                    source_pos=keep_source_pos,
                    path_length_prefix=keep_path_length_prefix,
                    first_interaction_pos=keep_first_interaction_pos,
                    edge_line_min=keep_edge_line_min,
                    edge_line_max=keep_edge_line_max,
                    incident_field=keep_incident_field,
                    incident_normal_derivative=keep_incident_normal_derivative,
                    incident_vector=keep_incident_vector,
                    incident_normal_derivative_vector=keep_incident_normal_derivative_vector,
                    is_direct_tx=(
                        dr.full(wt.Bool, False, dr.width(keep_idx))
                        if retain_cold_metadata
                        else None
                    ),
                    face0_operator=face0_operator,
                    face1_operator=face1_operator,
                    face0_material=face0_material,
                    face1_material=face1_material,
                    source_type_code=keep_source_type_code,
                    prefix_reflection_depth=keep_prefix_reflection_depth,
                    intermediate_reflection_depth=keep_intermediate_reflection_depth,
                    suffix_reflection_depth=keep_suffix_reflection_depth,
                    approximation_mode_code=dr.full(
                        wt.UInt32,
                        APPROX_MODE_SAMPLED_INSERTED_REFLECTION,
                        dr.width(keep_idx),
                    ),
                    order=keep_order,
                    lineage_parent_state_id=keep_parent_state_id,
                    lineage_last_edge_idx=wt.Int32(keep_edge_idx),
                    lineage_last_reflection_depth_delta=dr.full(
                        wt.UInt32,
                        1,
                        dr.width(keep_idx),
                    ),
                    lineage_store=keep_lineage_store,
                    retain_cold_metadata=retain_cold_metadata,
                )
            )
            stats["timing"]["state_array_assembly_seconds"] += time.perf_counter() - assembly_start

        if len(per_candidate_states) > 0:
            chunk_concat_start = time.perf_counter()
            chunk_states.append(concat_state_arrays(per_candidate_states))
            stats["timing"]["chunk_concat_seconds"] += time.perf_counter() - chunk_concat_start

    concat_start = time.perf_counter()
    result = concat_state_arrays(chunk_states)
    stats["timing"]["concat_seconds"] = time.perf_counter() - concat_start
    stats["output_states"] = int(result["n_states"])
    stats["timing"]["total_seconds"] = time.perf_counter() - total_start
    return result
