from __future__ import annotations

import drjit as dr
import witwin as wt

from ..utils.constants import EDGE_2D_EPS, EPS, SMALL_EPS
from ..utils import scalar
from .builder import (
    MATERIAL_EPS_R_DEFAULT,
    MATERIAL_SIGMA_E_DEFAULT,
    _ensure_edge_runtime,
    empty_edge_selection_summary,
)
from .projection import project_to_2d
from .types import DiffractionPoint


def get_triangle_surface_edge_candidates(scene, prim_idx):
    _ensure_edge_runtime(scene)
    triangle_surface_data = scene._triangle_surface_data
    max_surface_edge_count = 0 if triangle_surface_data is None else int(
        triangle_surface_data.get("max_surface_edge_count", 0)
    )
    if triangle_surface_data is None or max_surface_edge_count <= 0:
        return {"count": wt.UInt32(0), "slots": ()}

    n_triangles = int(scene._tri_edge_indices["n_triangles"])
    prim_idx_i32 = wt.Int32(prim_idx)
    valid = (prim_idx_i32 >= 0) & (prim_idx_i32 < wt.Int32(n_triangles))
    safe_idx = wt.UInt32(dr.select(valid, prim_idx_i32, wt.Int32(0)))
    count = dr.select(valid, dr.gather(wt.UInt32, triangle_surface_data["surface_edge_size"], safe_idx), wt.UInt32(0))
    slots = []
    for slot in range(max_surface_edge_count):
        flat_idx = safe_idx * wt.UInt32(max_surface_edge_count) + wt.UInt32(slot)
        slot_value = dr.select(
            valid,
            dr.gather(wt.Int32, triangle_surface_data["surface_edge_indices"], flat_idx),
            wt.Int32(-1),
        )
        slots.append(slot_value)
    return {"count": count, "slots": tuple(slots)}


def get_triangle_material_data(scene, prim_idx, valid_mask=None):
    if scene.tri_data_gpu is None:
        prim_idx_i32 = wt.Int32(prim_idx)
        width = int(dr.width(prim_idx_i32))
        return {
            "eps_r": dr.full(wt.Float, MATERIAL_EPS_R_DEFAULT, width),
            "sigma_e": dr.full(wt.Float, MATERIAL_SIGMA_E_DEFAULT, width),
            "specified": dr.full(wt.Bool, False, width),
            "structure_idx": dr.full(wt.Int32, -1, width),
            "valid": dr.full(wt.Bool, False, width),
        }

    if valid_mask is None:
        valid_mask = wt.Bool(True) if dr.width(wt.Int32(prim_idx)) == 1 else (wt.Int32(prim_idx) >= 0)

    n_triangles = int(scene.tri_data_gpu["n_triangles"])
    prim_idx_i32 = wt.Int32(prim_idx)
    valid = valid_mask & (prim_idx_i32 >= 0) & (prim_idx_i32 < wt.Int32(n_triangles))
    safe_idx = wt.UInt32(dr.select(valid, prim_idx_i32, wt.Int32(0)))
    return {
        "eps_r": dr.select(
            valid,
            dr.gather(wt.Float, scene.tri_data_gpu["material_eps_r"], safe_idx),
            wt.Float(MATERIAL_EPS_R_DEFAULT),
        ),
        "sigma_e": dr.select(
            valid,
            dr.gather(wt.Float, scene.tri_data_gpu["material_sigma_e"], safe_idx),
            wt.Float(MATERIAL_SIGMA_E_DEFAULT),
        ),
        "specified": dr.select(
            valid,
            dr.gather(wt.Bool, scene.tri_data_gpu["material_specified"], safe_idx),
            wt.Bool(False),
        ),
        "structure_idx": dr.select(
            valid,
            dr.gather(wt.Int32, scene.tri_data_gpu["material_structure_idx"], safe_idx),
            wt.Int32(-1),
        ),
        "valid": valid,
    }


def gather_structure_indices(scene, prim_idx, *, valid_mask=None):
    width = dr.width(prim_idx)
    if scene is None or scene.tri_data_gpu is None or "material_structure_idx" not in scene.tri_data_gpu:
        return dr.full(wt.Int32, -1, width)

    prim_idx_i32 = wt.Int32(prim_idx)
    if valid_mask is None:
        valid_mask = prim_idx_i32 >= 0

    n_triangles = int(scene.tri_data_gpu["n_triangles"])
    valid = valid_mask & (prim_idx_i32 >= 0) & (prim_idx_i32 < wt.Int32(n_triangles))
    safe_idx = wt.UInt32(dr.select(valid, prim_idx_i32, wt.Int32(0)))
    return dr.select(
        valid,
        dr.gather(wt.Int32, scene.tri_data_gpu["material_structure_idx"], safe_idx),
        wt.Int32(-1),
    )


def get_diffraction_edge_data(scene, edge_idx, valid_mask=None):
    _ensure_edge_runtime(scene)
    if scene._diffraction_edge_gpu is None:
        return {
            "pos": wt.Point3f(0),
            "edge_dir": wt.Vector3f(0, 0, 1),
            "n0": wt.Vector3f(0, 0, 1),
            "n_face_n": wt.Vector3f(0, 0, -1),
            "wedge_n": wt.Float(1.5),
            "length": wt.Float(0),
            "line_min": wt.Float(0),
            "line_max": wt.Float(0),
            "valid": wt.Bool(False),
        }

    if valid_mask is None:
        valid_mask = edge_idx >= 0

    safe_idx = dr.select(valid_mask, wt.UInt32(edge_idx), wt.UInt32(0))
    return {
        "pos": dr.gather(wt.Point3f, scene._diffraction_edge_gpu["pos"], safe_idx),
        "edge_dir": dr.gather(wt.Vector3f, scene._diffraction_edge_gpu["edge_dir"], safe_idx),
        "n0": dr.gather(wt.Vector3f, scene._diffraction_edge_gpu["n0"], safe_idx),
        "n_face_n": dr.gather(wt.Vector3f, scene._diffraction_edge_gpu["n_face_n"], safe_idx),
        "wedge_n": dr.gather(wt.Float, scene._diffraction_edge_gpu["wedge_n"], safe_idx),
        "length": dr.gather(wt.Float, scene._diffraction_edge_gpu["length"], safe_idx),
        "line_min": dr.gather(wt.Float, scene._diffraction_edge_gpu["line_min"], safe_idx),
        "line_max": dr.gather(wt.Float, scene._diffraction_edge_gpu["line_max"], safe_idx),
        "valid": valid_mask,
    }


def get_global_diffraction_edge_indices(scene):
    _ensure_edge_runtime(scene)
    return tuple(scene._global_diffraction_edge_indices)


def get_adjacent_diffraction_edge_indices_for_triangle(scene, prim_idx: int, include_sibling: bool = True):
    del include_sibling
    _ensure_edge_runtime(scene)
    if scene._tri_edge_indices is None:
        return ()

    n_triangles = int(scene._tri_edge_indices["n_triangles"])
    prim_idx = int(prim_idx)
    if prim_idx < 0 or prim_idx >= n_triangles:
        return ()
    if prim_idx >= len(scene._triangle_surface_group_by_triangle):
        return ()
    group_idx = scene._triangle_surface_group_by_triangle[prim_idx]
    return tuple(scene._triangle_surface_edge_groups[group_idx])


def edge_anchor_for_height(scene, edge_info, calculation_height: float):
    del scene
    z0 = edge_info.p0.z
    z1 = edge_info.p1.z
    dz = z1 - z0
    if scalar(dr.abs(dz)) <= EPS:
        t = wt.Float(0.5)
    else:
        t = dr.clip((wt.Float(calculation_height) - z0) / dz, 0.0, 1.0)
    is_endpoint_anchor = scalar((t <= wt.Float(SMALL_EPS)) | (t >= wt.Float(1.0 - SMALL_EPS)))
    return edge_info.p0 + edge_info.edge_vector * t, t, bool(is_endpoint_anchor)


def _finite_edge_bounds(scene, edge_info, anchor_fraction):
    edge_length = wt.Float(edge_info.length)
    anchor_fraction = wt.Float(anchor_fraction)
    return -anchor_fraction * edge_length, (wt.Float(1.0) - anchor_fraction) * edge_length


def _build_vertical_only_diffraction_points(scene, calculation_height: float):
    diffraction_points = []
    global_indices = []
    for edge_idx, edge_info in enumerate(scene.vertical_edges):
        if edge_info.wedge_n is None or not (scalar(edge_info.wedge_n) > 1.0 + SMALL_EPS):
            continue
        z0 = scalar(edge_info.p0.z)
        z1 = scalar(edge_info.p1.z)
        if not (min(z0, z1) <= calculation_height <= max(z0, z1)):
            continue
        if scalar(dr.norm(wt.Vector2f(edge_info.edge_vector.x, edge_info.edge_vector.y))) > EDGE_2D_EPS:
            continue
        face_normals = edge_info.face_normals_3d or []
        if len(face_normals) < 2:
            continue

        position, anchor_fraction, _ = edge_anchor_for_height(scene, edge_info, calculation_height)
        line_min, line_max = _finite_edge_bounds(scene, edge_info, anchor_fraction)
        diffraction_points.append(
            DiffractionPoint(
                position=position,
                edge_vector=edge_info.edge_vector,
                length=edge_info.length,
                wedge_n=edge_info.wedge_n,
                face_normals_3d=face_normals,
                adjacent_faces=tuple(int(face_idx) for face_idx in edge_info.adjacent_faces),
                vertex_indices=tuple(int(v_idx) for v_idx in edge_info.vertex_indices),
                global_index=int(edge_info.global_index),
                line_min=line_min,
                line_max=line_max,
            )
        )
        global_indices.append(edge_idx)
    return diffraction_points, global_indices


def build_generic_diffraction_points(scene, calculation_height: float):
    diffraction_points = []
    global_indices = []
    for edge_idx, edge_info in enumerate(scene.vertical_edges):
        if edge_info.wedge_n is None or not (scalar(edge_info.wedge_n) > 1.0 + SMALL_EPS):
            continue
        face_normals = edge_info.face_normals_3d or []
        if len(face_normals) < 2:
            continue

        anchor, anchor_fraction, _is_endpoint_anchor = edge_anchor_for_height(scene, edge_info, calculation_height)
        line_min, line_max = _finite_edge_bounds(scene, edge_info, anchor_fraction)

        diffraction_points.append(
            DiffractionPoint(
                position=anchor,
                edge_vector=edge_info.edge_vector,
                length=edge_info.length,
                wedge_n=edge_info.wedge_n,
                face_normals_3d=face_normals,
                adjacent_faces=tuple(int(face_idx) for face_idx in edge_info.adjacent_faces),
                vertex_indices=tuple(int(v_idx) for v_idx in edge_info.vertex_indices),
                global_index=int(edge_info.global_index),
                line_min=line_min,
                line_max=line_max,
            )
        )
        global_indices.append(edge_idx)
    return diffraction_points, global_indices


def _ensure_edge_projection_cache(scene, calculation_height, cache_entry):
    if cache_entry.get("edges_2d") is not None and cache_entry.get("corners_2d") is not None:
        return cache_entry

    edges_2d, corners_2d = project_to_2d(scene.vertical_edges, calculation_height, scene.vertices)
    cache_entry["edges_2d"] = edges_2d
    cache_entry["corners_2d"] = corners_2d
    return cache_entry


def get_edge_data(scene, calculation_height, include_projection: bool = True):
    from ..trace.diffraction import preload_diffraction_edges

    _ensure_edge_runtime(scene)
    cache_key = (scene._mesh_version, calculation_height)
    cached = scene._edge_cache.get(cache_key)
    if cached is not None:
        if include_projection:
            return _ensure_edge_projection_cache(scene, calculation_height, cached)
        return cached

    if scene.edge_selection_mode == "all_edges":
        scene.edge_selection_summary["excluded_endpoint_anchors"] = 0

    if scene.edge_selection_mode == "all_edges":
        diffraction_points, global_indices = build_generic_diffraction_points(scene, calculation_height)
    else:
        diffraction_points, global_indices = _build_vertical_only_diffraction_points(scene, calculation_height)

    edge_data = preload_diffraction_edges(diffraction_points, global_indices=global_indices)
    cache_entry = {
        "edge_data": edge_data,
        "edges_2d": None,
        "corners_2d": None,
        "diffraction_points": diffraction_points,
        "boundary_edge_policy": scene.boundary_edge_policy,
    }
    scene._edge_cache[cache_key] = cache_entry
    if include_projection:
        return _ensure_edge_projection_cache(scene, calculation_height, cache_entry)
    return cache_entry
