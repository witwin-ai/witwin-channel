"""Surface-boundary support queries for reflection transition weighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import drjit as dr

from witwin.channel.deterministic import types as wt

from .detail import REFLECTION_TRANSITION_MODES

if TYPE_CHECKING:
    from witwin.channel.core.scene.edge_policy import EdgePolicy


@dataclass(frozen=True)
class ReflectionBoundarySupport:
    edge_idx: wt.Int32
    global_edge_idx: wt.Int32
    edge_pos: wt.Point3f
    edge_dir: wt.Vector3f
    edge_v0: wt.Point3f
    edge_v1: wt.Point3f
    edge_length: wt.Float
    distance: wt.Float
    adjacent_face0: wt.Int32
    adjacent_face1: wt.Int32
    n0: wt.Vector3f
    n_face_n: wt.Vector3f
    valid: wt.Bool


def _width_of(value) -> int:
    try:
        return int(dr.width(value))
    except TypeError:
        return 1


def _input_width(prim_idx: wt.Int32, hit_p: wt.Point3f) -> int:
    return max(1, _width_of(prim_idx), _width_of(hit_p.x))


def _zero_point(width: int) -> wt.Point3f:
    return dr.zeros(wt.Point3f, width)


def _zero_vector(width: int) -> wt.Vector3f:
    return dr.zeros(wt.Vector3f, width)


def _select_point(mask: wt.Bool, yes: wt.Point3f, no: wt.Point3f) -> wt.Point3f:
    return wt.Point3f(
        dr.select(mask, yes.x, no.x),
        dr.select(mask, yes.y, no.y),
        dr.select(mask, yes.z, no.z),
    )


def _select_vector(mask: wt.Bool, yes: wt.Vector3f, no: wt.Vector3f) -> wt.Vector3f:
    return wt.Vector3f(
        dr.select(mask, yes.x, no.x),
        dr.select(mask, yes.y, no.y),
        dr.select(mask, yes.z, no.z),
    )


def _empty_support(width: int) -> ReflectionBoundarySupport:
    zero_p = _zero_point(width)
    zero_v = _zero_vector(width)
    return ReflectionBoundarySupport(
        edge_idx=dr.full(wt.Int32, -1, width),
        global_edge_idx=dr.full(wt.Int32, -1, width),
        edge_pos=zero_p,
        edge_dir=zero_v,
        edge_v0=zero_p,
        edge_v1=zero_p,
        edge_length=dr.zeros(wt.Float, width),
        distance=dr.full(wt.Float, float("inf"), width),
        adjacent_face0=dr.full(wt.Int32, -1, width),
        adjacent_face1=dr.full(wt.Int32, -1, width),
        n0=zero_v,
        n_face_n=zero_v,
        valid=dr.full(wt.Bool, False, width),
    )


def nearest_surface_boundary_edge(
    *,
    scene,
    prim_idx,
    hit_p: wt.Point3f,
    mode: str,
    wavelength,
    boundary_radius_wavelengths: float,
    max_edges_per_slot: int = 1,
    edge_policy: "EdgePolicy | None" = None,
) -> ReflectionBoundarySupport:
    """Return the nearest eligible surface-boundary edge for a reflection hit.

    The helper intentionally returns only the nearest candidate. Production
    reflection F-weighting keeps the per-slot edge budget capped at one by
    default; wider validation modes can call this primitive repeatedly later if
    they need more candidates.
    """
    prim_idx_i32 = wt.Int32(prim_idx)
    width = _input_width(prim_idx_i32, hit_p)

    transition_mode = str(mode)
    if transition_mode not in REFLECTION_TRANSITION_MODES:
        raise ValueError(
            "mode must be one of "
            f"{sorted(REFLECTION_TRANSITION_MODES)}; got {transition_mode!r}."
        )
    if max_edges_per_slot <= 0:
        raise ValueError("max_edges_per_slot must be > 0.")
    if boundary_radius_wavelengths <= 0.0:
        raise ValueError("boundary_radius_wavelengths must be > 0.")
    if transition_mode == "hard":
        return _empty_support(width)

    edge_runtime = scene._selected_edge_runtime(edge_policy=edge_policy)
    if edge_runtime is None or int(edge_runtime.get("n_edges", 0)) <= 0:
        return _empty_support(width)

    candidates = scene.get_triangle_surface_edge_candidates(prim_idx_i32)
    candidate_slots = tuple(candidates["slots"])
    if not candidate_slots:
        return _empty_support(width)

    candidate_count = wt.UInt32(candidates["count"])
    best_edge_idx = dr.full(wt.Int32, -1, width)
    best_distance = dr.full(wt.Float, float("inf"), width)

    for slot, edge_idx in enumerate(candidate_slots):
        edge_idx_i32 = wt.Int32(edge_idx)
        slot_active = (
            (wt.UInt32(slot) < candidate_count)
            & (edge_idx_i32 >= wt.Int32(0))
        )
        safe_edge_idx = wt.UInt32(dr.select(slot_active, edge_idx_i32, wt.Int32(0)))
        edge_pos = dr.gather(wt.Point3f, edge_runtime["pos"], safe_edge_idx)
        edge_dir = dr.gather(wt.Vector3f, edge_runtime["edge_dir"], safe_edge_idx)
        line_min = dr.gather(wt.Float, edge_runtime["line_min"], safe_edge_idx)
        line_max = dr.gather(wt.Float, edge_runtime["line_max"], safe_edge_idx)

        projection = dr.dot(hit_p - edge_pos, edge_dir)
        clamped_projection = dr.minimum(dr.maximum(projection, line_min), line_max)
        closest = edge_pos + edge_dir * clamped_projection
        distance = dr.norm(hit_p - closest)
        better = slot_active & (distance < best_distance)
        best_distance = dr.select(better, distance, best_distance)
        best_edge_idx = dr.select(better, edge_idx_i32, best_edge_idx)

    radius = wt.Float(wavelength) * wt.Float(boundary_radius_wavelengths)
    valid = (best_edge_idx >= wt.Int32(0)) & (best_distance <= radius)
    safe_best_idx = wt.UInt32(dr.select(valid, best_edge_idx, wt.Int32(0)))

    edge_pos = dr.gather(wt.Point3f, edge_runtime["pos"], safe_best_idx)
    edge_dir = dr.gather(wt.Vector3f, edge_runtime["edge_dir"], safe_best_idx)
    line_min = dr.gather(wt.Float, edge_runtime["line_min"], safe_best_idx)
    line_max = dr.gather(wt.Float, edge_runtime["line_max"], safe_best_idx)
    edge_v0 = edge_pos + edge_dir * line_min
    edge_v1 = edge_pos + edge_dir * line_max
    zero_p = _zero_point(width)
    zero_v = _zero_vector(width)

    return ReflectionBoundarySupport(
        edge_idx=dr.select(valid, best_edge_idx, wt.Int32(-1)),
        global_edge_idx=dr.select(
            valid,
            dr.gather(wt.Int32, edge_runtime["global_idx"], safe_best_idx),
            wt.Int32(-1),
        ),
        edge_pos=_select_point(valid, edge_pos, zero_p),
        edge_dir=_select_vector(valid, edge_dir, zero_v),
        edge_v0=_select_point(valid, edge_v0, zero_p),
        edge_v1=_select_point(valid, edge_v1, zero_p),
        edge_length=dr.select(
            valid,
            dr.gather(wt.Float, edge_runtime["length"], safe_best_idx),
            wt.Float(0.0),
        ),
        distance=dr.select(valid, best_distance, wt.Float(float("inf"))),
        adjacent_face0=dr.select(
            valid,
            dr.gather(wt.Int32, edge_runtime["adjacent_face0"], safe_best_idx),
            wt.Int32(-1),
        ),
        adjacent_face1=dr.select(
            valid,
            dr.gather(wt.Int32, edge_runtime["adjacent_face1"], safe_best_idx),
            wt.Int32(-1),
        ),
        n0=_select_vector(
            valid,
            dr.gather(wt.Vector3f, edge_runtime["n0"], safe_best_idx),
            zero_v,
        ),
        n_face_n=_select_vector(
            valid,
            dr.gather(wt.Vector3f, edge_runtime["n_face_n"], safe_best_idx),
            zero_v,
        ),
        valid=valid,
    )


__all__ = [
    "ReflectionBoundarySupport",
    "nearest_surface_boundary_edge",
]
