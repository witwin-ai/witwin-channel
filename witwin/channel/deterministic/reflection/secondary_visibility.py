"""Secondary visibility support for reflection segment F-weighting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import drjit as dr

from witwin.channel.core.numerics.arrays import broadcast
from witwin.channel.core.numerics.constants import EPS, RAY_ORIGIN_BIAS
from witwin.channel.deterministic import types as wt


SECONDARY_VISIBILITY_MODES = {"hard", "f_weight"}


@dataclass(frozen=True)
class SecondaryVisibilitySupport:
    blocker_prim_idx: wt.Int32
    blocker_surface_group: wt.Int32
    silhouette_edge_idx: wt.Int32
    edge_pos: wt.Point3f
    edge_dir: wt.Vector3f
    edge_v0: wt.Point3f
    edge_v1: wt.Point3f
    edge_length: wt.Float
    n0: wt.Vector3f
    n_face_n: wt.Vector3f
    adjacent_face0: wt.Int32
    adjacent_face1: wt.Int32
    gamma: wt.Float
    effective_L: wt.Float
    is_occluded: wt.Bool
    valid: wt.Bool


def _width_of_point(point: wt.Point3f) -> int:
    return max(1, int(dr.width(point.x)))


def _zero_point(width: int) -> wt.Point3f:
    return dr.zeros(wt.Point3f, width)


def _zero_vector(width: int) -> wt.Vector3f:
    return dr.zeros(wt.Vector3f, width)


def _empty_support(width: int, *, is_occluded=None, blocker_prim_idx=None, blocker_surface_group=None) -> SecondaryVisibilitySupport:
    zero_p = _zero_point(width)
    zero_v = _zero_vector(width)
    return SecondaryVisibilitySupport(
        blocker_prim_idx=dr.full(
            wt.Int32,
            -1,
            width,
        ) if blocker_prim_idx is None else blocker_prim_idx,
        blocker_surface_group=dr.full(
            wt.Int32,
            -1,
            width,
        ) if blocker_surface_group is None else blocker_surface_group,
        silhouette_edge_idx=dr.full(wt.Int32, -1, width),
        edge_pos=zero_p,
        edge_dir=zero_v,
        edge_v0=zero_p,
        edge_v1=zero_p,
        edge_length=dr.zeros(wt.Float, width),
        n0=zero_v,
        n_face_n=zero_v,
        adjacent_face0=dr.full(wt.Int32, -1, width),
        adjacent_face1=dr.full(wt.Int32, -1, width),
        gamma=dr.full(wt.Float, float("inf"), width),
        effective_L=dr.zeros(wt.Float, width),
        is_occluded=dr.full(wt.Bool, False, width) if is_occluded is None else is_occluded,
        valid=dr.full(wt.Bool, False, width),
    )


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


def _ignore_entries(scene, groups, width: int) -> list[tuple[wt.Int32, wt.Bool]]:
    if groups is None:
        return []
    candidates = groups if isinstance(groups, (tuple, list)) else (groups,)
    entries: list[tuple[wt.Int32, wt.Bool]] = []
    for group in candidates:
        if group is None:
            continue
        group_i32 = broadcast(wt.Int32(group), width)
        entries.append((group_i32, group_i32 >= wt.Int32(0)))
    return entries


def _segment_edge_distance(
    segment_start: wt.Point3f,
    segment_end: wt.Point3f,
    edge_v0: wt.Point3f,
    edge_v1: wt.Point3f,
) -> tuple[wt.Float, wt.Point3f]:
    seg = segment_end - segment_start
    edge = edge_v1 - edge_v0
    rel = segment_start - edge_v0
    a = dr.dot(seg, seg) + wt.Float(EPS)
    b = dr.dot(seg, edge)
    c = dr.dot(edge, edge) + wt.Float(EPS)
    d = dr.dot(seg, rel)
    e = dr.dot(edge, rel)
    denom = a * c - b * b
    parallel = dr.abs(denom) <= wt.Float(EPS)
    seg_t = dr.select(parallel, wt.Float(0.0), (b * e - c * d) / denom)
    edge_t = dr.select(parallel, e / c, (a * e - b * d) / denom)
    seg_t = dr.clip(seg_t, wt.Float(0.0), wt.Float(1.0))
    edge_t = dr.clip(edge_t, wt.Float(0.0), wt.Float(1.0))
    closest_seg = segment_start + seg * seg_t
    closest_edge = edge_v0 + edge * edge_t
    return dr.norm(closest_seg - closest_edge), closest_edge


def _first_non_ignored_blocker(
    *,
    scene,
    ray_origin: wt.Point3f,
    ray_dir: wt.Vector3f,
    remaining: wt.Float,
    active: wt.Bool,
    ignore_groups: Sequence[tuple[wt.Int32, wt.Bool]],
    max_ignored_hits: int,
):
    width = _width_of_point(ray_origin)
    blocker_prim_idx = dr.full(wt.Int32, -1, width)
    blocker_distance = dr.full(wt.Float, float("inf"), width)
    blocked = dr.full(wt.Bool, False, width)
    unresolved = active
    cur_origin = ray_origin
    cur_remaining = remaining

    for _ in range(max(1, int(max_ignored_hits))):
        hit, hit_distance, prim_idx_u32 = scene.intersect_rays_raw_with_prim(
            cur_origin,
            ray_dir,
            unresolved,
            tmax=cur_remaining,
        )
        within = hit & (hit_distance < cur_remaining)
        prim_idx = wt.Int32(prim_idx_u32)
        hit_group = scene.triangle_group_id(prim_idx)
        ignored = dr.full(wt.Bool, False, width)
        for ignore_group, has_ignore in ignore_groups:
            ignored = ignored | (within & has_ignore & (hit_group == ignore_group))

        accepted = within & ~ignored & ~blocked
        blocker_prim_idx = dr.select(accepted, prim_idx, blocker_prim_idx)
        blocker_distance = dr.select(accepted, hit_distance, blocker_distance)
        blocked = blocked | accepted

        advance_distance = hit_distance + wt.Float(RAY_ORIGIN_BIAS)
        advance = ignored & (advance_distance < cur_remaining)
        cur_origin = _select_point(advance, cur_origin + ray_dir * advance_distance, cur_origin)
        cur_remaining = dr.select(advance, cur_remaining - advance_distance, cur_remaining)
        unresolved = advance & ~blocked & (cur_remaining > wt.Float(EPS))
        if (not bool(dr.flag(dr.JitFlag.Recording))) and (not bool(dr.any(unresolved))):
            break

    return blocked, blocker_prim_idx, blocker_distance


def _nearest_selected_blocker_edge(
    *,
    edge_runtime,
    segment_start: wt.Point3f,
    segment_end: wt.Point3f,
    blocker_prim_idx: wt.Int32,
    blocker_surface_group: wt.Int32,
    is_occluded: wt.Bool,
) -> dict[str, object]:
    width = _width_of_point(segment_end)
    zero_p = _zero_point(width)
    zero_v = _zero_vector(width)
    best = {
        "local_edge_idx": dr.full(wt.Int32, -1, width),
        "edge_pos": zero_p,
        "edge_dir": zero_v,
        "edge_v0": zero_p,
        "edge_v1": zero_p,
        "edge_length": dr.zeros(wt.Float, width),
        "n0": zero_v,
        "n_face_n": zero_v,
        "adjacent_face0": dr.full(wt.Int32, -1, width),
        "adjacent_face1": dr.full(wt.Int32, -1, width),
        "gamma": dr.full(wt.Float, float("inf"), width),
        "diffraction_point": zero_p,
    }
    adjacent_group0 = edge_runtime.get("adjacent_surface_group0")
    adjacent_group1 = edge_runtime.get("adjacent_surface_group1")

    for edge_idx in range(int(edge_runtime["n_edges"])):
        safe_edge_idx = wt.UInt32(edge_idx)
        edge_pos = dr.gather(wt.Point3f, edge_runtime["pos"], safe_edge_idx)
        edge_dir = dr.gather(wt.Vector3f, edge_runtime["edge_dir"], safe_edge_idx)
        line_min = dr.gather(wt.Float, edge_runtime["line_min"], safe_edge_idx)
        line_max = dr.gather(wt.Float, edge_runtime["line_max"], safe_edge_idx)
        edge_v0 = edge_pos + edge_dir * line_min
        edge_v1 = edge_pos + edge_dir * line_max
        edge_length = dr.gather(wt.Float, edge_runtime["length"], safe_edge_idx)
        adjacent_face0 = dr.gather(wt.Int32, edge_runtime["adjacent_face0"], safe_edge_idx)
        adjacent_face1 = dr.gather(wt.Int32, edge_runtime["adjacent_face1"], safe_edge_idx)
        n0 = dr.gather(wt.Vector3f, edge_runtime["n0"], safe_edge_idx)
        n_face_n = dr.gather(wt.Vector3f, edge_runtime["n_face_n"], safe_edge_idx)
        belongs_to_blocker = (adjacent_face0 == blocker_prim_idx) | (adjacent_face1 == blocker_prim_idx)
        if adjacent_group0 is not None:
            belongs_to_blocker = belongs_to_blocker | (
                dr.gather(wt.Int32, adjacent_group0, safe_edge_idx) == blocker_surface_group
            )
        if adjacent_group1 is not None:
            belongs_to_blocker = belongs_to_blocker | (
                dr.gather(wt.Int32, adjacent_group1, safe_edge_idx) == blocker_surface_group
            )
        gamma, diffraction_point = _segment_edge_distance(segment_start, segment_end, edge_v0, edge_v1)
        take = is_occluded & belongs_to_blocker & (gamma < best["gamma"])
        best["local_edge_idx"] = dr.select(take, wt.Int32(edge_idx), best["local_edge_idx"])
        best["edge_pos"] = _select_point(take, edge_pos, best["edge_pos"])
        best["edge_dir"] = _select_vector(take, edge_dir, best["edge_dir"])
        best["edge_v0"] = _select_point(take, edge_v0, best["edge_v0"])
        best["edge_v1"] = _select_point(take, edge_v1, best["edge_v1"])
        best["edge_length"] = dr.select(take, edge_length, best["edge_length"])
        best["n0"] = _select_vector(take, n0, best["n0"])
        best["n_face_n"] = _select_vector(take, n_face_n, best["n_face_n"])
        best["adjacent_face0"] = dr.select(take, adjacent_face0, best["adjacent_face0"])
        best["adjacent_face1"] = dr.select(take, adjacent_face1, best["adjacent_face1"])
        best["gamma"] = dr.select(take, gamma, best["gamma"])
        best["diffraction_point"] = _select_point(take, diffraction_point, best["diffraction_point"])

    return best


def nearest_blocker_silhouette_edge(
    *,
    scene,
    hit_p: wt.Point3f,
    rx_pos: wt.Point3f,
    primary_surface_group,
    mode: str,
    wavelength,
    boundary_radius_wavelengths: float,
    edge_policy=None,
    max_ignored_hits: int = 4,
) -> SecondaryVisibilitySupport:
    """Return the nearest blocker silhouette edge for a reflection segment."""
    segment_mode = str(mode)
    width = max(_width_of_point(hit_p), _width_of_point(rx_pos))
    if segment_mode not in SECONDARY_VISIBILITY_MODES:
        raise ValueError(
            "mode must be one of "
            f"{sorted(SECONDARY_VISIBILITY_MODES)}; got {segment_mode!r}."
        )
    if float(boundary_radius_wavelengths) <= 0.0:
        raise ValueError("boundary_radius_wavelengths must be > 0.")
    if segment_mode == "hard":
        return _empty_support(width)

    seg = rx_pos - hit_p
    seg_len = dr.norm(seg)
    active = seg_len > wt.Float(2.0 * RAY_ORIGIN_BIAS + EPS)
    ray_dir = seg / (seg_len + wt.Float(EPS))
    ray_origin = hit_p + ray_dir * wt.Float(RAY_ORIGIN_BIAS)
    remaining = dr.maximum(seg_len - wt.Float(2.0 * RAY_ORIGIN_BIAS), wt.Float(0.0))
    ignore_groups = _ignore_entries(scene, primary_surface_group, width)
    is_occluded, blocker_prim_idx, _blocker_distance = _first_non_ignored_blocker(
        scene=scene,
        ray_origin=ray_origin,
        ray_dir=ray_dir,
        remaining=remaining,
        active=active,
        ignore_groups=ignore_groups,
        max_ignored_hits=max_ignored_hits,
    )
    blocker_surface_group = scene.triangle_group_id(blocker_prim_idx)
    if not bool(dr.any(is_occluded)):
        return _empty_support(
            width,
            is_occluded=is_occluded,
            blocker_prim_idx=blocker_prim_idx,
            blocker_surface_group=blocker_surface_group,
        )
    edge_runtime = scene._selected_edge_runtime(edge_policy=edge_policy)
    if edge_runtime is None or int(edge_runtime.get("n_edges", 0)) <= 0:
        return _empty_support(
            width,
            is_occluded=is_occluded,
            blocker_prim_idx=blocker_prim_idx,
            blocker_surface_group=blocker_surface_group,
        )

    nearest = _nearest_selected_blocker_edge(
        edge_runtime=edge_runtime,
        segment_start=hit_p,
        segment_end=rx_pos,
        blocker_prim_idx=blocker_prim_idx,
        blocker_surface_group=blocker_surface_group,
        is_occluded=is_occluded,
    )
    edge_query_valid = nearest["local_edge_idx"] >= wt.Int32(0)
    gamma = nearest["gamma"]
    diffraction_point = nearest["diffraction_point"]
    s = dr.norm(hit_p - diffraction_point) + wt.Float(EPS)
    s_next = dr.norm(rx_pos - diffraction_point) + wt.Float(EPS)
    effective_l = (s * s_next) / (s + s_next + wt.Float(EPS))
    radius = wt.Float(wavelength) * wt.Float(boundary_radius_wavelengths)
    valid = edge_query_valid & (gamma <= radius)

    zero_p = _zero_point(width)
    zero_v = _zero_vector(width)
    return SecondaryVisibilitySupport(
        blocker_prim_idx=dr.select(is_occluded, blocker_prim_idx, wt.Int32(-1)),
        blocker_surface_group=dr.select(is_occluded, blocker_surface_group, wt.Int32(-1)),
        silhouette_edge_idx=dr.select(edge_query_valid, nearest["local_edge_idx"], wt.Int32(-1)),
        edge_pos=_select_point(edge_query_valid, nearest["edge_pos"], zero_p),
        edge_dir=_select_vector(edge_query_valid, nearest["edge_dir"], zero_v),
        edge_v0=_select_point(edge_query_valid, nearest["edge_v0"], zero_p),
        edge_v1=_select_point(edge_query_valid, nearest["edge_v1"], zero_p),
        edge_length=dr.select(edge_query_valid, nearest["edge_length"], wt.Float(0.0)),
        n0=_select_vector(edge_query_valid, nearest["n0"], zero_v),
        n_face_n=_select_vector(edge_query_valid, nearest["n_face_n"], zero_v),
        adjacent_face0=dr.select(edge_query_valid, nearest["adjacent_face0"], wt.Int32(-1)),
        adjacent_face1=dr.select(edge_query_valid, nearest["adjacent_face1"], wt.Int32(-1)),
        gamma=dr.select(edge_query_valid, gamma, wt.Float(float("inf"))),
        effective_L=dr.select(edge_query_valid, effective_l, wt.Float(0.0)),
        is_occluded=is_occluded,
        valid=valid,
    )


__all__ = [
    "SECONDARY_VISIBILITY_MODES",
    "SecondaryVisibilitySupport",
    "nearest_blocker_silhouette_edge",
]
