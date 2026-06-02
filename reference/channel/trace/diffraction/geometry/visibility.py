"""Visibility, validity masks, and intersection helpers for diffraction geometry."""

import drjit as dr
import rayd
import witwin as wt

from ....scene.runtime_queries import gather_structure_indices
from ....utils import scalar
from ....utils.constants import EPS, RAY_ORIGIN_BIAS, SMALL_EPS
from ....utils.geometry import point_in_triangle_3d, reflect_point_across_plane
from ....utils.drjit_ops import Broadcast, Concat, broadcast_point, broadcast_vector, repeat_int
from ..constants import _distance_to_cot_pole
from .angles import _project_to_wedge_plane


def _slope_derivative_safe_mask(phi, phi_prime, wedge_n, step):
    n_pi = wedge_n * dr.pi
    interior = (
        (phi >= step) & (phi <= (n_pi - step))
        & (phi_prime >= step) & (phi_prime <= (n_pi - step))
    )
    pole_guard = step / (2.0 * wedge_n)
    return interior & _cotangent_pole_safe_mask(phi, phi_prime, wedge_n, pole_guard)


def _cotangent_pole_safe_mask(phi, phi_prime, wedge_n, pole_guard):
    two_n = 2.0 * wedge_n
    dif_phi = phi - phi_prime
    sum_phi = phi + phi_prime
    args = [
        (dr.pi + dif_phi) / two_n,
        (dr.pi - dif_phi) / two_n,
        (dr.pi + sum_phi) / two_n,
        (dr.pi - sum_phi) / two_n,
    ]
    safe = dr.full(wt.Bool, True, dr.width(phi))
    for arg in args:
        safe = safe & (_distance_to_cot_pole(arg) > pole_guard)
    return safe



def _triangle_surface_group_id(tri_data, prim_idx_i32):
    valid_prim = prim_idx_i32 >= 0
    if tri_data is None or "surface_group_id" not in tri_data:
        return dr.select(valid_prim, prim_idx_i32, wt.Int32(-1))
    safe_prim_idx = wt.UInt32(dr.select(valid_prim, prim_idx_i32, wt.Int32(0)))
    return dr.select(valid_prim, dr.gather(wt.Int32, tri_data["surface_group_id"], safe_prim_idx), wt.Int32(-1))


def _broadcast_i32(value, width: int):
    value_i32 = wt.Int32(value)
    value_width = int(dr.width(value_i32))
    if value_width == width:
        return value_i32
    if value_width == 1:
        return repeat_int(value_i32, width)
    raise ValueError(f"Expected scalar or width {width} Int32 input, got width {value_width}.")


def _point_grad_enabled(point) -> bool:
    if point is None:
        return False
    for axis in ("x", "y", "z"):
        component = getattr(point, axis, None)
        if component is None:
            continue
        try:
            if bool(dr.grad_enabled(component)):
                return True
        except Exception:
            continue
    return False


def _scene_geometry_grad_enabled(scene) -> bool:
    if scene is None:
        return False
    vertices = getattr(scene, "vertices", None)
    if _point_grad_enabled(vertices):
        return True
    tri_data = getattr(scene, "tri_data_gpu", None)
    if isinstance(tri_data, dict):
        for key in ("v0", "v1", "v2"):
            value = tri_data.get(key)
            if _point_grad_enabled(value):
                return True
    return False


def _triangle_surface_canonical_prim(tri_data, prim_idx_i32):
    valid_prim = prim_idx_i32 >= 0
    if tri_data is None or "surface_canonical_prim" not in tri_data:
        return dr.select(valid_prim, prim_idx_i32, wt.Int32(-1))
    safe_prim_idx = wt.UInt32(dr.select(valid_prim, prim_idx_i32, wt.Int32(0)))
    return dr.select(valid_prim, dr.gather(wt.Int32, tri_data["surface_canonical_prim"], safe_prim_idx), wt.Int32(-1))


def _edge_owner_structure_idx(scene, adjacent_face0, adjacent_face1):
    face0_owner = gather_structure_indices(scene, adjacent_face0)
    face1_owner = gather_structure_indices(scene, adjacent_face1)
    return dr.select(face0_owner >= 0, face0_owner, face1_owner)


def _triangle_surface_contains_point(p, prim_idx_i32, tri_data):
    valid_prim = prim_idx_i32 >= 0
    if tri_data is None:
        return valid_prim, dr.select(valid_prim, prim_idx_i32, wt.Int32(-1))

    safe_prim_idx = wt.UInt32(dr.select(valid_prim, prim_idx_i32, wt.Int32(0)))
    max_group_size = int(tri_data.get("surface_max_group_size", 0))
    if max_group_size <= 0 or "surface_group_members" not in tri_data:
        v0 = dr.gather(wt.Point3f, tri_data["v0"], safe_prim_idx)
        v1 = dr.gather(wt.Point3f, tri_data["v1"], safe_prim_idx)
        v2 = dr.gather(wt.Point3f, tri_data["v2"], safe_prim_idx)
        hit = valid_prim & point_in_triangle_3d(p, v0, v1, v2)
        return hit, _triangle_surface_canonical_prim(tri_data, prim_idx_i32)

    group_size = dr.gather(wt.UInt32, tri_data["surface_group_size"], safe_prim_idx)
    surface_hit = dr.zeros(wt.Bool, dr.width(p.x))
    for slot in range(max_group_size):
        slot_active = valid_prim & (group_size > wt.UInt32(slot))
        flat_idx = safe_prim_idx * wt.UInt32(max_group_size) + wt.UInt32(slot)
        member_idx_i32 = dr.gather(wt.Int32, tri_data["surface_group_members"], flat_idx)
        slot_active = slot_active & (member_idx_i32 >= 0)
        safe_member_idx = wt.UInt32(dr.select(slot_active, member_idx_i32, wt.Int32(0)))
        v0 = dr.gather(wt.Point3f, tri_data["v0"], safe_member_idx)
        v1 = dr.gather(wt.Point3f, tri_data["v1"], safe_member_idx)
        v2 = dr.gather(wt.Point3f, tri_data["v2"], safe_member_idx)
        surface_hit = surface_hit | (slot_active & point_in_triangle_3d(p, v0, v1, v2))

    return surface_hit, _triangle_surface_canonical_prim(tri_data, prim_idx_i32)


def _intersect_rays_ad(ray_origin, ray_dir, active, scene, tri_data):
    hit, blocker_dist, hit_p, geom_n, _ = _intersect_rays_ad_with_prim(
        ray_origin, ray_dir, active, scene, tri_data
    )
    return hit, blocker_dist, hit_p, geom_n


def _intersect_rays_raw_with_prim(ray_origin, ray_dir, active, scene, *, tmax=None):
    rayd_scene = scene._require_rayd_scene() if hasattr(scene, "_require_rayd_scene") else scene
    ray = rayd.Ray(ray_origin, ray_dir)
    if tmax is not None:
        ray.tmax = tmax
    with dr.suspend_grad():
        raw = rayd_scene.intersect(
            ray,
            active=active,
            flags=rayd.RayFlags.All,
        )
        hit = raw.is_valid() & active
        blocker_dist = dr.select(hit, wt.Float(raw.t), wt.Float(1e10))
        prim_idx_i32 = wt.Int32(raw.prim_id)
    return hit, blocker_dist, wt.UInt32(dr.select(hit, prim_idx_i32, wt.Int32(-1)))


def _intersect_rays_ad_with_prim(ray_origin, ray_dir, active, scene, tri_data):
    ray = rayd.Ray(ray_origin, ray_dir)
    with dr.suspend_grad():
        si = scene.ray_intersect(ray, active)
        hit = si.is_valid() & active
        blocker_dist = dr.select(hit, si.t, wt.Float(1e10))
    prim_idx_i32 = wt.Int32(si.prim_index)
    resolved_prim_idx_i32 = dr.select(hit, prim_idx_i32, wt.Int32(-1))
    triangle_count = max(1, int(dr.width(tri_data["v0"].x))) if tri_data is not None else 1
    tri_count_i32 = wt.Int32(triangle_count)
    max_prim_i32 = wt.Int32(triangle_count - 1)
    prim_idx_valid = hit & (prim_idx_i32 >= 0) & (prim_idx_i32 < tri_count_i32)
    clamped_prim_idx_i32 = dr.minimum(dr.maximum(prim_idx_i32, wt.Int32(0)), max_prim_i32)
    safe_prim_idx = wt.UInt32(clamped_prim_idx_i32)

    if tri_data is None:
        return hit, blocker_dist, si.p, si.n, wt.UInt32(resolved_prim_idx_i32)

    # Keep invalid prim_index=-1 lanes away from triangle gathers. They are
    # resolved through the non-AD intersection data and masked out below.
    v0 = dr.gather(wt.Point3f, tri_data["v0"], safe_prim_idx)
    v1 = dr.gather(wt.Point3f, tri_data["v1"], safe_prim_idx)
    v2 = dr.gather(wt.Point3f, tri_data["v2"], safe_prim_idx)
    recon_n = dr.cross(v1 - v0, v2 - v0)
    recon_n = recon_n / (dr.norm(recon_n) + EPS)
    denom = dr.dot(ray_dir, recon_n)
    t_hit = dr.dot(v0 - ray_origin, recon_n) / (denom + EPS)
    blocker_dist = dr.select(prim_idx_valid, t_hit, blocker_dist)
    hit_p = dr.select(prim_idx_valid, ray_origin + t_hit * ray_dir, si.p)
    geom_n = dr.select(prim_idx_valid, recon_n, si.n)

    surface_hit, resolved_prim_idx = _triangle_surface_contains_point(hit_p, resolved_prim_idx_i32, tri_data)
    hit = hit & surface_hit
    blocker_dist = dr.select(hit, blocker_dist, wt.Float(1e10))
    return hit, blocker_dist, hit_p, geom_n, wt.UInt32(dr.select(hit, resolved_prim_idx, wt.Int32(-1)))


def _reflected_path_support_mask(image_source, target_pos, prim_idx, scene):
    width = dr.width(target_pos.x)
    if scene is None or scene.tri_data_gpu is None:
        return dr.full(wt.Bool, True, width)

    image_source_b = broadcast_point(image_source, width)
    prim_idx = repeat_int(wt.Int32(prim_idx), width)
    valid_prim = prim_idx >= 0
    safe_prim_idx = wt.UInt32(dr.select(valid_prim, prim_idx, wt.Int32(0)))

    tri_data = scene.tri_data_gpu
    v0 = dr.gather(wt.Point3f, tri_data["v0"], safe_prim_idx)
    v1 = dr.gather(wt.Point3f, tri_data["v1"], safe_prim_idx)
    v2 = dr.gather(wt.Point3f, tri_data["v2"], safe_prim_idx)
    geom_n = dr.cross(v1 - v0, v2 - v0)
    geom_n = geom_n / (dr.norm(geom_n) + EPS)

    segment = target_pos - image_source_b
    denom = dr.dot(segment, geom_n)
    valid_denom = dr.abs(denom) > EPS
    t_hit = dr.dot(v0 - image_source_b, geom_n) / (denom + EPS)
    hit_p = image_source_b + t_hit * segment
    surface_hit, _ = _triangle_surface_contains_point(hit_p, prim_idx, tri_data)
    return valid_prim & valid_denom & (t_hit > EPS) & (t_hit < (1.0 - EPS)) & surface_hit


def _segment_visibility_mask(
    start_pos,
    end_pos,
    scene,
    ignore_prim_idx=None,
    ignore_surface_group_idx=None,
    ignore_structure_idx=None,
    max_ignored_hits=4,
):
    width = dr.width(end_pos.x)
    if scene is None:
        return dr.full(wt.Bool, True, width)
    symbolic_recording = bool(dr.flag(dr.JitFlag.Recording))
    start_pos_b = broadcast_point(start_pos, width)
    seg_vec = end_pos - start_pos_b
    seg_len = dr.norm(seg_vec)
    min_seg_len = wt.Float(2.0 * RAY_ORIGIN_BIAS + EPS)
    active = seg_len > min_seg_len
    seg_dir = seg_vec / (seg_len + EPS)
    ray_origin = start_pos_b + seg_dir * RAY_ORIGIN_BIAS
    remaining = dr.maximum(seg_len - wt.Float(2.0 * RAY_ORIGIN_BIAS), wt.Float(0.0))

    if ignore_prim_idx is None and ignore_surface_group_idx is None and ignore_structure_idx is None:
        if hasattr(scene, "ray_test"):
            rays = rayd.Ray(ray_origin, seg_dir)
            rays.tmax = remaining
            with dr.suspend_grad():
                blocked = active & scene.ray_test(rays, active)
            return active & ~blocked
        hit, blocker_dist, _, _, _ = _intersect_rays_ad_with_prim(
            ray_origin, seg_dir, active, scene, scene.tri_data_gpu
        )
        blocked = hit & (blocker_dist < remaining)
        return active & ~blocked

    ignore_entries = []
    ignore_surface_group_candidates = (
        ignore_surface_group_idx
        if isinstance(ignore_surface_group_idx, (tuple, list))
        else (ignore_surface_group_idx,)
    )
    for surface_group_idx in ignore_surface_group_candidates:
        if surface_group_idx is None:
            continue
        ignore_group_i32 = _broadcast_i32(surface_group_idx, width)
        has_ignore = ignore_group_i32 >= 0
        ignore_entries.append((ignore_group_i32, has_ignore))
    ignore_candidates = ignore_prim_idx if isinstance(ignore_prim_idx, (tuple, list)) else (ignore_prim_idx,)
    for prim_idx in ignore_candidates:
        if prim_idx is None:
            continue
        prim_idx_i32 = _broadcast_i32(prim_idx, width)
        has_ignore = prim_idx_i32 >= 0
        ignore_group = _triangle_surface_group_id(scene.tri_data_gpu, prim_idx_i32)
        ignore_entries.append((ignore_group, has_ignore))
    ignore_structure_entries = []
    ignore_structure_candidates = (
        ignore_structure_idx
        if isinstance(ignore_structure_idx, (tuple, list))
        else (ignore_structure_idx,)
    )
    for structure_idx in ignore_structure_candidates:
        if structure_idx is None:
            continue
        structure_idx_i32 = repeat_int(wt.Int32(structure_idx), width)
        has_ignore = structure_idx_i32 >= 0
        ignore_structure_entries.append((structure_idx_i32, has_ignore))

    if len(ignore_entries) == 0 and len(ignore_structure_entries) == 0:
        if hasattr(scene, "ray_test"):
            rays = rayd.Ray(ray_origin, seg_dir)
            rays.tmax = remaining
            with dr.suspend_grad():
                blocked = active & scene.ray_test(rays, active)
            return active & ~blocked
        hit, blocker_dist, _, _, _ = _intersect_rays_ad_with_prim(
            ray_origin, seg_dir, active, scene, scene.tri_data_gpu
        )
        blocked = hit & (blocker_dist < remaining)
        return active & ~blocked

    blocked = dr.zeros(wt.Bool, width)
    unresolved = active
    cur_origin = ray_origin
    cur_remaining = remaining
    use_raw_ignore_loop = (
        not symbolic_recording
        and not _point_grad_enabled(start_pos)
        and not _point_grad_enabled(end_pos)
        and not _scene_geometry_grad_enabled(scene)
    )

    n_ignored_sets = max(1, len(ignore_entries) + len(ignore_structure_entries))
    for _ in range(max(max_ignored_hits, 2 * n_ignored_sets)):
        if use_raw_ignore_loop:
            hit, blocker_dist, prim_idx = _intersect_rays_raw_with_prim(
                cur_origin,
                seg_dir,
                unresolved,
                scene,
                tmax=cur_remaining,
            )
        else:
            hit, blocker_dist, _, _, prim_idx = _intersect_rays_ad_with_prim(
                cur_origin,
                seg_dir,
                unresolved,
                scene,
                scene.tri_data_gpu,
            )
        within = hit & (blocker_dist < cur_remaining)
        prim_idx_i32 = wt.Int32(prim_idx)
        hit_group = _triangle_surface_group_id(scene.tri_data_gpu, prim_idx_i32)
        ignored = dr.zeros(wt.Bool, width)
        for ignore_group, has_ignore in ignore_entries:
            ignored = ignored | (within & has_ignore & (hit_group == ignore_group))
        if len(ignore_structure_entries) > 0:
            hit_structure_idx = gather_structure_indices(scene, prim_idx_i32, valid_mask=within)
            for ignore_idx, has_ignore in ignore_structure_entries:
                ignored = ignored | (within & has_ignore & (hit_structure_idx == ignore_idx))
        blocked = blocked | (within & ~ignored)

        advance_dist = blocker_dist + wt.Float(RAY_ORIGIN_BIAS)
        advance = ignored & (advance_dist < cur_remaining)
        cur_origin = dr.select(advance, cur_origin + seg_dir * advance_dist, cur_origin)
        cur_remaining = dr.select(advance, cur_remaining - advance_dist, cur_remaining)
        unresolved = advance & ~blocked & (cur_remaining > EPS)
        if (not symbolic_recording) and (not dr.any(unresolved)):
            break

    return active & ~blocked & ~unresolved


def _segment_visibility_masks_batched(segment_starts, segment_ends, scene):
    if len(segment_starts) != len(segment_ends):
        raise ValueError("segment_starts and segment_ends must have the same length")
    if len(segment_starts) == 0:
        return tuple()
    if len(segment_starts) == 1:
        return (_segment_visibility_mask(segment_starts[0], segment_ends[0], scene),)

    widths = [dr.width(end_pos.x) for end_pos in segment_ends]
    if sum(widths) <= 0:
        return tuple(dr.zeros(wt.Bool, width) for width in widths)

    batched_start = concat_points(
        [broadcast_point(start_pos, width) for start_pos, width in zip(segment_starts, widths, strict=True)]
    )
    batched_end = concat_points(segment_ends)
    batched_visible = _segment_visibility_mask(batched_start, batched_end, scene)

    masks = []
    offset = 0
    for width in widths:
        if width <= 0:
            masks.append(dr.zeros(wt.Bool, 0))
        else:
            gather_idx = dr.arange(wt.UInt32, width) + wt.UInt32(offset)
            masks.append(dr.gather(wt.Bool, batched_visible, gather_idx))
        offset += width
    return tuple(masks)


def _fused_diffraction_visibility_masks(
    *,
    source_pos,
    diff_point,
    diff_point_offset,
    target_pos,
    target_valid,
    scene,
):
    safe_target_diff_point = dr.select(target_valid, diff_point, target_pos)
    safe_target_diff_point_offset = dr.select(target_valid, diff_point_offset, target_pos)
    return _segment_visibility_masks_batched(
        (source_pos, source_pos, target_pos, target_pos),
        (
            diff_point,
            diff_point_offset,
            safe_target_diff_point,
            safe_target_diff_point_offset,
        ),
        scene,
    )


def _constant_ray_direction(direction):
    vector = wt.Vector3f(float(direction[0]), float(direction[1]), float(direction[2]))
    return vector / (dr.norm(vector) + EPS)


def _point_inside_closed_mesh_mask_single(point_pos, scene, ray_dir, *, active=None):
    width = dr.width(point_pos.x)
    inside = dr.zeros(wt.Bool, width)
    if scene is None:
        return inside
    tri_data = getattr(scene, "tri_data_gpu", None)
    if tri_data is None or int(tri_data.get("n_triangles", 0)) <= 0:
        return inside

    active_mask = dr.full(wt.Bool, True, width) if active is None else active

    if not _point_grad_enabled(point_pos) and not _scene_geometry_grad_enabled(scene):
        ray_dir_b = wt.Vector3f(
            dr.full(wt.Float, scalar(ray_dir.x), width),
            dr.full(wt.Float, scalar(ray_dir.y), width),
            dr.full(wt.Float, scalar(ray_dir.z), width),
        )
        ray_origin = point_pos + ray_dir_b * RAY_ORIGIN_BIAS
        with dr.suspend_grad():
            si = scene.ray_intersect(rayd.Ray(ray_origin, ray_dir_b), active_mask)
            hit = si.is_valid() & active_mask
        return active_mask & hit & (dr.dot(si.geo_n, ray_dir_b) > wt.Float(0.0))
    ray_dir_b = broadcast_vector(ray_dir, width)
    ray_origin = point_pos + ray_dir_b * RAY_ORIGIN_BIAS
    hit, _, _, geom_n, _ = _intersect_rays_ad_with_prim(
        ray_origin,
        ray_dir_b,
        active_mask,
        scene,
        tri_data,
    )
    return active_mask & hit & (dr.dot(geom_n, ray_dir_b) > wt.Float(0.0))


def _point_inside_closed_mesh_mask(point_pos, scene, *, active=None):
    width = dr.width(point_pos.x)
    inside = dr.zeros(wt.Bool, width)
    if scene is None:
        return inside
    tri_data = getattr(scene, "tri_data_gpu", None)
    if tri_data is None or int(tri_data.get("n_triangles", 0)) <= 0:
        return inside

    active_mask = dr.full(wt.Bool, True, width) if active is None else active

    directions = (
        _constant_ray_direction((0.81234133, 0.52311241, 0.25843197)),
        _constant_ray_direction((-0.37139068, 0.60114462, 0.70757474)),
    )
    inside = active_mask
    for ray_dir in directions:
        inside = inside & _point_inside_closed_mesh_mask_single(
            point_pos,
            scene,
            ray_dir,
            active=active_mask,
        )
    return inside


def _triangle_surface_intersection(image_source, target_pos, prim_idx, scene):
    width = dr.width(target_pos.x)
    tri_data = None if scene is None else scene.tri_data_gpu
    if tri_data is None:
        zero_point = wt.Point3f(
            dr.zeros(wt.Float, width),
            dr.zeros(wt.Float, width),
            dr.zeros(wt.Float, width),
        )
        zero_normal = wt.Vector3f(
            dr.zeros(wt.Float, width),
            dr.zeros(wt.Float, width),
            dr.zeros(wt.Float, width),
        )
        return dr.zeros(wt.Bool, width), zero_point, zero_normal, wt.Int32(prim_idx)

    image_source_b = broadcast_point(image_source, width)
    prim_idx_i32 = repeat_int(wt.Int32(prim_idx), width)
    valid_prim = prim_idx_i32 >= 0
    safe_prim_idx = wt.UInt32(dr.select(valid_prim, prim_idx_i32, wt.Int32(0)))

    v0 = dr.gather(wt.Point3f, tri_data["v0"], safe_prim_idx)
    v1 = dr.gather(wt.Point3f, tri_data["v1"], safe_prim_idx)
    v2 = dr.gather(wt.Point3f, tri_data["v2"], safe_prim_idx)
    geom_n = dr.cross(v1 - v0, v2 - v0)
    geom_n = geom_n / (dr.norm(geom_n) + EPS)

    segment = target_pos - image_source_b
    denom = dr.dot(segment, geom_n)
    valid_denom = dr.abs(denom) > EPS
    t_hit = dr.dot(v0 - image_source_b, geom_n) / (denom + EPS)
    hit_p = image_source_b + t_hit * segment
    surface_hit, resolved_prim_idx = _triangle_surface_contains_point(hit_p, prim_idx_i32, tri_data)
    valid = valid_prim & valid_denom & (t_hit > EPS) & (t_hit < (1.0 - EPS)) & surface_hit
    return valid, hit_p, geom_n, resolved_prim_idx



def _wedge_exterior_region_mask(direction_from_edge, edge_dir, n0, nn):
    """
    Check whether a point direction lies on or outside the wedge boundary.

    This replaces the old hard angular clipping with a direct half-space test
    against the two wedge faces in the plane perpendicular to the edge.
    """
    direction_proj = _project_to_wedge_plane(direction_from_edge, edge_dir)
    signed_distance_0 = dr.dot(direction_proj, n0)
    signed_distance_n = dr.dot(direction_proj, nn)
    return (
        (dr.norm(direction_proj) > wt.Float(SMALL_EPS))
        & ((signed_distance_0 >= -wt.Float(SMALL_EPS)) | (signed_distance_n >= -wt.Float(SMALL_EPS)))
    )


__all__ = [
    "_slope_derivative_safe_mask",
    "_cotangent_pole_safe_mask",
    "point_in_triangle_3d",
    "_triangle_surface_group_id",
    "_triangle_surface_canonical_prim",
    "_edge_owner_structure_idx",
    "_triangle_surface_contains_point",
    "_intersect_rays_ad",
    "_intersect_rays_ad_with_prim",
    "_point_inside_closed_mesh_mask",
    "_point_inside_closed_mesh_mask_single",
    "_reflected_path_support_mask",
    "_fused_diffraction_visibility_masks",
    "_segment_visibility_mask",
    "_segment_visibility_masks_batched",
    "_triangle_surface_intersection",
    "reflect_point_across_plane",
    "_wedge_exterior_region_mask",
]
