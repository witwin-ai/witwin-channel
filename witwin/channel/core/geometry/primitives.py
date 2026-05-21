"""Pure geometric primitives: axes, axis-aligned planes, triangle/surface tests.

All helpers are GPU-vectorized DrJit math with no scene/solver state.
"""

from __future__ import annotations

from types import MappingProxyType

import drjit as dr
from witwin.channel import types as wt

from witwin.channel.core.numerics.constants import BARY_EPS, SMALL_EPS


_AXES = ("x", "y", "z")

_AXIS_TO_TANGENTIAL_AXES = MappingProxyType({
    "z": ("x", "y"),
    "x": ("y", "z"),
    "y": ("x", "z"),
})


def normalize_axis(axis: str) -> str:
    axis_name = str(axis).lower()
    if axis_name not in _AXES:
        raise ValueError("axis must be one of 'x', 'y', or 'z'.")
    return axis_name


def tangential_axes_for_axis(axis: str) -> tuple[str, str]:
    return _AXIS_TO_TANGENTIAL_AXES[normalize_axis(axis)]


def normal_axis_for_tangential_axes(tangential_axes: tuple[str, str]) -> str:
    if len(tangential_axes) != 2:
        raise ValueError("tangential_axes must contain exactly two axis labels.")
    axis_0, axis_1 = normalize_axis(tangential_axes[0]), normalize_axis(tangential_axes[1])
    if axis_0 == axis_1:
        raise ValueError("tangential_axes must use two distinct axes.")
    remaining = tuple(a for a in _AXES if a not in {axis_0, axis_1})
    return remaining[0]


def point_on_axis_aligned_plane(*, axis: str, position, tangential_0, tangential_1):
    axis_name = normalize_axis(axis)
    width = max(dr.width(tangential_0), dr.width(tangential_1))
    fixed = dr.full(wt.Float, position, width) if width > 1 else wt.Float(position)
    if axis_name == "x":
        return wt.Point3f(fixed, tangential_0, tangential_1)
    if axis_name == "y":
        return wt.Point3f(tangential_0, fixed, tangential_1)
    return wt.Point3f(tangential_0, tangential_1, fixed)


def point_in_triangle_3d(p, v0, v1, v2):
    """Barycentric-coordinate point-in-triangle test in 3D."""
    edge1, edge2, vp = v1 - v0, v2 - v0, p - v0
    dot00 = dr.dot(edge1, edge1)
    dot01 = dr.dot(edge1, edge2)
    dot02 = dr.dot(edge1, vp)
    dot11 = dr.dot(edge2, edge2)
    dot12 = dr.dot(edge2, vp)
    inv_denom = dr.rcp(dot00 * dot11 - dot01 * dot01 + BARY_EPS)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    eps = wt.Float(SMALL_EPS)
    return (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps)


def reflect_point_across_plane(point, plane_point, plane_normal):
    return point - 2.0 * dr.dot(point - plane_point, plane_normal) * plane_normal


def _triangle_hit(p, valid, member_idx_i32, tri_v0, tri_v1, tri_v2):
    active = valid & (member_idx_i32 >= 0)
    safe = wt.UInt32(dr.select(active, member_idx_i32, wt.Int32(0)))
    return active & point_in_triangle_3d(
        p,
        dr.gather(wt.Point3f, tri_v0, safe),
        dr.gather(wt.Point3f, tri_v1, safe),
        dr.gather(wt.Point3f, tri_v2, safe),
    )


def surface_contains_point(p, prim_idx_i32, tri_v0, tri_v1, tri_v2, tri_surface_data):
    """Test whether *p* lies on the surface group containing the given primitive."""
    valid = prim_idx_i32 >= 0
    safe_idx = wt.UInt32(dr.select(valid, prim_idx_i32, wt.Int32(0)))

    max_group_size = (
        int(tri_surface_data.get("max_group_size", 0))
        if tri_surface_data is not None and "group_members" in tri_surface_data
        else 0
    )
    if max_group_size <= 0:
        return _triangle_hit(p, valid, prim_idx_i32, tri_v0, tri_v1, tri_v2)

    group_members = tri_surface_data["group_members"]
    # max_group_size in {1,2} stores every slot densely; >2 needs a per-row group_size mask.
    needs_size_mask = max_group_size > 2
    group_size = (
        dr.gather(wt.UInt32, tri_surface_data["group_size"], safe_idx)
        if needs_size_mask else None
    )

    stride = wt.UInt32(max_group_size)
    surface_hit = dr.zeros(wt.Bool, dr.width(p.x))
    for slot in range(max_group_size):
        flat_idx = safe_idx * stride + wt.UInt32(slot)
        member_idx_i32 = dr.gather(wt.Int32, group_members, flat_idx)
        if needs_size_mask:
            slot_active = group_size > wt.UInt32(slot)
            member_idx_i32 = dr.select(slot_active, member_idx_i32, wt.Int32(-1))
        surface_hit = surface_hit | _triangle_hit(p, valid, member_idx_i32, tri_v0, tri_v1, tri_v2)
    return surface_hit


def triangles_are_coplanar_mask(
    face_idx_a,
    face_idx_b,
    face_normals: wt.Vector3f,
    vertices: wt.Point3f,
    faces: wt.Vector3u,
    *,
    shared_v0=None,
    shared_v1=None,
    normal_cos_tol: float = 1.0 - 1e-5,
    plane_tol: float = 1e-5,
) -> wt.Bool:
    """Per-edge coplanarity mask for adjacent triangle pairs."""
    face_idx_a_i32 = wt.Int32(face_idx_a)
    face_idx_b_i32 = wt.Int32(face_idx_b)
    valid = (face_idx_a_i32 >= 0) & (face_idx_b_i32 >= 0)

    safe_a = wt.UInt32(dr.select(valid, face_idx_a_i32, wt.Int32(0)))
    safe_b = wt.UInt32(dr.select(valid, face_idx_b_i32, wt.Int32(0)))

    normal_a = dr.gather(wt.Vector3f, face_normals, safe_a)
    normal_b = dr.gather(wt.Vector3f, face_normals, safe_b)
    aligned = dr.abs(dr.dot(normal_a, normal_b)) >= normal_cos_tol

    face_a = dr.gather(wt.Vector3u, faces, safe_a)
    face_b = dr.gather(wt.Vector3u, faces, safe_b)

    if shared_v0 is None or shared_v1 is None:
        shared0, shared1 = face_a.x, face_a.y
    else:
        shared0, shared1 = wt.UInt32(shared_v0), wt.UInt32(shared_v1)

    def _opposite(face):
        x_other = (face.x != shared0) & (face.x != shared1)
        y_other = (face.y != shared0) & (face.y != shared1)
        return dr.select(x_other, face.x, dr.select(y_other, face.y, face.z))

    plane_point = dr.gather(wt.Point3f, vertices, shared0)
    point_a = dr.gather(wt.Point3f, vertices, _opposite(face_a))
    point_b = dr.gather(wt.Point3f, vertices, _opposite(face_b))

    plane_dist_a = dr.abs(dr.dot(point_a - plane_point, normal_a))
    plane_dist_b = dr.abs(dr.dot(point_b - plane_point, normal_a))
    return valid & aligned & (plane_dist_a <= plane_tol) & (plane_dist_b <= plane_tol)


__all__ = [
    "normal_axis_for_tangential_axes",
    "normalize_axis",
    "point_in_triangle_3d",
    "point_on_axis_aligned_plane",
    "reflect_point_across_plane",
    "surface_contains_point",
    "tangential_axes_for_axis",
    "triangles_are_coplanar_mask",
]
