"""Pure geometric primitives (no scene/trace domain dependencies)."""

from collections import defaultdict

import drjit as dr
import witwin as wt

from .constants import BARY_EPS, EPS, SMALL_EPS
from .conversion import scalar


def point_in_triangle_3d(p, v0, v1, v2):
    """Barycentric-coordinate point-in-triangle test in 3D."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    vp = p - v0

    dot00 = dr.dot(edge1, edge1)
    dot01 = dr.dot(edge1, edge2)
    dot02 = dr.dot(edge1, vp)
    dot11 = dr.dot(edge2, edge2)
    dot12 = dr.dot(edge2, vp)

    inv_denom = dr.rcp(dot00 * dot11 - dot01 * dot01 + BARY_EPS)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    epsilon = wt.Float(SMALL_EPS)
    return (u >= -epsilon) & (v >= -epsilon) & (u + v <= 1 + epsilon)


def reflect_point_across_plane(point, plane_point, plane_normal):
    """Reflect a point across a plane defined by a point and normal."""
    d_to_plane = dr.dot(point - plane_point, plane_normal)
    return point - 2.0 * d_to_plane * plane_normal


def _surface_contains_point_group1(p, valid, safe_idx, group_members, tri_v0, tri_v1, tri_v2):
    member_idx_i32 = dr.gather(wt.Int32, group_members, safe_idx)
    active = valid & (member_idx_i32 >= 0)
    safe_member_idx = wt.UInt32(dr.select(active, member_idx_i32, wt.Int32(0)))
    return active & point_in_triangle_3d(
        p,
        dr.gather(wt.Point3f, tri_v0, safe_member_idx),
        dr.gather(wt.Point3f, tri_v1, safe_member_idx),
        dr.gather(wt.Point3f, tri_v2, safe_member_idx),
    )


def _surface_contains_point_group2(p, valid, safe_idx, group_members, tri_v0, tri_v1, tri_v2):
    flat_idx0 = safe_idx * wt.UInt32(2)
    flat_idx1 = flat_idx0 + wt.UInt32(1)
    member_idx0_i32 = dr.gather(wt.Int32, group_members, flat_idx0)
    member_idx1_i32 = dr.gather(wt.Int32, group_members, flat_idx1)
    active0 = valid & (member_idx0_i32 >= 0)
    active1 = valid & (member_idx1_i32 >= 0)
    safe_member_idx0 = wt.UInt32(dr.select(active0, member_idx0_i32, wt.Int32(0)))
    safe_member_idx1 = wt.UInt32(dr.select(active1, member_idx1_i32, wt.Int32(0)))
    hit0 = active0 & point_in_triangle_3d(
        p,
        dr.gather(wt.Point3f, tri_v0, safe_member_idx0),
        dr.gather(wt.Point3f, tri_v1, safe_member_idx0),
        dr.gather(wt.Point3f, tri_v2, safe_member_idx0),
    )
    hit1 = active1 & point_in_triangle_3d(
        p,
        dr.gather(wt.Point3f, tri_v0, safe_member_idx1),
        dr.gather(wt.Point3f, tri_v1, safe_member_idx1),
        dr.gather(wt.Point3f, tri_v2, safe_member_idx1),
    )
    return hit0 | hit1


def surface_contains_point(p, prim_idx_i32, tri_v0, tri_v1, tri_v2, tri_surface_data):
    """Test whether *p* lies on the surface group containing the given primitive."""
    valid = prim_idx_i32 >= 0
    safe_idx = wt.UInt32(dr.select(valid, prim_idx_i32, wt.Int32(0)))
    if tri_surface_data is None:
        return valid & point_in_triangle_3d(
            p,
            dr.gather(wt.Point3f, tri_v0, safe_idx),
            dr.gather(wt.Point3f, tri_v1, safe_idx),
            dr.gather(wt.Point3f, tri_v2, safe_idx),
        )

    max_group_size = int(tri_surface_data.get("max_group_size", 0))
    if max_group_size <= 0 or "group_members" not in tri_surface_data:
        return valid & point_in_triangle_3d(
            p,
            dr.gather(wt.Point3f, tri_v0, safe_idx),
            dr.gather(wt.Point3f, tri_v1, safe_idx),
            dr.gather(wt.Point3f, tri_v2, safe_idx),
        )

    group_members = tri_surface_data["group_members"]
    if max_group_size == 1:
        return _surface_contains_point_group1(
            p,
            valid,
            safe_idx,
            group_members,
            tri_v0,
            tri_v1,
            tri_v2,
        )

    if max_group_size == 2:
        return _surface_contains_point_group2(
            p,
            valid,
            safe_idx,
            group_members,
            tri_v0,
            tri_v1,
            tri_v2,
        )

    group_size = dr.gather(wt.UInt32, tri_surface_data["group_size"], safe_idx)
    surface_hit = dr.zeros(wt.Bool, dr.width(p.x))
    for slot in range(max_group_size):
        slot_active = valid & (group_size > wt.UInt32(slot))
        flat_idx = safe_idx * wt.UInt32(max_group_size) + wt.UInt32(slot)
        member_idx_i32 = dr.gather(wt.Int32, group_members, flat_idx)
        slot_active = slot_active & (member_idx_i32 >= 0)
        safe_member_idx = wt.UInt32(dr.select(slot_active, member_idx_i32, wt.Int32(0)))
        surface_hit = surface_hit | (
            slot_active
            & point_in_triangle_3d(
                p,
                dr.gather(wt.Point3f, tri_v0, safe_member_idx),
                dr.gather(wt.Point3f, tri_v1, safe_member_idx),
                dr.gather(wt.Point3f, tri_v2, safe_member_idx),
            )
        )
    return surface_hit


def extract_edges_with_adjacency(vertices, faces):
    """Extract all edges and their adjacent faces from a triangle mesh."""
    del vertices
    edge_to_faces = defaultdict(list)

    f0 = faces.x
    f1 = faces.y
    f2 = faces.z
    n_faces = dr.width(f0)

    for face_idx in range(n_faces):
        v0 = int(f0[face_idx])
        v1 = int(f1[face_idx])
        v2 = int(f2[face_idx])
        edges_in_face = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0)),
        ]
        for edge in edges_in_face:
            edge_to_faces[edge].append(face_idx)

    return edge_to_faces


def compute_face_normals(vertices, faces, mesh_center_3d=None, n_verts=None):
    """Compute face normals from vertex winding order via cross product."""
    del mesh_center_3d, n_verts
    f0 = faces.x
    f1 = faces.y
    f2 = faces.z
    n_faces = dr.width(f0)

    face_idx = dr.arange(wt.UInt32, n_faces)
    va = dr.gather(wt.Point3f, vertices, dr.gather(wt.UInt32, f0, face_idx))
    vb = dr.gather(wt.Point3f, vertices, dr.gather(wt.UInt32, f1, face_idx))
    vc = dr.gather(wt.Point3f, vertices, dr.gather(wt.UInt32, f2, face_idx))

    normal = dr.cross(vb - va, vc - va)
    norm_len = dr.norm(normal) + EPS
    return normal / norm_len


def triangles_are_coplanar(
    face_idx_a,
    face_idx_b,
    face_normals,
    vertices,
    faces,
    *,
    normal_cos_tol: float = 1.0 - 1e-5,
    plane_tol: float = 1e-5,
) -> bool:
    """Return whether two adjacent triangles lie on the same plane."""
    normal_a = (
        scalar(face_normals.x[face_idx_a]),
        scalar(face_normals.y[face_idx_a]),
        scalar(face_normals.z[face_idx_a]),
    )
    normal_b = (
        scalar(face_normals.x[face_idx_b]),
        scalar(face_normals.y[face_idx_b]),
        scalar(face_normals.z[face_idx_b]),
    )
    dot_normals = (
        normal_a[0] * normal_b[0]
        + normal_a[1] * normal_b[1]
        + normal_a[2] * normal_b[2]
    )
    if abs(dot_normals) < normal_cos_tol:
        return False

    face_a = (
        int(faces.x[face_idx_a]),
        int(faces.y[face_idx_a]),
        int(faces.z[face_idx_a]),
    )
    face_b = (
        int(faces.x[face_idx_b]),
        int(faces.y[face_idx_b]),
        int(faces.z[face_idx_b]),
    )
    shared_vertices = set(face_a) & set(face_b)
    if len(shared_vertices) < 2:
        return False

    def _vertex_position(vertex_idx: int) -> tuple[float, float, float]:
        return (
            scalar(vertices.x[vertex_idx]),
            scalar(vertices.y[vertex_idx]),
            scalar(vertices.z[vertex_idx]),
        )

    plane_point = _vertex_position(face_a[0])
    other_a = next((vertex_idx for vertex_idx in face_a if vertex_idx not in shared_vertices), face_a[0])
    other_b = next((vertex_idx for vertex_idx in face_b if vertex_idx not in shared_vertices), face_b[0])
    point_a = _vertex_position(other_a)
    point_b = _vertex_position(other_b)

    def _distance_to_plane(point: tuple[float, float, float]) -> float:
        dx = point[0] - plane_point[0]
        dy = point[1] - plane_point[1]
        dz = point[2] - plane_point[2]
        return abs(dx * normal_a[0] + dy * normal_a[1] + dz * normal_a[2])

    return _distance_to_plane(point_a) <= plane_tol and _distance_to_plane(point_b) <= plane_tol


__all__ = [
    "compute_face_normals",
    "extract_edges_with_adjacency",
    "point_in_triangle_3d",
    "reflect_point_across_plane",
    "surface_contains_point",
    "triangles_are_coplanar",
]
