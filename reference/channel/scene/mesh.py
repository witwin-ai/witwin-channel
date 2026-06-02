"""Mesh geometry, edge filtering, and edge-geometry computation."""

from __future__ import annotations

import math

import drjit as dr
import torch
import witwin as wt

from witwin.core import GeometryBase

from ..utils.constants import EPS, SMALL_EPS
from ..utils import scalar
from ..utils.mesh_buffers import faces_array, vertices_array
from .types import VerticalEdge


class DrJitMesh(GeometryBase):
    """Explicit mesh geometry that preserves DrJit-backed vertex buffers."""

    kind = "drjit_mesh"

    def __init__(self, vertices, faces):
        super().__init__(position=(0.0, 0.0, 0.0), rotation=None, device="cpu")
        self._vertices = self._validate_vertices(vertices)
        self._faces = self._validate_faces(faces)

    @staticmethod
    def _validate_vertices(vertices):
        if isinstance(vertices, wt.Point3f):
            if dr.width(vertices) == 0:
                raise ValueError("vertices must contain at least one vertex.")
            return vertices
        if isinstance(vertices, torch.Tensor):
            if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
                raise ValueError("vertices must have shape (N, 3) with N > 0.")
            return vertices.contiguous()
        vertices_array(vertices)
        return vertices

    @staticmethod
    def _validate_faces(faces):
        if isinstance(faces, wt.Vector3u):
            if dr.width(faces) == 0:
                raise ValueError("faces must contain at least one triangle.")
            return faces
        if isinstance(faces, torch.Tensor):
            if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
                raise ValueError("faces must have shape (M, 3) with M > 0.")
            return faces.to(dtype=torch.int32).contiguous()
        faces_array(faces)
        return faces

    def to_mesh(self, segments=16, *, device=None):
        del segments
        vertices = self._vertices
        faces = self._faces
        if device is not None and isinstance(vertices, torch.Tensor):
            vertices = vertices.to(device=device, dtype=torch.float32).contiguous()
        if device is not None and isinstance(faces, torch.Tensor):
            faces = faces.to(device=device, dtype=torch.int32).contiguous()
        return vertices, faces


# ---------------------------------------------------------------------------
# Edge filtering and geometry
# ---------------------------------------------------------------------------


def filter_diffraction_edges(
    vertices,
    edge_entries,
    vertical_ratio=0.7,
    edge_selection_mode: str = "vertical_only",
    boundary_edge_policy: str = "exclude",
):
    """Filter edges that participate in diffraction."""
    if edge_selection_mode not in {"vertical_only", "all_edges"}:
        raise ValueError(
            f"Unsupported edge_selection_mode '{edge_selection_mode}'. "
            "Supported values are 'vertical_only' and 'all_edges'."
        )
    if boundary_edge_policy not in {"exclude", "half_plane"}:
        raise ValueError(
            f"Unsupported boundary_edge_policy '{boundary_edge_policy}'. "
            "Supported values are 'exclude' and 'half_plane'."
        )

    vertical_edges = []
    summary = {
        "total_vertical_edges": 0,
        "interior_vertical_edges": 0,
        "boundary_vertical_edges": 0,
        "included_vertical_edges": 0,
        "total_selected_edges": 0,
        "interior_selected_edges": 0,
        "boundary_selected_edges": 0,
        "included_selected_edges": 0,
        "included_boundary_edges": 0,
        "excluded_boundary_edges": 0,
        "selection_mode": edge_selection_mode,
        "total_mesh_edges": 0,
        "total_candidate_edges": 0,
        "included_edges": 0,
        "edges_matching_vertical_ratio": 0,
    }

    for edge_key, face_list in edge_entries:
        v0_idx, v1_idx = edge_key

        idx0 = wt.UInt32(v0_idx)
        idx1 = wt.UInt32(v1_idx)
        p0 = dr.gather(wt.Point3f, vertices, idx0)
        p1 = dr.gather(wt.Point3f, vertices, idx1)
        edge_vec = p1 - p0
        edge_len = dr.norm(edge_vec) + EPS

        z_component = dr.abs(edge_vec.z)
        vertical_ratio_val = z_component / edge_len
        summary["total_mesh_edges"] += 1
        matches_vertical_ratio = bool((edge_len > SMALL_EPS) & (vertical_ratio_val > vertical_ratio))
        if matches_vertical_ratio:
            summary["edges_matching_vertical_ratio"] += 1

        if edge_selection_mode == "vertical_only":
            is_valid = (edge_len > SMALL_EPS) & (vertical_ratio_val > vertical_ratio)
        else:
            is_valid = edge_len > SMALL_EPS
        if not bool(is_valid):
            continue

        summary["total_candidate_edges"] += 1
        is_boundary = len(face_list) == 1
        summary["total_vertical_edges"] += 1
        summary["total_selected_edges"] += 1
        if is_boundary:
            summary["boundary_vertical_edges"] += 1
            summary["boundary_selected_edges"] += 1
            if boundary_edge_policy == "exclude":
                summary["excluded_boundary_edges"] += 1
                continue
            summary["included_boundary_edges"] += 1
        else:
            summary["interior_vertical_edges"] += 1
            summary["interior_selected_edges"] += 1

        vertical_edges.append(
            VerticalEdge(
                vertex_indices=(v0_idx, v1_idx),
                p0=p0,
                p1=p1,
                adjacent_faces=tuple(face_list),
                is_boundary=is_boundary,
                edge_vector=edge_vec,
                length=edge_len,
                global_index=len(vertical_edges),
            )
        )

    summary["included_vertical_edges"] = len(vertical_edges)
    summary["included_selected_edges"] = len(vertical_edges)
    summary["included_edges"] = len(vertical_edges)
    return vertical_edges, summary


def compute_edge_geometry(
    edge_info,
    vertices,
    faces,
    mesh_center_3d=None,
    mesh_center_2d=None,
    n_verts=None,
    face_normals=None,
    boundary_edge_policy: str = "exclude",
):
    """Compute edge normals and wedge angle."""
    v0_idx, v1_idx = edge_info.vertex_indices
    face_indices = edge_info.adjacent_faces

    idx0 = wt.UInt32(v0_idx)
    idx1 = wt.UInt32(v1_idx)
    p0 = dr.gather(wt.Point3f, vertices, idx0)
    p1 = dr.gather(wt.Point3f, vertices, idx1)
    edge_vec = p1 - p0

    if n_verts is None:
        n_verts = dr.width(vertices)
    if mesh_center_3d is None:
        mesh_center_3d = wt.Point3f(
            dr.sum(vertices.x) / n_verts,
            dr.sum(vertices.y) / n_verts,
            dr.sum(vertices.z) / n_verts,
        )
    if mesh_center_2d is None:
        mesh_center_2d = wt.Vector2f(
            dr.sum(vertices.x) / n_verts,
            dr.sum(vertices.y) / n_verts,
        )
    del mesh_center_3d, mesh_center_2d

    f0 = faces.x
    f1 = faces.y
    f2 = faces.z

    face_normals_3d = []
    for face_idx in face_indices:
        if face_normals is not None:
            idx = wt.UInt32(int(face_idx))
            normal = wt.Vector3f(
                dr.gather(wt.Float, face_normals.x, idx),
                dr.gather(wt.Float, face_normals.y, idx),
                dr.gather(wt.Float, face_normals.z, idx),
            )
            face_normals_3d.append(normal)
            continue

        idx = wt.UInt32(int(face_idx))
        va_idx = dr.gather(wt.UInt32, f0, idx)
        vb_idx = dr.gather(wt.UInt32, f1, idx)
        vc_idx = dr.gather(wt.UInt32, f2, idx)
        va = dr.gather(wt.Point3f, vertices, va_idx)
        vb = dr.gather(wt.Point3f, vertices, vb_idx)
        vc = dr.gather(wt.Point3f, vertices, vc_idx)
        normal = dr.cross(vb - va, vc - va)
        norm_len = dr.norm(normal) + EPS
        face_normals_3d.append(normal / norm_len)

    if len(face_normals_3d) == 2:
        avg_normal_3d = (face_normals_3d[0] + face_normals_3d[1]) / 2
        normal_2d = wt.Vector2f(avg_normal_3d.x, avg_normal_3d.y)
    elif len(face_normals_3d) == 1:
        normal_2d = wt.Vector2f(face_normals_3d[0].x, face_normals_3d[0].y)
    else:
        normal_2d = wt.Vector2f(0.0, 0.0)

    norm_2d_len = dr.norm(normal_2d) + EPS
    normal_2d = dr.select(norm_2d_len > EPS, normal_2d / norm_2d_len, wt.Vector2f(0, 0))

    if len(face_normals_3d) == 2:
        n0_candidate = face_normals_3d[0]
        n1_candidate = face_normals_3d[1]
        edge_vec_normalized = edge_vec / (dr.norm(edge_vec) + EPS)

        to_hat_1 = dr.cross(n0_candidate, edge_vec_normalized)
        tn_hat_1 = dr.cross(n1_candidate, edge_vec_normalized)
        to_hat_2 = dr.cross(n1_candidate, edge_vec_normalized)
        tn_hat_2 = dr.cross(n0_candidate, edge_vec_normalized)

        to_hat_1 = to_hat_1 / (dr.norm(to_hat_1) + EPS)
        tn_hat_1 = tn_hat_1 / (dr.norm(tn_hat_1) + EPS)
        to_hat_2 = to_hat_2 / (dr.norm(to_hat_2) + EPS)
        tn_hat_2 = tn_hat_2 / (dr.norm(tn_hat_2) + EPS)

        cross_1 = dr.cross(to_hat_1, tn_hat_1)
        dot_1 = dr.dot(to_hat_1, tn_hat_1)
        sign_1 = dr.sign(dr.dot(cross_1, edge_vec_normalized))
        angle_1 = dr.atan2(sign_1 * dr.norm(cross_1), dot_1)
        angle_1 = dr.select(angle_1 < 0, angle_1 + 2 * dr.pi, angle_1)

        cross_2 = dr.cross(to_hat_2, tn_hat_2)
        dot_2 = dr.dot(to_hat_2, tn_hat_2)
        sign_2 = dr.sign(dr.dot(cross_2, edge_vec_normalized))
        angle_2 = dr.atan2(sign_2 * dr.norm(cross_2), dot_2)
        angle_2 = dr.select(angle_2 < 0, angle_2 + 2 * dr.pi, angle_2)

        choose_first = angle_1 < angle_2
        n0 = dr.select(choose_first, n0_candidate, n1_candidate)
        n1 = dr.select(choose_first, n1_candidate, n0_candidate)

        dot_product = dr.clip(-dr.dot(n0, n1), -1.0, 1.0)
        interior_angle = dr.acos(dot_product)
        exterior_angle = 2 * dr.pi - interior_angle
        wedge_n = exterior_angle / dr.pi
        face_normals_3d = [n0, n1]
    elif edge_info.is_boundary and len(face_normals_3d) == 1 and boundary_edge_policy == "half_plane":
        boundary_normal = face_normals_3d[0]
        face_normals_3d = [boundary_normal, -boundary_normal]
        wedge_n = wt.Float(2.0)
    else:
        wedge_n = None

    edge_info.normal_2d = normal_2d
    edge_info.wedge_n = wedge_n
    edge_info.face_normals_3d = face_normals_3d
    return edge_info
