from __future__ import annotations

import math

import drjit as dr
import witwin as wt

from ..utils.constants import EDGE_2D_EPS, EPS
from ..utils import scalar
from .types import Corner2D, DiffractionPoint, Edge2D


def drjit_to_key(v):
    """Convert a DrJit vector/point to a hashable tuple key."""
    if isinstance(v, wt.Vector2f):
        x = scalar(v.x)
        y = scalar(v.y)
        return (round(x, 6), round(y, 6))
    if isinstance(v, wt.Point3f):
        x = scalar(v.x)
        y = scalar(v.y)
        return (round(x, 6), round(y, 6))
    if hasattr(v, "__iter__"):
        return tuple(round(float(x), 6) for x in v)
    return (round(float(v), 6),)


def vectors_close(v1, v2, tol=1e-5):
    """Check if two DrJit vectors are close."""
    if isinstance(v1, (wt.Vector2f, wt.Vector3f, wt.Point3f)):
        v1_x = scalar(v1.x)
        v1_y = scalar(v1.y)
        v2_x = scalar(v2.x)
        v2_y = scalar(v2.y)
        return abs(v1_x - v2_x) < tol and abs(v1_y - v2_y) < tol
    return all(abs(a - b) < tol for a, b in zip(v1, v2))


def project_to_2d(vertical_edges, calculation_height, vertices):
    """
    Project selected diffraction edges to the calculation plane.

    Returns:
        edges_2d: list[Edge2D]
        corners_2d: list[Corner2D]
    """
    del vertices
    edges_2d = []
    corner_vertices = {}

    for index, edge_info in enumerate(vertical_edges):
        p0_3d = edge_info.p0
        p1_3d = edge_info.p1

        z0 = scalar(p0_3d.z)
        z1 = scalar(p1_3d.z)
        z_min = min(z0, z1)
        z_max = max(z0, z1)
        if not (z_min <= calculation_height <= z_max):
            continue

        p0_2d = wt.Vector2f(p0_3d.x, p0_3d.y)
        p1_2d = wt.Vector2f(p1_3d.x, p1_3d.y)
        edge_len_2d = dr.norm(p1_2d - p0_2d)
        edge_len_2d_val = scalar(edge_len_2d)

        if edge_len_2d_val > EDGE_2D_EPS:
            normal_2d = edge_info.normal_2d if edge_info.normal_2d is not None else wt.Vector2f(0, 0)
            edges_2d.append(Edge2D(p0_2d, p1_2d, normal_2d, f"edge_{index}"))

            corner_key_0 = drjit_to_key(p0_2d)
            corner_key_1 = drjit_to_key(p1_2d)
            if corner_key_0 not in corner_vertices:
                corner_vertices[corner_key_0] = {"pos": p0_2d, "edges": [], "vertical_edge": None}
            corner_vertices[corner_key_0]["edges"].append(edge_info)

            if corner_key_1 not in corner_vertices:
                corner_vertices[corner_key_1] = {"pos": p1_2d, "edges": [], "vertical_edge": None}
            corner_vertices[corner_key_1]["edges"].append(edge_info)
        else:
            corner_key = drjit_to_key(p0_2d)
            if corner_key not in corner_vertices:
                corner_vertices[corner_key] = {"pos": p0_2d, "edges": [], "vertical_edge": edge_info}
            else:
                corner_vertices[corner_key]["vertical_edge"] = edge_info

    corners_2d = []
    for vertex_data in corner_vertices.values():
        connected_edges = vertex_data["edges"]
        vertical_edge = vertex_data["vertical_edge"]
        if len(connected_edges) == 0 and vertical_edge is None:
            continue

        vertex_pos = vertex_data["pos"]
        if vertical_edge is not None:
            wedge_n = vertical_edge.wedge_n if vertical_edge.wedge_n is not None else wt.Float(1.5)
        elif len(connected_edges) == 1:
            wedge_n = wt.Float(2.0)
        else:
            edge0 = connected_edges[0]
            edge1 = connected_edges[1]
            wn0 = edge0.wedge_n if edge0.wedge_n is not None else wt.Float(1.5)
            wn1 = edge1.wedge_n if edge1.wedge_n is not None else wt.Float(1.5)
            wedge_n = (wn0 + wn1) / 2

        if vertical_edge is not None:
            edge_vector = vertical_edge.edge_vector
            e_hat = edge_vector / (dr.norm(edge_vector) + EPS)
            face_normals = vertical_edge.face_normals_3d or []

            if len(face_normals) >= 2:
                n0 = face_normals[0]
                nn = face_normals[1]
                to_hat = dr.cross(n0, e_hat)
                to_hat = to_hat / (dr.norm(to_hat) + EPS)
                tn_hat = dr.cross(nn, e_hat)
                tn_hat = tn_hat / (dr.norm(tn_hat) + EPS)

                to_hat_2d = wt.Vector2f(to_hat.x, to_hat.y)
                to_hat_2d = to_hat_2d / (dr.norm(to_hat_2d) + EPS)
                tn_hat_2d = wt.Vector2f(tn_hat.x, tn_hat.y)
                tn_hat_2d = tn_hat_2d / (dr.norm(tn_hat_2d) + EPS)

                face0_dir = to_hat_2d
                face_n_dir = tn_hat_2d
                corner_face_normals = [n0, nn]
                face0_point = vertex_pos + face0_dir * 0.1
                face_n_point = vertex_pos + face_n_dir * 0.1
                position_3d = wt.Point3f(vertex_pos.x, vertex_pos.y, wt.Float(calculation_height))
                corner_specific_edge_info = DiffractionPoint(
                    position=position_3d,
                    edge_vector=vertical_edge.edge_vector,
                    length=vertical_edge.length,
                    wedge_n=wedge_n,
                    face_normals_3d=corner_face_normals,
                    adjacent_faces=tuple(int(face_idx) for face_idx in vertical_edge.adjacent_faces),
                    vertex_indices=tuple(int(v_idx) for v_idx in vertical_edge.vertex_indices),
                    global_index=int(vertical_edge.global_index),
                )
            else:
                face0_point = vertex_pos + wt.Vector2f(0.1, 0.0)
                face_n_point = vertex_pos + wt.Vector2f(0.0, 0.1)
                corner_specific_edge_info = None
        elif len(connected_edges) == 0:
            face0_point = vertex_pos + wt.Vector2f(0.1, 0.0)
            face_n_point = vertex_pos + wt.Vector2f(0.0, 0.1)
            corner_specific_edge_info = None
        elif len(connected_edges) == 1:
            edge0 = connected_edges[0]
            p0_edge = wt.Vector2f(edge0.p0.x, edge0.p0.y)
            p1_edge = wt.Vector2f(edge0.p1.x, edge0.p1.y)
            if vectors_close(p0_edge, vertex_pos):
                edge_dir = p1_edge - p0_edge
            else:
                edge_dir = p0_edge - p1_edge

            edge_dir = edge_dir / (dr.norm(edge_dir) + EPS)
            face0_point = vertex_pos + edge_dir * 0.1
            ed_x = scalar(edge_dir.x)
            ed_y = scalar(edge_dir.y)
            perp_dir = wt.Vector2f(-ed_y, ed_x)
            face_n_point = vertex_pos + perp_dir * 0.1
            corner_specific_edge_info = None
        else:
            edge0 = connected_edges[0]
            edge1 = connected_edges[1]
            p0_e0 = wt.Vector2f(edge0.p0.x, edge0.p0.y)
            p1_e0 = wt.Vector2f(edge0.p1.x, edge0.p1.y)
            if vectors_close(p0_e0, vertex_pos):
                dir0 = p1_e0 - p0_e0
            else:
                dir0 = p0_e0 - p1_e0

            p0_e1 = wt.Vector2f(edge1.p0.x, edge1.p0.y)
            p1_e1 = wt.Vector2f(edge1.p1.x, edge1.p1.y)
            if vectors_close(p0_e1, vertex_pos):
                dir1 = p1_e1 - p0_e1
            else:
                dir1 = p0_e1 - p1_e1

            dir0 = dir0 / (dr.norm(dir0) + EPS)
            dir1 = dir1 / (dr.norm(dir1) + EPS)
            d0_x = scalar(dir0.x)
            d0_y = scalar(dir0.y)
            d1_x = scalar(dir1.x)
            d1_y = scalar(dir1.y)
            cross_val = d0_x * d1_y - d0_y * d1_x
            dot_val = d0_x * d1_x + d0_y * d1_y
            angle = math.atan2(cross_val, dot_val)
            if angle < 0:
                angle = angle + 2 * math.pi
            if angle < math.pi:
                face0_dir = dir1
                face_n_dir = dir0
            else:
                face0_dir = dir0
                face_n_dir = dir1
            face0_point = vertex_pos + face0_dir * 0.1
            face_n_point = vertex_pos + face_n_dir * 0.1
            corner_specific_edge_info = None

        corners_2d.append(
            Corner2D(
                vertex_pos,
                face0_point,
                face_n_point,
                f"corner_{len(corners_2d)}",
                corner_specific_edge_info,
            )
        )

    return edges_2d, corners_2d

