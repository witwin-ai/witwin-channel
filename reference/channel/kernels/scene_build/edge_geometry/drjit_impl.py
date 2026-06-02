"""
Pure-DrJit reference implementation of the batch edge geometry kernel.

Interface matches native/edge_geometry/edge_geometry.h.

The C++ kernel computes all edges in one parallel launch.
This DrJit path vectorizes the same logic over all edges at once,
replacing the per-edge Python loop + per-component ``scalar()`` sync.
"""

from __future__ import annotations

import math

import drjit as dr

import witwin as wt

EPS = 1e-10


def batch_edge_geometry(
    vertices,
    face_normals,
    edge_v0,
    edge_v1,
    edge_face0,
    edge_face1,
    n_edges: int,
):
    """
    Compute diffraction edge geometry for all interior edges in one pass.

    Parameters
    ----------
    vertices : wt.Point3f
        Mesh vertex positions ``[n_verts]``.
    face_normals : wt.Vector3f
        Unit face normals ``[n_faces]``.
    edge_v0 : wt.UInt32
        First vertex index per edge ``[n_edges]``.
    edge_v1 : wt.UInt32
        Second vertex index per edge ``[n_edges]``.
    edge_face0 : wt.UInt32
        First adjacent face per edge ``[n_edges]``.
    edge_face1 : wt.UInt32
        Second adjacent face per edge ``[n_edges]``.
    n_edges : int
        Number of edges.

    Returns
    -------
    dict with keys:
        pos : wt.Point3f - edge midpoints ``[n_edges]``
        edge_dir : wt.Vector3f - normalized edge direction ``[n_edges]``
        n0 : wt.Vector3f - face 0 normal (oriented outward) ``[n_edges]``
        nn : wt.Vector3f - face 1 normal (oriented outward) ``[n_edges]``
        wedge_n : wt.Float - wedge angle parameter ``[n_edges]``
        length : wt.Float - edge length ``[n_edges]``
    """
    if n_edges <= 0:
        z3 = wt.Vector3f(0, 0, 0)
        return {
            "pos": wt.Point3f(0, 0, 0),
            "edge_dir": z3, "n0": z3, "nn": z3,
            "wedge_n": wt.Float(0), "length": wt.Float(0),
        }

    # Gather vertex positions
    p0 = dr.gather(wt.Point3f, vertices, edge_v0)
    p1 = dr.gather(wt.Point3f, vertices, edge_v1)

    # Edge vector and length
    edge_vec = p1 - p0
    edge_len = dr.norm(edge_vec)
    edge_hat = edge_vec / (edge_len + EPS)

    # Midpoint
    mid = (p0 + p1) * 0.5

    # Face normals
    fn0 = dr.gather(wt.Vector3f, face_normals, edge_face0)
    fn1 = dr.gather(wt.Vector3f, face_normals, edge_face1)

    # Orient normals for consistent UTD wedge convention
    t0 = dr.cross(fn0, edge_hat)
    t1 = dr.cross(fn1, edge_hat)

    # If tangents point same direction, flip fn1
    same_dir = dr.dot(t0, t1) > 0.0
    fn1 = dr.select(same_dir, -fn1, fn1)

    # Ensure t0 points outward (away from fn1 side)
    t0_new = dr.cross(fn0, edge_hat)
    outward = dr.dot(t0_new, fn1) >= 0.0
    fn0 = dr.select(outward, fn0, -fn0)
    fn1 = dr.select(outward, fn1, -fn1)

    # Wedge angle: exterior_angle = 2*pi - acos(clamp(-dot(n0, nn), -1, 1))
    cos_int = dr.clip(-dr.dot(fn0, fn1), -1.0, 1.0)
    interior_angle = dr.acos(cos_int)
    exterior_angle = 2.0 * math.pi - interior_angle
    wedge_n = exterior_angle / math.pi

    return {
        "pos": mid,
        "edge_dir": edge_hat,
        "n0": fn0,
        "nn": fn1,
        "wedge_n": wedge_n,
        "length": edge_len,
    }
