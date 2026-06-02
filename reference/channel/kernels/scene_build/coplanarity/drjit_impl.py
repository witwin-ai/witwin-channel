"""
Pure-DrJit reference implementation of the batch coplanarity check.

Interface matches native/coplanarity/coplanarity.h.

The C++ kernel does all edge-pair checks in a single parallel launch.
This DrJit path vectorizes the same logic over all edges at once,
replacing the per-edge ``scalar()`` GPU->CPU sync loop.
"""

from __future__ import annotations

import drjit as dr

import witwin as wt

SURFACE_GROUP_NORMAL_COS_TOL = 0.99999
SURFACE_GROUP_PLANE_TOL = 1e-5


def batch_coplanarity_check(
    face_normals,
    edge_face_a,
    edge_face_b,
    vertices,
    faces,
    n_edges: int,
    normal_cos_tol: float = SURFACE_GROUP_NORMAL_COS_TOL,
    plane_tol: float = SURFACE_GROUP_PLANE_TOL,
):
    """
    Vectorized coplanarity check for all edges with 2 adjacent faces.

    Parameters
    ----------
    face_normals : wt.Vector3f
        Unit face normals ``[n_faces]``.
    edge_face_a : wt.Int32
        First adjacent face index per edge ``[n_edges]``.
    edge_face_b : wt.Int32
        Second adjacent face index per edge ``[n_edges]``.
    vertices : wt.Point3f
        Mesh vertex positions ``[n_verts]``.
    faces : tuple of wt.UInt32
        Face vertex indices ``(v0, v1, v2)`` each ``[n_faces]``.
    n_edges : int
        Number of edges.
    normal_cos_tol : float
        Cosine threshold for normal alignment.
    plane_tol : float
        Distance threshold for planarity.

    Returns
    -------
    is_coplanar : wt.Bool
        Per-edge boolean ``[n_edges]``.
    """
    if n_edges <= 0:
        return dr.zeros(wt.Bool, 0)

    fa = wt.UInt32(edge_face_a)
    fb = wt.UInt32(edge_face_b)

    # Gather face normals
    na = dr.gather(wt.Vector3f, face_normals, fa)
    nb = dr.gather(wt.Vector3f, face_normals, fb)

    # Stage 1: normal alignment
    dot_n = dr.dot(na, nb)
    normals_ok = dr.abs(dot_n) >= normal_cos_tol

    # Stage 2: plane distance
    # Get face vertex indices
    v0a = dr.gather(wt.UInt32, faces[0], fa)
    v1a = dr.gather(wt.UInt32, faces[1], fa)
    v2a = dr.gather(wt.UInt32, faces[2], fa)
    v0b = dr.gather(wt.UInt32, faces[0], fb)
    v1b = dr.gather(wt.UInt32, faces[1], fb)
    v2b = dr.gather(wt.UInt32, faces[2], fb)

    # Find non-shared vertex from each face (simplified: use v2 as candidate,
    # which is correct when the shared edge is v0-v1; for the general case
    # we pick the vertex not equal to any vertex of the other face)
    # Use plane_point = vertex 0 of face_a
    plane_pt = dr.gather(wt.Point3f, vertices, v0a)

    # For a robust check, test all three vertices of face_b against plane of face_a
    p0b = dr.gather(wt.Point3f, vertices, v0b)
    p1b = dr.gather(wt.Point3f, vertices, v1b)
    p2b = dr.gather(wt.Point3f, vertices, v2b)

    def plane_dist(pt):
        d = pt - plane_pt
        return dr.abs(dr.dot(d, na))

    dist0 = plane_dist(p0b)
    dist1 = plane_dist(p1b)
    dist2 = plane_dist(p2b)

    # At least the non-shared vertex must be within tolerance.
    # The shared vertices are on the edge and should have dist~0.
    # Check max distance of all face_b vertices.
    max_dist = dr.maximum(dist0, dr.maximum(dist1, dist2))
    plane_ok = max_dist <= plane_tol

    return normals_ok & plane_ok
