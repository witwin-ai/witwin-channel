"""Native CUDA implementation of the batch coplanarity check."""

from __future__ import annotations

import drjit as dr

import witwin as wt
from witwin.channel._native import _extension
from witwin.channel.kernels.scene_build.coplanarity import drjit_impl as _drjit_impl


def _can_use_native(face_normals, vertices) -> bool:
    for value in (
        face_normals.x, face_normals.y, face_normals.z,
        vertices.x, vertices.y, vertices.z,
    ):
        if dr.grad_enabled(value):
            return False
    return True


def batch_coplanarity_check(
    face_normals,
    edge_face_a,
    edge_face_b,
    vertices,
    faces,
    n_edges: int,
    normal_cos_tol: float = 0.99999,
    plane_tol: float = 1e-5,
):
    """
    Native CUDA batch coplanarity check.

    Same signature as ``drjit_impl.batch_coplanarity_check``.
    """
    if n_edges <= 0:
        return dr.zeros(wt.Bool, 0)

    if not _can_use_native(face_normals, vertices):
        return _drjit_impl.batch_coplanarity_check(
            face_normals,
            edge_face_a,
            edge_face_b,
            vertices,
            faces,
            n_edges,
            normal_cos_tol=normal_cos_tol,
            plane_tol=plane_tol,
        )

    ext = _extension()
    edge_face_a_i32 = wt.Int32(edge_face_a)
    edge_face_b_i32 = wt.Int32(edge_face_b)
    face0_i32 = wt.Int32(faces[0])
    face1_i32 = wt.Int32(faces[1])
    face2_i32 = wt.Int32(faces[2])
    result = ext.batch_coplanarity_check_arrays(
        dr.detach(face_normals.x),
        dr.detach(face_normals.y),
        dr.detach(face_normals.z),
        edge_face_a_i32,
        edge_face_b_i32,
        dr.detach(vertices.x),
        dr.detach(vertices.y),
        dr.detach(vertices.z),
        face0_i32,
        face1_i32,
        face2_i32,
        n_edges,
        dr.width(face_normals.x),
        dr.width(vertices.x),
        normal_cos_tol,
        plane_tol,
    )
    return wt.Int32(result) != 0
