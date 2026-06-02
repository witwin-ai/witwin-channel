"""Native CUDA implementation of batch edge geometry computation."""

from __future__ import annotations

import drjit as dr

import witwin as wt
from witwin.channel._native import _extension
from witwin.channel.kernels.scene_build.edge_geometry import drjit_impl as _drjit_impl


def _can_use_native(vertices, face_normals) -> bool:
    for value in (
        vertices.x, vertices.y, vertices.z,
        face_normals.x, face_normals.y, face_normals.z,
    ):
        if dr.grad_enabled(value):
            return False
    return True


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
    Native CUDA batch edge geometry computation.

    Same signature as ``drjit_impl.batch_edge_geometry``.
    """
    if n_edges <= 0:
        z3 = wt.Vector3f(0.0, 0.0, 0.0)
        return {
            "pos": wt.Point3f(0.0, 0.0, 0.0),
            "edge_dir": z3,
            "n0": z3,
            "nn": z3,
            "wedge_n": wt.Float(0.0),
            "length": wt.Float(0.0),
        }

    if not _can_use_native(vertices, face_normals):
        return _drjit_impl.batch_edge_geometry(
            vertices,
            face_normals,
            edge_v0,
            edge_v1,
            edge_face0,
            edge_face1,
            n_edges,
        )

    ext = _extension()
    edge_v0_i32 = wt.Int32(edge_v0)
    edge_v1_i32 = wt.Int32(edge_v1)
    edge_face0_i32 = wt.Int32(edge_face0)
    edge_face1_i32 = wt.Int32(edge_face1)
    outputs = ext.batch_edge_geometry_arrays(
        dr.detach(vertices.x),
        dr.detach(vertices.y),
        dr.detach(vertices.z),
        dr.detach(face_normals.x),
        dr.detach(face_normals.y),
        dr.detach(face_normals.z),
        edge_v0_i32,
        edge_v1_i32,
        edge_face0_i32,
        edge_face1_i32,
        n_edges,
    )
    dr.eval(*outputs)
    return {
        "pos": wt.Point3f(wt.Float(outputs[0]), wt.Float(outputs[1]), wt.Float(outputs[2])),
        "edge_dir": wt.Vector3f(wt.Float(outputs[3]), wt.Float(outputs[4]), wt.Float(outputs[5])),
        "n0": wt.Vector3f(wt.Float(outputs[6]), wt.Float(outputs[7]), wt.Float(outputs[8])),
        "nn": wt.Vector3f(wt.Float(outputs[9]), wt.Float(outputs[10]), wt.Float(outputs[11])),
        "wedge_n": wt.Float(outputs[12]),
        "length": wt.Float(outputs[13]),
    }
