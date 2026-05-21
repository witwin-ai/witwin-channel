"""Functional tests for the live ``witwin.channel.core`` surface.

Helpers that were dropped during the channel_utils consolidation
(``to_numpy*`` family, ``compute_face_normals``, ``triangles_are_coplanar``
CPU-scalar variant, ``WrappedOps``, ``TorchBridge.argsort_stable/lexsort/
to_torch``) are no longer covered here — the deterministic and Monte Carlo
solvers consume the underlying GPU buffers directly via
:mod:`witwin.channel.core`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import drjit as dr
import witwin.channel as wt

from witwin.channel.core.numerics.arrays import scalar
from witwin.channel.core.geometry import (
    reflect_point_across_plane,
    surface_contains_point,
    triangles_are_coplanar_mask,
)
from witwin.channel.core.geometry.mesh_buffers import (
    faces_array,
    to_point3f,
    to_vector3u,
    vertices_array,
)
from witwin.channel.core.numerics.tensors import to_torch_view


def _tolist(value) -> list:
    return np.asarray(value).reshape(-1).tolist()


def test_mesh_buffer_helpers_round_trip_numpy_and_torch() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)

    vertex_array = vertices_array(vertices)
    face_array = faces_array(faces)
    vertex_dr = to_point3f(vertex_array)
    face_dr = to_vector3u(face_array)

    assert vertex_array.dtype == np.float32
    assert face_array.dtype == np.int32
    assert np.allclose(vertex_array, vertices.astype(np.float32))
    assert _tolist(face_dr.x) == [0]
    assert _tolist(face_dr.y) == [1]
    assert _tolist(face_dr.z) == [2]
    assert _tolist(vertex_dr.x) == [0.0, 1.0, 0.0]
    assert _tolist(vertex_dr.y) == [0.0, 0.0, 1.0]


def test_scalar_extracts_python_float_from_drjit_array() -> None:
    assert scalar(wt.Float([2.5])) == pytest.approx(2.5)
    assert scalar(3.5) == pytest.approx(3.5)


def test_to_torch_view_shares_drjit_storage_via_dlpack() -> None:
    src = wt.Float([1.0, 2.0, 3.0, 4.0])
    tensor = to_torch_view(src, dtype=torch.float32)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_geometry_helpers_cover_reflection_surface_membership_and_coplanarity() -> None:
    vertices = to_point3f(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    faces = to_vector3u(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    # Synthetic face normals: both triangles lie on the XY plane → +Z.
    n_faces = int(dr.width(faces.x))
    face_normals = wt.Vector3f(
        dr.zeros(wt.Float, n_faces),
        dr.zeros(wt.Float, n_faces),
        dr.full(wt.Float, 1.0, n_faces),
    )
    tri_v0 = dr.gather(wt.Point3f, vertices, faces.x)
    tri_v1 = dr.gather(wt.Point3f, vertices, faces.y)
    tri_v2 = dr.gather(wt.Point3f, vertices, faces.z)
    tri_surface_data = {
        "group_size": wt.UInt32([2, 2]),
        "group_members": wt.Int32([0, 1, 0, 1]),
        "max_group_size": 2,
    }

    mask = triangles_are_coplanar_mask(
        wt.Int32([0]),
        wt.Int32([1]),
        face_normals,
        vertices,
        faces,
        shared_v0=wt.UInt32([0]),
        shared_v1=wt.UInt32([2]),
    )
    point = wt.Point3f(0.25, 0.75, 0.0)
    reflected = reflect_point_across_plane(
        wt.Point3f(1.0, 2.0, 3.0),
        wt.Point3f(0.0, 0.0, 0.0),
        wt.Vector3f(0.0, 0.0, 1.0),
    )

    assert _tolist(mask) == [True]
    assert not _tolist(surface_contains_point(point, wt.Int32(0), tri_v0, tri_v1, tri_v2, None))[0]
    assert _tolist(surface_contains_point(point, wt.Int32(0), tri_v0, tri_v1, tri_v2, tri_surface_data)) == [True]
    assert _tolist(reflected.z) == [-3.0]
