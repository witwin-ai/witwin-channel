from __future__ import annotations

import drjit as dr
import numpy as np
import torch
from witwin.channel import types as wt


def _to_n3_array(value, *, np_dtype, drjit_type, label: str) -> np.ndarray:
    if isinstance(value, drjit_type):
        array = np.stack(
            [np.asarray(getattr(value, axis), dtype=np_dtype) for axis in ("x", "y", "z")],
            axis=-1,
        )
    elif isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value, dtype=np_dtype)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{label} must have shape (N, 3).")
    return np.ascontiguousarray(array.astype(np_dtype, copy=False))


def vertices_array(vertices: wt.Point3f | torch.Tensor | np.ndarray) -> np.ndarray:
    return _to_n3_array(vertices, np_dtype=np.float32, drjit_type=wt.Point3f, label="vertices")


def faces_array(faces: wt.Vector3u | torch.Tensor | np.ndarray) -> np.ndarray:
    return _to_n3_array(faces, np_dtype=np.int32, drjit_type=wt.Vector3u, label="faces")


def to_point3f(vertices: wt.Point3f | torch.Tensor | np.ndarray) -> wt.Point3f:
    if isinstance(vertices, wt.Point3f):
        return vertices
    if isinstance(vertices, torch.Tensor):
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("Expected vertices torch tensor with shape (N, 3).")
        return wt.Point3f(
            wt.Float(vertices[:, 0].contiguous()),
            wt.Float(vertices[:, 1].contiguous()),
            wt.Float(vertices[:, 2].contiguous()),
        )
    array = vertices_array(vertices)
    return wt.Point3f(
        wt.Float(array[:, 0].tolist()),
        wt.Float(array[:, 1].tolist()),
        wt.Float(array[:, 2].tolist()),
    )


def to_vector3u(faces: wt.Vector3u | torch.Tensor | np.ndarray) -> wt.Vector3u:
    if isinstance(faces, wt.Vector3u):
        return faces
    if isinstance(faces, torch.Tensor):
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Expected faces torch tensor with shape (M, 3).")
        faces = faces.to(dtype=torch.int32)
        return wt.Vector3u(
            wt.UInt32(faces[:, 0].contiguous()),
            wt.UInt32(faces[:, 1].contiguous()),
            wt.UInt32(faces[:, 2].contiguous()),
        )
    array = faces_array(faces)
    return wt.Vector3u(
        wt.UInt32(array[:, 0].tolist()),
        wt.UInt32(array[:, 1].tolist()),
        wt.UInt32(array[:, 2].tolist()),
    )


def mesh_buffer_count(buffer: wt.Point3f | wt.Vector3u | torch.Tensor | np.ndarray) -> int:
    if isinstance(buffer, (wt.Point3f, wt.Vector3u)):
        return int(dr.width(buffer))
    if isinstance(buffer, torch.Tensor):
        if buffer.ndim != 2 or buffer.shape[1] != 3:
            raise ValueError("Expected a torch mesh buffer with shape (N, 3).")
        return int(buffer.shape[0])
    array = np.asarray(buffer)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Expected a mesh buffer with shape (N, 3).")
    return int(array.shape[0])


__all__ = [
    "faces_array",
    "mesh_buffer_count",
    "to_point3f",
    "to_vector3u",
    "vertices_array",
]
