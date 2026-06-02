from __future__ import annotations

import numpy as np
import torch
import witwin as wt


def vertices_array(vertices) -> np.ndarray:
    if isinstance(vertices, wt.Point3f):
        array = np.stack(
            [
                np.asarray(vertices.x, dtype=np.float32),
                np.asarray(vertices.y, dtype=np.float32),
                np.asarray(vertices.z, dtype=np.float32),
            ],
            axis=-1,
        )
    elif isinstance(vertices, torch.Tensor):
        array = vertices.detach().cpu().numpy()
    else:
        array = np.asarray(vertices, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3).")
    return np.ascontiguousarray(array.astype(np.float32, copy=False))


def faces_array(faces) -> np.ndarray:
    if isinstance(faces, wt.Vector3u):
        array = np.stack(
            [
                np.asarray(faces.x, dtype=np.int32),
                np.asarray(faces.y, dtype=np.int32),
                np.asarray(faces.z, dtype=np.int32),
            ],
            axis=-1,
        )
    elif isinstance(faces, torch.Tensor):
        array = faces.detach().cpu().numpy()
    else:
        array = np.asarray(faces, dtype=np.int32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3).")
    return np.ascontiguousarray(array.astype(np.int32, copy=False))


def to_point3f(vertices) -> wt.Point3f:
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


def to_vector3u(faces) -> wt.Vector3u:
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


__all__ = [
    "faces_array",
    "to_point3f",
    "to_vector3u",
    "vertices_array",
]
