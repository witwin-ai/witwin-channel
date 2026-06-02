from __future__ import annotations

from collections.abc import Sequence

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


def to_point3f(value, *, role: str = "point", allow_single: bool = True) -> wt.Point3f:
    """Coerce ``value`` into ``wt.Point3f``.

    Accepts a ``wt.Point3f``, an object with ``.x/.y/.z`` attributes, a length-3
    scalar sequence or ``torch.Tensor`` of shape ``(3,)`` (when ``allow_single``),
    a ``torch.Tensor``/``numpy.ndarray`` of shape ``(N, 3)``, or a nested sequence
    of 3-vectors.
    """
    if isinstance(value, wt.Point3f):
        if dr.width(value.x) == 0:
            raise ValueError(f"{role} must contain at least one point.")
        return value
    if all(hasattr(value, axis) for axis in ("x", "y", "z")):
        return wt.Point3f(value.x, value.y, value.z)
    if isinstance(value, torch.Tensor):
        return _tensor_to_point3f(value, role=role, allow_single=allow_single)
    if isinstance(value, np.ndarray):
        return _tensor_to_point3f(torch.as_tensor(value, dtype=torch.float32), role=role, allow_single=allow_single)
    if isinstance(value, Sequence):
        if allow_single and len(value) == 3 and all(not hasattr(v, "__len__") for v in value):
            return wt.Point3f(value[0], value[1], value[2])
        if not value:
            raise ValueError(f"{role} sequence must not be empty.")
        return _tensor_to_point3f(
            torch.as_tensor(list(value), dtype=torch.float32), role=role, allow_single=allow_single
        )
    raise TypeError(
        f"{role} must be a wt.Point3f, length-3 sequence, numpy array, or torch tensor of shape (3,)/(N, 3)."
    )


def _tensor_to_point3f(tensor: torch.Tensor, *, role: str, allow_single: bool) -> wt.Point3f:
    if allow_single and tensor.ndim == 1 and tensor.shape[0] == 3:
        tensor = tensor.reshape(1, 3)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        expected = "(3,) or (N, 3)" if allow_single else "(N, 3)"
        raise ValueError(f"{role} tensor must have shape {expected}.")
    if tensor.shape[0] == 0:
        raise ValueError(f"{role} must contain at least one point.")
    tensor = tensor.to(dtype=torch.float32).contiguous()
    return wt.Point3f(wt.Float(tensor[:, 0]), wt.Float(tensor[:, 1]), wt.Float(tensor[:, 2]))


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
