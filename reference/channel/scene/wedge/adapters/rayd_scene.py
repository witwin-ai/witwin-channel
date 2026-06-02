"""RayD-backed adapter for the backend-neutral wedge runtime."""

from __future__ import annotations

from typing import Any

import witwin as wt
import torch

from ..types import EdgeInfoBuffer, EdgeTopologyBuffer


_RAYD_SCENE_METHODS = (
    "edge_info",
    "edge_topology",
    "triangle_edge_indices",
    "edge_adjacent_faces",
    "mesh_face_offsets",
    "mesh_edge_offsets",
    "intersect",
    "shadow_test",
)


def is_rayd_scene_like(source: Any) -> bool:
    return all(hasattr(source, name) for name in _RAYD_SCENE_METHODS) and hasattr(source, "version") and hasattr(
        source, "edge_version"
    )


def _to_torch_scalar_array(value: Any, *, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    elif hasattr(value, "torch"):
        tensor = value.torch()
    else:
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if device is not None and tensor.device != device:
        tensor = tensor.to(device=device)
    return tensor.to(dtype=dtype).reshape(-1).contiguous()


def _to_torch_vec3_array(value: Any, *, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    elif hasattr(value, "torch"):
        tensor = value.torch()
    else:
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 2 and tensor.shape[0] == 3 and tensor.shape[1] != 3:
        tensor = tensor.transpose(0, 1)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError("Expected a vector array with shape (N, 3).")
    if device is not None and tensor.device != device:
        tensor = tensor.to(device=device)
    return tensor.to(dtype=dtype).contiguous()


def _to_mi_point3(value: Any) -> wt.Point3f:
    tensor = _to_torch_vec3_array(value, dtype=torch.float32)
    return wt.Point3f(
        wt.Float(tensor[:, 0]),
        wt.Float(tensor[:, 1]),
        wt.Float(tensor[:, 2]),
    )


def _to_mi_vector3(value: Any) -> wt.Vector3f:
    tensor = _to_torch_vec3_array(value, dtype=torch.float32)
    return wt.Vector3f(
        wt.Float(tensor[:, 0]),
        wt.Float(tensor[:, 1]),
        wt.Float(tensor[:, 2]),
    )


def _to_mi_float1(value: Any) -> wt.Float:
    return wt.Float(_to_torch_scalar_array(value, dtype=torch.float32))


def _to_mi_int1(value: Any) -> wt.Int32:
    return wt.Int32(_to_torch_scalar_array(value, dtype=torch.int32))


def _to_mi_bool1(value: Any) -> wt.Bool:
    return wt.Bool(_to_torch_scalar_array(value, dtype=torch.bool))


def _version_value(owner: Any, name: str) -> int:
    value = getattr(owner, name)
    if callable(value):
        value = value()
    return int(value)


class RayDSceneAdapter:
    """Adapt RayD-style scene topology to the wedge runtime contract."""

    def __init__(self, scene: Any):
        if not is_rayd_scene_like(scene):
            raise TypeError("RayDSceneAdapter expects a RayD-like scene object.")
        self._scene = scene
        self._device: torch.device | None = None

    def version(self) -> int:
        return _version_value(self._scene, "version")

    def edge_version(self) -> int:
        return _version_value(self._scene, "edge_version")

    def n_triangles(self) -> int:
        offsets = _to_torch_scalar_array(self._scene.mesh_face_offsets(), dtype=torch.int32)
        if offsets.numel() == 0:
            return 0
        return int(offsets.reshape(-1)[-1].item())

    def edge_info(self) -> EdgeInfoBuffer:
        info = self._scene.edge_info()
        return EdgeInfoBuffer(
            start=_to_mi_point3(info.start),
            edge=_to_mi_vector3(info.edge),
            end=_to_mi_point3(info.end),
            length=_to_mi_float1(info.length),
            normal0=_to_mi_vector3(info.normal0),
            normal1=_to_mi_vector3(info.normal1),
            is_boundary=_to_mi_bool1(info.is_boundary),
            shape_id=_to_mi_int1(info.shape_id),
            local_edge_id=_to_mi_int1(info.local_edge_id),
            global_edge_id=_to_mi_int1(info.global_edge_id),
        )

    def edge_topology(self) -> EdgeTopologyBuffer:
        topology = self._scene.edge_topology()
        return EdgeTopologyBuffer(
            v0=_to_mi_int1(topology.v0),
            v1=_to_mi_int1(topology.v1),
            face0_local=_to_mi_int1(topology.face0_local),
            face1_local=_to_mi_int1(topology.face1_local),
            face0_global=_to_mi_int1(topology.face0_global),
            face1_global=_to_mi_int1(topology.face1_global),
            opposite_vertex0=_to_mi_int1(topology.opposite_vertex0),
            opposite_vertex1=_to_mi_int1(topology.opposite_vertex1),
        )

    def triangle_edge_indices(self, prim_id, global_: bool = True):
        prim_idx = _to_torch_scalar_array(prim_id, dtype=torch.int32, device=self._torch_device())
        edge0, edge1, edge2 = self._scene.triangle_edge_indices(prim_idx, global_=global_)
        return _to_mi_int1(edge0), _to_mi_int1(edge1), _to_mi_int1(edge2)

    def edge_adjacent_faces(self, edge_id, global_: bool = True):
        edge_idx = _to_torch_scalar_array(edge_id, dtype=torch.int32, device=self._torch_device())
        face0, face1 = self._scene.edge_adjacent_faces(edge_idx, global_=global_)
        return _to_mi_int1(face0), _to_mi_int1(face1)

    def mesh_face_offsets(self):
        return _to_mi_int1(self._scene.mesh_face_offsets())

    def mesh_edge_offsets(self):
        return _to_mi_int1(self._scene.mesh_edge_offsets())

    def shadow_test(self, ray, active=True):
        return self._scene.shadow_test(ray, active=active)

    def intersect(self, ray, active=True, flags=None):
        del flags
        return self._scene.intersect(ray, active=active)

    def _torch_device(self) -> torch.device:
        if self._device is None:
            offsets = _to_torch_scalar_array(self._scene.mesh_face_offsets(), dtype=torch.int32)
            self._device = offsets.device if offsets.numel() > 0 else torch.device("cpu")
        return self._device

