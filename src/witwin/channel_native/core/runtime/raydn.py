from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel_native.core.kernels import raydn_backend


@dataclass(frozen=True, slots=True)
class RayDNEdgeRecords:
    vertices: torch.Tensor
    faces: torch.Tensor
    face_normals: torch.Tensor
    edge_v0: torch.Tensor
    edge_v1: torch.Tensor
    face0: torch.Tensor
    face1: torch.Tensor
    shape_id: torch.Tensor
    local_edge_id: torch.Tensor
    opposite: torch.Tensor


@dataclass(frozen=True, slots=True)
class RayDNScene:
    """Opaque wrapper for a native RayDN scene/cache handle."""

    handle: object | None = None
    mesh_tensors: tuple[tuple[torch.Tensor, ...], ...] = ()
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.handle is not None

    def require_handle(self) -> object:
        if self.handle is None:
            reason = "unknown" if self.reason is None else self.reason
            raise RuntimeError(f"RayDN native scene is unavailable: {reason}")
        return self.handle

    def edge_records(self) -> RayDNEdgeRecords:
        values = self.require_handle().edge_records()
        if len(values) != 12:
            raise RuntimeError(f"RayDN edge_records returned {len(values)} tensors, expected 12")
        face_normals = torch.stack((values[2], values[3], values[4]), dim=1).contiguous()
        return RayDNEdgeRecords(
            vertices=values[0],
            faces=values[1],
            face_normals=face_normals,
            edge_v0=values[5],
            edge_v1=values[6],
            face0=values[7],
            face1=values[8],
            shape_id=values[9],
            local_edge_id=values[10],
            opposite=values[11],
        )


def _empty_tensor(shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def _mesh_flags(*, use_face_normals: bool, edges_enabled: bool, dynamic: bool) -> int:
    flags = 0
    if use_face_normals:
        flags |= 1
    if edges_enabled:
        flags |= 2
    if dynamic:
        flags |= 4
    return flags


def build_scene_from_structures(structures: tuple[object, ...]) -> RayDNScene:
    """Build a RayDN native scene from Channel Native structures.

    This function uses only the `_raydn` native extension and Torch custom class
    registration. It intentionally does not import the Python `raydn` package.
    """

    if not structures:
        return RayDNScene(reason="scene has no structures")
    if not torch.cuda.is_available():
        return RayDNScene(reason="CUDA is unavailable")

    raydn_backend.require_native_extension()
    device = torch.device("cuda")
    vertices: list[torch.Tensor] = []
    faces: list[torch.Tensor] = []
    uv: list[torch.Tensor] = []
    face_uv: list[torch.Tensor] = []
    to_world_left: list[torch.Tensor] = []
    to_world_right: list[torch.Tensor] = []
    mesh_flags: list[int] = []
    keepalive: list[tuple[torch.Tensor, ...]] = []

    for structure in structures:
        mesh_vertices = structure.vertices.to(device=device, dtype=torch.float32).contiguous()
        mesh_faces = structure.faces.to(device=device, dtype=torch.int32).contiguous()
        mesh_uv = _empty_tensor((0, 2), dtype=torch.float32, device=device)
        mesh_face_uv = _empty_tensor((0, 3), dtype=torch.int32, device=device)
        mesh_to_world_left = _empty_tensor((0, 4), dtype=torch.float32, device=device)
        mesh_to_world_right = _empty_tensor((0, 4), dtype=torch.float32, device=device)
        vertices.append(mesh_vertices)
        faces.append(mesh_faces)
        uv.append(mesh_uv)
        face_uv.append(mesh_face_uv)
        to_world_left.append(mesh_to_world_left)
        to_world_right.append(mesh_to_world_right)
        mesh_flags.append(_mesh_flags(use_face_normals=False, edges_enabled=True, dynamic=False))
        keepalive.append(
            (
                mesh_vertices,
                mesh_faces,
                mesh_uv,
                mesh_face_uv,
                mesh_to_world_left,
                mesh_to_world_right,
            )
        )

    with torch._C._DisableFuncTorch():
        handle = torch.classes.raydn.Scene(
            vertices,
            faces,
            uv,
            face_uv,
            to_world_left,
            to_world_right,
            mesh_flags,
        )
    return RayDNScene(handle=handle, mesh_tensors=tuple(keepalive))
