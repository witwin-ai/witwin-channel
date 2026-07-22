from __future__ import annotations

from dataclasses import dataclass, field

import torch

from witwin.channel.runtime.native_buffers import mc_pack_vec3
from witwin.channel.runtime.symbols import required_symbol as _required_native_op


def rayd_scene_create(
    vertices: list[torch.Tensor],
    faces: list[torch.Tensor],
    uv: list[torch.Tensor],
    face_uv: list[torch.Tensor],
    to_world_left: list[torch.Tensor],
    to_world_right: list[torch.Tensor],
    mesh_flags: list[int],
) -> object:
    resource = _required_native_op("rayd_scene_create")(
        vertices,
        faces,
        uv,
        face_uv,
        to_world_left,
        to_world_right,
        mesh_flags,
    )
    if resource is None or not bool(getattr(resource, "available", False)):
        raise RuntimeError("_channel_native.rayd_scene_create returned an invalid resource")
    return resource


def rayd_scene_edge_records(resource: object) -> tuple[torch.Tensor, ...]:
    out = _required_native_op("rayd_scene_edge_records")(resource)
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel_native.rayd_scene_edge_records must return a tensor sequence"
        )
    return tuple(out)


@dataclass(frozen=True, slots=True)
class RayDEdgeRecords:
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
class RayDSceneResource:
    """Typed, owning wrapper for a native RayD scene resource."""

    resource: object | None = None
    mesh_tensors: tuple[tuple[torch.Tensor, ...], ...] = ()
    reason: str | None = None
    runtime_cache: dict[str, object] = field(
        default_factory=dict, compare=False, repr=False
    )

    @property
    def available(self) -> bool:
        return self.resource is not None and bool(
            getattr(self.resource, "available", False)
        )

    def require_resource(self) -> object:
        if not self.available:
            reason = "unknown" if self.reason is None else self.reason
            raise RuntimeError(f"RayD native scene is unavailable: {reason}")
        assert self.resource is not None
        return self.resource

    def edge_records(self) -> RayDEdgeRecords:
        cached = self.runtime_cache.get("edge_records")
        if cached is not None:
            return cached  # type: ignore[return-value]
        values = rayd_scene_edge_records(self.require_resource())
        if len(values) != 12:
            raise RuntimeError(
                f"RayD edge_records returned {len(values)} tensors, expected 12"
            )
        face_normals = mc_pack_vec3(values[2], values[3], values[4])
        records = RayDEdgeRecords(
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
        self.runtime_cache["edge_records"] = records
        return records


def _empty_tensor(
    shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
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


def build_scene_from_structures(structures: tuple[object, ...]) -> RayDSceneResource:
    """Build a typed RayD native scene from Channel Native structures.

    This function uses the RayD native core source-linked into
    `_channel_native`. It does not import a RayD Python package or dispatcher.
    """

    if not structures:
        return RayDSceneResource(reason="scene has no structures")
    if not torch.cuda.is_available():
        return RayDSceneResource(reason="CUDA is unavailable")

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
        mesh_vertices = structure.vertices.to(
            device=device, dtype=torch.float32
        ).contiguous()
        mesh_faces = structure.faces.to(device=device, dtype=torch.int32).contiguous()
        # Structures that carry a UV parametrization forward it to the native
        # mesh (RayD carries UV end-to-end); structures without UV keep the
        # empty per-mesh tensors, preserving the pre-UV behavior exactly.
        structure_uv = getattr(structure, "uv", None)
        structure_face_uv = getattr(structure, "face_uv", None)
        if structure_uv is not None and structure_face_uv is not None:
            mesh_uv = structure_uv.to(device=device, dtype=torch.float32).contiguous()
            mesh_face_uv = structure_face_uv.to(
                device=device, dtype=torch.int32
            ).contiguous()
        else:
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
        mesh_flags.append(
            _mesh_flags(use_face_normals=False, edges_enabled=True, dynamic=False)
        )
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

    resource = rayd_scene_create(
        vertices,
        faces,
        uv,
        face_uv,
        to_world_left,
        to_world_right,
        mesh_flags,
    )
    return RayDSceneResource(resource=resource, mesh_tensors=tuple(keepalive))


__all__ = [
    "RayDEdgeRecords",
    "RayDSceneResource",
    "_empty_tensor",
    "_mesh_flags",
    "build_scene_from_structures",
    "rayd_scene_create",
    "rayd_scene_edge_records",
]
