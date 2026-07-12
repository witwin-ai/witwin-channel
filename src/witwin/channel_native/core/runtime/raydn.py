from __future__ import annotations

from dataclasses import dataclass, field

import torch

from witwin.channel_native.core.kernels.ops import mc_pack_vec3, raydn_scene_create, raydn_scene_edge_records


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

    handle: int | None = None
    owner: object | None = None
    mesh_tensors: tuple[tuple[torch.Tensor, ...], ...] = ()
    reason: str | None = None
    runtime_cache: dict[str, object] = field(default_factory=dict, compare=False, repr=False)

    @property
    def available(self) -> bool:
        return self.handle is not None

    def require_handle(self) -> int:
        if self.handle is None:
            reason = "unknown" if self.reason is None else self.reason
            raise RuntimeError(f"RayDN native scene is unavailable: {reason}")
        return self.handle

    def edge_records(self) -> RayDNEdgeRecords:
        cached = self.runtime_cache.get("edge_records")
        if cached is not None:
            return cached  # type: ignore[return-value]
        values = raydn_scene_edge_records(self.require_handle())
        if len(values) != 12:
            raise RuntimeError(f"RayDN edge_records returned {len(values)} tensors, expected 12")
        face_normals = mc_pack_vec3(values[2], values[3], values[4])
        records = RayDNEdgeRecords(
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

    This function uses the RayD native core source-linked into
    `_channel_native`. It does not import a RayD Python package or dispatcher.
    """

    if not structures:
        return RayDNScene(reason="scene has no structures")
    if not torch.cuda.is_available():
        return RayDNScene(reason="CUDA is unavailable")

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
        # Structures that carry a UV parametrization forward it to the native
        # mesh (RayD carries UV end-to-end); structures without UV keep the
        # empty per-mesh tensors, preserving the pre-UV behavior exactly.
        structure_uv = getattr(structure, "uv", None)
        structure_face_uv = getattr(structure, "face_uv", None)
        if structure_uv is not None and structure_face_uv is not None:
            mesh_uv = structure_uv.to(device=device, dtype=torch.float32).contiguous()
            mesh_face_uv = structure_face_uv.to(device=device, dtype=torch.int32).contiguous()
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

    handle, owner = raydn_scene_create(
        vertices,
        faces,
        uv,
        face_uv,
        to_world_left,
        to_world_right,
        mesh_flags,
    )
    return RayDNScene(handle=handle, owner=owner, mesh_tensors=tuple(keepalive))
