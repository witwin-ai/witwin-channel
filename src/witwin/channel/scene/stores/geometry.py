from __future__ import annotations

from dataclasses import dataclass

import torch

from ._validation import require_tensor


@dataclass(frozen=True, slots=True)
class GeometryStore:
    vertices: torch.Tensor
    faces: torch.Tensor
    face_normals: torch.Tensor
    edges: torch.Tensor
    edge_adj_faces: torch.Tensor
    edge_param_range: torch.Tensor
    face_structure_id: torch.Tensor
    face_surface_id: torch.Tensor
    structure_uv_presence: tuple[tuple[bool, bool], ...]
    version: int

    def __post_init__(self) -> None:
        require_tensor("vertices", self.vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,))
        require_tensor("faces", self.faces, dtype=torch.int32, ndim=2, trailing_shape=(3,))
        require_tensor(
            "face_normals", self.face_normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        require_tensor("edges", self.edges, dtype=torch.int32, ndim=2, trailing_shape=(2,))
        require_tensor(
            "edge_adj_faces", self.edge_adj_faces, dtype=torch.int32, ndim=2, trailing_shape=(2,)
        )
        require_tensor(
            "edge_param_range",
            self.edge_param_range,
            dtype=torch.float32,
            ndim=2,
            trailing_shape=(2,),
        )
        require_tensor("face_structure_id", self.face_structure_id, dtype=torch.int32, ndim=1)
        require_tensor("face_surface_id", self.face_surface_id, dtype=torch.int32, ndim=1)
        if self.faces.shape[0] != self.face_normals.shape[0]:
            raise ValueError("face_normals length must match faces")
        if self.faces.shape[0] != self.face_structure_id.shape[0]:
            raise ValueError("face_structure_id length must match faces")
        if self.faces.shape[0] != self.face_surface_id.shape[0]:
            raise ValueError("face_surface_id length must match faces")
        if self.edges.shape[0] != self.edge_adj_faces.shape[0]:
            raise ValueError("edge_adj_faces length must match edges")
        if self.edges.shape[0] != self.edge_param_range.shape[0]:
            raise ValueError("edge_param_range length must match edges")
        if any(
            type(uv_present) is not bool or type(face_uv_present) is not bool
            for uv_present, face_uv_present in self.structure_uv_presence
        ):
            raise ValueError("structure_uv_presence entries must contain bool values")


GeometryStore.__module__ = "witwin.channel.core.runtime.geometry"
