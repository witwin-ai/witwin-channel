from __future__ import annotations

from dataclasses import dataclass

import torch

from ._validation import require_tensor


@dataclass(frozen=True, slots=True)
class AssignmentStore:
    face_material_id: torch.Tensor
    edge_material_id0: torch.Tensor
    edge_material_id1: torch.Tensor
    surface_material_id: torch.Tensor
    structure_material_id: torch.Tensor
    num_faces: int
    num_edges: int
    version: int

    def __post_init__(self) -> None:
        require_tensor("face_material_id", self.face_material_id, dtype=torch.int32, ndim=1)
        require_tensor("edge_material_id0", self.edge_material_id0, dtype=torch.int32, ndim=1)
        require_tensor("edge_material_id1", self.edge_material_id1, dtype=torch.int32, ndim=1)
        require_tensor("surface_material_id", self.surface_material_id, dtype=torch.int32, ndim=1)
        require_tensor(
            "structure_material_id", self.structure_material_id, dtype=torch.int32, ndim=1
        )
        if self.face_material_id.shape[0] != self.num_faces:
            raise ValueError("face_material_id length must match num_faces")
        if self.edge_material_id0.shape[0] != self.num_edges:
            raise ValueError("edge_material_id0 length must match num_edges")
        if self.edge_material_id1.shape[0] != self.num_edges:
            raise ValueError("edge_material_id1 length must match num_edges")
