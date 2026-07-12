from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..materials import PhaseScreen
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
    # ABI v3: per-structure phase-screen bindings from SurfaceAssignment.
    structure_phase_screens: dict[int, PhaseScreen] = field(default_factory=dict)

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
        num_structures = self.structure_material_id.shape[0]
        for index, screen in self.structure_phase_screens.items():
            if not isinstance(index, int) or not 0 <= index < num_structures:
                raise ValueError(
                    "structure_phase_screens keys must be structure indices in "
                    f"[0, {num_structures})"
                )
            if not isinstance(screen, PhaseScreen):
                raise ValueError("structure_phase_screens values must be PhaseScreen")
