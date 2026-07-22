"""Fixed-capacity straight-transmission topology contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel.runtime.capacity import (
    CapacityFailureState,
    require_capacity_failure_state,
)

from .capacity import (
    CapacityExecutionCounts,
    _require_cuda_tensor,
    _require_host_count,
)


@dataclass(frozen=True, slots=True, eq=False)
class TransmissionTopologyCapacity:
    """Pair-major component-5 rows with CUDA-resident actual counts."""

    candidate_capacity: int
    sequence_width: int
    failure_state: CapacityFailureState
    execution: CapacityExecutionCounts
    valid: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    depth: torch.Tensor
    component_id: torch.Tensor
    primitive_id: torch.Tensor
    edge_id: torch.Tensor
    path_length_m: torch.Tensor
    delay_s: torch.Tensor
    path_gain: torch.Tensor
    path_field: torch.Tensor
    interaction_position: torch.Tensor
    interaction_normal: torch.Tensor
    material_id: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor

    def __post_init__(self) -> None:
        capacity = _require_host_count("candidate_capacity", self.candidate_capacity)
        width = _require_host_count("sequence_width", self.sequence_width)
        valid = _require_cuda_tensor(
            "valid", self.valid, dtype=torch.bool, shape=(capacity,)
        )
        require_capacity_failure_state(self.failure_state, device=valid.device)
        if not isinstance(self.execution, CapacityExecutionCounts):
            raise TypeError("execution must be CapacityExecutionCounts")
        if self.execution.candidate_capacity != capacity:
            raise ValueError("execution capacity must match candidate_capacity")
        if self.execution.failure_state is not self.failure_state:
            raise ValueError("execution must retain the exact failure_state")
        for name in (
            "tx_id",
            "rx_id",
            "depth",
            "component_id",
            "primitive_id",
            "edge_id",
            "material_id",
        ):
            _require_cuda_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=(capacity,),
                device=valid.device,
            )
        for name in ("path_length_m", "delay_s", "path_gain"):
            _require_cuda_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(capacity,),
                device=valid.device,
            )
        _require_cuda_tensor(
            "path_field",
            self.path_field,
            dtype=torch.complex64,
            shape=(capacity,),
            device=valid.device,
        )
        for name in ("interaction_position", "interaction_normal"):
            _require_cuda_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(capacity, 3),
                device=valid.device,
            )
        for name in ("primitive_sequence", "material_sequence"):
            _require_cuda_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=(capacity, width),
                device=valid.device,
            )
        for name in ("interaction_positions", "interaction_normals"):
            _require_cuda_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(capacity, width, 3),
                device=valid.device,
            )

    @property
    def device(self) -> torch.device:
        return self.valid.device

    def as_block(self) -> dict[str, torch.Tensor]:
        """Return the topology block without copying or reordering tensors."""

        return {
            "valid": self.valid,
            "tx_id": self.tx_id,
            "rx_id": self.rx_id,
            "depth": self.depth,
            "component_id": self.component_id,
            "primitive_id": self.primitive_id,
            "edge_id": self.edge_id,
            "path_length_m": self.path_length_m,
            "delay_s": self.delay_s,
            "path_gain": self.path_gain,
            "path_field": self.path_field,
            "interaction_position": self.interaction_position,
            "interaction_normal": self.interaction_normal,
            "material_id": self.material_id,
            "primitive_sequence": self.primitive_sequence,
            "material_sequence": self.material_sequence,
            "interaction_positions": self.interaction_positions,
            "interaction_normals": self.interaction_normals,
        }


__all__ = ["TransmissionTopologyCapacity"]
