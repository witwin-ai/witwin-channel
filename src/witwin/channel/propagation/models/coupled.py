"""Fixed-capacity discrete candidates for coupled path discovery."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel.runtime.capacity import (
    CapacityFailureState,
    require_capacity_failure_state,
)

from .capacity import _require_cuda_tensor, _require_host_count


@dataclass(frozen=True, slots=True, eq=False)
class CoupledCandidateCapacity:
    """Stable coupled candidate axes with device-resident status."""

    candidate_capacity: int
    failure_state: CapacityFailureState
    valid: torch.Tensor
    candidate_count: torch.Tensor
    overflow: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    component_id: torch.Tensor
    face_id: torch.Tensor
    edge1_id: torch.Tensor
    edge2_id: torch.Tensor

    def __post_init__(self) -> None:
        capacity = _require_host_count("candidate_capacity", self.candidate_capacity)
        valid = _require_cuda_tensor(
            "valid", self.valid, dtype=torch.bool, shape=(capacity,)
        )
        require_capacity_failure_state(self.failure_state, device=valid.device)
        _require_cuda_tensor(
            "candidate_count",
            self.candidate_count,
            dtype=torch.int32,
            shape=(1,),
            device=valid.device,
        )
        _require_cuda_tensor(
            "overflow",
            self.overflow,
            dtype=torch.bool,
            shape=(1,),
            device=valid.device,
        )
        for name in (
            "tx_id",
            "rx_id",
            "component_id",
            "face_id",
            "edge1_id",
            "edge2_id",
        ):
            _require_cuda_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=(capacity,),
                device=valid.device,
            )

    @property
    def device(self) -> torch.device:
        return self.valid.device


__all__ = ["CoupledCandidateCapacity"]
