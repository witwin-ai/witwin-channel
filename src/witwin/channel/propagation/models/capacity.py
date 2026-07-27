"""Device-resident capacity execution-count contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel.runtime.capacity import (
    CapacityFailureState,
    require_capacity_failure_state,
)


def _require_host_count(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_cuda_tensor(
    name: str,
    value: object,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if value.dtype != dtype:
        raise ValueError(f"{name} must use {dtype}, got {value.dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    return value

@dataclass(frozen=True, slots=True, eq=False)
class CapacityExecutionCounts:
    """Host capacity plus CUDA-resident actual diagnostic counts.

    Public metadata may expose ``candidate_capacity``. The actual candidate
    and guardrail counts remain device sidecars so result assembly never hides
    a device-to-host synchronization behind metadata construction.
    """

    candidate_capacity: int
    failure_state: CapacityFailureState
    device_candidate_count: torch.Tensor
    device_guardrail_count: torch.Tensor

    def __post_init__(self) -> None:
        _require_host_count("candidate_capacity", self.candidate_capacity)
        candidate_count = _require_cuda_tensor(
            "device_candidate_count",
            self.device_candidate_count,
            dtype=torch.int32,
            shape=(1,),
        )
        require_capacity_failure_state(
            self.failure_state, device=candidate_count.device
        )
        _require_cuda_tensor(
            "device_guardrail_count",
            self.device_guardrail_count,
            dtype=torch.int32,
            shape=(1,),
            device=candidate_count.device,
        )

    @property
    def device(self) -> torch.device:
        return self.device_candidate_count.device


__all__ = ["CapacityExecutionCounts"]
