# Copyright Xingyu Chen.
# Typed fixed-capacity contracts for RayD segment penetration.

"""Typed fixed-capacity contracts for RayD segment penetration."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import torch

from witwin.channel.runtime import (
    CapacityFailureState,
    require_capacity_failure_state,
    require_host_count,
    require_tensor,
)


class SegmentPenetrationPolicy(enum.IntEnum):
    """Explicit RayD straight-segment geometry policy."""

    EnumeratedFullDistance = 0
    MonteCarloTargetInset = 1


@dataclass(frozen=True, slots=True, eq=False)
class SegmentPenetrationResult:
    """Resident ``[segment, hit_capacity]`` penetration result."""

    hit_capacity: int
    failure_state: CapacityFailureState
    valid: torch.Tensor
    num_hits: torch.Tensor
    reached_target: torch.Tensor
    overflow: torch.Tensor
    distance: torch.Tensor
    direction: torch.Tensor
    t: torch.Tensor
    position: torch.Tensor
    normal: torch.Tensor
    geometric_normal: torch.Tensor
    global_primitive_id: torch.Tensor

    def __post_init__(self) -> None:
        capacity = require_host_count("hit_capacity", self.hit_capacity)
        if not isinstance(self.valid, torch.Tensor) or self.valid.ndim != 2:
            raise ValueError("valid must have shape (N, hit_capacity)")
        rows = int(self.valid.shape[0])
        valid = require_tensor(
            "valid",
            self.valid,
            dtype=torch.bool,
            shape=(rows, capacity),
            cuda=True,
            contiguous=True,
        )
        require_capacity_failure_state(self.failure_state, device=valid.device)
        for name, dtype in (
            ("num_hits", torch.int32),
            ("reached_target", torch.bool),
            ("overflow", torch.bool),
            ("distance", torch.float32),
        ):
            require_tensor(
                name,
                getattr(self, name),
                dtype=dtype,
                shape=(rows,),
                device=valid.device,
                cuda=True,
                contiguous=True,
            )
        require_tensor(
            "direction",
            self.direction,
            dtype=torch.float32,
            shape=(rows, 3),
            device=valid.device,
            cuda=True,
            contiguous=True,
        )
        for name, dtype in (
            ("t", torch.float32),
            ("global_primitive_id", torch.int32),
        ):
            require_tensor(
                name,
                getattr(self, name),
                dtype=dtype,
                shape=(rows, capacity),
                device=valid.device,
                cuda=True,
                contiguous=True,
            )
        for name in ("position", "normal", "geometric_normal"):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(rows, capacity, 3),
                device=valid.device,
                cuda=True,
                contiguous=True,
            )

    @property
    def segment_count(self) -> int:
        return int(self.valid.shape[0])

    @property
    def device(self) -> torch.device:
        return self.valid.device


@dataclass(frozen=True, slots=True, eq=False)
class SegmentPenetrationTapeResult:
    """Primal result plus opaque RayD fixed-winner derivative tape."""

    result: SegmentPenetrationResult
    tape_primitive_id: torch.Tensor
    tape_barycentric: torch.Tensor
    tape_restart_epsilon: torch.Tensor
    tape_restart_branch: torch.Tensor
    tape_restart_tie_mask: torch.Tensor
    tape_direction_denominator_branch: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.result, SegmentPenetrationResult):
            raise TypeError("result must be a SegmentPenetrationResult")
        rows = self.result.segment_count
        capacity = self.result.hit_capacity
        device = self.result.device
        require_tensor(
            "tape_primitive_id",
            self.tape_primitive_id,
            dtype=torch.int32,
            shape=(rows, capacity),
            device=device,
            cuda=True,
            contiguous=True,
        )
        require_tensor(
            "tape_barycentric",
            self.tape_barycentric,
            dtype=torch.float32,
            shape=(rows, capacity, 2),
            device=device,
            cuda=True,
            contiguous=True,
        )
        require_tensor(
            "tape_restart_epsilon",
            self.tape_restart_epsilon,
            dtype=torch.float32,
            shape=(rows, capacity),
            device=device,
            cuda=True,
            contiguous=True,
        )
        for name in ("tape_restart_branch", "tape_restart_tie_mask"):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.uint8,
                shape=(rows, capacity),
                device=device,
                cuda=True,
                contiguous=True,
            )
        require_tensor(
            "tape_direction_denominator_branch",
            self.tape_direction_denominator_branch,
            dtype=torch.bool,
            shape=(rows,),
            device=device,
            cuda=True,
            contiguous=True,
        )

    @property
    def failure_state(self) -> CapacityFailureState:
        return self.result.failure_state


@dataclass(frozen=True, slots=True, eq=False)
class SegmentPenetrationBackwardResult:
    """Named native cotangents for the three continuous inputs."""

    grad_vertices: torch.Tensor | None
    grad_origins: torch.Tensor | None
    grad_targets: torch.Tensor | None


@dataclass(frozen=True, slots=True, eq=False)
class SegmentPenetrationJvpResult:
    """Named native tangents for the six continuous outputs."""

    tangent_distance: torch.Tensor
    tangent_direction: torch.Tensor
    tangent_t: torch.Tensor
    tangent_position: torch.Tensor
    tangent_normal: torch.Tensor
    tangent_geometric_normal: torch.Tensor


__all__ = [
    "SegmentPenetrationBackwardResult",
    "SegmentPenetrationJvpResult",
    "SegmentPenetrationPolicy",
    "SegmentPenetrationResult",
    "SegmentPenetrationTapeResult",
]