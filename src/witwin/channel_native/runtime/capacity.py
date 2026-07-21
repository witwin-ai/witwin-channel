"""Device-resident shared failure state for capacity transactions."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import torch

from .tensor_contracts import validate_cuda_tensor


class CapacityFailureBit(enum.IntFlag):
    """Stable device failure bits recorded by ADR-029 intermediate owners."""

    DIFFRACTION_STATE_OVERFLOW = 1 << 0
    DIFFRACTION_PATH_OVERFLOW = 1 << 1
    DIFFRACTION_PATH_CONTRACT_ERROR = 1 << 2
    PAIR_CAPACITY_OVERFLOW = 1 << 3
    PAIR_CONTRACT_ERROR = 1 << 4
    COUPLED_CANDIDATE_OVERFLOW = 1 << 5
    REFLECTION_CANDIDATE_OVERFLOW = 1 << 6


@dataclass(frozen=True, slots=True, eq=False)
class CapacityFailureState:
    """One solve-owned CUDA ``int32[1]`` failure bitmask.

    Construction is metadata-only. Intermediate native operations atomically
    accumulate bits on the active CUDA stream and never read them on the host.
    """

    bits: torch.Tensor

    def __post_init__(self) -> None:
        validate_cuda_tensor("bits", self.bits, dtype=torch.int32, ndim=1)
        if self.bits.shape != (1,):
            raise ValueError("bits must have shape (1,)")

    @property
    def device(self) -> torch.device:
        return self.bits.device


@dataclass(slots=True, eq=False)
class SolveCapacityTransaction:
    """Solve-scoped owner of one failure state and one terminal observation.

    The transaction is orchestration state only. It never reads the CUDA
    bitmask. Solvers pass ``failure_state`` unchanged to every capacity
    intermediate and call ``terminal_check`` once after result sanitization.
    """

    failure_state: CapacityFailureState
    _terminal_enqueued: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.failure_state, CapacityFailureState):
            raise TypeError("failure_state must be a CapacityFailureState")

    @property
    def device(self) -> torch.device:
        return self.failure_state.device

    @property
    def terminal_enqueued(self) -> bool:
        return self._terminal_enqueued

    def terminal_check(self) -> None:
        """Enqueue the runtime terminal observer exactly once."""

        if self._terminal_enqueued:
            raise RuntimeError("capacity transaction terminal check already enqueued")
        capacity_failure_terminal_check(self.failure_state)
        self._terminal_enqueued = True


def create_capacity_failure_state(reference: torch.Tensor) -> CapacityFailureState:
    """Create a native-zeroed failure state on ``reference``'s CUDA device."""

    from .symbols import required_symbol

    if not isinstance(reference, torch.Tensor):
        raise TypeError("reference must be a torch.Tensor")
    if not reference.is_cuda:
        raise ValueError("reference must be a CUDA tensor")
    bits = required_symbol("capacity_failure_state_create")(reference)
    if not isinstance(bits, torch.Tensor):
        raise TypeError("native capacity failure state must be a tensor")
    return CapacityFailureState(bits=bits)


def create_solve_capacity_transaction(
    reference: torch.Tensor,
) -> SolveCapacityTransaction:
    """Create the one ADR-029 capacity transaction owned by a solve."""

    return SolveCapacityTransaction(
        failure_state=create_capacity_failure_state(reference)
    )


def require_capacity_failure_state(
    state: object, *, device: torch.device
) -> CapacityFailureState:
    """Validate a required typed state without reading its device value."""

    if not isinstance(state, CapacityFailureState):
        raise TypeError("failure_state must be a CapacityFailureState")
    if state.device != device:
        raise ValueError("failure_state must share the input device")
    return state


def capacity_failure_terminal_check(failure_state: CapacityFailureState) -> None:
    """Enqueue the one terminal failure observation for a capacity solve."""

    if not isinstance(failure_state, CapacityFailureState):
        raise TypeError("failure_state must be a CapacityFailureState")
    from .symbols import required_symbol

    required_symbol("capacity_failure_terminal_check")(failure_state.bits)


__all__ = [
    "CapacityFailureBit",
    "CapacityFailureState",
    "SolveCapacityTransaction",
    "capacity_failure_terminal_check",
    "create_capacity_failure_state",
    "create_solve_capacity_transaction",
]
