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


def require_capacity_failure_state(
    state: object, *, device: torch.device
) -> CapacityFailureState:
    """Validate a required typed state without reading its device value."""

    if not isinstance(state, CapacityFailureState):
        raise TypeError("failure_state must be a CapacityFailureState")
    if state.device != device:
        raise ValueError("failure_state must share the input device")
    return state


__all__ = [
    "CapacityFailureBit",
    "CapacityFailureState",
    "create_capacity_failure_state",
]
