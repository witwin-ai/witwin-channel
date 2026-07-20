"""Native fixed-capacity coupled candidate generation."""

from __future__ import annotations

import torch

from witwin.channel_native.propagation.models.coupled import (
    CoupledCandidateCapacity,
)
from witwin.channel_native.runtime.capacity import (
    CapacityFailureState,
    require_capacity_failure_state,
)
from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor


_MAX_COUPLED_CANDIDATES = 1_000_000
_OUTPUT_FIELDS = {
    "valid",
    "candidate_count",
    "overflow",
    "tx_id",
    "rx_id",
    "component_id",
    "face_id",
    "edge1_id",
    "edge2_id",
}


def _host_int(name: str, value: object, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _theoretical_candidate_count(
    *, tx_count: int, rx_count: int, group_count: int, edge_count: int
) -> int:
    pair_count = tx_count * rx_count
    rd_per_pair = group_count * edge_count
    dd_per_pair = edge_count * (edge_count - 1)
    return pair_count * (2 * rd_per_pair + dd_per_pair)


def coupled_candidate_capacity_block(
    representative_faces: torch.Tensor,
    selected_edges: torch.Tensor,
    *,
    failure_state: CapacityFailureState,
    tx_count: int,
    rx_count: int,
    rx_id_offset: int,
    candidate_capacity: int,
    candidate_limit: int,
) -> CoupledCandidateCapacity:
    """Generate dormant coupled candidate axes without selecting result rows."""

    validate_cuda_tensor(
        "representative_faces", representative_faces, dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor("selected_edges", selected_edges, dtype=torch.int32, ndim=1)
    if selected_edges.device != representative_faces.device:
        raise ValueError("selected_edges must share representative_faces device")
    tx_count = _host_int("tx_count", tx_count)
    rx_count = _host_int("rx_count", rx_count)
    rx_id_offset = _host_int("rx_id_offset", rx_id_offset)
    candidate_capacity = _host_int("candidate_capacity", candidate_capacity)
    candidate_limit = _host_int("candidate_limit", candidate_limit, positive=True)
    effective_limit = min(candidate_limit, _MAX_COUPLED_CANDIDATES)
    theoretical = _theoretical_candidate_count(
        tx_count=tx_count,
        rx_count=rx_count,
        group_count=int(representative_faces.shape[0]),
        edge_count=int(selected_edges.shape[0]),
    )
    if theoretical > effective_limit:
        raise RuntimeError(
            "coupled reflection-diffraction topology requires "
            f"{theoretical} candidates, exceeding "
            f"coupled_candidate_limit={effective_limit}"
        )
    if candidate_capacity > effective_limit:
        raise ValueError("candidate_capacity exceeds coupled candidate guardrail")
    require_capacity_failure_state(
        failure_state, device=representative_faces.device
    )

    raw = _required_native_op("coupled_candidate_capacity_block")(
        failure_state.bits,
        representative_faces,
        selected_edges,
        tx_count,
        rx_count,
        rx_id_offset,
        candidate_capacity,
        candidate_limit,
    )
    if not isinstance(raw, dict) or set(raw) != _OUTPUT_FIELDS:
        raise TypeError("native coupled candidate capacity returned bad fields")
    return CoupledCandidateCapacity(
        candidate_capacity=candidate_capacity,
        failure_state=failure_state,
        **raw,
    )


__all__ = ["coupled_candidate_capacity_block"]
