"""Native component-5 topology packing for RayD segment penetration."""

from __future__ import annotations

import torch

from witwin.channel_native.propagation.models.capacity import CapacityExecutionCounts
from witwin.channel_native.propagation.models.penetration import SegmentPenetrationResult
from witwin.channel_native.propagation.models.transmission import (
    TransmissionTopologyCapacity,
)
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor
from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op


_BLOCK_FIELDS = {
    "valid",
    "tx_id",
    "rx_id",
    "depth",
    "component_id",
    "primitive_id",
    "edge_id",
    "path_length_m",
    "delay_s",
    "path_gain",
    "path_field",
    "interaction_position",
    "interaction_normal",
    "material_id",
    "primitive_sequence",
    "material_sequence",
    "interaction_positions",
    "interaction_normals",
}
_COUNT_FIELDS = {"device_candidate_count", "device_guardrail_count"}


def _host_count(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def enumerated_transmission_topology_pack(
    penetration: SegmentPenetrationResult,
    face_material_id: torch.Tensor,
    geometry_mode_id: torch.Tensor,
    *,
    tx_count: int,
    rx_count: int,
) -> TransmissionTopologyCapacity:
    """Pack one inert row per endpoint pair without reading device counts."""

    if not isinstance(penetration, SegmentPenetrationResult):
        raise TypeError("penetration must be a SegmentPenetrationResult")
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "geometry_mode_id", geometry_mode_id, dtype=torch.int32, ndim=1
    )
    if face_material_id.device != penetration.device:
        raise ValueError("face_material_id must share the penetration device")
    if geometry_mode_id.device != penetration.device:
        raise ValueError("geometry_mode_id must share the penetration device")
    tx_count = _host_count("tx_count", tx_count)
    rx_count = _host_count("rx_count", rx_count)
    candidate_capacity = tx_count * rx_count
    if penetration.segment_count != candidate_capacity:
        raise ValueError("penetration rows must equal tx_count * rx_count")

    raw = _required_native_op("enumerated_transmission_topology_pack")(
        penetration.failure_state.bits,
        penetration.valid,
        penetration.num_hits,
        penetration.reached_target,
        penetration.overflow,
        penetration.distance,
        penetration.position,
        penetration.normal,
        penetration.global_primitive_id,
        face_material_id,
        geometry_mode_id,
        tx_count,
        rx_count,
    )
    if not isinstance(raw, dict) or set(raw) != _BLOCK_FIELDS | _COUNT_FIELDS:
        raise TypeError("native transmission topology pack returned bad fields")
    block = {name: raw[name] for name in _BLOCK_FIELDS}
    execution = CapacityExecutionCounts(
        candidate_capacity=candidate_capacity,
        failure_state=penetration.failure_state,
        device_candidate_count=raw["device_candidate_count"],
        device_guardrail_count=raw["device_guardrail_count"],
    )
    return TransmissionTopologyCapacity(
        candidate_capacity=candidate_capacity,
        sequence_width=penetration.hit_capacity,
        failure_state=penetration.failure_state,
        execution=execution,
        **block,
    )


__all__ = ["enumerated_transmission_topology_pack"]
