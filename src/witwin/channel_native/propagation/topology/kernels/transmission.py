"""Native component-5 topology packing for RayD segment penetration."""

from __future__ import annotations

import torch

from witwin.channel_native.propagation.models.capacity import CapacityExecutionCounts
from witwin.channel_native.propagation.models.penetration import (
    SegmentPenetrationResult,
)
from witwin.channel_native.propagation.models.transmission import (
    TransmissionTopologyCapacity,
)
from witwin.channel_native.runtime import torch_compat
from witwin.channel_native.runtime.autograd_contracts import (
    _ad_native_tangent_or_none,
    _ad_native_tensor,
)
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor
from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op


_BLOCK_FIELDS = (
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
)
_COUNT_FIELDS = ("device_candidate_count", "device_guardrail_count")
_OUTPUT_FIELDS = (*_BLOCK_FIELDS, *_COUNT_FIELDS)
_CONTINUOUS_OUTPUT_FIELDS = (
    "path_length_m",
    "delay_s",
    "interaction_position",
    "interaction_normal",
    "interaction_positions",
    "interaction_normals",
)


class _EnumeratedTransmissionTopologyPackFunction(torch.autograd.Function):
    """Native fixed-valid topology copy with native VJP/JVP companions."""

    @staticmethod
    def forward(*inputs):
        raw = _required_native_op("enumerated_transmission_topology_pack")(*inputs)
        if not isinstance(raw, dict) or set(raw) != set(_OUTPUT_FIELDS):
            raise TypeError("native transmission topology pack returned bad fields")
        return tuple(raw[name] for name in _OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        saved = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (output[0], inputs[1])
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        continuous = set(_CONTINUOUS_OUTPUT_FIELDS)
        ctx.mark_non_differentiable(
            *(
                value
                for name, value in zip(_OUTPUT_FIELDS, output, strict=True)
                if name not in continuous
            )
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        none_grads = (None,) * 13
        continuous_grads = tuple(
            grad_outputs[_OUTPUT_FIELDS.index(name)]
            for name in _CONTINUOUS_OUTPUT_FIELDS
        )
        if all(value is None for value in continuous_grads):
            return none_grads
        if not any(ctx.needs_input_grad[index] for index in (5, 6, 7)):
            return none_grads
        topology_valid, hit_valid = ctx.saved_tensors
        # Reductions commonly supply expanded stride-zero cotangents.  The
        # typed CUDA companion consumes contiguous row-major buffers, so
        # normalize only live cotangent slots at the dispatch boundary.
        continuous_grads = tuple(
            None if value is None else value.contiguous() for value in continuous_grads
        )
        raw = _required_native_op("enumerated_transmission_topology_pack_backward")(
            topology_valid, hit_valid, *continuous_grads
        )
        expected = {"grad_distance", "grad_position", "grad_normal"}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise TypeError("native transmission topology backward returned bad fields")
        return (
            None,
            None,
            None,
            None,
            None,
            raw["grad_distance"] if ctx.needs_input_grad[5] else None,
            raw["grad_position"] if ctx.needs_input_grad[6] else None,
            raw["grad_normal"] if ctx.needs_input_grad[7] else None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        native_tangents = tuple(
            _ad_native_tangent_or_none(tangents[index]) for index in (5, 6, 7)
        )
        if all(value is None for value in native_tangents):
            return (None,) * len(_OUTPUT_FIELDS)
        topology_valid, hit_valid = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with torch_compat.disable_functorch():
            raw = _required_native_op("enumerated_transmission_topology_pack_jvp")(
                topology_valid, hit_valid, *native_tangents
            )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_OUTPUT_FIELDS):
            raise TypeError("native transmission topology JVP returned bad fields")
        return tuple(raw.get(name) for name in _OUTPUT_FIELDS)


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

    values = _EnumeratedTransmissionTopologyPackFunction.apply(
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
    raw = dict(zip(_OUTPUT_FIELDS, values, strict=True))
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


__all__ = [
    "_EnumeratedTransmissionTopologyPackFunction",
    "enumerated_transmission_topology_pack",
]
