"""Native component-5 topology packing for RayD segment penetration.

This facade owns both the dispatch and the fixed-capacity row table it
publishes: ``TransmissionTopologyCapacity`` is the named typed contract this
one native operation converts its result into, and it has no other producer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel.propagation.penetration import (
    SegmentPenetrationResult,
)
from witwin.channel.runtime import (
    CapacityExecutionCounts,
    CapacityFailureState,
    _ad_first_order_only,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    disable_functorch,
    require_capacity_failure_state,
    require_host_count,
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)
from witwin.channel.tensor_math import require_tensor


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
    @_ad_first_order_only
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
        with disable_functorch():
            raw = _required_native_op("enumerated_transmission_topology_pack_jvp")(
                topology_valid, hit_valid, *native_tangents
            )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_OUTPUT_FIELDS):
            raise TypeError("native transmission topology JVP returned bad fields")
        return tuple(raw.get(name) for name in _OUTPUT_FIELDS)


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
        capacity = require_host_count("candidate_capacity", self.candidate_capacity)
        width = require_host_count("sequence_width", self.sequence_width)
        valid = require_tensor(
            "valid",
            self.valid,
            dtype=torch.bool,
            shape=(capacity,),
            cuda=True,
            contiguous=True,
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
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=(capacity,),
                device=valid.device,
                cuda=True,
                contiguous=True,
            )
        for name in ("path_length_m", "delay_s", "path_gain"):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(capacity,),
                device=valid.device,
                cuda=True,
                contiguous=True,
            )
        require_tensor(
            "path_field",
            self.path_field,
            dtype=torch.complex64,
            shape=(capacity,),
            device=valid.device,
            cuda=True,
            contiguous=True,
        )
        for name in ("interaction_position", "interaction_normal"):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(capacity, 3),
                device=valid.device,
                cuda=True,
                contiguous=True,
            )
        for name in ("primitive_sequence", "material_sequence"):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=(capacity, width),
                device=valid.device,
                cuda=True,
                contiguous=True,
            )
        for name in ("interaction_positions", "interaction_normals"):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(capacity, width, 3),
                device=valid.device,
                cuda=True,
                contiguous=True,
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
    tx_count = require_host_count("tx_count", tx_count)
    rx_count = require_host_count("rx_count", rx_count)
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
    "TransmissionTopologyCapacity",
    "_EnumeratedTransmissionTopologyPackFunction",
    "enumerated_transmission_topology_pack",
]
