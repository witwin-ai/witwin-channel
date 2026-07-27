"""Native failure sanitizers for complete evaluated enumerated path rows."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from witwin.channel.propagation.topology.export import (
        EvaluatedPathSidecars,
    )

from witwin.channel.propagation.models.evaluated import EvaluatedPaths
from witwin.channel.propagation.models.fields import PathFields
from witwin.channel.propagation.models.geometry import PathGeometry
from witwin.channel.propagation.models.topology import PathTopology
from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.autograd_contracts import (
    _ad_first_order_only,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
)
from witwin.channel.runtime.capacity import (
    CapacityFailureState,
    require_capacity_failure_state,
)
from witwin.channel.runtime.symbols import required_symbol as _required_native_op


_TOPOLOGY_INPUT_FIELDS = (
    "valid",
    "tx_id",
    "rx_id",
    "depth",
    "component_id",
    "primitive_id",
    "edge_id",
    "material_id",
    "primitive_sequence",
    "material_sequence",
    "interaction_type",
)
_CONTINUOUS_FIELDS = (
    "path_length_m",
    "delay_s",
    "field_direction",
    "interaction_position",
    "interaction_normal",
    "interaction_positions",
    "interaction_normals",
    "path_gain",
    "path_field",
    "field_xyz",
    "coefficient",
)
_SANITIZE_OUTPUT_FIELDS = (
    "selected_row_index",
    *_TOPOLOGY_INPUT_FIELDS,
    *_CONTINUOUS_FIELDS,
)
_SANITIZE_DISCRETE_OUTPUT_COUNT = 1 + len(_TOPOLOGY_INPUT_FIELDS)


def _evaluated_paths_capacity_pack_backward_native(*args: object) -> object:
    return _required_native_op("evaluated_paths_capacity_pack_backward")(*args)


def _evaluated_paths_capacity_pack_jvp_native(*args: object) -> object:
    return _required_native_op("evaluated_paths_capacity_pack_jvp")(*args)


def _enumerated_capacity_failure_vector_sanitize_native(
    failure_state_bits: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    output = _required_native_op("enumerated_capacity_failure_vector_sanitize")(
        failure_state_bits, values
    )
    if not isinstance(output, torch.Tensor):
        raise TypeError("native enumerated vector sanitizer returned a non-tensor")
    return output


def _candidate_tensors(paths: EvaluatedPaths) -> tuple[torch.Tensor, ...]:
    topology = paths.topology
    geometry = paths.geometry
    fields = paths.fields
    return (
        *(getattr(topology, name) for name in _TOPOLOGY_INPUT_FIELDS),
        geometry.path_length_m,
        geometry.delay_s,
        geometry.field_direction,
        geometry.interaction_position,
        geometry.interaction_normal,
        geometry.interaction_positions,
        geometry.interaction_normals,
        fields.path_gain,
        fields.path_field,
        fields.field_xyz,
        fields.coefficient,
    )


def _validate_candidate(paths: EvaluatedPaths) -> tuple[torch.Tensor, ...]:
    if not isinstance(paths, EvaluatedPaths):
        raise TypeError("paths must be EvaluatedPaths")
    tensors = _candidate_tensors(paths)
    device = tensors[0].device
    if device.type != "cuda":
        raise ValueError("evaluated path capacity packing requires CUDA tensors")
    for name, tensor in zip(
        (*_TOPOLOGY_INPUT_FIELDS, *_CONTINUOUS_FIELDS), tensors, strict=True
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must share evaluated path device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    return tensors


class _EnumeratedCapacityFailureSanitizeFunction(torch.autograd.Function):
    """Shape-preserving final failure sanitizer with native AD companions."""

    @staticmethod
    def forward(*inputs):
        raw = _required_native_op("enumerated_capacity_failure_sanitize")(
            inputs[22], *inputs[:22]
        )
        if not isinstance(raw, dict) or set(raw) != set(_SANITIZE_OUTPUT_FIELDS):
            raise TypeError("native enumerated failure sanitizer returned bad fields")
        return tuple(raw[name] for name in _SANITIZE_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.candidate_count = int(inputs[0].shape[0])
        ctx.sequence_width = int(inputs[8].shape[1])
        saved = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (output[1], output[0])
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.mark_non_differentiable(*output[:_SANITIZE_DISCRETE_OUTPUT_COUNT])

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *grad_outputs):
        none_grads = (None,) * 23
        continuous_grads = grad_outputs[_SANITIZE_DISCRETE_OUTPUT_COUNT:]
        if all(value is None for value in continuous_grads):
            return none_grads
        if not any(ctx.needs_input_grad[11:22]):
            return none_grads
        valid, selected_row_index = ctx.saved_tensors
        raw = _evaluated_paths_capacity_pack_backward_native(
            valid,
            selected_row_index,
            *continuous_grads,
            ctx.candidate_count,
            ctx.sequence_width,
        )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_FIELDS):
            raise TypeError(
                "native enumerated failure sanitizer backward returned bad fields"
            )
        return (
            *(None for _ in range(11)),
            *(
                raw[name] if ctx.needs_input_grad[index] else None
                for index, name in enumerate(_CONTINUOUS_FIELDS, start=11)
            ),
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        continuous_tangents = tuple(
            _ad_native_tangent_or_none(value) for value in tangents[11:22]
        )
        if all(value is None for value in continuous_tangents):
            return (None,) * len(_SANITIZE_OUTPUT_FIELDS)
        valid, selected_row_index = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with torch_compat.disable_functorch():
            raw = _evaluated_paths_capacity_pack_jvp_native(
                valid,
                selected_row_index,
                *continuous_tangents,
                ctx.candidate_count,
                ctx.sequence_width,
            )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_FIELDS):
            raise TypeError(
                "native enumerated failure sanitizer JVP returned bad fields"
            )
        return (
            *(None for _ in range(_SANITIZE_DISCRETE_OUTPUT_COUNT)),
            *(raw[name] for name in _CONTINUOUS_FIELDS),
        )


def enumerated_capacity_failure_sanitize(
    paths: EvaluatedPaths,
    *,
    failure_state: CapacityFailureState,
) -> EvaluatedPaths:
    """Make every final enumerated row inert after any transaction failure."""

    tensors = _validate_candidate(paths)
    require_capacity_failure_state(failure_state, device=tensors[0].device)
    outputs = _EnumeratedCapacityFailureSanitizeFunction.apply(
        *tensors, failure_state.bits
    )
    raw = dict(zip(_SANITIZE_OUTPUT_FIELDS, outputs, strict=True))
    topology = PathTopology(
        valid=raw["valid"],
        tx_id=raw["tx_id"],
        rx_id=raw["rx_id"],
        depth=raw["depth"],
        component_id=raw["component_id"],
        primitive_id=raw["primitive_id"],
        edge_id=raw["edge_id"],
        material_id=raw["material_id"],
        primitive_sequence=raw["primitive_sequence"],
        material_sequence=raw["material_sequence"],
        interaction_type=raw["interaction_type"],
    )
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        path_length_m=raw["path_length_m"],
        delay_s=raw["delay_s"],
        field_direction=raw["field_direction"],
        interaction_position=raw["interaction_position"],
        interaction_normal=raw["interaction_normal"],
        interaction_positions=raw["interaction_positions"],
        interaction_normals=raw["interaction_normals"],
    )
    fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=raw["path_gain"],
        path_field=raw["path_field"],
        field_xyz=raw["field_xyz"],
        coefficient=raw["coefficient"],
    )
    return EvaluatedPaths(topology=topology, geometry=geometry, fields=fields)


class _EnumeratedCapacityFailureVectorSanitizeFunction(torch.autograd.Function):
    """Failure-aware complex-vector identity with native VJP/JVP copies."""

    @staticmethod
    def forward(failure_state_bits, values):
        return _enumerated_capacity_failure_vector_sanitize_native(
            failure_state_bits, values
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        del output
        ctx.set_materialize_grads(False)
        failure_state_bits = torch.autograd.forward_ad.unpack_dual(inputs[0]).primal
        ctx.save_for_backward(failure_state_bits)
        ctx.save_for_forward(failure_state_bits)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_output):
        if grad_output is None or not ctx.needs_input_grad[1]:
            return None, None
        (failure_state_bits,) = ctx.saved_tensors
        grad_values = _enumerated_capacity_failure_vector_sanitize_native(
            failure_state_bits, grad_output
        )
        return None, grad_values

    @staticmethod
    def jvp(ctx, _failure_tangent, values_tangent):
        values_tangent = _ad_native_tangent_or_none(values_tangent)
        if values_tangent is None:
            return None
        (failure_state_bits,) = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with torch_compat.disable_functorch():
            return _enumerated_capacity_failure_vector_sanitize_native(
                failure_state_bits, values_tangent
            )


def enumerated_capacity_failure_vector_sanitize(
    values: torch.Tensor,
    *,
    failure_state: CapacityFailureState,
) -> torch.Tensor:
    """Sanitize the deterministic diffraction vector sidecar on failure."""

    if not isinstance(values, torch.Tensor):
        raise TypeError("diffraction_vector_field must be a torch.Tensor")
    if not values.is_cuda or values.dtype != torch.complex64:
        raise ValueError("diffraction_vector_field must be a CUDA complex64 tensor")
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("diffraction_vector_field must have shape (T, R, 3)")
    require_capacity_failure_state(failure_state, device=values.device)
    return _EnumeratedCapacityFailureVectorSanitizeFunction.apply(
        failure_state.bits, values
    )


def sanitize_enumerated_capacity_transaction(
    paths: EvaluatedPaths,
    sidecars: EvaluatedPathSidecars,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    """Sanitize all enumerated payloads before outer solver result assembly."""

    from witwin.channel.propagation.topology.export import (
        EvaluatedPathSidecars,
    )

    if not isinstance(sidecars, EvaluatedPathSidecars):
        raise TypeError("sidecars must be EvaluatedPathSidecars")
    transaction = sidecars.capacity_transaction
    if transaction is None:
        return paths, sidecars
    sanitized = enumerated_capacity_failure_sanitize(
        paths, failure_state=transaction.failure_state
    )
    vector_field = sidecars.diffraction_vector_field
    if vector_field is not None:
        vector_field = enumerated_capacity_failure_vector_sanitize(
            vector_field, failure_state=transaction.failure_state
        )
    return sanitized, replace(sidecars, diffraction_vector_field=vector_field)


__all__ = [
    "_EnumeratedCapacityFailureSanitizeFunction",
    "_EnumeratedCapacityFailureVectorSanitizeFunction",
    "enumerated_capacity_failure_sanitize",
    "enumerated_capacity_failure_vector_sanitize",
    "sanitize_enumerated_capacity_transaction",
]
