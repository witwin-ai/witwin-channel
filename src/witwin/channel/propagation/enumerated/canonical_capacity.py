"""Dormant native gather from canonical selection to fixed-capacity rows."""

from __future__ import annotations

import torch

from witwin.channel.propagation.models.capacity import (
    CanonicalEvaluatedPaths,
    CanonicalPathSelection,
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
from witwin.channel.runtime.symbols import required_symbol as _required_native_op


_TOPOLOGY_FIELDS = (
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
_SELECTION_FIELDS = (
    "selected_row_index",
    "valid",
    "num_selected",
    "num_paths",
)
_OUTPUT_FIELDS = (
    *_SELECTION_FIELDS,
    *_TOPOLOGY_FIELDS[1:],
    *_CONTINUOUS_FIELDS,
)
_DISCRETE_OUTPUT_COUNT = len(_SELECTION_FIELDS) + len(_TOPOLOGY_FIELDS) - 1


def _candidate_tensors(paths: EvaluatedPaths) -> tuple[torch.Tensor, ...]:
    topology = paths.topology
    geometry = paths.geometry
    fields = paths.fields
    return (
        *(getattr(topology, name) for name in _TOPOLOGY_FIELDS),
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


def _validate_inputs(
    paths: EvaluatedPaths,
    selection: CanonicalPathSelection,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(paths, EvaluatedPaths):
        raise TypeError("paths must be EvaluatedPaths")
    if not isinstance(selection, CanonicalPathSelection):
        raise TypeError("selection must be CanonicalPathSelection")
    if paths.row_count != selection.candidate_capacity:
        raise ValueError("paths must match selection candidate capacity")
    tensors = _candidate_tensors(paths)
    device = tensors[0].device
    if device.type != "cuda":
        raise ValueError("canonical evaluated-path gather requires CUDA tensors")
    if selection.device != device:
        raise ValueError("selection and paths must share a CUDA device")
    for name, tensor in zip(
        (*_TOPOLOGY_FIELDS, *_CONTINUOUS_FIELDS), tensors, strict=True
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must share evaluated path device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    return tensors


class _EvaluatedPathsCanonicalCapacityGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(*inputs):
        failure_state_bits = inputs[0]
        selection_tensors = inputs[1:5]
        candidate_tensors = inputs[5:27]
        num_tx, num_rx = inputs[27:29]
        raw = _required_native_op("evaluated_paths_canonical_capacity_gather")(
            failure_state_bits,
            *selection_tensors,
            *candidate_tensors,
            int(num_tx),
            int(num_rx),
        )
        if not isinstance(raw, dict) or set(raw) != set(_OUTPUT_FIELDS):
            raise TypeError(
                "native canonical evaluated-path gather returned bad fields"
            )
        return tuple(raw[name] for name in _OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.candidate_count = int(inputs[5].shape[0])
        ctx.sequence_width = int(inputs[13].shape[1])
        saved = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (inputs[0], output[1], output[0])
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.mark_non_differentiable(*output[:_DISCRETE_OUTPUT_COUNT])

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *grad_outputs):
        none_grads = (None,) * 29
        continuous_grads = grad_outputs[_DISCRETE_OUTPUT_COUNT:]
        if all(value is None for value in continuous_grads):
            return none_grads
        if not any(ctx.needs_input_grad[16:27]):
            return none_grads
        failure_state_bits, valid, selected_row_index = ctx.saved_tensors
        raw = _required_native_op(
            "evaluated_paths_canonical_capacity_gather_backward"
        )(
            failure_state_bits,
            valid,
            selected_row_index,
            *continuous_grads,
            ctx.candidate_count,
            ctx.sequence_width,
        )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_FIELDS):
            raise TypeError(
                "native canonical evaluated-path backward returned bad fields"
            )
        return (
            *(None for _ in range(16)),
            *(
                raw[name] if ctx.needs_input_grad[index] else None
                for index, name in enumerate(_CONTINUOUS_FIELDS, start=16)
            ),
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        continuous_tangents = tuple(
            _ad_native_tangent_or_none(value) for value in tangents[16:27]
        )
        if all(value is None for value in continuous_tangents):
            return (None,) * len(_OUTPUT_FIELDS)
        failure_state_bits, valid, selected_row_index = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with torch_compat.disable_functorch():
            raw = _required_native_op(
                "evaluated_paths_canonical_capacity_gather_jvp"
            )(
                failure_state_bits,
                valid,
                selected_row_index,
                *continuous_tangents,
                ctx.candidate_count,
                ctx.sequence_width,
            )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_FIELDS):
            raise TypeError("native canonical evaluated-path JVP returned bad fields")
        return (
            *(None for _ in range(_DISCRETE_OUTPUT_COUNT)),
            *(raw[name] for name in _CONTINUOUS_FIELDS),
        )


def _contract_from_outputs(
    outputs: tuple[torch.Tensor, ...],
    *,
    source_selection: CanonicalPathSelection,
) -> CanonicalEvaluatedPaths:
    raw = dict(zip(_OUTPUT_FIELDS, outputs, strict=True))
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
    selection = CanonicalPathSelection(
        candidate_capacity=source_selection.candidate_capacity,
        pair_count=source_selection.pair_count,
        num_tx=source_selection.num_tx,
        num_rx=source_selection.num_rx,
        failure_state=source_selection.failure_state,
        selected_row_index=raw["selected_row_index"],
        valid=topology.valid,
        num_selected=raw["num_selected"],
        num_paths=raw["num_paths"],
    )
    return CanonicalEvaluatedPaths(
        selection=selection,
        evaluated=EvaluatedPaths(topology=topology, geometry=geometry, fields=fields),
    )


def evaluated_paths_canonical_capacity_gather(
    paths: EvaluatedPaths,
    *,
    selection: CanonicalPathSelection,
) -> CanonicalEvaluatedPaths:
    """Gather canonical selected rows without reading device cardinality."""

    candidate_tensors = _validate_inputs(paths, selection)
    outputs = _EvaluatedPathsCanonicalCapacityGatherFunction.apply(
        selection.failure_state.bits,
        selection.selected_row_index,
        selection.valid,
        selection.num_selected,
        selection.num_paths,
        *candidate_tensors,
        selection.num_tx,
        selection.num_rx,
    )
    return _contract_from_outputs(outputs, source_selection=selection)


__all__ = ["evaluated_paths_canonical_capacity_gather"]
