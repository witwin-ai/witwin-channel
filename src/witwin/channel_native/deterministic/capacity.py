"""Dormant deterministic fixed-capacity PathTable packing."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel_native.deterministic.result import PathTable
from witwin.channel_native.propagation.models.capacity import (
    CapacityEvaluatedPaths,
    CapacityPathLayout,
)
from witwin.channel_native.runtime import torch_compat
from witwin.channel_native.runtime.capacity import (
    CapacityFailureState,
    require_capacity_failure_state,
)
from witwin.channel_native.runtime.autograd_contracts import (
    _ad_native_tangent_or_none,
    _ad_native_tensor,
)
from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op


_TABLE_INPUT_FIELDS = (
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
)
_CONTINUOUS_INPUT_FIELDS = (
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
_NONDIFFERENTIABLE_OUTPUT_FIELDS = (
    "valid",
    "num_paths",
    "tx_id",
    "rx_id",
    "depth",
    "component_id",
    "primitive_id",
    "edge_id",
    "material_id",
    "primitive_sequence",
    "material_sequence",
    "interaction_count",
    "phase_rad",
)
_DIFFERENTIABLE_OUTPUT_FIELDS = (
    "path_length_m",
    "delay_s",
    "path_gain",
    "interaction_position",
    "interaction_normal",
    "interaction_positions",
    "interaction_normals",
    "field_real",
    "field_imag",
    "coefficient",
    "field_xyz",
    "field_direction",
)
_OUTPUT_FIELDS = (*_NONDIFFERENTIABLE_OUTPUT_FIELDS, *_DIFFERENTIABLE_OUTPUT_FIELDS)
_FUNCTION_INPUT_FIELDS = (
    "failure_state",
    *_TABLE_INPUT_FIELDS,
    *_CONTINUOUS_INPUT_FIELDS,
    "num_paths",
    "overflow",
    "include_fields",
    "pair_count",
    "path_capacity_per_pair",
)
_CONTINUOUS_INPUT_SLICE = slice(11, 22)


@dataclass(frozen=True, slots=True, eq=False)
class _CapacityPathTable:
    """Strictly internal dormant result; never used as ``Result.paths``."""

    table: PathTable
    layout: CapacityPathLayout

    def __post_init__(self) -> None:
        if not isinstance(self.table, PathTable):
            raise TypeError("table must be a PathTable")
        if not isinstance(self.layout, CapacityPathLayout):
            raise TypeError("layout must be a CapacityPathLayout")
        if self.table.valid is not self.layout.valid:
            raise ValueError("table must share layout validity")
        for name in PathTable.__dataclass_fields__:
            value = getattr(self.table, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"table.{name} must be a torch.Tensor")
            if value.device != self.layout.device:
                raise ValueError(f"table.{name} must share layout CUDA device")
            if value.shape[0] != self.layout.row_capacity:
                raise ValueError(f"table.{name} must match layout row capacity")
            if not value.is_contiguous():
                raise ValueError(f"table.{name} must be contiguous")

    @property
    def pair_count(self) -> int:
        return self.layout.pair_count

    @property
    def path_capacity_per_pair(self) -> int:
        return self.layout.path_capacity_per_pair

    @property
    def row_capacity(self) -> int:
        return self.layout.row_capacity

    @property
    def valid(self) -> torch.Tensor:
        return self.layout.valid

    @property
    def num_paths(self) -> torch.Tensor:
        return self.layout.num_paths


def _path_table_inputs(paths: CapacityEvaluatedPaths) -> tuple[torch.Tensor, ...]:
    topology = paths.evaluated.topology
    geometry = paths.evaluated.geometry
    fields = paths.evaluated.fields
    return (
        *(getattr(topology, name) for name in _TABLE_INPUT_FIELDS),
        *(getattr(geometry, name) for name in _CONTINUOUS_INPUT_FIELDS[:7]),
        *(getattr(fields, name) for name in _CONTINUOUS_INPUT_FIELDS[7:]),
    )


def _validate_capacity(paths: CapacityEvaluatedPaths) -> tuple[torch.Tensor, ...]:
    if not isinstance(paths, CapacityEvaluatedPaths):
        raise TypeError("paths must be a CapacityEvaluatedPaths")
    tensors = _path_table_inputs(paths)
    device = paths.selection.device
    for name, tensor in zip(
        (*_TABLE_INPUT_FIELDS, *_CONTINUOUS_INPUT_FIELDS), tensors, strict=True
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must share capacity CUDA device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    return tensors


class _DeterministicPathTableCapacityPackFunction(torch.autograd.Function):
    @staticmethod
    def forward(*inputs):
        if len(inputs) != len(_FUNCTION_INPUT_FIELDS):
            raise TypeError("deterministic PathTable capacity pack expects 27 inputs")
        failure_state = inputs[0]
        tensors = inputs[1:22]
        num_paths, overflow, include_fields, pair_count, capacity = inputs[22:]
        raw = _required_native_op("deterministic_path_table_capacity_pack")(
            failure_state,
            *tensors,
            num_paths,
            overflow,
            bool(include_fields),
            int(pair_count),
            int(capacity),
        )
        if not isinstance(raw, dict) or set(raw) != set(_OUTPUT_FIELDS):
            raise TypeError("native deterministic PathTable capacity pack returned bad fields")
        return tuple(raw[name] for name in _OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.include_fields = bool(inputs[24])
        ctx.sequence_width = int(inputs[9].shape[1])
        # Saved order is contractual: canonical output validity only. Failure bits
        # are a discrete forward input and never enter derivative companions.
        saved = (torch.autograd.forward_ad.unpack_dual(output[0]).primal,)
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.mark_non_differentiable(*output[: len(_NONDIFFERENTIABLE_OUTPUT_FIELDS)])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        none_grads = (None,) * 27
        continuous_grads = grad_outputs[len(_NONDIFFERENTIABLE_OUTPUT_FIELDS) :]
        if all(value is None for value in continuous_grads):
            return none_grads
        if not any(ctx.needs_input_grad[_CONTINUOUS_INPUT_SLICE]):
            return none_grads
        (valid,) = ctx.saved_tensors
        raw = _required_native_op("deterministic_path_table_capacity_pack_backward")(
            valid,
            ctx.include_fields,
            *continuous_grads,
            ctx.sequence_width,
        )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_INPUT_FIELDS):
            raise TypeError("native deterministic PathTable capacity backward returned bad fields")
        return (
            *(None for _ in range(11)),
            *(
                raw[name] if ctx.needs_input_grad[index] else None
                for index, name in enumerate(_CONTINUOUS_INPUT_FIELDS, start=11)
            ),
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        continuous_tangents = tuple(
            _ad_native_tangent_or_none(value)
            for value in tangents[_CONTINUOUS_INPUT_SLICE]
        )
        if all(value is None for value in continuous_tangents):
            return (None,) * len(_OUTPUT_FIELDS)
        (valid,) = (_ad_native_tensor(value) for value in ctx.saved_tensors)
        with torch_compat.disable_functorch():
            raw = _required_native_op("deterministic_path_table_capacity_pack_jvp")(
                valid,
                ctx.include_fields,
                *continuous_tangents,
                ctx.sequence_width,
            )
        if not isinstance(raw, dict) or set(raw) != set(_DIFFERENTIABLE_OUTPUT_FIELDS):
            raise TypeError("native deterministic PathTable capacity JVP returned bad fields")
        return (
            *(None for _ in _NONDIFFERENTIABLE_OUTPUT_FIELDS),
            *(raw[name] for name in _DIFFERENTIABLE_OUTPUT_FIELDS),
        )


def _contract_from_outputs(
    outputs: tuple[torch.Tensor, ...],
    *,
    failure_state: CapacityFailureState,
    overflow: torch.Tensor,
    pair_count: int,
    path_capacity_per_pair: int,
) -> _CapacityPathTable:
    raw = dict(zip(_OUTPUT_FIELDS, outputs, strict=True))
    table = PathTable(
        valid=raw["valid"],
        tx_id=raw["tx_id"],
        rx_id=raw["rx_id"],
        depth=raw["depth"],
        component_id=raw["component_id"],
        primitive_id=raw["primitive_id"],
        edge_id=raw["edge_id"],
        path_length_m=raw["path_length_m"],
        delay_s=raw["delay_s"],
        path_gain=raw["path_gain"],
        interaction_position=raw["interaction_position"],
        interaction_normal=raw["interaction_normal"],
        material_id=raw["material_id"],
        primitive_sequence=raw["primitive_sequence"],
        material_sequence=raw["material_sequence"],
        interaction_positions=raw["interaction_positions"],
        interaction_normals=raw["interaction_normals"],
        field_real=raw["field_real"],
        field_imag=raw["field_imag"],
        coefficient=raw["coefficient"],
        field_xyz=raw["field_xyz"],
        field_direction=raw["field_direction"],
        phase_rad=raw["phase_rad"],
        interaction_count=raw["interaction_count"],
    )
    return _CapacityPathTable(
        table=table,
        layout=CapacityPathLayout(
            pair_count=pair_count,
            path_capacity_per_pair=path_capacity_per_pair,
            failure_state=failure_state,
            valid=raw["valid"],
            num_paths=raw["num_paths"],
            overflow=overflow,
        ),
    )


def deterministic_path_table_capacity_pack(
    paths: CapacityEvaluatedPaths,
    *,
    include_fields: bool = True,
) -> _CapacityPathTable:
    """Build a dormant pair-major fixed-capacity PathTable on CUDA.

    ``phase_rad`` intentionally preserves the current export contract and is
    non-differentiable. The other eleven continuous evaluated-path inputs use
    registered native VJP/JVP companions; invalid and failed rows have zero AD.
    """

    if type(include_fields) is not bool:
        raise TypeError("include_fields must be a bool")
    tensors = _validate_capacity(paths)
    layout = paths.selection.layout
    failure_state = require_capacity_failure_state(
        layout.failure_state, device=layout.device
    )
    outputs = _DeterministicPathTableCapacityPackFunction.apply(
        failure_state.bits,
        *tensors,
        layout.num_paths,
        layout.overflow,
        include_fields,
        layout.pair_count,
        layout.path_capacity_per_pair,
    )
    return _contract_from_outputs(
        outputs,
        failure_state=failure_state,
        overflow=layout.overflow,
        pair_count=layout.pair_count,
        path_capacity_per_pair=layout.path_capacity_per_pair,
    )


__all__ = ["deterministic_path_table_capacity_pack"]
