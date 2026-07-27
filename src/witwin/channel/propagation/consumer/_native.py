"""Internal native finalization for compact consumer path rows."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel.propagation.models.evaluated import EvaluatedPaths
from witwin.channel.propagation.models.fields import PathFields
from witwin.channel.propagation.models.geometry import PathGeometry
from witwin.channel.propagation.models.topology import PathTopology
from witwin.channel.propagation.topology.kernels import (
    evaluated_paths_compact_finalize_backward,
    evaluated_paths_compact_finalize_jvp,
)
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
_STRUCTURAL_FIELDS = (
    "selected_row_index",
    "pair_index",
    "pair_offsets",
    "source_id",
    "sink_id",
)
_OUTPUT_FIELDS = (*_STRUCTURAL_FIELDS, *_TOPOLOGY_FIELDS, *_CONTINUOUS_FIELDS)
_DISCRETE_OUTPUT_COUNT = len(_STRUCTURAL_FIELDS) + len(_TOPOLOGY_FIELDS)
COMPACT_COUNT_D2H_COPIES = 1
COMPACT_COUNT_D2H_BYTES = 8
COMPACT_COUNT_SYNCHRONIZATIONS = 1


@dataclass(frozen=True, slots=True)
class CompactEvaluatedPaths:
    path_count: int
    pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    source_id: torch.Tensor
    sink_id: torch.Tensor
    evaluated: EvaluatedPaths
    count_d2h_copies: int
    count_d2h_bytes: int
    count_synchronizations: int
    native_launch_count: int


@dataclass(frozen=True, slots=True)
class LoSJonesRows:
    matrix: torch.Tensor
    source_basis: torch.Tensor
    sink_basis: torch.Tensor
    native_launch_count: int


def _candidate_tensors(paths: EvaluatedPaths) -> tuple[torch.Tensor, ...]:
    return (
        *(getattr(paths.topology, name) for name in _TOPOLOGY_FIELDS),
        *(getattr(paths.geometry, name) for name in _CONTINUOUS_FIELDS[:7]),
        *(getattr(paths.fields, name) for name in _CONTINUOUS_FIELDS[7:]),
    )


def _validate_inputs(
    paths: EvaluatedPaths,
    source_stable_ids: torch.Tensor,
    sink_stable_ids: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(paths, EvaluatedPaths):
        raise TypeError("paths must be EvaluatedPaths")
    tensors = _candidate_tensors(paths)
    device = tensors[0].device
    if device.type != "cuda":
        raise ValueError("compact evaluated-path finalization requires CUDA")
    for name, tensor in zip(
        (*_TOPOLOGY_FIELDS, *_CONTINUOUS_FIELDS), tensors, strict=True
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must share evaluated path device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    for name, lookup in (
        ("source_stable_ids", source_stable_ids),
        ("sink_stable_ids", sink_stable_ids),
    ):
        if not isinstance(lookup, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (
            lookup.device != device
            or lookup.dtype != torch.int64
            or lookup.ndim != 1
            or not lookup.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be a contiguous CUDA int64 vector on {device}"
            )
    return tensors


def evaluated_paths_compact_finalize(
    *inputs: torch.Tensor, rows_are_compact: bool
) -> dict[str, object]:
    return _required_native_op("evaluated_paths_compact_finalize")(
        *inputs, rows_are_compact
    )


class _CompactEvaluatedPathsFunction(torch.autograd.Function):
    @staticmethod
    def forward(*inputs):
        raw = evaluated_paths_compact_finalize(
            *inputs, rows_are_compact=False
        )
        expected = {"path_count", *_OUTPUT_FIELDS}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise TypeError("native compact finalizer returned bad fields")
        if type(raw["path_count"]) is not int:
            raise TypeError("native compact finalizer returned a non-int path_count")
        outputs = tuple(raw[name] for name in _OUTPUT_FIELDS)
        if raw["path_count"] != outputs[1].shape[0]:
            raise RuntimeError("native compact finalizer returned inconsistent K")
        return outputs

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.candidate_count = int(inputs[0].shape[0])
        ctx.sequence_width = int(inputs[8].shape[1])
        saved = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (output[5], output[0])
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.mark_non_differentiable(*output[:_DISCRETE_OUTPUT_COUNT])

    @staticmethod
    @_ad_first_order_only
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        none_grads = (None,) * 24
        continuous_grads = grad_outputs[_DISCRETE_OUTPUT_COUNT:]
        if all(value is None for value in continuous_grads):
            return none_grads
        if not any(ctx.needs_input_grad[11:22]):
            return none_grads
        valid, selected_row_index = ctx.saved_tensors
        raw = evaluated_paths_compact_finalize_backward(
            valid,
            selected_row_index,
            *continuous_grads,
            candidate_count=ctx.candidate_count,
            sequence_width=ctx.sequence_width,
        )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_FIELDS):
            raise TypeError("native compact finalizer backward returned bad fields")
        return (
            *(None for _ in range(11)),
            *(
                raw[name] if ctx.needs_input_grad[index] else None
                for index, name in enumerate(_CONTINUOUS_FIELDS, start=11)
            ),
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        continuous_tangents = tuple(
            _ad_native_tangent_or_none(value) for value in tangents[11:22]
        )
        if all(value is None for value in continuous_tangents):
            return (None,) * len(_OUTPUT_FIELDS)
        valid, selected_row_index = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with torch_compat.disable_functorch():
            raw = evaluated_paths_compact_finalize_jvp(
                valid,
                selected_row_index,
                *continuous_tangents,
                candidate_count=ctx.candidate_count,
                sequence_width=ctx.sequence_width,
            )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_FIELDS):
            raise TypeError("native compact finalizer JVP returned bad fields")
        return (
            *(None for _ in range(_DISCRETE_OUTPUT_COUNT)),
            *(raw[name] for name in _CONTINUOUS_FIELDS),
        )


def compact_evaluated_paths(
    paths: EvaluatedPaths,
    *,
    source_stable_ids: torch.Tensor,
    sink_stable_ids: torch.Tensor,
    rows_are_compact: bool = False,
) -> CompactEvaluatedPaths:
    """Publish exact valid rows and pair segmentation from the sole native owner."""

    tensors = _validate_inputs(paths, source_stable_ids, sink_stable_ids)
    if rows_are_compact:
        native = evaluated_paths_compact_finalize(
            *tensors,
            source_stable_ids,
            sink_stable_ids,
            rows_are_compact=True,
        )
        expected = {"path_count", *_OUTPUT_FIELDS}
        if not isinstance(native, dict) or set(native) != expected:
            raise TypeError("native exact-row finalizer returned bad fields")
        if native["path_count"] != paths.row_count:
            raise RuntimeError("native exact-row finalizer changed K")
        for name, tensor in zip(
            (*_TOPOLOGY_FIELDS, *_CONTINUOUS_FIELDS), tensors, strict=True
        ):
            alias = native[name]
            if (
                alias.data_ptr() != tensor.data_ptr()
                or alias.shape != tensor.shape
                or alias.stride() != tensor.stride()
            ):
                raise RuntimeError(
                    f"native exact-row finalizer copied payload field {name}"
                )
        raw = native
        evaluated = paths
    else:
        outputs = _CompactEvaluatedPathsFunction.apply(
            *tensors, source_stable_ids, sink_stable_ids
        )
        raw = dict(zip(_OUTPUT_FIELDS, outputs, strict=True))
        topology = PathTopology(
            **{name: raw[name] for name in _TOPOLOGY_FIELDS}
        )
        geometry = PathGeometry(
            row_identity=topology.row_identity,
            **{name: raw[name] for name in _CONTINUOUS_FIELDS[:7]},
        )
        fields = PathFields(
            row_identity=topology.row_identity,
            **{name: raw[name] for name in _CONTINUOUS_FIELDS[7:]},
        )
        evaluated = EvaluatedPaths(
            topology=topology,
            geometry=geometry,
            fields=fields,
        )
    return CompactEvaluatedPaths(
        path_count=int(raw["pair_index"].shape[0]),
        pair_index=raw["pair_index"],
        pair_offsets=raw["pair_offsets"],
        source_id=raw["source_id"],
        sink_id=raw["sink_id"],
        evaluated=evaluated,
        count_d2h_copies=(
            COMPACT_COUNT_D2H_COPIES
            if paths.row_count > 0 and not rows_are_compact
            else 0
        ),
        count_d2h_bytes=(
            COMPACT_COUNT_D2H_BYTES
            if paths.row_count > 0 and not rows_are_compact
            else 0
        ),
        count_synchronizations=(
            COMPACT_COUNT_SYNCHRONIZATIONS
            if paths.row_count > 0 and not rows_are_compact
            else 0
        ),
        native_launch_count=1,
    )


def consumer_los_jones(
    *,
    pair_index: torch.Tensor,
    source_positions: torch.Tensor,
    sink_positions: torch.Tensor,
    source_reference_basis: torch.Tensor,
    sink_reference_basis: torch.Tensor,
    frequency_hz: float,
) -> LoSJonesRows:
    """Evaluate primal-only LoS transport between row-specific transverse bases."""

    tensors = (
        pair_index,
        source_positions,
        sink_positions,
        source_reference_basis,
        sink_reference_basis,
    )
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("LoS Jones inputs must be torch tensors")
    if any(value.requires_grad for value in tensors):
        raise RuntimeError("LoS Jones transport does not support AD")
    raw = _required_native_op("consumer_los_jones")(
        pair_index,
        source_positions,
        sink_positions,
        source_reference_basis,
        sink_reference_basis,
        float(frequency_hz),
    )
    if not isinstance(raw, dict) or set(raw) != {
        "matrix",
        "source_basis",
        "sink_basis",
    }:
        raise TypeError("native LoS Jones operator returned bad fields")
    return LoSJonesRows(
        matrix=raw["matrix"],
        source_basis=raw["source_basis"],
        sink_basis=raw["sink_basis"],
        native_launch_count=1 if pair_index.shape[0] > 0 else 0,
    )


__all__ = [
    "COMPACT_COUNT_D2H_BYTES",
    "COMPACT_COUNT_D2H_COPIES",
    "COMPACT_COUNT_SYNCHRONIZATIONS",
    "CompactEvaluatedPaths",
    "LoSJonesRows",
    "compact_evaluated_paths",
    "consumer_los_jones",
]
