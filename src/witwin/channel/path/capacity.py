"""Dormant ADR-029 native fixed-capacity :class:`PathResult` packing."""

from __future__ import annotations

from typing import Any

import torch

from witwin.channel.propagation.models.capacity import CapacityEvaluatedPaths
from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.autograd_contracts import (
    _ad_first_order_only,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
)
from witwin.channel.runtime.capacity import require_capacity_failure_state
from witwin.channel.runtime.symbols import required_symbol as _required_native_op

from .result import PathResult


_OUTPUT_FIELDS = (
    "a",
    "tau",
    "theta_t",
    "phi_t",
    "theta_r",
    "phi_r",
    "valid",
    "interaction_type",
    "primitive_id",
    "material_id",
    "position",
    "normal",
    "num_paths",
    "field_xyz",
    "field_direction",
)
_BACKWARD_FIELDS = (
    "delay_s",
    "field_direction",
    "interaction_positions",
    "interaction_normals",
    "coefficient",
    "field_xyz",
    "tx_positions",
    "rx_positions",
)
_TANGENT_OUTPUT_FIELDS = (
    "a",
    "tau",
    "theta_t",
    "phi_t",
    "theta_r",
    "phi_r",
    "position",
    "normal",
    "field_xyz",
    "field_direction",
)
_INPUT_COUNT = 23


class _PathResultCapacityPackFunction(torch.autograd.Function):
    @staticmethod
    def forward(*inputs):
        raw = _required_native_op("path_result_capacity_pack")(*inputs)
        if not isinstance(raw, dict) or set(raw) != set(_OUTPUT_FIELDS):
            raise TypeError("native path result capacity pack returned bad fields")
        return tuple(raw[name] for name in _OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.num_tx = int(inputs[20])
        ctx.num_rx = int(inputs[21])
        ctx.capacity = int(inputs[22])
        saved_values = (
            output[6],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[11],
            inputs[14],
            inputs[15],
            inputs[18],
            inputs[19],
        )
        saved = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in saved_values
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.mark_non_differentiable(
            output[6], output[7], output[8], output[9], output[12]
        )

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *grad_outputs):
        if not any(ctx.needs_input_grad[12:20]):
            return (None,) * _INPUT_COUNT
        continuous_output_grads = (
            grad_outputs[0],
            grad_outputs[1],
            grad_outputs[2],
            grad_outputs[3],
            grad_outputs[4],
            grad_outputs[5],
            grad_outputs[10],
            grad_outputs[11],
            grad_outputs[13],
            grad_outputs[14],
        )
        if all(value is None for value in continuous_output_grads):
            return (None,) * _INPUT_COUNT
        saved = tuple(_ad_native_tensor(value) for value in ctx.saved_tensors)
        raw = _required_native_op("path_result_capacity_pack_backward")(
            *saved,
            *continuous_output_grads,
            ctx.num_tx,
            ctx.num_rx,
            ctx.capacity,
        )
        if not isinstance(raw, dict) or set(raw) != set(_BACKWARD_FIELDS):
            raise TypeError("native path result capacity backward returned bad fields")
        return (
            *(None for _ in range(12)),
            *(
                raw[name] if ctx.needs_input_grad[index] else None
                for index, name in enumerate(_BACKWARD_FIELDS, start=12)
            ),
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        continuous_tangents = tuple(
            _ad_native_tangent_or_none(value) for value in tangents[12:20]
        )
        if all(value is None for value in continuous_tangents):
            return (None,) * len(_OUTPUT_FIELDS)
        saved = tuple(_ad_native_tensor(value) for value in ctx.saved_tensors)
        with torch_compat.disable_functorch():
            raw = _required_native_op("path_result_capacity_pack_jvp")(
                *saved,
                *continuous_tangents,
                ctx.num_tx,
                ctx.num_rx,
                ctx.capacity,
            )
        if not isinstance(raw, dict) or set(raw) != set(_TANGENT_OUTPUT_FIELDS):
            raise TypeError("native path result capacity JVP returned bad fields")
        return (
            raw["a"],
            raw["tau"],
            raw["theta_t"],
            raw["phi_t"],
            raw["theta_r"],
            raw["phi_r"],
            None,
            None,
            None,
            None,
            raw["position"],
            raw["normal"],
            None,
            raw["field_xyz"],
            raw["field_direction"],
        )


def _require_host_count(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_endpoint_positions(
    name: str,
    value: object,
    *,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must use torch.float32")
    if tuple(value.shape) != (count, 3):
        raise ValueError(f"{name} must have shape {(count, 3)}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}")
    return value


def from_capacity_evaluated_paths(
    paths: CapacityEvaluatedPaths,
    *,
    num_rx: int,
    num_tx: int,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> PathResult:
    """Pack pair-major evaluated capacity rows without host compaction.

    This entry is dormant until the ADR-029 atomic solver switch. The shared
    failure state is inherited from the capacity layout and is never exposed
    by the public result.
    """

    if not isinstance(paths, CapacityEvaluatedPaths):
        raise TypeError("paths must be a CapacityEvaluatedPaths")
    num_rx = _require_host_count("num_rx", num_rx)
    num_tx = _require_host_count("num_tx", num_tx)
    layout = paths.selection.layout
    if layout.pair_count != num_rx * num_tx:
        raise ValueError("capacity pair_count must equal num_rx * num_tx")
    topology = paths.evaluated.topology
    geometry = paths.evaluated.geometry
    fields = paths.evaluated.fields
    device = topology.device
    failure_state = require_capacity_failure_state(
        layout.failure_state, device=device
    )
    tx_positions = _require_endpoint_positions(
        "tx_positions", tx_positions, count=num_tx, device=device
    )
    rx_positions = _require_endpoint_positions(
        "rx_positions", rx_positions, count=num_rx, device=device
    )
    outputs = _PathResultCapacityPackFunction.apply(
        failure_state.bits,
        layout.overflow,
        topology.valid,
        layout.num_paths,
        topology.tx_id,
        topology.rx_id,
        topology.depth,
        topology.component_id,
        topology.edge_id,
        topology.primitive_sequence,
        topology.material_sequence,
        topology.interaction_type,
        geometry.delay_s,
        geometry.field_direction,
        geometry.interaction_positions,
        geometry.interaction_normals,
        fields.coefficient,
        fields.field_xyz,
        tx_positions,
        rx_positions,
        num_tx,
        num_rx,
        layout.path_capacity_per_pair,
    )
    raw = dict(zip(_OUTPUT_FIELDS, outputs, strict=True))
    capacity = layout.path_capacity_per_pair
    width = topology.sequence_width
    path_shape = (num_rx, 1, num_tx, 1, capacity)
    depth_shape = (*path_shape, width)
    result_metadata = dict(metadata or {})
    result_metadata.update(
        {
            "schema": "PathResult",
            "schema_version": 1,
            "max_paths_scope": "per_pair",
            "path_capacity_per_pair": capacity,
            "stable_order": "receiver_major_pair_then_selected_input_order",
            "coefficient_semantics": (
                "unit_excitation_dimensionless_receiver_projection"
            ),
            "coupled_coefficient_semantics": "unified_complex3_jones",
            "interaction_geometry": "canonical_topology",
        }
    )
    return PathResult(
        a=raw["a"].reshape(*path_shape, 1),
        tau=raw["tau"].reshape(path_shape),
        theta_t=raw["theta_t"].reshape(path_shape),
        phi_t=raw["phi_t"].reshape(path_shape),
        theta_r=raw["theta_r"].reshape(path_shape),
        phi_r=raw["phi_r"].reshape(path_shape),
        valid=raw["valid"].reshape(path_shape),
        interaction_type=raw["interaction_type"].reshape(depth_shape),
        primitive_id=raw["primitive_id"].reshape(depth_shape),
        material_id=raw["material_id"].reshape(depth_shape),
        position=raw["position"].reshape(*depth_shape, 3),
        normal=raw["normal"].reshape(*depth_shape, 3),
        num_paths=raw["num_paths"].reshape(num_rx, 1, num_tx, 1),
        metadata=result_metadata,
        field_xyz=raw["field_xyz"].reshape(*path_shape, 3),
        field_direction=raw["field_direction"].reshape(*path_shape, 3),
    )


__all__ = ["from_capacity_evaluated_paths"]
