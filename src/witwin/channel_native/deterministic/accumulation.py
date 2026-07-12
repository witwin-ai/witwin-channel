from __future__ import annotations

from collections.abc import Mapping

import torch

from witwin.channel_native.core.kernels import ops

from .result import PathTable
from witwin.channel_native.core.path_topology import (
    ReceiverLayout,
    TopologyBatch,
    apply_receiver_layout,
)


# Canonical component_id -> name identity. 3=reflection->diffraction and
# 4=diffraction->reflection are path-solver coupled classes that the
# deterministic accumulator folds into their primary component, so they are not
# listed here.
_COMPONENT_NAME = {
    0: "los",
    1: "reflection",
    2: "diffraction",
    5: "transmission",
    6: "scattering",
}
_COMPONENT_ID = {name: component_id for component_id, name in _COMPONENT_NAME.items()}

# Component slots materialized by the native accumulator (kComponentCount=3).
# The remaining components are accumulated in Python from the same flat path
# tensors: transmission carries real paths since wave 2; scattering still
# carries zero paths and therefore accumulates structurally valid zeros.
_NATIVE_COMPONENT_SLOTS = {"los": 0, "reflection": 1, "diffraction": 2}
_PYTHON_ACCUMULATED_COMPONENTS = ("transmission", "scattering")


def empty_field_like_power(path_gain: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), device=path_gain.device, dtype=torch.complex64)


def accumulate_flat_components(
    *,
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    component_id: torch.Tensor,
    path_gain: torch.Tensor,
    path_field: torch.Tensor,
    num_tx: int,
    num_rx: int,
    coherent: bool,
    extra_components: tuple[str, ...] = (),
) -> tuple[
    torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    exported = ops.deterministic_accumulate_flat(
        tx_id.to(dtype=torch.int32).contiguous(),
        rx_id.to(dtype=torch.int32).contiguous(),
        component_id.to(dtype=torch.int32).contiguous(),
        path_gain.to(dtype=torch.float32).contiguous(),
        path_field.real.to(dtype=torch.float32).contiguous(),
        path_field.imag.to(dtype=torch.float32).contiguous(),
        num_tx=int(num_tx),
        num_rx=int(num_rx),
        coherent=bool(coherent),
    )
    power_total = exported["power_total"]
    field_total = ops.deterministic_pack_complex(
        exported["field_total_real"].reshape(-1).contiguous(),
        exported["field_total_imag"].reshape(-1).contiguous(),
    ).reshape(exported["field_total_real"].shape)
    component_power_tensor = exported["component_power"]
    component_field_tensor = ops.deterministic_pack_complex(
        exported["component_field_real"].reshape(-1).contiguous(),
        exported["component_field_imag"].reshape(-1).contiguous(),
    ).reshape(exported["component_field_real"].shape)
    component_power = {
        name: component_power_tensor[cid].contiguous()
        for name, cid in _NATIVE_COMPONENT_SLOTS.items()
    }
    component_fields = {
        name: component_field_tensor[cid].contiguous()
        for name, cid in _NATIVE_COMPONENT_SLOTS.items()
    }
    # The native accumulator materializes only the three slots above; the
    # remaining requested components accumulate here from the same flat paths
    # and fold into the totals with the same coherent/incoherent semantics.
    extra_field_sum = torch.zeros_like(field_total)
    extra_power_sum = torch.zeros_like(power_total)
    has_extra_paths = False
    for name in extra_components:
        rows = torch.nonzero(
            component_id == _COMPONENT_ID[name], as_tuple=False
        ).reshape(-1)
        if int(rows.shape[0]) == 0:
            component_power[name] = torch.zeros_like(component_power["los"])
            component_fields[name] = torch.zeros_like(component_fields["los"])
            continue
        has_extra_paths = True
        cell = tx_id[rows].to(dtype=torch.int64) * int(num_rx) + rx_id[rows].to(
            dtype=torch.int64
        )
        cells = int(num_tx) * int(num_rx)
        power_map = torch.zeros(
            (cells,), device=power_total.device, dtype=torch.float32
        ).index_add_(0, cell, path_gain[rows].to(dtype=torch.float32))
        field_map = torch.complex(
            torch.zeros(
                (cells,), device=power_total.device, dtype=torch.float32
            ).index_add_(0, cell, path_field.real[rows].to(dtype=torch.float32)),
            torch.zeros(
                (cells,), device=power_total.device, dtype=torch.float32
            ).index_add_(0, cell, path_field.imag[rows].to(dtype=torch.float32)),
        )
        power_map = power_map.reshape(int(num_tx), int(num_rx))
        field_map = field_map.reshape(int(num_tx), int(num_rx))
        component_power[name] = (
            field_map.abs().square() if coherent else power_map
        ).contiguous()
        component_fields[name] = field_map.contiguous()
        extra_field_sum = extra_field_sum + field_map
        extra_power_sum = extra_power_sum + component_power[name]
    if has_extra_paths:
        if coherent:
            field_total = field_total + extra_field_sum
            power_total = field_total.abs().square()
        else:
            power_total = power_total + extra_power_sum
            field_total = torch.complex(
                power_total.clamp_min(0.0).sqrt(), torch.zeros_like(power_total)
            )
    return power_total, field_total, component_power, component_fields


def apply_layout_to_accumulation(
    *,
    path_gain: torch.Tensor,
    field: torch.Tensor,
    component_power: Mapping[str, torch.Tensor],
    component_fields: Mapping[str, torch.Tensor],
    layout: ReceiverLayout,
    return_field: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    laid_out_power = apply_receiver_layout(path_gain, layout)
    laid_out_field = (
        apply_receiver_layout(field, layout)
        if return_field
        else torch.empty((0,), device=field.device, dtype=torch.complex64)
    )
    laid_out_component_power = {
        name: apply_receiver_layout(value, layout)
        for name, value in component_power.items()
    }
    laid_out_component_fields = {
        name: apply_receiver_layout(value, layout)
        if return_field
        else torch.empty((0,), device=value.device, dtype=torch.complex64)
        for name, value in component_fields.items()
    }
    return (
        laid_out_power,
        laid_out_field,
        laid_out_component_power,
        laid_out_component_fields,
    )


def accumulate_path_result(
    paths: TopologyBatch,
    *,
    frequency_hz: float,
    num_tx: int,
    num_rx: int,
    layout: ReceiverLayout,
    coherent: bool,
    return_field: bool,
    extra_components: tuple[str, ...] = (),
) -> tuple[
    torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    power, field, component_power, component_fields = accumulate_flat_components(
        tx_id=paths.tx_id,
        rx_id=paths.rx_id,
        component_id=paths.component_id,
        path_gain=paths.path_gain,
        path_field=paths.path_field,
        num_tx=num_tx,
        num_rx=num_rx,
        coherent=coherent,
        extra_components=extra_components,
    )
    return apply_layout_to_accumulation(
        path_gain=power,
        field=field,
        component_power=component_power,
        component_fields=component_fields,
        layout=layout,
        return_field=return_field,
    )


def build_path_table(
    paths: TopologyBatch, *, frequency_hz: float, include_fields: bool = True
) -> PathTable:
    if include_fields:
        path_field = paths.path_field.to(dtype=torch.complex64).contiguous()
        phase = ops.deterministic_phase_from_field(
            path_field.real.to(dtype=torch.float32).contiguous(),
            path_field.imag.to(dtype=torch.float32).contiguous(),
        )
    else:
        zero_field_phase = ops.deterministic_zero_field_phase(
            paths.path_gain.to(dtype=torch.float32).contiguous()
        )
        path_field = zero_field_phase["path_field"]
        phase = zero_field_phase["phase_rad"]
    return PathTable(
        valid=paths.valid.contiguous(),
        tx_id=paths.tx_id.to(dtype=torch.int32).contiguous(),
        rx_id=paths.rx_id.to(dtype=torch.int32).contiguous(),
        depth=paths.depth.to(dtype=torch.int32).contiguous(),
        component_id=paths.component_id.to(dtype=torch.int32).contiguous(),
        primitive_id=paths.primitive_id.to(dtype=torch.int32).contiguous(),
        edge_id=paths.edge_id.to(dtype=torch.int32).contiguous(),
        path_length_m=paths.path_length_m.to(dtype=torch.float32).contiguous(),
        delay_s=paths.delay_s.to(dtype=torch.float32).contiguous(),
        path_gain=paths.path_gain.to(dtype=torch.float32).contiguous(),
        interaction_position=paths.interaction_position.to(
            dtype=torch.float32
        ).contiguous(),
        interaction_normal=paths.interaction_normal.to(
            dtype=torch.float32
        ).contiguous(),
        material_id=paths.material_id.to(dtype=torch.int32).contiguous(),
        primitive_sequence=paths.primitive_sequence.to(dtype=torch.int32).contiguous(),
        material_sequence=paths.material_sequence.to(dtype=torch.int32).contiguous(),
        interaction_positions=paths.interaction_positions.to(
            dtype=torch.float32
        ).contiguous(),
        interaction_normals=paths.interaction_normals.to(
            dtype=torch.float32
        ).contiguous(),
        field_real=path_field.real.to(dtype=torch.float32).contiguous(),
        field_imag=path_field.imag.to(dtype=torch.float32).contiguous(),
        coefficient=paths.coefficient.to(dtype=torch.complex64).contiguous(),
        field_xyz=paths.field_xyz.to(dtype=torch.complex64).contiguous(),
        field_direction=paths.field_direction.to(dtype=torch.float32).contiguous(),
        phase_rad=phase,
        interaction_count=paths.depth.to(dtype=torch.int32).contiguous(),
    )
