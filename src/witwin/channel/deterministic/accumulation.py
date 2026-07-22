from __future__ import annotations

from collections.abc import Mapping

import torch

from .result import PathTable
from .kernels import accumulation as accumulation_kernels
from witwin.channel.propagation.fields.kernels import (
    deterministic as field_kernels,
)
from witwin.channel.propagation.geometry.endpoints import (
    ReceiverLayout,
    apply_receiver_layout,
)
from witwin.channel.propagation.models.evaluated import EvaluatedPaths


# Component slots materialized by the native accumulator
# (kAccumSlotCount=6 in kernels/deterministic_accum.cu); the path component
# ids map to them as 0/1/2 -> 0/1/2, 5 -> 3, 6 -> 4, and 3/4 -> 5 (ADR-011:
# reflection->diffraction and diffraction->reflection are the coupled classes,
# both summed coherently into the single coupled field slot). Under
# ad_mode != "none" the same native forward runs inside a dispatch-only
# autograd.Function whose backward/jvp are native CUDA companions
# (accumulation_kernels.deterministic_accumulate_flat_ad), so the accumulated
# result keeps the
# autograd graph with no torch mirror of the kernel math. transmission
# carries specular wall-penetration paths (wave 2) and joins the coherent
# field total like the first three slots; scattering carries Kirchhoff
# rough-surface patch paths (wave 3) and is an incoherent POWER slot (plan
# 05 sections 6.7.3 / 7.3): its rows always fold into the totals in the
# power domain, never as zero-phase amplitudes, and its complex cell field
# is a diagnostic only. coupled carries the reflection-diffraction
# compensator (ADR-011) and is an ordinary coherent field slot that joins
# the coherent field total like the first three slots.
_NATIVE_COMPONENT_SLOTS = {
    "los": 0,
    "reflection": 1,
    "diffraction": 2,
    "transmission": 3,
    "scattering": 4,
    "coupled": 5,
}
_BASE_COMPONENTS = ("los", "reflection", "diffraction")
_OPTIONAL_COMPONENTS = ("transmission", "scattering")


def empty_field_like_power(path_gain: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), device=path_gain.device, dtype=torch.complex64)


def accumulate_flat_components(
    *,
    valid: torch.Tensor,
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    component_id: torch.Tensor,
    path_gain: torch.Tensor,
    path_field: torch.Tensor,
    num_tx: int,
    num_rx: int,
    coherent: bool,
    extra_components: tuple[str, ...] = (),
    differentiable: bool = False,
    scattering_coherent: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    # ADR-021 D3 opt-in coherent scattering combine. Default OFF threads NO new
    # argument to the native accumulator, so the call stays byte-identical to
    # today; ON requests the native scattering_combine_domain=1 path where the
    # scattering slot's summed complex field squares into its power instead of
    # summing per-row powers (the ADR-019 per-component phasor precedent).
    combine_kwargs = {"scattering_combine_domain": 1} if scattering_coherent else {}
    if differentiable:
        # AD modes (plan 07): the same native accumulator kernels run inside
        # a dispatch-only autograd.Function with native backward/jvp
        # companions, so Result.path_gain / field / component_power carry
        # the complete graph of the per-path fields and powers.
        exported = accumulation_kernels.deterministic_accumulate_flat_ad(
            valid.contiguous(),
            tx_id.to(dtype=torch.int32).contiguous(),
            rx_id.to(dtype=torch.int32).contiguous(),
            component_id.to(dtype=torch.int32).contiguous(),
            path_gain.to(dtype=torch.float32).contiguous(),
            path_field.real.to(dtype=torch.float32).contiguous(),
            path_field.imag.to(dtype=torch.float32).contiguous(),
            num_tx=int(num_tx),
            num_rx=int(num_rx),
            coherent=bool(coherent),
            **combine_kwargs,
        )
        power_total = exported["power_total"]
        field_total = torch.complex(
            exported["field_total_real"], exported["field_total_imag"]
        )
        component_power_tensor = exported["component_power"]
        component_field_tensor = torch.complex(
            exported["component_field_real"], exported["component_field_imag"]
        )
    else:
        exported = accumulation_kernels.deterministic_accumulate_flat(
            valid.contiguous(),
            tx_id.to(dtype=torch.int32).contiguous(),
            rx_id.to(dtype=torch.int32).contiguous(),
            component_id.to(dtype=torch.int32).contiguous(),
            path_gain.to(dtype=torch.float32).contiguous(),
            path_field.real.to(dtype=torch.float32).contiguous(),
            path_field.imag.to(dtype=torch.float32).contiguous(),
            num_tx=int(num_tx),
            num_rx=int(num_rx),
            coherent=bool(coherent),
            **combine_kwargs,
        )
        power_total = exported["power_total"]
        field_total = field_kernels.deterministic_pack_complex(
            exported["field_total_real"].reshape(-1).contiguous(),
            exported["field_total_imag"].reshape(-1).contiguous(),
        ).reshape(exported["field_total_real"].shape)
        component_power_tensor = exported["component_power"]
        component_field_tensor = field_kernels.deterministic_pack_complex(
            exported["component_field_real"].reshape(-1).contiguous(),
            exported["component_field_imag"].reshape(-1).contiguous(),
        ).reshape(exported["component_field_real"].shape)
    # All five slots come out of the one native accumulator (forward and,
    # in the AD modes, its native backward/jvp companions); the result dicts
    # expose the base components plus the requested optional ones.
    exported_names = _BASE_COMPONENTS + tuple(extra_components)
    component_power = {
        name: component_power_tensor[_NATIVE_COMPONENT_SLOTS[name]].contiguous()
        for name in exported_names
    }
    component_fields = {
        name: component_field_tensor[_NATIVE_COMPONENT_SLOTS[name]].contiguous()
        for name in exported_names
    }
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
    paths: EvaluatedPaths,
    *,
    frequency_hz: float,
    num_tx: int,
    num_rx: int,
    layout: ReceiverLayout,
    coherent: bool,
    return_field: bool,
    extra_components: tuple[str, ...] = (),
    differentiable: bool = False,
    scattering_coherent: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    topology = paths.topology
    fields = paths.fields
    power, field, component_power, component_fields = accumulate_flat_components(
        valid=topology.valid,
        tx_id=topology.tx_id,
        rx_id=topology.rx_id,
        component_id=topology.component_id,
        path_gain=fields.path_gain,
        path_field=fields.path_field,
        num_tx=num_tx,
        num_rx=num_rx,
        coherent=coherent,
        extra_components=extra_components,
        differentiable=differentiable,
        scattering_coherent=scattering_coherent,
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
    paths: EvaluatedPaths, *, frequency_hz: float, include_fields: bool = True
) -> PathTable:
    topology = paths.topology
    geometry = paths.geometry
    fields = paths.fields
    if include_fields:
        path_field = fields.path_field.to(dtype=torch.complex64).contiguous()
        phase = field_kernels.deterministic_phase_from_field(
            path_field.real.to(dtype=torch.float32).contiguous(),
            path_field.imag.to(dtype=torch.float32).contiguous(),
        )
    else:
        zero_field_phase = field_kernels.deterministic_zero_field_phase(
            fields.path_gain.to(dtype=torch.float32).contiguous()
        )
        path_field = zero_field_phase["path_field"]
        phase = zero_field_phase["phase_rad"]
    return PathTable(
        valid=topology.valid.contiguous(),
        tx_id=topology.tx_id.to(dtype=torch.int32).contiguous(),
        rx_id=topology.rx_id.to(dtype=torch.int32).contiguous(),
        depth=topology.depth.to(dtype=torch.int32).contiguous(),
        component_id=topology.component_id.to(dtype=torch.int32).contiguous(),
        primitive_id=topology.primitive_id.to(dtype=torch.int32).contiguous(),
        edge_id=topology.edge_id.to(dtype=torch.int32).contiguous(),
        path_length_m=geometry.path_length_m.to(dtype=torch.float32).contiguous(),
        delay_s=geometry.delay_s.to(dtype=torch.float32).contiguous(),
        path_gain=fields.path_gain.to(dtype=torch.float32).contiguous(),
        interaction_position=geometry.interaction_position.to(
            dtype=torch.float32
        ).contiguous(),
        interaction_normal=geometry.interaction_normal.to(
            dtype=torch.float32
        ).contiguous(),
        material_id=topology.material_id.to(dtype=torch.int32).contiguous(),
        primitive_sequence=topology.primitive_sequence.to(
            dtype=torch.int32
        ).contiguous(),
        material_sequence=topology.material_sequence.to(dtype=torch.int32).contiguous(),
        interaction_positions=geometry.interaction_positions.to(
            dtype=torch.float32
        ).contiguous(),
        interaction_normals=geometry.interaction_normals.to(
            dtype=torch.float32
        ).contiguous(),
        field_real=path_field.real.to(dtype=torch.float32).contiguous(),
        field_imag=path_field.imag.to(dtype=torch.float32).contiguous(),
        coefficient=fields.coefficient.to(dtype=torch.complex64).contiguous(),
        field_xyz=fields.field_xyz.to(dtype=torch.complex64).contiguous(),
        field_direction=geometry.field_direction.to(dtype=torch.float32).contiguous(),
        phase_rad=phase,
        interaction_count=topology.depth.to(dtype=torch.int32).contiguous(),
    )
