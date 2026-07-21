"""Typed diffraction edge and first-order path geometry queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel_native.core.diffraction_geometry import (
    cached_diffraction_edge_geometry as _cached_diffraction_edge_geometry,
    diffraction_edge_geometry as _diffraction_edge_geometry,
)
from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge
from witwin.channel_native.runtime.profiling import (
    CudaProfileMark,
    cuda_profile_mark,
)


@dataclass(frozen=True, slots=True)
class DiffractionEdgeGeometry:
    selected: torch.Tensor
    edge_position: torch.Tensor
    edge_direction: torch.Tensor
    edge_length: torch.Tensor
    line_min: torch.Tensor
    line_max: torch.Tensor
    n0: torch.Tensor
    n1: torch.Tensor
    face0: torch.Tensor
    face1: torch.Tensor
    exterior_angle: torch.Tensor


def query_diffraction_edges(
    rayd: object,
    *,
    preserve_imported_edges: bool,
) -> DiffractionEdgeGeometry:
    raw = (
        _diffraction_edge_geometry(rayd.edge_records())
        if preserve_imported_edges
        else _cached_diffraction_edge_geometry(rayd)
    )
    return DiffractionEdgeGeometry(
        selected=raw[0],
        edge_position=raw[1],
        edge_direction=raw[2],
        edge_length=raw[3],
        line_min=raw[4],
        line_max=raw[5],
        n0=raw[6],
        n1=raw[7],
        face0=raw[8],
        face1=raw[9],
        exterior_angle=raw[10],
    )


@dataclass(frozen=True, slots=True)
class DiffractionStateGeometry:
    edge_index: torch.Tensor
    edge_position: torch.Tensor
    edge_direction: torch.Tensor
    edge_t_min: torch.Tensor
    edge_t_max: torch.Tensor
    n0: torch.Tensor
    n1: torch.Tensor
    prim0: torch.Tensor
    prim1: torch.Tensor
    exterior_angle: torch.Tensor
    source: torch.Tensor
    source_power: torch.Tensor


def name_diffraction_states(
    states: tuple[torch.Tensor, ...],
) -> DiffractionStateGeometry:
    if len(states) != 12:
        raise ValueError("diffraction state tuple must contain exactly 12 tensors")
    return DiffractionStateGeometry(
        edge_index=states[0],
        edge_position=states[1],
        edge_direction=states[2],
        edge_t_min=states[3],
        edge_t_max=states[4],
        n0=states[5],
        n1=states[6],
        prim0=states[7],
        prim1=states[8],
        exterior_angle=states[9],
        source=states[10],
        source_power=states[11],
    )


@dataclass(frozen=True, slots=True)
class DiffractionVisibleStatePlan:
    edge_index: torch.Tensor
    edge_position: torch.Tensor
    edge_direction: torch.Tensor
    edge_t_min: torch.Tensor
    edge_t_max: torch.Tensor
    n0: torch.Tensor
    n1: torch.Tensor
    prim0: torch.Tensor
    prim1: torch.Tensor
    exterior_angle: torch.Tensor
    source: torch.Tensor
    source_power: torch.Tensor
    active: torch.Tensor


def plan_tx_visible_diffraction_states(
    rayd: object,
    states: tuple[torch.Tensor, ...],
    tx: torch.Tensor,
) -> DiffractionVisibleStatePlan:
    named = name_diffraction_states(states)
    active = geometry_bridge.diffraction_tx_visible_state_plan(
        rayd.require_resource(),
        tx,
        named.edge_index,
        named.edge_position,
        named.edge_direction,
        named.edge_t_min,
        named.edge_t_max,
        named.n0,
        named.n1,
        named.prim0,
        named.prim1,
        named.exterior_angle,
        named.source,
        named.source_power,
    )
    return DiffractionVisibleStatePlan(
        edge_index=named.edge_index,
        edge_position=named.edge_position,
        edge_direction=named.edge_direction,
        edge_t_min=named.edge_t_min,
        edge_t_max=named.edge_t_max,
        n0=named.n0,
        n1=named.n1,
        prim0=named.prim0,
        prim1=named.prim1,
        exterior_angle=named.exterior_angle,
        source=named.source,
        source_power=named.source_power,
        active=active,
    )


@dataclass(frozen=True, slots=True)
class DiffractionOrder1Query:
    handle: object
    tx_position: torch.Tensor
    tx_polarization: torch.Tensor
    rx_positions: torch.Tensor
    active: torch.Tensor
    states: DiffractionVisibleStatePlan
    material_eta_r: torch.Tensor
    material_sigma: torch.Tensor
    material_mu_r: torch.Tensor
    material_gain: torch.Tensor
    material_valid: torch.Tensor
    state_count: int
    capacity: int
    wavelength: float
    # ISB boundary taper (ADR-017), D member. 0.0 (default) reproduces the hard
    # RayD GO step; > 0 notches the incident-boundary odd part over the congruent
    # window inside the shared UTD header (pair.isbTaperWidthScale).
    isb_taper_width_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.active is not self.states.active:
            raise ValueError("diffraction query active must alias the visible state plan")
        if self.state_count != int(self.states.edge_index.shape[0]):
            raise ValueError("diffraction query state_count must equal plan capacity N")


@dataclass(frozen=True, slots=True)
class DiffractionOrder1Geometry:
    valid: torch.Tensor
    rx_id: torch.Tensor
    depth: torch.Tensor
    edge_id: torch.Tensor
    delay_s: torch.Tensor
    x_re: torch.Tensor
    x_im: torch.Tensor
    y_re: torch.Tensor
    y_im: torch.Tensor
    z_re: torch.Tensor
    z_im: torch.Tensor
    interaction_position: torch.Tensor


def query_diffraction_order1(
    query: DiffractionOrder1Query,
) -> DiffractionOrder1Geometry:
    states = query.states
    cuda_profile_mark(CudaProfileMark.OPTIX_TRAVERSAL)
    cuda_profile_mark(CudaProfileMark.DIFFRACTION_EXPORTER_REQUEST)
    raw = geometry_bridge.rayd_diffraction_paths_order1_forward(
        query.handle,
        query.tx_position,
        query.tx_polarization,
        query.rx_positions,
        query.active,
        states.edge_index,
        states.edge_position,
        states.edge_direction,
        states.edge_t_min,
        states.edge_t_max,
        states.n0,
        states.n1,
        states.prim0,
        states.prim1,
        states.exterior_angle,
        states.source,
        states.source_power,
        query.material_eta_r,
        query.material_sigma,
        query.material_mu_r,
        query.material_gain,
        query.material_valid,
        query.state_count,
        query.capacity,
        query.wavelength,
        query.isb_taper_width_scale,
    )
    return DiffractionOrder1Geometry(
        valid=raw[1],
        rx_id=raw[3],
        depth=raw[4],
        edge_id=raw[5],
        delay_s=raw[8],
        x_re=raw[9],
        x_im=raw[10],
        y_re=raw[11],
        y_im=raw[12],
        z_re=raw[13],
        z_im=raw[14],
        interaction_position=raw[15],
    )
