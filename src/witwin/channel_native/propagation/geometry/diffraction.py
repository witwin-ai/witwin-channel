"""Typed diffraction edge and first-order path geometry queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel_native.core.diffraction_geometry import (
    cached_diffraction_edge_geometry as _cached_diffraction_edge_geometry,
    diffraction_edge_geometry as _diffraction_edge_geometry,
)
from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge


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


_DIFFRACTION_PREFILTER_EDGE_FRACTIONS = (0.02, 1.0 / 3.0, 2.0 / 3.0, 0.98)


def _tx_visible_diffraction_states(
    rayd: object,
    states: tuple[torch.Tensor, ...],
    tx: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Drop edge states that are occluded from the transmitter.

    The UTD kernel checks visibility at the per-receiver stationary point, so
    a state is only culled when the transmitter cannot see the edge at any of
    several sample points along it. This shrinks the rx x state workspace and
    pair launches on city-scale scenes (mirrors the original tx_first
    pruning) while keeping states whose midpoint happens to be occluded.
    """

    state_count = int(states[0].shape[0])
    if state_count <= 0:
        return states
    edge_anchor = states[1]
    edge_dir = states[2]
    line_min = states[3]
    line_max = states[4]
    starts = tx.reshape(1, 3).expand(state_count, 3).contiguous()
    visible = torch.zeros((state_count,), device=edge_anchor.device, dtype=torch.bool)
    for fraction in _DIFFRACTION_PREFILTER_EDGE_FRACTIONS:
        t = line_min + fraction * (line_max - line_min)
        point = (edge_anchor + t.unsqueeze(1) * edge_dir).contiguous()
        visible |= geometry_bridge.rayd_visibility_forward(
            rayd.require_resource(), starts, point, None
        )[0]
    if bool(visible.all()):
        return states
    return tuple(
        tensor[visible] if tensor.shape[:1] == (state_count,) else tensor
        for tensor in states
    )


@dataclass(frozen=True, slots=True)
class DiffractionOrder1Query:
    handle: object
    tx_position: torch.Tensor
    tx_polarization: torch.Tensor
    rx_positions: torch.Tensor
    active: torch.Tensor | None
    states: DiffractionStateGeometry
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
