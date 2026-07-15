"""Typed coupled reflection-diffraction geometry queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge


@dataclass(frozen=True, slots=True)
class CoupledGeometryQuery:
    raydn_handle: object
    source: torch.Tensor
    receiver: torch.Tensor
    face_id: torch.Tensor
    face_anchor: torch.Tensor
    face_normal: torch.Tensor
    edge_id: torch.Tensor
    edge_position: torch.Tensor
    edge_direction: torch.Tensor
    edge_t_min: torch.Tensor
    edge_t_max: torch.Tensor
    surface_group_id: torch.Tensor
    surface_group_size: torch.Tensor
    surface_group_members: torch.Tensor
    reverse: bool


@dataclass(frozen=True, slots=True)
class CoupledGeometry:
    valid: torch.Tensor
    interaction_type_sequence: torch.Tensor
    primitive_sequence: torch.Tensor
    edge_sequence: torch.Tensor
    face_id: torch.Tensor
    edge_id: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    reflection_position: torch.Tensor
    reflection_normal: torch.Tensor
    edge_position: torch.Tensor
    edge_direction: torch.Tensor
    path_length_m: torch.Tensor
    delay_s: torch.Tensor


def query_coupled_geometry(query: CoupledGeometryQuery) -> CoupledGeometry:
    raw = geometry_bridge.raydn_coupled_rd_geometry_forward(
        query.raydn_handle,
        query.source,
        query.receiver,
        query.face_id,
        query.face_anchor,
        query.face_normal,
        query.edge_id,
        query.edge_position,
        query.edge_direction,
        query.edge_t_min,
        query.edge_t_max,
        query.surface_group_id,
        query.surface_group_size,
        query.surface_group_members,
        query.reverse,
    )
    return CoupledGeometry(
        valid=raw["valid"],
        interaction_type_sequence=raw["interaction_type_sequence"],
        primitive_sequence=raw["primitive_sequence"],
        edge_sequence=raw["edge_sequence"],
        face_id=raw["face_id"],
        edge_id=raw["edge_id"],
        interaction_positions=raw["interaction_positions"],
        interaction_normals=raw["interaction_normals"],
        reflection_position=raw["reflection_position"],
        reflection_normal=raw["reflection_normal"],
        edge_position=raw["edge_position"],
        edge_direction=raw["edge_direction"],
        path_length_m=raw["path_length_m"],
        delay_s=raw["delay_s"],
    )
