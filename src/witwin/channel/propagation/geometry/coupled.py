"""Typed coupled reflection-diffraction geometry queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel.propagation.geometry.kernels import bridge as geometry_bridge


@dataclass(frozen=True, slots=True)
class CoupledGeometryQuery:
    rayd_resource: object
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


@dataclass(frozen=True, slots=True)
class CoupledDdGeometryQuery:
    rayd_resource: object
    source: torch.Tensor
    receiver: torch.Tensor
    edge1_id: torch.Tensor
    edge1_position: torch.Tensor
    edge1_direction: torch.Tensor
    edge1_t_min: torch.Tensor
    edge1_t_max: torch.Tensor
    edge2_id: torch.Tensor
    edge2_position: torch.Tensor
    edge2_direction: torch.Tensor
    edge2_t_min: torch.Tensor
    edge2_t_max: torch.Tensor


@dataclass(frozen=True, slots=True)
class CoupledDdGeometry:
    valid: torch.Tensor
    interaction_type_sequence: torch.Tensor
    primitive_sequence: torch.Tensor
    edge_sequence: torch.Tensor
    edge1_id: torch.Tensor
    edge2_id: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    edge1_position: torch.Tensor
    edge2_position: torch.Tensor
    path_length_m: torch.Tensor
    delay_s: torch.Tensor


def query_coupled_geometry(query: CoupledGeometryQuery) -> CoupledGeometry:
    raw = geometry_bridge.coupled_rd_geometry_forward(
        query.rayd_resource,
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


def query_coupled_dd_geometry(query: CoupledDdGeometryQuery) -> CoupledDdGeometry:
    raw = geometry_bridge.coupled_dd_geometry_forward(
        query.rayd_resource,
        query.source,
        query.receiver,
        query.edge1_id,
        query.edge1_position,
        query.edge1_direction,
        query.edge1_t_min,
        query.edge1_t_max,
        query.edge2_id,
        query.edge2_position,
        query.edge2_direction,
        query.edge2_t_min,
        query.edge2_t_max,
    )
    return CoupledDdGeometry(
        valid=raw["valid"],
        interaction_type_sequence=raw["interaction_type_sequence"],
        primitive_sequence=raw["primitive_sequence"],
        edge_sequence=raw["edge_sequence"],
        edge1_id=raw["edge1_id"],
        edge2_id=raw["edge2_id"],
        interaction_positions=raw["interaction_positions"],
        interaction_normals=raw["interaction_normals"],
        edge1_position=raw["edge1_position"],
        edge2_position=raw["edge2_position"],
        path_length_m=raw["path_length_m"],
        delay_s=raw["delay_s"],
    )
