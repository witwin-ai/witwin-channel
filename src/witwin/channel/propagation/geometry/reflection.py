"""Typed reflection endpoint-connection geometry queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel.propagation.geometry.kernels import bridge as geometry_bridge


@dataclass(frozen=True, slots=True)
class ReflectionEpcQuery:
    rayd: object
    source: torch.Tensor
    receiver: torch.Tensor
    active: torch.Tensor | None
    expected_prim_ids: torch.Tensor
    direct_plane_points: torch.Tensor
    direct_plane_normals: torch.Tensor
    surface_group_id: torch.Tensor
    surface_group_size: torch.Tensor
    surface_group_members: torch.Tensor
    max_bounces: int
    visibility_ignore_mode: int


@dataclass(frozen=True, slots=True)
class ReflectionEpcGeometry:
    visible: torch.Tensor
    path_length_m: torch.Tensor
    resolved_prim_ids: torch.Tensor
    surface_group_ids: torch.Tensor
    hit_positions: torch.Tensor
    normals: torch.Tensor


def query_reflection_epc(query: ReflectionEpcQuery) -> ReflectionEpcGeometry:
    raw = geometry_bridge.rayd_reflection_epc_paths_forward(
        query.rayd.require_resource(),
        query.source,
        query.receiver,
        query.active,
        query.expected_prim_ids,
        query.direct_plane_points,
        query.direct_plane_normals,
        query.surface_group_id,
        query.surface_group_size,
        query.surface_group_members,
        query.max_bounces,
        query.visibility_ignore_mode,
    )
    return ReflectionEpcGeometry(
        visible=raw[0],
        path_length_m=raw[1],
        resolved_prim_ids=raw[2],
        surface_group_ids=raw[3],
        hit_positions=raw[4],
        normals=raw[5],
    )
