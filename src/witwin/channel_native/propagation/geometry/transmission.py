"""Typed closest-hit geometry queries for straight transmission paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge


@dataclass(frozen=True, slots=True)
class TransmissionClosestHitQuery:
    handle: object
    origin: torch.Tensor
    direction: torch.Tensor
    ray_tmax: torch.Tensor
    active: torch.Tensor | None
    flags: int


@dataclass(frozen=True, slots=True)
class TransmissionClosestHitResult:
    t: torch.Tensor
    position: torch.Tensor
    geometric_normal: torch.Tensor
    global_primitive_id: torch.Tensor


def query_transmission_closest_hit(
    query: TransmissionClosestHitQuery,
) -> TransmissionClosestHitResult:
    raw = geometry_bridge.bdpt_intersect_forward(
        query.handle,
        query.origin,
        query.direction,
        query.ray_tmax,
        query.active,
        flags=query.flags,
    )
    return TransmissionClosestHitResult(
        t=raw["t"],
        position=raw["p"],
        geometric_normal=raw["geo_n"],
        global_primitive_id=raw["global_prim_id"],
    )
