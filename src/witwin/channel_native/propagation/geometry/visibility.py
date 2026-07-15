"""Typed RayDN visibility queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge


@dataclass(frozen=True, slots=True)
class VisibilityQuery:
    raydn: object
    start: torch.Tensor
    end: torch.Tensor
    active: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class VisibilityResult:
    visible: torch.Tensor


def run_visibility_query(query: VisibilityQuery) -> VisibilityResult:
    return VisibilityResult(
        visible=geometry_bridge.raydn_visibility_forward(
            query.raydn.require_handle(),
            query.start,
            query.end,
            query.active,
        )[0]
    )


def _raydn_visibility_mask(
    raydn: object, start: torch.Tensor, end: torch.Tensor
) -> torch.Tensor:
    if start.shape[0] == 0:
        return torch.empty((0,), device=start.device, dtype=torch.bool)
    return geometry_bridge.raydn_visibility_forward(
        raydn.require_handle(), start.contiguous(), end.contiguous(), None
    )[0]


def _los_visibility_mask(
    raydn: object,
    tx_for_path: torch.Tensor,
    rx_for_path: torch.Tensor,
    *,
    has_structures: bool,
) -> torch.Tensor | None:
    if not has_structures or tx_for_path.shape[0] == 0:
        return None
    if not raydn.available:
        raise RuntimeError("LoS visibility requires RayDN native scene capability")
    return _raydn_visibility_mask(raydn, tx_for_path, rx_for_path)
