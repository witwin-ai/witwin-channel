"""Typed RayD visibility queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel.propagation.geometry.kernels import bridge as geometry_bridge


@dataclass(frozen=True, slots=True)
class VisibilityQuery:
    rayd: object
    start: torch.Tensor
    end: torch.Tensor
    active: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class VisibilityResult:
    visible: torch.Tensor


def run_visibility_query(query: VisibilityQuery) -> VisibilityResult:
    return VisibilityResult(
        visible=geometry_bridge.rayd_visibility_forward(
            query.rayd.require_resource(),
            query.start,
            query.end,
            query.active,
        )[0]
    )


def _rayd_visibility_mask(
    rayd: object, start: torch.Tensor, end: torch.Tensor
) -> torch.Tensor:
    if start.shape[0] == 0:
        return torch.empty((0,), device=start.device, dtype=torch.bool)
    return geometry_bridge.rayd_visibility_forward(
        rayd.require_resource(), start.contiguous(), end.contiguous(), None
    )[0]


def _los_visibility_mask(
    rayd: object,
    tx_for_path: torch.Tensor,
    rx_for_path: torch.Tensor,
    *,
    has_structures: bool,
) -> torch.Tensor | None:
    if not has_structures or tx_for_path.shape[0] == 0:
        return None
    if not rayd.available:
        raise RuntimeError("LoS visibility requires RayD native scene capability")
    return _rayd_visibility_mask(rayd, tx_for_path, rx_for_path)
