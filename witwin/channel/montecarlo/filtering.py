"""Differentiable power-domain filters for Monte Carlo radio maps."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Mapping

import drjit as dr

from . import types as wt
from .config import ComponentFilterConfig, FilterConfig, _coerce_component, _coerce_filter


_FILTER_CONTRACT = "differentiable_post_accumulation_power_denoising"
_DIFFRACTION_SHADOW_BOUNDARY_POWER_KEYS = (
    "diffraction_incident_transition_power",
    "diffraction_reflection_transition_power",
)


def apply_component_filter(
    values,
    config: ComponentFilterConfig | Mapping[str, object] | None,
    *,
    grid,
):
    """Return ``values`` filtered over ``grid`` with a differentiable operator."""
    component_config = _coerce_component(config)
    if component_config is None or component_config.radius <= 0 or component_config.blend <= 0.0:
        return values
    if int(dr.width(values)) != int(grid.n_cells):
        raise ValueError("filter input width must match grid.n_cells.")
    candidate = _normalized_filter(values, component_config, grid=grid)
    blend = wt.Float(component_config.blend)
    return (wt.Float(1.0) - blend) * values + blend * candidate


def apply_power_filtering(
    weighted_diagnostics: dict[str, object],
    *,
    filtering: FilterConfig | Mapping[str, object] | None,
    grid,
) -> None:
    """Apply configured component filters to a weighted diagnostics payload."""
    filter_config = _coerce_filter(filtering)
    if filter_config is None or not filter_config.enabled:
        return
    incoherent = weighted_diagnostics["incoherent"]
    if filter_config.reflection is not None:
        incoherent["reflection"] = apply_component_filter(
            incoherent["reflection"], filter_config.reflection, grid=grid,
        )
    if filter_config.diffraction is not None:
        incoherent["diffraction"] = apply_component_filter(
            incoherent["diffraction"], filter_config.diffraction, grid=grid,
        )
        for key in _DIFFRACTION_SHADOW_BOUNDARY_POWER_KEYS:
            if key in incoherent:
                incoherent[key] = apply_component_filter(
                    incoherent[key], filter_config.diffraction, grid=grid,
                )


def filtering_metadata(
    filtering: FilterConfig | Mapping[str, object] | None,
) -> dict[str, object]:
    """Return public metadata for the configured power filtering transform."""
    filter_config = _coerce_filter(filtering)
    if filter_config is None or not filter_config.enabled:
        return {"enabled": False}
    components = {}
    if filter_config.reflection is not None:
        components["reflection"] = _component_dict(filter_config.reflection)
    if filter_config.diffraction is not None:
        components["diffraction"] = _component_dict(filter_config.diffraction)
    return {
        "enabled": True,
        "domain": "incoherent_power",
        "components": components,
        "shadow_boundary_transition_power": (
            "filtered_with_diffraction_component_when_diffraction_filtering_is_enabled"
        ),
        "contract": _FILTER_CONTRACT,
    }


def _component_dict(component: ComponentFilterConfig) -> dict[str, object]:
    data = asdict(component)
    if data["range_sigma"] is None:
        del data["range_sigma"]
    return data


def _normalized_filter(values, config: ComponentFilterConfig, *, grid):
    nx = int(grid.grid_shape[0])
    ny = int(grid.grid_shape[1])
    n_cells = int(grid.n_cells)
    cell_idx = dr.arange(wt.UInt32, n_cells)
    ix = wt.Int32(cell_idx % wt.UInt32(nx))
    iy = wt.Int32(cell_idx // wt.UInt32(nx))
    center = wt.Float(values)
    weighted_sum = dr.zeros(wt.Float, n_cells)
    weight_sum = dr.zeros(wt.Float, n_cells)
    radius = int(config.radius)
    inv_two_sigma2 = 0.5 / (float(config.sigma) * float(config.sigma))
    inv_two_range_sigma2 = (
        0.0 if config.range_sigma is None
        else 0.5 / (float(config.range_sigma) * float(config.range_sigma))
    )

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor_x = ix + wt.Int32(dx)
            neighbor_y = iy + wt.Int32(dy)
            valid = (
                (neighbor_x >= wt.Int32(0))
                & (neighbor_x < wt.Int32(nx))
                & (neighbor_y >= wt.Int32(0))
                & (neighbor_y < wt.Int32(ny))
            )
            flat_idx = neighbor_y * wt.Int32(nx) + neighbor_x
            safe_idx = wt.UInt32(dr.select(valid, flat_idx, wt.Int32(0)))
            neighbor = dr.gather(wt.Float, values, safe_idx)
            spatial_weight = wt.Float(
                math.exp(-float(dx * dx + dy * dy) * inv_two_sigma2)
            )
            if config.method == "bilateral":
                range_delta = neighbor - center
                weight = spatial_weight * dr.exp(
                    -dr.square(range_delta) * wt.Float(inv_two_range_sigma2)
                )
            else:
                weight = spatial_weight
            active_weight = dr.select(valid, weight, wt.Float(0.0))
            weighted_sum = weighted_sum + active_weight * neighbor
            weight_sum = weight_sum + active_weight

    return weighted_sum / dr.maximum(weight_sum, wt.Float(1.0e-30))


__all__ = [
    "apply_component_filter",
    "apply_power_filtering",
    "filtering_metadata",
]
