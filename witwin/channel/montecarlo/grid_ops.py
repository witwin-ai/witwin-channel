"""Monte Carlo receiver-grid operations: plane-hit, scatter, store, point-source.

The shared :class:`~witwin.channel.core.grid.Grid` /
:class:`~witwin.channel.core.grid.GridSpec` types live in
:mod:`witwin.channel.core.grid`. This module adds the MC-specific
plane-hit, scatter, contribution-store, and point-source helpers
required by the Monte Carlo integrators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import drjit as dr
from witwin.channel.montecarlo import types as wt

from witwin.channel.core.numerics.arrays import broadcast_complex, broadcast_point
from witwin.channel.core.numerics.constants import EPS, RAY_ORIGIN_BIAS
from witwin.channel.core.geometry import point_on_axis_aligned_plane


_SCATTER_COMPONENTS = (
    "los",
    "reflection",
    "diffraction",
    "diffraction_incident_transition_power",
    "diffraction_reflection_transition_power",
)


@dataclass(slots=True)
class PlaneHit:
    """Ray / receiver-plane intersection result."""
    valid: object
    coord_0: object
    coord_1: object
    target_pos: object
    distance: object
    cos_theta: object


# ---------------------------------------------------------------------------
# Grid intersection and scatter operations
# ---------------------------------------------------------------------------


def scatter_component_deltas(
    *,
    grid,
    coord_0,
    coord_1,
    component_power: dict[str, object],
    active=None,
) -> dict[str, object]:
    width = int(dr.width(coord_0))
    n_cells = int(grid.n_cells)
    if width <= 0:
        zero_map = dr.zeros(wt.Float, n_cells)
        return {component: zero_map for component in _SCATTER_COMPONENTS}
    active_mask = (
        dr.full(wt.Bool, True, width)
        if active is None
        else wt.Bool(active)
    )
    zero = dr.zeros(wt.Float, width)
    deltas = {
        component: dr.zeros(wt.Float, n_cells)
        for component in _SCATTER_COMPONENTS
    }
    _scatter_components_into(
        coord_0=coord_0,
        coord_1=coord_1,
        component_power={
            component: component_power.get(component, zero)
            for component in _SCATTER_COMPONENTS
        },
        targets=deltas,
        grid=grid,
        active=active_mask,
    )
    return deltas


def scatter_components(
    *,
    grid,
    weighted_diagnostics,
    coord_0,
    coord_1,
    component_power: dict[str, object],
    active=None,
) -> None:
    width = int(dr.width(coord_0))
    if width <= 0:
        return
    active_mask = (
        dr.full(wt.Bool, True, width)
        if active is None
        else wt.Bool(active)
    )
    zero = dr.zeros(wt.Float, width)
    _scatter_components_into(
        grid=grid,
        coord_0=coord_0,
        coord_1=coord_1,
        component_power={
            component: component_power.get(component, zero)
            for component in _SCATTER_COMPONENTS
        },
        targets=weighted_diagnostics["incoherent"],
        active=active_mask,
    )


def plane_hit(*, ray_origin, ray_dir, blocker_dist, grid, active):
    axis = str(grid.axis)
    axis_dir = getattr(ray_dir, axis)
    safe_axis_dir = axis_dir + dr.select(axis_dir >= 0.0, wt.Float(EPS), -wt.Float(EPS))
    t_plane = (wt.Float(grid.position) - getattr(ray_origin, axis)) / safe_axis_dir
    hit_point = ray_origin + ray_dir * t_plane
    if axis == "x":
        coord_0 = hit_point.y
        coord_1 = hit_point.z
    elif axis == "y":
        coord_0 = hit_point.x
        coord_1 = hit_point.z
    else:
        coord_0 = hit_point.x
        coord_1 = hit_point.y
    within_bounds = (
        (coord_0 >= wt.Float(grid.bounds[0][0]))
        & (coord_0 < wt.Float(grid.bounds[0][1]))
        & (coord_1 >= wt.Float(grid.bounds[1][0]))
        & (coord_1 < wt.Float(grid.bounds[1][1]))
    )
    valid = (
        active
        & (dr.abs(axis_dir) > wt.Float(EPS))
        & (t_plane > wt.Float(RAY_ORIGIN_BIAS))
        & (t_plane < blocker_dist)
        & within_bounds
    )
    target_pos = point_on_axis_aligned_plane(
        axis=axis,
        position=grid.position,
        tangential_0=coord_0,
        tangential_1=coord_1,
    )
    return PlaneHit(
        valid=valid,
        coord_0=coord_0,
        coord_1=coord_1,
        target_pos=target_pos,
        distance=t_plane,
        cos_theta=dr.abs(axis_dir),
    )


def cell_index(*, grid, coord_0, coord_1):
    (coord_0_min, _), (coord_1_min, _) = grid.bounds
    nx, ny = grid.grid_shape
    ix = dr.clip(wt.Int32((coord_0 - coord_0_min) / grid.cell_size[0]), 0, nx - 1)
    iy = dr.clip(wt.Int32((coord_1 - coord_1_min) / grid.cell_size[1]), 0, ny - 1)
    return wt.UInt32(iy * nx + ix)


def _coords_in_bounds(*, grid, coord_0, coord_1):
    return (
        (coord_0 >= wt.Float(grid.bounds[0][0]))
        & (coord_0 < wt.Float(grid.bounds[0][1]))
        & (coord_1 >= wt.Float(grid.bounds[1][0]))
        & (coord_1 < wt.Float(grid.bounds[1][1]))
    )


def _scatter_components_into(
    *,
    grid,
    coord_0,
    coord_1,
    component_power: dict[str, object],
    targets: dict[str, object],
    active,
) -> None:
    index = cell_index(grid=grid, coord_0=coord_0, coord_1=coord_1)
    in_bounds = _coords_in_bounds(grid=grid, coord_0=coord_0, coord_1=coord_1)
    zero = dr.zeros(wt.Float, int(dr.width(coord_0)))
    for component in _SCATTER_COMPONENTS:
        power = wt.Float(component_power.get(component, zero))
        scatter_active = active & in_bounds & (power != wt.Float(0.0))
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            targets[component],
            dr.select(scatter_active, power, zero),
            index,
            scatter_active,
        )


def scatter(
    *,
    grid,
    weighted_diagnostics,
    component: str,
    coord_0,
    coord_1,
    power,
    active,
):
    active_mask = active & (power != wt.Float(0.0))
    if int(dr.width(power)) <= 0:
        return
    scatter_components(
        grid=grid,
        weighted_diagnostics=weighted_diagnostics,
        coord_0=coord_0,
        coord_1=coord_1,
        component_power={
            str(component): dr.select(active_mask, power, wt.Float(0.0)),
        },
        active=active_mask,
    )


def point_source(source_pos, source_weight, target_pos, wavelength, k):
    """Free-space point-source field (Green's function) with phase."""
    width = dr.width(target_pos.x)
    source_pos_b = broadcast_point(source_pos, width)
    distance = dr.norm(target_pos - source_pos_b) + EPS
    phase = dr.exp(wt.Complex2f(0, -wt.Float(k) * distance))
    source_w = broadcast_complex(source_weight, width)
    fspl = wt.Float(wavelength / (4.0 * math.pi)) / distance
    return source_w * fspl * phase


# ---------------------------------------------------------------------------
# Per-hit cell contribution store (used inside symbolic loops)
# ---------------------------------------------------------------------------


class GridContributionStore:
    """Record per-hit Monte Carlo cell contributions inside symbolic loops."""

    def __init__(self, *, capacity: int, grid=None, weighted_diagnostics=None) -> None:
        if (grid is None) != (weighted_diagnostics is None):
            raise ValueError("grid and weighted_diagnostics must be provided together.")
        self.capacity = max(0, int(capacity))
        self.next_slot = wt.UInt32(0)
        self.grid = grid
        self.weighted_diagnostics = weighted_diagnostics
        self.direct = grid is not None
        if self.direct:
            return
        self.coord_0 = dr.zeros(wt.Float, self.capacity)
        self.coord_1 = dr.zeros(wt.Float, self.capacity)
        self.los_power = dr.zeros(wt.Float, self.capacity)
        self.reflection_power = dr.zeros(wt.Float, self.capacity)
        self.diffraction_power = dr.zeros(wt.Float, self.capacity)
        self.diffraction_incident_transition_power = dr.zeros(wt.Float, self.capacity)
        self.diffraction_reflection_transition_power = dr.zeros(wt.Float, self.capacity)

    def store(
        self,
        *,
        coord_0,
        coord_1,
        component_power: dict[str, object],
        active,
    ) -> None:
        if self.capacity <= 0:
            return
        slot = dr.scatter_inc(self.next_slot, wt.UInt32(0), active)
        store_mask = active & (slot < wt.UInt32(self.capacity))
        width = int(dr.width(coord_0))
        zero = dr.zeros(wt.Float, width)
        if self.direct:
            scatter_components(
                grid=self.grid,
                weighted_diagnostics=self.weighted_diagnostics,
                coord_0=coord_0,
                coord_1=coord_1,
                component_power=component_power,
                active=store_mask,
            )
            return
        dr.scatter(self.coord_0, coord_0, slot, store_mask)
        dr.scatter(self.coord_1, coord_1, slot, store_mask)
        dr.scatter(self.los_power, component_power.get("los", zero), slot, store_mask)
        dr.scatter(
            self.reflection_power,
            component_power.get("reflection", zero),
            slot,
            store_mask,
        )
        dr.scatter(
            self.diffraction_power,
            component_power.get("diffraction", zero),
            slot,
            store_mask,
        )
        dr.scatter(
            self.diffraction_incident_transition_power,
            component_power.get("diffraction_incident_transition_power", zero),
            slot,
            store_mask,
        )
        dr.scatter(
            self.diffraction_reflection_transition_power,
            component_power.get("diffraction_reflection_transition_power", zero),
            slot,
            store_mask,
        )

    def scatter_into(self, *, grid, weighted_diagnostics) -> None:
        if self.direct:
            return
        scatter_components(
            grid=grid,
            weighted_diagnostics=weighted_diagnostics,
            coord_0=self.coord_0,
            coord_1=self.coord_1,
            component_power={
                "los": self.los_power,
                "reflection": self.reflection_power,
                "diffraction": self.diffraction_power,
                "diffraction_incident_transition_power": (
                    self.diffraction_incident_transition_power
                ),
                "diffraction_reflection_transition_power": (
                    self.diffraction_reflection_transition_power
                ),
            },
        )


__all__ = [
    "GridContributionStore",
    "PlaneHit",
    "cell_index",
    "plane_hit",
    "point_source",
    "scatter",
    "scatter_component_deltas",
    "scatter_components",
]
