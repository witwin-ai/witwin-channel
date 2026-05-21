"""Axis-aligned dense field specifications and boundary-sampled grids."""

from __future__ import annotations

from collections.abc import Mapping
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import drjit as dr
from witwin.channel.core.scene import Scene

from . import config as cfg, types as wt
from .config import Config, FieldSolveSpec
from .trace import diffraction, los, reflection
from .reflection.common import grad_sensitive_workload
from witwin.channel.core.runtime import TraceContext, TraceCtx, assert_scene_materials_complete
from witwin.channel.core.physics import polarization
from witwin.channel.core.numerics.arrays import complex_abs_sqr, eval_complex, eval_nested, scalar
from witwin.channel.core.geometry import (
    normal_axis_for_tangential_axes,
    normalize_axis,
    point_on_axis_aligned_plane,
    tangential_axes_for_axis,
)
from witwin.channel.core.results.ray_mode import normalize_ray_mode

_RAY_SAMPLING = {"auto", "circle", "full_sphere", "hemisphere", "hemisphere_facing_monitor"}


def _normalize_bounds(
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(bounds) != 2 or len(bounds[0]) != 2 or len(bounds[1]) != 2:
        raise ValueError("bounds must be ((min_0, max_0), (min_1, max_1)).")
    resolved = (
        (float(bounds[0][0]), float(bounds[0][1])),
        (float(bounds[1][0]), float(bounds[1][1])),
    )
    if resolved[0][0] >= resolved[0][1] or resolved[1][0] >= resolved[1][1]:
        raise ValueError("bounds must be ordered as ((min_0, max_0), (min_1, max_1)).")
    return resolved


def _normalize_grid_shape(grid_shape) -> tuple[int, int] | None:
    if grid_shape is None:
        return None
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain exactly two dimensions.")
    resolved = (int(grid_shape[0]), int(grid_shape[1]))
    if resolved[0] <= 0 or resolved[1] <= 0:
        raise ValueError("grid_shape must contain positive integers.")
    return resolved


def _normalize_resolution(resolution) -> float | None:
    if resolution is None:
        return None
    resolved = float(resolution)
    if resolved <= 0.0:
        raise ValueError("resolution must be > 0.")
    return resolved


def _normalize_ray_sampling(ray_sampling: str) -> str:
    resolved = str(ray_sampling).lower()
    if resolved not in _RAY_SAMPLING:
        raise ValueError(
            "ray_sampling must be one of 'auto', 'circle', 'full_sphere', "
            "'hemisphere', or 'hemisphere_facing_monitor'."
        )
    return resolved


@dataclass(slots=True)
class FieldSpec:
    """Public description of one dense axis-aligned receiver plane."""

    axis: str = "z"
    position: float = 0.0
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((-8.0, 8.0), (-8.0, 8.0))
    grid_shape: tuple[int, int] | None = None
    resolution: float | None = None
    ray_mode: str = "2d"
    ray_sampling: str = "full_sphere"
    max_diffractions: int | None = None

    def __post_init__(self) -> None:
        self.axis = normalize_axis(self.axis)
        self.position = float(self.position)
        self.bounds = _normalize_bounds(self.bounds)
        self.grid_shape = _normalize_grid_shape(self.grid_shape)
        self.resolution = _normalize_resolution(self.resolution)
        self.ray_mode = normalize_ray_mode(self.ray_mode)
        self.ray_sampling = _normalize_ray_sampling(self.ray_sampling)
        if self.grid_shape is None and self.resolution is None:
            raise ValueError("FieldSpec requires grid_shape or resolution.")
        if self.max_diffractions is not None and int(self.max_diffractions) < 0:
            raise ValueError("max_diffractions must be >= 0 when provided.")
        self.max_diffractions = None if self.max_diffractions is None else int(self.max_diffractions)

    @property
    def tangential_axes(self) -> tuple[str, str]:
        return tangential_axes_for_axis(self.axis)

    def resolve_grid_shape(
        self,
        wavelength: float,
        *,
        default_resolution: float | None = None,
    ) -> tuple[int, int]:
        if self.grid_shape is not None:
            return self.grid_shape
        resolved_resolution = self.resolution
        if resolved_resolution is None and default_resolution is not None:
            resolved_resolution = _normalize_resolution(default_resolution)
        if resolved_resolution is None:
            raise ValueError("FieldSpec requires grid_shape or resolution.")
        cell_size = float(resolved_resolution) * float(wavelength)
        nx = max(1, int(math.ceil((self.bounds[0][1] - self.bounds[0][0]) / cell_size)))
        ny = max(1, int(math.ceil((self.bounds[1][1] - self.bounds[1][0]) / cell_size)))
        return (nx, ny)

    def to_field(
        self,
        wavelength: float,
        *,
        default_resolution: float | None = None,
    ) -> "FieldGrid":
        return FieldGrid(
            bounds=self.bounds,
            size=self.resolve_grid_shape(wavelength, default_resolution=default_resolution),
            axis=self.axis,
            position=self.position,
        )

    def to_solve_spec(
        self,
        *,
        wavelength: float,
        default_resolution: float,
        shadow_support_cutoff_db: float | None = None,
    ) -> FieldSolveSpec:
        return FieldSolveSpec(
            axis=self.axis,
            position=self.position,
            bounds=self.bounds,
            grid_shape=self.resolve_grid_shape(
                wavelength,
                default_resolution=default_resolution,
            ),
            ray_mode=self.ray_mode,
            ray_sampling=self.ray_sampling,
            shadow_support_cutoff_db=shadow_support_cutoff_db,
        )


class FieldGrid:
    """Boundary-aligned field monitor grid with legacy ``span / n`` bins."""

    _coord_cache: ClassVar[dict] = {}

    def __init__(
        self,
        *,
        bounds: tuple[tuple[float, float], tuple[float, float]],
        size: tuple[int, int],
        axis: str = "z",
        position: float = 0.0,
        tangential_axes: tuple[str, str] | None = None,
    ) -> None:
        self.bounds = _normalize_bounds(bounds)
        self.size = _normalize_grid_shape(size)
        self.n_cells = int(self.size[0] * self.size[1])
        self.axis = normalize_axis(axis)
        self.position = float(position)
        self.tangential_axes = (
            tangential_axes_for_axis(self.axis)
            if tangential_axes is None
            else tuple(normalize_axis(axis_name) for axis_name in tangential_axes)
        )
        if len(self.tangential_axes) != 2 or self.tangential_axes[0] == self.tangential_axes[1]:
            raise ValueError("tangential_axes must contain two distinct axes.")
        self.normal_axis = normal_axis_for_tangential_axes(self.tangential_axes)
        if self.normal_axis != self.axis:
            raise ValueError(
                f"tangential_axes {self.tangential_axes!r} are incompatible with axis={self.axis!r}."
            )
        (x_min, x_max), (y_min, y_max) = self.bounds
        self.cell_size = ((x_max - x_min) / self.size[0], (y_max - y_min) / self.size[1])

    @property
    def grid_size(self) -> int:
        return int(self.size[0])

    @property
    def axis_x(self) -> str:
        return self.tangential_axes[0]

    @property
    def axis_y(self) -> str:
        return self.tangential_axes[1]

    @property
    def X(self):
        return self.get_coordinates()["X"]

    @property
    def Y(self):
        return self.get_coordinates()["Y"]

    @property
    def receivers(self):
        return self.receiver_positions_3d()

    def pos_to_idx(self, coord_0: wt.Float, coord_1: wt.Float) -> wt.UInt32:
        (x_min, _), (y_min, _) = self.bounds
        nx, ny = self.size
        ix = dr.clip(wt.Int32((coord_0 - x_min) / self.cell_size[0]), 0, nx - 1)
        iy = dr.clip(wt.Int32((coord_1 - y_min) / self.cell_size[1]), 0, ny - 1)
        return wt.UInt32(iy * nx + ix)

    def get_coordinates(self) -> dict[str, object]:
        cache_key = (self.size, self.bounds, self.axis, self.position, self.tangential_axes)
        cached = FieldGrid._coord_cache.get(cache_key)
        if cached is not None:
            return cached

        (x_min, x_max), (y_min, y_max) = self.bounds
        nx, ny = self.size
        x_step = (x_max - x_min) / (nx - 1) if nx > 1 else 0.0
        y_step = (y_max - y_min) / (ny - 1) if ny > 1 else 0.0
        x_coords = wt.Float(x_min) + dr.arange(wt.Float, nx) * wt.Float(x_step)
        y_coords = wt.Float(y_min) + dr.arange(wt.Float, ny) * wt.Float(y_step)
        X = dr.tile(x_coords, ny)
        Y = dr.repeat(y_coords, nx)
        eval_nested(x_coords, y_coords, X, Y)
        coord_data = {
            "axis_x": self.axis_x,
            "axis_y": self.axis_y,
            "axis": self.axis,
            "position": self.position,
            "tangential_axes": self.tangential_axes,
            "x_coords": x_coords,
            "y_coords": y_coords,
            "X": X,
            "Y": Y,
        }
        FieldGrid._coord_cache[cache_key] = coord_data
        return coord_data

    def receiver_positions_3d(self, axis: str | None = None, position: float | None = None):
        resolved_axis = self.axis if axis is None else normalize_axis(axis)
        resolved_position = self.position if position is None else float(position)
        if resolved_axis != self.normal_axis:
            raise ValueError(
                f"FieldGrid tangential axes {self.tangential_axes!r} are incompatible "
                f"with axis={resolved_axis!r}."
            )
        coords = self.get_coordinates()
        return point_on_axis_aligned_plane(
            axis=resolved_axis,
            position=resolved_position,
            tangential_0=coords["X"],
            tangential_1=coords["Y"],
        )


def _mapping_proxy(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


def _component_proxy(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({str(key): component for key, component in dict(value).items()})


@dataclass(frozen=True)
class Coordinates:
    """Coordinate arrays for a solved receiver field."""

    grid_x: object
    grid_y: object
    x: object
    y: object
    axis_x: str = "x"
    axis_y: str = "y"

    @property
    def tangential_axes(self) -> tuple[str, str]:
        return (self.axis_x, self.axis_y)


@dataclass(frozen=True)
class FieldComponents:
    """Scalar complex field components."""

    los: object
    reflection: object
    diffraction_direct: object
    diffraction_mixed: object
    diffraction: object
    total: object


@dataclass(frozen=True)
class VectorComponents:
    """World-vector complex field components."""

    los: Mapping[str, object]
    reflection: Mapping[str, object]
    diffraction_direct: Mapping[str, object]
    diffraction_mixed: Mapping[str, object]
    diffraction: Mapping[str, object]
    total: Mapping[str, object]


@dataclass(frozen=True)
class JonesComponents:
    """Tangential Jones complex field components."""

    los: Mapping[str, object]
    reflection: Mapping[str, object]
    diffraction_direct: Mapping[str, object]
    diffraction_mixed: Mapping[str, object]
    diffraction: Mapping[str, object]
    total: Mapping[str, object]


@dataclass(frozen=True)
class FieldResult:
    """Full field result for one ``FieldSpec`` solve."""

    name: str
    kind: str
    axis: str
    plane_position: float
    ray_mode: str
    ray_sampling: str
    bounds: tuple[tuple[float, float], tuple[float, float]]
    grid_shape: tuple[int, int]
    coords: Coordinates
    field: FieldComponents
    vector: VectorComponents
    jones: JonesComponents
    power: Mapping[str, object]
    metadata: Mapping[str, object]
    tx_pos: tuple[float, float, float]
    diffraction_detail: Mapping[str, object] | None = None

    @property
    def primary(self) -> "FieldResult":
        return self

    @property
    def range_x(self) -> tuple[float, float]:
        return self.bounds[0]

    @property
    def range_y(self) -> tuple[float, float]:
        return self.bounds[1]

    @property
    def tangential_axes(self) -> tuple[str, str]:
        return self.coords.tangential_axes

    @property
    def total_power(self):
        return self.power["total"]


def _reflection_model_metadata(reflection_detail) -> dict[str, object]:
    if reflection_detail is None:
        return {"reflection_model": "none"}
    return {
        "reflection_model": getattr(reflection_detail, "reflection_model", "unknown"),
        "reflection_model_source": getattr(reflection_detail, "reflection_model_source", "unknown"),
        "reflection_gain": getattr(reflection_detail, "reflection_gain", None),
    }


def _diffraction_metadata(diffraction_payload: Mapping[str, object]) -> dict[str, object]:
    source = dict(diffraction_payload.get("metadata", {}))
    keys = (
        "diffraction_skipped",
        "diffraction_skip_reason",
        "max_diffractions",
        "enable_rd_diffraction",
        "edge_selection_mode",
        "edge_diffraction",
        "boundary_edge_policy",
    )
    return {key: source[key] for key in keys if key in source}


def _build_metadata(
    *,
    field: FieldSpec,
    grid_size: tuple[int, int],
    config: cfg.ResolvedTraceConfig,
    reflection_detail,
    diffraction_payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "field": {
            "axis": field.axis,
            "position": field.position,
            "bounds": field.bounds,
            "grid_shape": grid_size,
            "ray_mode": field.ray_mode,
            "ray_sampling": field.ray_sampling,
            "tangential_axes": field.tangential_axes,
            "resolution_wavelength": (
                field.resolution if field.resolution is not None else config.resolution_wavelength
            ),
        },
        "reflection": _reflection_model_metadata(reflection_detail),
        "diffraction": _diffraction_metadata(diffraction_payload),
        "frequency": float(config.frequency),
        "wavelength": float(config.wavelength),
    }


def solve_field(
    *,
    scene: Scene,
    frequency: float,
    tx_pos,
    field: FieldSpec,
    config: Config | None = None,
    return_diffraction_audit: bool = False,
) -> FieldResult:
    """Solve LoS, reflection, and diffraction field components on one plane."""

    assert_scene_materials_complete(scene)
    user_config = Config() if config is None else config
    resolved = cfg.resolve_trace_config(frequency=frequency, config=user_config)
    max_diffractions = (
        int(user_config.max_diffraction_order)
        if field.max_diffractions is None
        else int(field.max_diffractions)
    )
    solver_controls = cfg.resolve_solver_controls(
        user_config,
        execution_intent="field",
        max_diffractions_override=max_diffractions,
    )
    scene.diffraction_edge_count(edge_policy=user_config.edge_policy)
    runtime = TraceContext.from_config(tx_pos=tx_pos, config=resolved)
    grid = field.to_field(
        resolved.wavelength,
        default_resolution=resolved.resolution_wavelength,
    )
    spec = field.to_solve_spec(
        wavelength=resolved.wavelength,
        default_resolution=resolved.resolution_wavelength,
        shadow_support_cutoff_db=resolved.shadow_support_cutoff_db,
    )
    coords = grid.get_coordinates()
    sample_runtime = runtime.with_rx(
        grid.receivers,
        polarization=resolved.rx_polarization,
    )
    n_rx = int(grid.n_cells)
    active_rx_polarization = polarization.effective_rx_polarization(
        resolved.rx_polarization,
        resolved.tx_polarization,
    )
    grad_sensitive = grad_sensitive_workload(tx=runtime.tx, scene=scene)

    polarization_los = los.trace_vector(scene=scene, runtime=sample_runtime)
    a_los = eval_complex(
        polarization.scalarize_vector_to_tangential_polarization(
            polarization_los,
            active_rx_polarization,
            axis=field.axis,
        )
    )

    ctx = TraceCtx(
        scene=scene,
        runtime=sample_runtime,
        sample_grid=grid,
        config=resolved,
        spec=spec,
        solver_controls=solver_controls,
        grad_preserving=bool(grad_sensitive),
        n_rx=n_rx,
    )

    reflection_payload = reflection.trace(ctx=ctx, reflection_detail=None)
    reflection_detail = reflection_payload["detail"]
    polarization_ref = polarization.vector_eval(reflection_payload["field"]["vector"])
    a_ref = eval_complex(
        polarization.scalarize_vector_to_tangential_polarization(
            polarization_ref,
            active_rx_polarization,
            axis=field.axis,
        )
    )

    diffraction_payload = diffraction.trace_components(
        ctx=ctx,
        reflection_detail=reflection_detail,
        return_audit=return_diffraction_audit,
    )
    a_dif_direct = eval_complex(diffraction_payload["a_direct"])
    a_dif_mixed = eval_complex(diffraction_payload["a_multi"])
    a_dif = eval_complex(a_dif_direct + a_dif_mixed)
    polarization_dif_direct = polarization.vector_eval(diffraction_payload["polarization_direct"])
    polarization_dif_mixed = polarization.vector_eval(diffraction_payload["polarization_multi"])
    polarization_dif = polarization.vector_eval(
        polarization.vector_add(polarization_dif_direct, polarization_dif_mixed)
    )

    polarization_tot = polarization.vector_eval(
        polarization.vector_add(
            polarization.vector_add(polarization_los, polarization_ref),
            polarization_dif,
        )
    )
    a_tot = eval_complex(a_los + a_ref + a_dif)
    field_components = FieldComponents(
        los=a_los,
        reflection=a_ref,
        diffraction_direct=a_dif_direct,
        diffraction_mixed=a_dif_mixed,
        diffraction=a_dif,
        total=a_tot,
    )
    field_component_map = {
        "los": field_components.los,
        "reflection": field_components.reflection,
        "diffraction_direct": field_components.diffraction_direct,
        "diffraction_mixed": field_components.diffraction_mixed,
        "diffraction": field_components.diffraction,
        "total": field_components.total,
    }
    power_components = {
        name: complex_abs_sqr(value)
        for name, value in field_component_map.items()
    }
    vector_components = VectorComponents(
        los=_component_proxy(polarization_los),
        reflection=_component_proxy(polarization_ref),
        diffraction_direct=_component_proxy(polarization_dif_direct),
        diffraction_mixed=_component_proxy(polarization_dif_mixed),
        diffraction=_component_proxy(polarization_dif),
        total=_component_proxy(polarization_tot),
    )
    jones_components = JonesComponents(
        los=_component_proxy(polarization.jones_tangential(polarization_los, axis=field.axis)),
        reflection=_component_proxy(polarization.jones_tangential(polarization_ref, axis=field.axis)),
        diffraction_direct=_component_proxy(
            polarization.jones_tangential(polarization_dif_direct, axis=field.axis)
        ),
        diffraction_mixed=_component_proxy(
            polarization.jones_tangential(polarization_dif_mixed, axis=field.axis)
        ),
        diffraction=_component_proxy(polarization.jones_tangential(polarization_dif, axis=field.axis)),
        total=_component_proxy(polarization.jones_tangential(polarization_tot, axis=field.axis)),
    )
    eval_nested(a_tot, power_components["total"])

    diffraction_detail = None
    if return_diffraction_audit:
        diffraction_detail = _mapping_proxy({
            key: value
            for key, value in diffraction_payload.items()
            if key in {"metadata", "state_audit"}
        })

    return FieldResult(
        name="field",
        kind="field",
        axis=field.axis,
        plane_position=field.position,
        ray_mode=field.ray_mode,
        ray_sampling=field.ray_sampling,
        bounds=field.bounds,
        grid_shape=grid.size,
        coords=Coordinates(
            grid_x=coords["X"],
            grid_y=coords["Y"],
            x=coords["x_coords"],
            y=coords["y_coords"],
            axis_x=coords["axis_x"],
            axis_y=coords["axis_y"],
        ),
        field=field_components,
        vector=vector_components,
        jones=jones_components,
        power=_component_proxy(power_components),
        metadata=_mapping_proxy(_build_metadata(
            field=field,
            grid_size=grid.size,
            config=resolved,
            reflection_detail=reflection_detail,
            diffraction_payload=diffraction_payload,
        )),
        tx_pos=(
            scalar(runtime.tx.position.x),
            scalar(runtime.tx.position.y),
            scalar(runtime.tx.position.z),
        ),
        diffraction_detail=diffraction_detail,
    )


__all__ = [
    "Coordinates",
    "FieldComponents",
    "FieldGrid",
    "FieldResult",
    "FieldSpec",
    "JonesComponents",
    "VectorComponents",
    "solve_field",
]
