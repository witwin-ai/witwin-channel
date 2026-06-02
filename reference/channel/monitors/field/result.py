from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


def _mapping_proxy(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


def _complex_component_map(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({str(key): component for key, component in dict(value).items()})


@dataclass(frozen=True)
class MonitorCoordinates:
    """Coordinate arrays for a traced field monitor."""

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
class MonitorField:
    """Scalar complex field components for a traced field monitor."""

    los: object
    reflection: object
    diffraction_direct: object
    diffraction_mixed: object
    diffraction: object
    total: object


@dataclass(frozen=True)
class MonitorJones:
    """Jones-vector field components for a traced field monitor."""

    los: Mapping[str, object]
    reflection: Mapping[str, object]
    diffraction_direct: Mapping[str, object]
    diffraction_mixed: Mapping[str, object]
    diffraction: Mapping[str, object]
    total: Mapping[str, object]


@dataclass(frozen=True)
class MonitorVector:
    """World-vector field components for a traced field monitor."""

    los: Mapping[str, object]
    reflection: Mapping[str, object]
    diffraction_direct: Mapping[str, object]
    diffraction_mixed: Mapping[str, object]
    diffraction: Mapping[str, object]
    total: Mapping[str, object]


@dataclass(frozen=True)
class MonitorResult:
    """Structured result payload for a single traced field monitor."""

    name: str
    kind: str
    axis: str
    plane_position: float
    ray_mode: str
    bounds: tuple[tuple[float, float], tuple[float, float]]
    grid_shape: tuple[int, int]
    coords: MonitorCoordinates
    field: MonitorField
    vector: MonitorVector
    jones: MonitorJones
    metadata: Mapping[str, object]
    tx_pos: tuple[float, float, float]
    diffraction_detail: Mapping[str, object] | None = None
    timing: Mapping[str, object] | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "MonitorResult":
        diffraction_detail = payload.get("diffraction_detail")
        timing = payload.get("timing")
        coords_payload = payload["coords"]
        metadata_payload = payload.get("metadata", {})
        receiver_sampling = metadata_payload.get("receiver_sampling", {})
        grid_shape = tuple(int(value) for value in payload["grid_shape"])
        tangential_axes = coords_payload.get("tangential_axes", receiver_sampling.get("tangential_axes"))
        axis_x = coords_payload.get("axis_x")
        axis_y = coords_payload.get("axis_y")
        if tangential_axes is not None and len(tangential_axes) == 2:
            if axis_x is None:
                axis_x = tangential_axes[0]
            if axis_y is None:
                axis_y = tangential_axes[1]
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            axis=str(payload["axis"]),
            plane_position=float(payload["plane_position"]),
            ray_mode=str(payload["ray_mode"]),
            bounds=(
                tuple(float(value) for value in payload["bounds"][0]),
                tuple(float(value) for value in payload["bounds"][1]),
            ),
            grid_shape=grid_shape,
            coords=MonitorCoordinates(
                grid_x=coords_payload["grid_x"],
                grid_y=coords_payload["grid_y"],
                x=coords_payload["x"],
                y=coords_payload["y"],
                axis_x="x" if axis_x is None else str(axis_x),
                axis_y="y" if axis_y is None else str(axis_y),
            ),
            field=MonitorField(
                los=payload["field"]["los"],
                reflection=payload["field"]["reflection"],
                diffraction_direct=payload["field"]["diffraction_direct"],
                diffraction_mixed=payload["field"]["diffraction_mixed"],
                diffraction=payload["field"]["diffraction"],
                total=payload["field"]["total"],
            ),
            vector=MonitorVector(
                los=_complex_component_map(payload["vector"]["los"]),
                reflection=_complex_component_map(payload["vector"]["reflection"]),
                diffraction_direct=_complex_component_map(payload["vector"]["diffraction_direct"]),
                diffraction_mixed=_complex_component_map(payload["vector"]["diffraction_mixed"]),
                diffraction=_complex_component_map(payload["vector"]["diffraction"]),
                total=_complex_component_map(payload["vector"]["total"]),
            ),
            jones=MonitorJones(
                los=_complex_component_map(payload["jones"]["los"]),
                reflection=_complex_component_map(payload["jones"]["reflection"]),
                diffraction_direct=_complex_component_map(payload["jones"]["diffraction_direct"]),
                diffraction_mixed=_complex_component_map(payload["jones"]["diffraction_mixed"]),
                diffraction=_complex_component_map(payload["jones"]["diffraction"]),
                total=_complex_component_map(payload["jones"]["total"]),
            ),
            metadata=_mapping_proxy(metadata_payload),
            tx_pos=tuple(float(value) for value in payload["tx_pos"]),
            diffraction_detail=(
                None
                if diffraction_detail is None
                else _mapping_proxy(diffraction_detail)
            ),
            timing=None if timing is None else _mapping_proxy(timing),
        )

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
    def primary(self) -> "MonitorResult":
        return self


__all__ = [
    "MonitorCoordinates",
    "MonitorField",
    "MonitorJones",
    "MonitorResult",
    "MonitorVector",
]
