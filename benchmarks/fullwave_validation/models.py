from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_NAME = "witwin.channel_native.fullwave-reference"
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    kind: str
    eps_r: float = 4.0
    sigma_e: float = 0.01

    def __post_init__(self) -> None:
        if self.kind not in {"metal", "dielectric"}:
            raise ValueError("material kind must be 'metal' or 'dielectric'")
        if self.eps_r <= 0.0:
            raise ValueError("eps_r must be positive")
        if self.sigma_e < 0.0:
            raise ValueError("sigma_e must be non-negative")


@dataclass(frozen=True, slots=True)
class CaseSpec:
    scenario: str
    material: MaterialSpec
    frequency_hz: float
    analysis_bounds_xy: tuple[tuple[float, float], tuple[float, float]]
    domain_bounds_xyz: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    plane_z: float
    tx_position: tuple[float, float, float]
    cube_centers: tuple[tuple[float, float, float], ...]
    cube_size_m: float
    receiver_shape: tuple[int, int]
    fullwave_dl_m: float
    fullwave_pml_layers: int
    max_depth: int
    source_layout: str

    def __post_init__(self) -> None:
        if self.scenario not in {"single_cube", "three_cube", "three_cube_320"}:
            raise ValueError(
                "scenario must be 'single_cube', 'three_cube', or 'three_cube_320'"
            )
        expected_cubes = 1 if self.scenario == "single_cube" else 3
        if len(self.cube_centers) != expected_cubes:
            raise ValueError(f"{self.scenario} requires {expected_cubes} cube centers")
        if self.frequency_hz <= 0.0 or self.cube_size_m <= 0.0:
            raise ValueError("frequency_hz and cube_size_m must be positive")
        if min(self.receiver_shape) < 2 or self.fullwave_dl_m <= 0.0:
            raise ValueError("receiver_shape and fullwave_dl_m must be positive")
        if self.fullwave_pml_layers <= 0:
            raise ValueError("fullwave_pml_layers must be positive")
        self._validate_fullwave_interior()

    def _validate_fullwave_interior(self) -> None:
        thickness = self.fullwave_dl_m * self.fullwave_pml_layers
        interior = tuple((lo + thickness, hi - thickness) for lo, hi in self.domain_bounds_xyz)
        if any(lo >= hi for lo, hi in interior):
            raise ValueError("PML layers consume the full-wave domain")

        for axis, (analysis_lo, analysis_hi) in enumerate(self.analysis_bounds_xy):
            interior_lo, interior_hi = interior[axis]
            if analysis_lo < interior_lo or analysis_hi > interior_hi:
                raise ValueError("analysis_bounds_xy overlaps the full-wave PML")
        for value, (interior_lo, interior_hi) in zip(self.tx_position, interior, strict=True):
            if not interior_lo < value < interior_hi:
                raise ValueError("tx_position lies inside or on the full-wave PML")
        if not interior[2][0] < self.plane_z < interior[2][1]:
            raise ValueError("plane_z lies inside or on the full-wave PML")

        half_size = self.cube_size_m / 2.0
        for center in self.cube_centers:
            for value, (interior_lo, interior_hi) in zip(center, interior, strict=True):
                if value - half_size < interior_lo or value + half_size > interior_hi:
                    raise ValueError("cube geometry overlaps the full-wave PML")

    @property
    def case_id(self) -> str:
        return f"{self.scenario}-{self.material.kind}"

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def x(self) -> np.ndarray:
        return _cell_center_axis(self.analysis_bounds_xy[0], self.receiver_shape[1])

    @property
    def y(self) -> np.ndarray:
        return _cell_center_axis(self.analysis_bounds_xy[1], self.receiver_shape[0])


def _cell_center_axis(bounds: tuple[float, float], count: int) -> np.ndarray:
    lo, hi = bounds
    step = (hi - lo) / count
    return lo + step * (np.arange(count, dtype=np.float64) + 0.5)


@dataclass(frozen=True, slots=True)
class FieldMap:
    x: np.ndarray
    y: np.ndarray
    field: np.ndarray
    components: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        values = np.asarray(self.field, dtype=np.complex128)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be one-dimensional")
        if values.shape != (y.size, x.size):
            raise ValueError(
                f"field must have shape (len(y), len(x)); got {values.shape}"
            )
        if (
            x.size < 2
            or y.size < 2
            or np.any(np.diff(x) <= 0)
            or np.any(np.diff(y) <= 0)
        ):
            raise ValueError(
                "x and y must be strictly increasing with at least two samples"
            )
        normalized_components: dict[str, np.ndarray] = {}
        for name, component in self.components.items():
            array = np.asarray(component, dtype=np.complex128)
            if array.shape != values.shape:
                raise ValueError(f"component {name!r} shape does not match field")
            normalized_components[str(name)] = array
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "field", values)
        object.__setattr__(self, "components", normalized_components)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "schema_name": np.asarray(SCHEMA_NAME),
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
            "x": self.x,
            "y": self.y,
            "field": self.field,
            "metadata_json": np.asarray(
                json.dumps(self.metadata, sort_keys=True, separators=(",", ":"))
            ),
        }
        arrays.update(
            {f"component_{name}": value for name, value in self.components.items()}
        )
        with output.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        return output

    @classmethod
    def load(cls, path: str | Path) -> FieldMap:
        with np.load(Path(path), allow_pickle=False) as data:
            if str(data["schema_name"].item()) != SCHEMA_NAME:
                raise ValueError("not a channel-native full-wave reference")
            if int(data["schema_version"].item()) != SCHEMA_VERSION:
                raise ValueError("unsupported full-wave reference schema version")
            components = {
                key.removeprefix("component_"): data[key]
                for key in data.files
                if key.startswith("component_")
            }
            return cls(
                x=data["x"],
                y=data["y"],
                field=data["field"],
                components=components,
                metadata=json.loads(str(data["metadata_json"].item())),
            )
