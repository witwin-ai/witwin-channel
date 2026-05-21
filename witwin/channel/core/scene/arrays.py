from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import drjit as dr
from witwin.channel import types as wt

from witwin.channel.core.runtime import to_point3f


_PATTERNS = {"iso", "dipole", "tr38901"}
_POL_PRESETS = {
    "v": ((1.0, 0.0, 0.0),),
    "h": ((0.0, 1.0, 0.0),),
    "vh": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "cross": (
        (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0),
        (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0),
    ),
}


def _coerce_pattern(pattern: str) -> str:
    value = str(pattern).lower()
    if value not in _PATTERNS:
        raise ValueError(f"pattern must be one of {sorted(_PATTERNS)}; got {pattern!r}.")
    return value


def _coerce_polarization_slots(polarization) -> tuple[tuple[complex, complex, complex], ...]:
    if isinstance(polarization, str):
        slots = _POL_PRESETS.get(polarization.lower())
        if slots is None:
            raise ValueError(f"polarization must be one of {sorted(_POL_PRESETS)} or a sequence of complex 3-vectors.")
        return tuple(tuple(complex(v) for v in slot) for slot in slots)
    if len(polarization) == 3 and not any(isinstance(item, (list, tuple)) for item in polarization):
        polarization = (polarization,)
    slots = []
    for slot in polarization:
        if len(slot) != 3:
            raise ValueError("Each polarization slot must contain exactly three components.")
        slots.append(tuple(complex(value) for value in slot))
    if not slots:
        raise ValueError("polarization must provide at least one slot.")
    return tuple(slots)


def _repeat_point3(value: wt.Point3f, width: int, *, role: str) -> wt.Point3f:
    current = int(dr.width(value.x))
    if current == width:
        return value
    if current == 1:
        return wt.Point3f(dr.repeat(value.x, width), dr.repeat(value.y, width), dr.repeat(value.z, width))
    raise ValueError(f"{role} must be scalar or contain {width} entries; got {current}.")


def _coerce_orientations(value, width: int):
    if value is None:
        return None
    return _repeat_point3(to_point3f(value, role="element_orientations"), width, role="element_orientations")


def _concat_float_exprs(values) -> wt.Float:
    return dr.concat([wt.Float(value) for value in values]) if values else dr.zeros(wt.Float, 0)


@dataclass(slots=True)
class AntennaArray:
    """Scene-level antenna array with optional per-element orientation."""

    element_positions: wt.Point3f
    element_orientations: wt.Point3f | None = None
    polarization_slots: tuple[tuple[complex, complex, complex], ...] = ((1.0 + 0.0j, 0.0j, 0.0j),)
    pattern: str = "iso"

    def __init__(
        self,
        *,
        element_positions,
        element_orientations=None,
        polarization="V",
        pattern: str = "iso",
    ) -> None:
        positions = to_point3f(element_positions, role="element_positions", allow_single=False)
        width = int(dr.width(positions.x))
        if width <= 0:
            raise ValueError("AntennaArray requires at least one element.")
        self.element_positions = positions
        self.element_orientations = _coerce_orientations(element_orientations, width)
        self.polarization_slots = _coerce_polarization_slots(polarization)
        self.pattern = _coerce_pattern(pattern)

    @property
    def num_elements(self) -> int:
        return int(dr.width(self.element_positions.x))

    @property
    def num_polarization_slots(self) -> int:
        return len(self.polarization_slots)

    @property
    def num_ant(self) -> int:
        return self.num_elements * self.num_polarization_slots


class PlanarArray(AntennaArray):
    """Uniform rectangular array in the local x/y plane, centered at origin."""

    def __init__(
        self,
        *,
        num_rows: int,
        num_cols: int,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        element_orientations=None,
        polarization="V",
        pattern: str = "iso",
    ) -> None:
        rows, cols = int(num_rows), int(num_cols)
        if rows <= 0 or cols <= 0:
            raise ValueError("num_rows and num_cols must be positive.")
        x_values, y_values, z_values = [], [], []
        row_center = 0.5 * float(rows - 1)
        col_center = 0.5 * float(cols - 1)
        for row in range(rows):
            for col in range(cols):
                x_values.append((float(col) - col_center) * horizontal_spacing)
                y_values.append((float(row) - row_center) * vertical_spacing)
                z_values.append(0.0)
        positions = wt.Point3f(
            _concat_float_exprs(x_values),
            _concat_float_exprs(y_values),
            _concat_float_exprs(z_values),
        )
        self.num_rows = rows
        self.num_cols = cols
        self.vertical_spacing = vertical_spacing
        self.horizontal_spacing = horizontal_spacing
        super().__init__(
            element_positions=positions,
            element_orientations=element_orientations,
            polarization=polarization,
            pattern=pattern,
        )


class UPA(PlanarArray):
    """Uniform planar array convenience constructor."""


class ULA(PlanarArray):
    """Uniform linear array convenience constructor."""

    def __init__(
        self,
        *,
        num_elements: int,
        spacing=0.5,
        axis: str = "x",
        element_orientations=None,
        polarization="V",
        pattern: str = "iso",
    ) -> None:
        axis_name = str(axis).lower()
        if axis_name == "x":
            super().__init__(
                num_rows=1,
                num_cols=int(num_elements),
                vertical_spacing=0.0,
                horizontal_spacing=spacing,
                element_orientations=element_orientations,
                polarization=polarization,
                pattern=pattern,
            )
            return
        if axis_name == "y":
            super().__init__(
                num_rows=int(num_elements),
                num_cols=1,
                vertical_spacing=spacing,
                horizontal_spacing=0.0,
                element_orientations=element_orientations,
                polarization=polarization,
                pattern=pattern,
            )
            return
        raise ValueError("ULA axis must be 'x' or 'y'.")


def default_array() -> AntennaArray:
    return AntennaArray(element_positions=[(0.0, 0.0, 0.0)], pattern="iso", polarization="V")


__all__ = ["AntennaArray", "PlanarArray", "ULA", "UPA", "default_array"]
