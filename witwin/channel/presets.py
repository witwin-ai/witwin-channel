# Copyright Xingyu Chen.
# Builds ITU-R P.2040 radio materials and TR 38.901/dipole antenna patterns as Core values.

"""ITU-R P.2040 material presets and TR 38.901/dipole antenna patterns as ``witwin.core`` values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal

import torch

from witwin.core import AntennaPattern, PhysicalMaterial, PowerLawDispersion

__all__ = [
    "ITU_MATERIALS",
    "ItuBand",
    "ItuBandDispersion",
    "dipole_pattern",
    "itu_material",
    "itu_material_names",
    "itu_properties",
    "tr38901_pattern",
]

_GHZ = 1.0e9


@dataclass(frozen=True, slots=True)
class ItuBand:
    """One ITU-R P.2040 row: ``eps_r = a * f_GHz**b`` and ``sigma = c * f_GHz**d`` S/m."""

    min_frequency_hz: float
    max_frequency_hz: float
    a: float
    b: float
    c: float
    d: float

    def contains(self, frequency_hz: float) -> bool:
        return self.min_frequency_hz <= frequency_hz <= self.max_frequency_hz

    def evaluate(self, frequency_hz: float) -> tuple[float, float]:
        f_ghz = frequency_hz / _GHZ
        return self.a * f_ghz**self.b, self.c * f_ghz**self.d


def _band(lo_ghz: float, hi_ghz: float, a: float, b: float, c: float, d: float) -> ItuBand:
    return ItuBand(lo_ghz * _GHZ, hi_ghz * _GHZ, a, b, c, d)


# ITU-R P.2040-3 Table 3. Rows of one material never overlap, so a frequency
# selects at most one row.
ITU_MATERIALS: Mapping[str, tuple[ItuBand, ...]] = MappingProxyType(
    {
        "concrete": (_band(1.0, 100.0, 5.24, 0.0, 0.0462, 0.7822),),
        "brick": (_band(1.0, 40.0, 3.91, 0.0, 0.0238, 0.16),),
        "plasterboard": (_band(1.0, 100.0, 2.73, 0.0, 0.0085, 0.9395),),
        "wood": (_band(0.001, 100.0, 1.99, 0.0, 0.0047, 1.0718),),
        "glass": (
            _band(0.1, 100.0, 6.31, 0.0, 0.0036, 1.3394),
            _band(220.0, 450.0, 5.79, 0.0, 0.0004, 1.658),
        ),
        "ceiling_board": (
            _band(1.0, 100.0, 1.48, 0.0, 0.0011, 1.075),
            _band(220.0, 450.0, 1.52, 0.0, 0.0029, 1.029),
        ),
        "chipboard": (_band(1.0, 100.0, 2.58, 0.0, 0.0217, 0.78),),
        "plywood": (_band(1.0, 40.0, 2.71, 0.0, 0.33, 0.0),),
        "marble": (_band(1.0, 60.0, 7.074, 0.0, 0.0055, 0.9262),),
        "floorboard": (_band(50.0, 100.0, 3.66, 0.0, 0.0044, 1.3515),),
        "metal": (_band(1.0, 100.0, 1.0, 0.0, 1.0e7, 0.0),),
        "very_dry_ground": (_band(1.0, 10.0, 3.0, 0.0, 0.00015, 2.52),),
        "medium_dry_ground": (_band(1.0, 10.0, 15.0, -0.1, 0.035, 1.63),),
        "wet_ground": (_band(1.0, 10.0, 30.0, -0.4, 0.15, 1.30),),
    }
)


def itu_material_names() -> tuple[str, ...]:
    """Return the sorted ITU-R P.2040 material names accepted by :func:`itu_material`."""

    return tuple(sorted(ITU_MATERIALS))


def _canonical_name(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key.startswith("itu_"):
        key = key[4:]
    if key not in ITU_MATERIALS:
        raise ValueError(
            f"unknown ITU-R P.2040 material {name!r}; known materials: "
            f"{', '.join(itu_material_names())}"
        )
    return key


def _format_bands(bands: tuple[ItuBand, ...]) -> str:
    return ", ".join(
        f"{band.min_frequency_hz / _GHZ:g}-{band.max_frequency_hz / _GHZ:g} GHz" for band in bands
    )


def _frequency_span(frequency_hz: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(frequency_hz, tuple):
        if len(frequency_hz) != 2:
            raise ValueError("frequency_hz band must be a (low_hz, high_hz) pair")
        low, high = float(frequency_hz[0]), float(frequency_hz[1])
    elif isinstance(frequency_hz, bool) or not isinstance(frequency_hz, (int, float)):
        raise TypeError("frequency_hz must be a positive float or a (low_hz, high_hz) pair")
    else:
        low = high = float(frequency_hz)
    if not (math.isfinite(low) and math.isfinite(high)) or low <= 0.0 or high < low:
        raise ValueError("frequency_hz must be finite and positive, with low_hz <= high_hz")
    return low, high


def _select_band(key: str, low: float, high: float) -> ItuBand:
    bands = ITU_MATERIALS[key]
    for band in bands:
        if band.contains(low) and band.contains(high):
            return band
    span = f"{low / _GHZ:g} GHz" if low == high else f"{low / _GHZ:g}-{high / _GHZ:g} GHz"
    raise ValueError(
        f"ITU-R P.2040 material {key!r} is defined for {_format_bands(bands)}; "
        f"{span} is outside every valid range"
    )


def itu_properties(name: str, frequency_hz: float) -> tuple[float, float]:
    """Return ``(eps_r, sigma_e)`` of ITU material ``name`` at ``frequency_hz``.

    Raises ``ValueError`` for an unknown name or a frequency outside the
    material's ITU-R P.2040 validity range.
    """

    key = _canonical_name(name)
    low, high = _frequency_span(frequency_hz)
    return _select_band(key, low, high).evaluate(low)


@dataclass(frozen=True)
class ItuBandDispersion(PowerLawDispersion):
    """Core power-law dispersion that also enforces one ITU-R P.2040 validity range.

    ``scale`` raises ``ValueError`` for a Python-scalar frequency outside
    ``[min_frequency_hz, max_frequency_hz]``. A tensor frequency is not read on
    the host; :func:`itu_material` validated the declared band instead.
    """

    min_frequency_hz: float = 0.0
    max_frequency_hz: float = math.inf
    material: str = ""

    def scale(self, frequency_hz: Any) -> tuple[Any, Any]:
        if isinstance(frequency_hz, Real) and not (
            self.min_frequency_hz <= float(frequency_hz) <= self.max_frequency_hz
        ):
            raise ValueError(
                f"ITU-R P.2040 material {self.material!r} is defined for "
                f"{self.min_frequency_hz / _GHZ:g}-{self.max_frequency_hz / _GHZ:g} GHz; "
                f"{float(frequency_hz) / _GHZ:g} GHz is outside that range"
            )
        scales: tuple[Any, Any] = super().scale(frequency_hz)
        return scales


def itu_material(
    name: str, frequency_hz: float | tuple[float, float], *, dispersive: bool = True,
    thickness_m: float = 0.1, material_name: str | None = None, **material_kwargs: Any,
) -> PhysicalMaterial:
    """Return an ITU-R P.2040 ``PhysicalMaterial`` valid at ``frequency_hz``.

    ``frequency_hz`` is the frequency, or ``(low_hz, high_hz)`` band, the material
    will be compiled at; it must lie inside one ITU validity range. With
    ``dispersive=True`` the material carries the ITU power law as a Core
    ``PowerLawDispersion`` (``eps_r = a * f_GHz**b``, ``sigma = c * f_GHz**d``),
    so ``scene.compile`` evaluates it at the compile frequency and rejects a
    frequency outside the range. With ``dispersive=False`` the law is evaluated
    once at ``frequency_hz`` and the material is frequency independent, which
    wideband offset replay accepts. Remaining keyword arguments go to
    ``PhysicalMaterial`` (for example ``scattering_coefficient``).
    """

    key = _canonical_name(name)
    low, high = _frequency_span(frequency_hz)
    band = _select_band(key, low, high)
    resolved_name = material_name if material_name is not None else f"itu_{key}"
    if not dispersive:
        if low != high:
            raise ValueError("dispersive=False needs one frequency_hz, not a band")
        eps_r, sigma_e = band.evaluate(low)
        return PhysicalMaterial(
            eps_r=eps_r, sigma_e=sigma_e, name=resolved_name, thickness_m=thickness_m,
            **material_kwargs,
        )
    dispersion = ItuBandDispersion(
        reference_frequency_hz=_GHZ, eps_r_exponent=band.b, sigma_e_exponent=band.d,
        min_frequency_hz=band.min_frequency_hz, max_frequency_hz=band.max_frequency_hz,
        material=key,
    )
    return PhysicalMaterial(
        eps_r=band.a, sigma_e=band.c, name=resolved_name, thickness_m=thickness_m,
        dispersion=dispersion, **material_kwargs,
    )


def _spherical_deg(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    horizontal = torch.sqrt(direction[..., 0].square() + direction[..., 1].square())
    theta = torch.rad2deg(torch.atan2(horizontal, direction[..., 2]))
    phi = torch.rad2deg(torch.atan2(direction[..., 1], direction[..., 0]))
    return theta, phi


@dataclass(frozen=True, slots=True)
class _Tr38901Element:
    max_gain_dbi: float
    beamwidth_deg: float
    side_lobe_db: float

    def __call__(self, direction: torch.Tensor) -> torch.Tensor:
        theta, phi = _spherical_deg(direction)
        vertical = -torch.clamp(
            12.0 * ((theta - 90.0) / self.beamwidth_deg).square(), max=self.side_lobe_db
        )
        horizontal = -torch.clamp(
            12.0 * (phi / self.beamwidth_deg).square(), max=self.side_lobe_db
        )
        attenuation = -torch.clamp(-(vertical + horizontal), max=self.side_lobe_db)
        power = torch.pow(10.0, (self.max_gain_dbi + attenuation) / 10.0)
        return torch.sqrt(power).to(torch.complex64)


@dataclass(frozen=True, slots=True)
class _Dipole:
    kind: str
    directivity: float

    def __call__(self, direction: torch.Tensor) -> torch.Tensor:
        cos_theta = direction[..., 2].clamp(-1.0, 1.0)
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta.square(), min=0.0))
        if self.kind == "short":
            shape = sin_theta
        else:
            tiny = torch.finfo(sin_theta.dtype).tiny
            shape = torch.where(
                sin_theta > 1.0e-6,
                torch.cos(0.5 * math.pi * cos_theta) / sin_theta.clamp_min(tiny),
                torch.zeros_like(sin_theta),
            )
        return (math.sqrt(self.directivity) * shape).to(torch.complex64)


def tr38901_pattern(
    *, max_gain_dbi: float = 8.0, beamwidth_deg: float = 65.0, side_lobe_db: float = 30.0,
) -> AntennaPattern:
    """Return the 3GPP TR 38.901 Table 7.3-1 element as a Core custom antenna pattern.

    Boresight is the endpoint-local ``+x`` axis and zenith is local ``+z``; orient
    the element with ``AntennaState.orientation``. The response is the scalar
    field amplitude ``sqrt(10**(G/10))`` with peak gain ``max_gain_dbi``;
    polarization comes from ``AntennaState.polarization``.
    """

    for value, label in ((beamwidth_deg, "beamwidth_deg"), (side_lobe_db, "side_lobe_db")):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{label} must be finite and positive")
    if not math.isfinite(max_gain_dbi):
        raise ValueError("max_gain_dbi must be finite")
    return AntennaPattern(
        kind="custom",
        custom=_Tr38901Element(float(max_gain_dbi), float(beamwidth_deg), float(side_lobe_db)),
    )


_DIPOLE_DIRECTIVITY = {"short": 1.5, "half_wave": 1.6409}


def dipole_pattern(kind: Literal["short", "half_wave"] = "half_wave") -> AntennaPattern:
    """Return a z-oriented dipole as a Core custom pattern normalized to its directivity.

    ``"short"`` is ``sqrt(1.5) * sin(theta)`` (1.76 dBi); ``"half_wave"`` is
    ``sqrt(1.641) * cos(pi/2 cos(theta)) / sin(theta)`` (2.15 dBi). Core's
    built-in ``"vertical"`` pattern is the short-dipole shape with unit peak.
    """

    if kind not in _DIPOLE_DIRECTIVITY:
        raise ValueError(f"kind must be one of {sorted(_DIPOLE_DIRECTIVITY)}")
    return AntennaPattern(kind="custom", custom=_Dipole(kind, _DIPOLE_DIRECTIVITY[kind]))
