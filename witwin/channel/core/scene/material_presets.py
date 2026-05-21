from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ITU_MATERIALS_PROPERTIES = {
    "concrete": {(1.0, 100.0): (5.24, 0.0, 0.0462, 0.7822)},
    "brick": {(1.0, 40.0): (3.91, 0.0, 0.0238, 0.16)},
    "plasterboard": {(1.0, 100.0): (2.73, 0.0, 0.0085, 0.9395)},
    "wood": {(0.001, 100.0): (1.99, 0.0, 0.0047, 1.0718)},
    "glass": {
        (0.1, 100.0): (6.31, 0.0, 0.0036, 1.3394),
        (220.0, 450.0): (5.79, 0.0, 0.0004, 1.658),
    },
    "ceiling_board": {
        (1.0, 100.0): (1.48, 0.0, 0.0011, 1.0750),
        (220.0, 450.0): (1.52, 0.0, 0.0029, 1.029),
    },
    "chipboard": {(1.0, 100.0): (2.58, 0.0, 0.0217, 0.7800)},
    "plywood": {(1.0, 40.0): (2.71, 0.0, 0.33, 0.0)},
    "marble": {(1.0, 60.0): (7.074, 0.0, 0.0055, 0.9262)},
    "floorboard": {(50.0, 100.0): (3.66, 0.0, 0.0044, 1.3515)},
    "metal": {(1.0, 100.0): (1.0, 0.0, 1.0e7, 0.0)},
    "very_dry_ground": {(1.0, 10.0): (3.0, 0.0, 0.00015, 2.52)},
    "medium_dry_ground": {(1.0, 10.0): (15.0, -0.1, 0.035, 1.63)},
    "wet_ground": {(1.0, 10.0): (30.0, -0.4, 0.15, 1.30)},
}

_ALIASES = {
    "itu_concrete": "concrete",
    "itu_brick": "brick",
    "itu_plasterboard": "plasterboard",
    "itu_wood": "wood",
    "itu_glass": "glass",
    "itu_metal": "metal",
}


@dataclass(frozen=True)
class FrequencyMaterialSample:
    eps_r: Any
    mu_r: Any = 1.0
    sigma_e: Any = 0.0


def normalize_itu_name(name: str) -> str:
    normalized = str(name).lower().replace("-", "_")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in ITU_MATERIALS_PROPERTIES:
        raise ValueError(f"Unknown ITU material '{name}'.")
    return normalized


def evaluate_itu_material(name: str, frequency: float) -> FrequencyMaterialSample:
    normalized = normalize_itu_name(name)
    f_ghz = float(frequency) / 1.0e9
    for (f_min, f_max), (a, b, c, d) in ITU_MATERIALS_PROPERTIES[normalized].items():
        if f_min <= f_ghz <= f_max:
            return FrequencyMaterialSample(
                eps_r=a * (f_ghz ** b),
                mu_r=1.0,
                sigma_e=c * (f_ghz ** d),
            )
    raise ValueError(f"Properties of ITU material '{normalized}' are not defined for {frequency} Hz.")


def install_material_from_itu(Material) -> None:
    if hasattr(Material, "from_itu"):
        return

    @classmethod
    def from_itu(cls, name: str, frequency: float | None = None):
        normalized = normalize_itu_name(name)
        material = cls(name=normalized)
        # Material lives in witwin.core and is frozen=True; use object.__setattr__ here only
        # because we are tagging an extension descriptor onto a foreign immutable dataclass.
        object.__setattr__(material, "itu_descriptor", (normalized, None if frequency is None else float(frequency)))
        return material

    Material.from_itu = from_itu


def material_sample_for_frequency(material, scene_frequency: float | None):
    descriptor = getattr(material, "itu_descriptor", None)
    if descriptor is None:
        return material.evaluate_static()
    name, freq_override = descriptor
    frequency = scene_frequency if freq_override is None else freq_override
    if frequency is None:
        raise ValueError(f"ITU material '{name}' requires Scene.frequency or a frequency override.")
    return evaluate_itu_material(name, float(frequency))


__all__ = [
    "FrequencyMaterialSample",
    "ITU_MATERIALS_PROPERTIES",
    "evaluate_itu_material",
    "install_material_from_itu",
    "material_sample_for_frequency",
    "normalize_itu_name",
]
