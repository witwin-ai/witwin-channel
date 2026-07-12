from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import Any


_CAPABILITIES: dict[str, Any] = {
    "schema_version": 1,
    "components": ["los", "reflection", "diffraction"],
    "max_reflection_depth": 5,
    "max_diffraction_order": 1,
    "supports_reflection_diffraction_coupling": True,
    "supports_complex_path_coefficients": True,
    "supports_polarization": True,
    "supports_arrays": True,
    "supports_ad": False,
    "receiver_types": ["point", "grid"],
    "solvers": {
        "path": {
            "max_reflection_depth": 5,
            "max_diffraction_order": 1,
            "supports_reflection_diffraction_coupling": True,
            "supports_reflection_diffraction_coupling_geometry": True,
            "reflection_diffraction_coupling_candidate_limit": 1_000_000,
            "supports_complex_path_coefficients": True,
            "supports_polarization": True,
            "supports_arrays": True,
            "supports_ad": False,
        },
        "deterministic": {
            "max_reflection_depth": 5,
            "max_diffraction_order": 1,
            "supports_reflection_diffraction_coupling": False,
            "supports_complex_path_coefficients": True,
            "supports_polarization": True,
            "supports_arrays": False,
            "supports_ad": False,
        },
        "montecarlo_basic": {
            "max_diffraction_order": 1,
            "supports_reflection_diffraction_coupling": False,
            "supports_complex_path_coefficients": False,
            "supports_polarization": False,
            "supports_arrays": False,
            "supports_ad": False,
        },
        "montecarlo_bdpt": {
            "max_diffraction_order": 1,
            "supports_reflection_diffraction_coupling": False,
            "supports_complex_path_coefficients": False,
            "supports_polarization": False,
            "supports_arrays": False,
            "supports_ad": False,
        },
    },
}


def capabilities() -> dict[str, Any]:
    """Return the versioned semantic capability manifest."""

    return deepcopy(_CAPABILITIES)


def config_metadata(
    *,
    requested: dict[str, Any],
    effective: dict[str, Any],
    component_max_depth: dict[str, int],
) -> dict[str, Any]:
    """Build the common requested/effective solver metadata contract."""

    return {
        "requested_config": requested,
        "effective_config": effective,
        "requested_max_depth": int(requested["max_depth"]),
        "effective_max_depth": int(effective["max_depth"]),
        "component_max_depth": {
            key: int(value) for key, value in component_max_depth.items()
        },
    }


def serialize_config(config: object) -> dict[str, Any]:
    """Serialize every dataclass config field to JSON-compatible values."""

    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError("config must be a dataclass instance")
    serialized: dict[str, Any] = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, (set, frozenset)):
            value = sorted(value)
        elif isinstance(value, tuple):
            value = list(value)
        serialized[field.name] = value
    return serialized


__all__ = ["capabilities", "config_metadata", "serialize_config"]
