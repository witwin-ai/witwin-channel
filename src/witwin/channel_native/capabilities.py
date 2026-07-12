from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import Any


_CAPABILITIES: dict[str, Any] = {
    "schema_version": 1,
    "components": ["los", "reflection", "diffraction", "transmission", "scattering"],
    # transmission physics is live in all four solvers: endpoint-connection
    # (path, deterministic) and shooting-context Monte Carlo (straight
    # penetration chains; BDPT adds event-selected mixed chains). Kirchhoff
    # scattering is live in all four solvers: deterministic/path use ensemble
    # patch quadrature plus realization_coherent phase screens; MC basic uses
    # an area-sampled diffuse radiomap; BDPT uses a three-way event sampler
    # with NEE connections.
    "component_solver_integration": {
        "transmission": {
            "path": True,
            "deterministic": True,
            "montecarlo_basic": True,
            "montecarlo_bdpt": True,
        },
        "scattering": {
            "path": True,
            "deterministic": True,
            "montecarlo_basic": True,
            "montecarlo_bdpt": True,
        },
    },
    "max_reflection_depth": 5,
    "max_diffraction_order": 1,
    "supports_reflection_diffraction_coupling": True,
    "supports_complex_path_coefficients": True,
    "supports_polarization": True,
    "supports_arrays": True,
    "supports_ad": False,
    "ad_contract": {
        "decision": "primal_only_first_replacement",
        "public_modes": ["none"],
        "fixed_topology_jvp": False,
        "fixed_topology_vjp": False,
        "visibility_discontinuity_estimator": False,
        "experimental_low_level_primitives": [
            "mc_los_path_gain_backward",
            "mc_los_path_gain_jvp",
        ],
    },
    "receiver_types": ["point", "grid"],
    "materials": {
        "abi_version": 3,
        "supports_dispersive_evaluation": True,
        "perfect_conductor_model": "explicit",
        "traceable_material_ids": True,
        "physical_surface": True,
        "layer_csr": True,
        "event_api": {
            "transmission_refraction": False,
            "absorption": False,
            "layered_media": False,
            "rough_scattering": False,
            "tabulated_polarimetric_bsdf": False,
            "medium_stack": False,
            "energy_accounting": False,
        },
        "runtime_material_abi_integration": {
            "path": True,
            "deterministic": True,
            "montecarlo_basic": True,
            "montecarlo_bdpt": True,
        },
        "event_solver_integration": {
            "path": False,
            "deterministic": False,
            "montecarlo_basic": False,
            "montecarlo_bdpt": False,
        },
    },
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
            "ad_modes": ["none"],
        },
        "deterministic": {
            "max_reflection_depth": 5,
            "max_diffraction_order": 1,
            "supports_reflection_diffraction_coupling": False,
            "supports_complex_path_coefficients": True,
            "supports_polarization": True,
            "supports_arrays": False,
            "supports_ad": False,
            "ad_modes": ["none"],
        },
        "montecarlo_basic": {
            "max_diffraction_order": 1,
            "supports_reflection_diffraction_coupling": False,
            "supports_complex_path_coefficients": False,
            "supports_polarization": False,
            "supports_arrays": False,
            "supports_ad": False,
            "ad_modes": ["none"],
        },
        "montecarlo_bdpt": {
            "max_diffraction_order": 1,
            "supports_reflection_diffraction_coupling": True,
            "supports_complex_path_coefficients": True,
            "supports_polarization": True,
            "supports_arrays": False,
            "supports_ad": False,
            "ad_modes": ["none"],
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
