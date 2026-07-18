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
    "reflection_diffraction_coupling_topology": "one_reflection_one_diffraction_both_orders",
    "supports_complex_path_coefficients": True,
    "supports_polarization": True,
    "supports_arrays": True,
    "supports_ad": True,
    # Plan 07 completion gate (AD-4b). Honest scope: fixed-topology forward
    # and reverse mode through the frozen discrete winner (path topology,
    # sampling tapes, validity masks, polarization frames). No estimator for
    # visibility / topology discontinuities: path birth/death and shadow
    # transitions are out of contract, and the solvers refuse the excluded
    # combinations below before any launch instead of returning silent zeros.
    "ad_contract": {
        "decision": "fixed_topology_jvp_vjp",
        "public_modes": ["none", "jvp", "vjp"],
        "fixed_topology_jvp": True,
        "fixed_topology_vjp": True,
        "visibility_discontinuity_estimator": False,
        "differentiable_solvers": ["path", "deterministic", "montecarlo_basic"],
        "differentiable_inputs": [
            "material_eps_r",
            "material_sigma_e",
            "material_gain",
            "material_thickness",
            "frequency",
            "tx_position",
            "rx_position",
            "mesh_vertices",
        ],
        # Per-solver refusals (fail loudly before any launch).
        "ad_excluded": {
            "path": ["scattering", "coupled_paths_mesh_vertex"],
            "deterministic": ["scattering", "coupled_paths_mesh_vertex"],
            "montecarlo_basic": ["scattering"],
            "montecarlo_bdpt": ["all"],
        },
        # Wired into the montecarlo.basic LoS Function since AD-3; the
        # standalone facades remain available.
        "low_level_primitives": [
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
            "reflection_diffraction_coupling_topology": "one_reflection_one_diffraction_both_orders",
            "max_reflections_in_coupled_path": 1,
            "reflection_diffraction_coupling_candidate_limit": 1_000_000,
            "supports_complex_path_coefficients": True,
            "supports_polarization": True,
            "supports_arrays": True,
            "supports_ad": True,
            "ad_modes": ["none", "jvp", "vjp"],
            # Fail-loudly refusals: Kirchhoff scattering rows, and mesh
            # vertex gradients through coupled R-D paths (the coupled
            # adjoints take the wall plane / edge tables as frozen winners).
            "ad_excluded": ["scattering", "coupled_paths_mesh_vertex"],
        },
        "deterministic": {
            "max_reflection_depth": 5,
            "max_diffraction_order": 1,
            # Coupled reflection-diffraction on the grid solver (ADR-011). The
            # coupling keys mirror the path solver; the deterministic engine
            # streams coupled discovery over receiver blocks under the same
            # per-block candidate limit.
            "supports_reflection_diffraction_coupling": True,
            "supports_reflection_diffraction_coupling_geometry": True,
            "reflection_diffraction_coupling_topology": "one_reflection_one_diffraction_both_orders",
            "max_reflections_in_coupled_path": 1,
            "reflection_diffraction_coupling_candidate_limit": 1_000_000,
            "supports_complex_path_coefficients": True,
            "supports_polarization": True,
            "supports_arrays": False,
            "supports_ad": True,
            "ad_modes": ["none", "jvp", "vjp"],
            # Mesh vertex gradients through coupled R-D paths are refused (the
            # coupled adjoints take the wall plane / edge tables as frozen
            # winners), matching the path solver.
            "ad_excluded": ["scattering", "coupled_paths_mesh_vertex"],
        },
        "montecarlo_basic": {
            "max_diffraction_order": 1,
            "supports_reflection_diffraction_coupling": False,
            "supports_complex_path_coefficients": False,
            "supports_polarization": False,
            "supports_arrays": False,
            "supports_ad": True,
            "ad_modes": ["none", "jvp", "vjp"],
            "ad_excluded": ["scattering"],
        },
        "montecarlo_bdpt": {
            "max_diffraction_order": 1,
            "supports_reflection_diffraction_coupling": True,
            "supports_complex_path_coefficients": True,
            "supports_polarization": True,
            "supports_arrays": False,
            "field_carrier": "complex3_jones_for_coherent_events",
            "scalar_throughput_role": "sampling_probability_proxy_only",
            "scattering_field_semantics": "incoherent_power_only",
            "supports_sensor_subpaths": False,
            "sensor_depth": "receiver_endpoint_only_always_zero",
            "pdf_measure": "proposal_density_excludes_geometry_jacobian",
            "endpoint_connection_strategies": 1,
            "diffraction_mis_strategies": ["direct", "keller"],
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
