# Copyright Xingyu Chen.
# Solver-level capability manifest for :mod:`witwin.channel`.

"""Solver-level capability manifest for :mod:`witwin.channel`."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import Any, Literal, TypedDict, cast


# BDPT AD scope: the parameters its jvp/vjp companions carry, and the leaves
# it refuses before launch (endpoint positions and mesh vertices).
_BDPT_AD_DIFFERENTIABLE_PARAMETERS = (
    "layer_eps_r",
    "layer_sigma_e",
    "layer_thickness",
    "roughness_sigma_h",
    "roughness_corr_x",
    "roughness_corr_y",
    "bsdf_table_values",
    "phase_screen_heights",
    "frequency",
    "tx_power",
)
_BDPT_AD_EXCLUDED = ("tx_position", "rx_position", "mesh_vertices")

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
    # the AD capability gate (AD). Honest scope: fixed-topology forward
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
        "differentiable_solvers": ["path", "deterministic", "montecarlo_basic", "montecarlo_bdpt"],
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
            # The sampled estimator carries material, frequency and transmitter
            # power derivatives; endpoint-position and mesh-vertex leaves are
            # refused before any launch.
            "montecarlo_bdpt": list(_BDPT_AD_EXCLUDED),
        },
        # Wired into the radiomap.Stochastic LoS Function since AD; the
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
            "reflection_diffraction_coupling_topology": "one_diffraction_anywhere_in_a_reflection_chain",
            # A chain carries up to max_depth reflections and the diffraction.
            "max_reflections_in_coupled_path": 5,
            # coupled_paths=True also enables double diffraction (cid 7), which shares
            # the coupled gate, validator chunk and accumulator slot.
            "coupled_double_diffraction": True,
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
            # Interaction chains on the grid solver; the coupling keys mirror the path solver.
            "supports_reflection_diffraction_coupling": True,
            "supports_reflection_diffraction_coupling_geometry": True,
            "reflection_diffraction_coupling_topology": "one_diffraction_anywhere_in_a_reflection_chain",
            "max_reflections_in_coupled_path": 4,
            # coupled_paths=True includes double diffraction (cid 7) on the grid solver,
            # sharing the coupled gate and slot.
            "coupled_double_diffraction": True,
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
            # coupled double diffraction: BDPT reads the shared enumerated coupled union
            # from the enumerated-path oracle, which now includes cid 7 D->D rows.
            "coupled_double_diffraction": True,
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
            # coherent combination: opt-in coherent combine (DEFAULT OFF). When enabled, the
            # enumerable delta/UTD family (los / reflection / diffraction, plus
            # the coupled compensator folded into diffraction) is summed as a
            # complex phasor per (tx, rx, component) and finalized |sum|^2; the
            # stochastic transmission/scattering samplers have no coherent field
            # and are refused. The coherent path carries the same AD as the power path.
            "coherent_combine": {
                "default": False,
                "combine_domains": ["power", "coherent"],
                "coherent_components": ["los", "reflection", "diffraction"],
                "coherent_excluded": ["transmission", "scattering"],
                "supports_ad": True,
            },
            "supports_ad": True,
            "ad_modes": ["none", "jvp", "vjp"],
            "ad_differentiable_inputs": list(_BDPT_AD_DIFFERENTIABLE_PARAMETERS),
            "ad_geometry": "refused",
            "ad_excluded": list(_BDPT_AD_EXCLUDED),
        },
    },
}


def _propagation_consumer_capabilities() -> dict[str, Any]:
    """Project the consumer contract record into this manifest's shape.

    The values are read from the consumer contract rather than restated, so the
    solver manifest and the cross-package contract cannot drift apart. The
    import is deferred to keep this module free of a package-level dependency
    on ``propagation``.
    """

    from witwin.channel.propagation.consumer import capabilities as consumer

    record = consumer()
    return {
        "contract_version": record.contract_version,
        "components": sorted(record.components),
        "responses": sorted(record.responses),
        "topology_modes": sorted(record.topology_modes),
        "ad_modes": sorted(record.ad_modes),
        "response_components": {
            response: sorted(components)
            for response, components in record.response_components
        },
        "response_ad_modes": {
            response: sorted(modes) for response, modes in record.response_ad_modes
        },
        "component_ad_modes": {
            component: sorted(modes) for component, modes in record.component_ad_modes
        },
        "fixed_topology_components": sorted(record.fixed_topology_components),
        "fixed_topology_responses": sorted(record.fixed_topology_responses),
        "supports_fixed_topology": record.supports_fixed_topology,
        "supports_los_jones": record.supports_los_jones,
    }


def capabilities() -> dict[str, Any]:
    """Return the versioned semantic solver capability manifest.

    ``components`` here is the solver-level set and includes ``scattering``.
    The narrower cross-package field-transport contract is reported under
    ``propagation_consumer`` and is owned by :func:`witwin.channel.propagation.consumer.capabilities`.
    """

    manifest = deepcopy(_CAPABILITIES)
    manifest["propagation_consumer"] = _propagation_consumer_capabilities()
    return manifest


def solver_capabilities(solver: str) -> dict[str, Any]:
    """Return a fresh copy of one solver's entry of the capability manifest.

    Equal to ``capabilities()["solvers"][solver]`` without copying the whole manifest.
    """

    return deepcopy(_CAPABILITIES["solvers"][solver])


class ConfigMetadata(TypedDict):
    """The requested/effective configuration keys every solver publishes in its metadata."""

    requested_config: dict[str, Any]
    effective_config: dict[str, Any]
    requested_max_depth: int
    effective_max_depth: int
    component_max_depth: dict[str, int]


def config_metadata(
    *, requested: dict[str, Any], effective: dict[str, Any], component_max_depth: dict[str, int],
) -> ConfigMetadata:
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


ESTIMATORS = frozenset({"coherent", "stochastic"})
QUANTITIES = frozenset({"point_sample", "cell_average"})
COMBINES = frozenset({"field", "power", "none"})

Estimator = Literal["coherent", "stochastic"]
Quantity = Literal["point_sample", "cell_average"]
Combine = Literal["field", "power", "none"]


class ResultSemantics(TypedDict):
    """The result-semantics keys every solver publishes in its metadata."""

    estimator: Estimator
    quantity: dict[str, Quantity]
    combine: Combine
    complete: bool | None


def estimator_metadata(
    *, estimator: str, quantity: Mapping[str, str], combine: str, complete: bool | None,
) -> ResultSemantics:
    """The result-semantics keys every solver publishes in its metadata.

    ``estimator`` is ``coherent`` (enumerated paths) or ``stochastic`` (sampled paths);
    ``quantity`` maps each requested component to ``point_sample`` (the value at the receiver
    position) or ``cell_average`` (averaged over the receiver's cell or capture region);
    ``combine`` is how rows reach a receiver value (``field``: complex sum, then squared;
    ``power``: sum of row powers; ``none``: rows returned uncombined); ``complete`` is whether
    every path within the requested bounds was found, or ``None`` when the solve does not
    report it.
    """

    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {sorted(ESTIMATORS)}, got {estimator!r}")
    if not isinstance(quantity, Mapping) or not quantity:
        raise TypeError("quantity must be a non-empty mapping from component to quantity")
    for component, value in quantity.items():
        if value not in QUANTITIES:
            raise ValueError(
                f"quantity[{component!r}] must be one of {sorted(QUANTITIES)}, got {value!r}"
            )
    if combine not in COMBINES:
        raise ValueError(f"combine must be one of {sorted(COMBINES)}, got {combine!r}")
    if complete is not None and type(complete) is not bool:
        raise TypeError("complete must be a bool or None")
    # The checks above narrow every value to its vocabulary.
    return {
        "estimator": cast(Estimator, estimator),
        "quantity": cast("dict[str, Quantity]", dict(sorted(quantity.items()))),
        "combine": cast(Combine, combine),
        "complete": complete,
    }


def component_quantities(
    components: Iterable[str], *, cell_average: Iterable[str] = (),
) -> dict[str, Quantity]:
    """``point_sample`` for every component except those listed in ``cell_average``."""

    averaged = frozenset(cell_average)
    return {
        name: "cell_average" if name in averaged else "point_sample" for name in sorted(components)
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


__all__ = [
    "Combine",
    "ConfigMetadata",
    "Estimator",
    "Quantity",
    "ResultSemantics",
    "capabilities",
    "config_metadata",
    "component_quantities",
    "estimator_metadata",
    "serialize_config",
    "solver_capabilities",
]
