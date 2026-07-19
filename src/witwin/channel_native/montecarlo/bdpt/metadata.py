from __future__ import annotations

from typing import Any

from witwin.channel_native.core.kernels.metadata import AdLaunchLedger, make_metadata
from witwin.channel_native.capabilities import (
    capabilities,
    config_metadata,
    serialize_config,
)
from witwin.channel_native.core.components import component_availability_status

from .config import Config


# ADR-022 differentiable-parameter inventory: the parameters BDPT AD carries
# gradients for, per estimator block. Reported in metadata so a caller can see
# exactly what is on the graph and what stays frozen. Geometry is differentiable
# only for the enumerated discrete blocks (fixed-winner endpoints / mesh
# vertices, inherited from the enumerated engine); the stochastic sampler keeps
# hit geometry frozen (ad_geometry='enumerated_blocks_only').
_AD_DIFFERENTIABLE_PARAMETERS = (
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
_AD_GEOMETRY_SCOPE = "enumerated_blocks_only"


_KERNEL_ACCUMULATION = {
    "atomic": "atomic_add",
    "staged": "cell_reduce",
    "compact": "compact_atomic_add",
}

# BDPT per-path component_mask bit scheme (see subpaths.component_mask).
# Transmitted subpaths set bit 8 (delta specular transmission events);
# scattered subpaths set bit 16 (continuous Kirchhoff scattering events).
COMPONENT_MASK_LOS = 1
COMPONENT_MASK_REFLECTION = 2
COMPONENT_MASK_DIFFRACTION = 4
COMPONENT_MASK_TRANSMISSION = 8
COMPONENT_MASK_SCATTERING = 16
_COMPONENT_MASK_BITS = {
    "los": COMPONENT_MASK_LOS,
    "reflection": COMPONENT_MASK_REFLECTION,
    "diffraction": COMPONENT_MASK_DIFFRACTION,
    "transmission": COMPONENT_MASK_TRANSMISSION,
    "scattering": COMPONENT_MASK_SCATTERING,
}


def select_accumulation_strategy(
    config: Config, *, grid_cells: int, estimated_valid_ratio: float
) -> str:
    if config.accumulation_strategy != "auto":
        return config.accumulation_strategy
    if grid_cells <= 0:
        return "atomic"
    samples_per_cell = config.samples / float(grid_cells)
    if estimated_valid_ratio < 0.1:
        return "compact"
    if samples_per_cell >= 64:
        return "staged"
    return "atomic"


def component_status(
    *,
    config: Config,
    reflection_available: bool,
    diffraction_available: bool,
) -> dict[str, str]:
    return component_availability_status(
        config.components,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        reflection_error="BDPT reflection requires RayDN native capability",
        diffraction_error="BDPT diffraction requires RayDN native capability",
    )


def _ad_launch_accounting(
    config: Config, ad_ledger: AdLaunchLedger | None
) -> tuple[int, int, int]:
    """ADR-022 companion accounting: backward/jvp launch counts and tape bytes.

    ad_mode='none' wires no companions and retains no tape (bitwise default).
    Under jvp/vjp report the companion launches this solve registered in the
    AdLaunchLedger, exactly as montecarlo.basic does."""

    ledger = ad_ledger if ad_ledger is not None else AdLaunchLedger()
    backward_launch_count = ledger.launches if config.ad_mode == "vjp" else 0
    jvp_launch_count = ledger.launches if config.ad_mode == "jvp" else 0
    tape_bytes = ledger.tape_bytes if config.ad_mode == "vjp" else 0
    return backward_launch_count, jvp_launch_count, tape_bytes


def _component_max_depth(
    config: Config, effective_max_depth: int
) -> dict[str, int]:
    depth = int(effective_max_depth)
    return {
        "los": 0 if "los" in config.components else -1,
        "reflection": depth if "reflection" in config.components else -1,
        "diffraction": min(1, depth) if "diffraction" in config.components else -1,
        # transmission chains are capped like reflection; scattering is
        # single-bounce in v1 and carries zero paths until its wave.
        "transmission": depth if "transmission" in config.components else -1,
        "scattering": min(1, depth) if "scattering" in config.components else -1,
    }


def make_solver_metadata(
    *,
    config: Config,
    selected_accumulation_strategy: str,
    path_counts_by_strategy: dict[str, int],
    valid_contribution_count: int,
    reflection_available: bool,
    diffraction_available: bool,
    cuda_available: bool,
    optix_available: bool,
    workspace_bytes: int,
    variance_enabled: bool,
    launch_count: int,
    effective_max_depth: int,
    ad_ledger: AdLaunchLedger | None = None,
) -> dict[str, Any]:
    raydn_component_enabled = (
        "reflection" in config.components and reflection_available
    ) or ("diffraction" in config.components and diffraction_available)
    # ADR-022: ad_mode='none' wires no companions and retains no tape (bitwise
    # default). Under jvp/vjp report the companion launches this solve
    # registered in the AdLaunchLedger, exactly as montecarlo.basic does.
    ad_active = config.ad_mode != "none"
    backward_launch_count, jvp_launch_count, tape_bytes = _ad_launch_accounting(
        config, ad_ledger
    )
    kernel_metadata = make_metadata(
        primitive="montecarlo_bdpt_primal",
        forward_launch_count=max(1, int(launch_count)),
        backward_launch_count=backward_launch_count,
        jvp_launch_count=jvp_launch_count,
        tape_bytes=tape_bytes,
        fused_stages=1 if raydn_component_enabled else 0,
        intermediate_bytes=int(workspace_bytes),
        accumulation_strategy=_KERNEL_ACCUMULATION[selected_accumulation_strategy],
        scheduling_strategy="native_fused"
        if raydn_component_enabled
        else "native_cuda",
        raydn_native=reflection_available or diffraction_available,
        ad_status=config.ad_mode if ad_active else "none",
    )
    requested_config = serialize_config(config)
    effective_config = dict(requested_config)
    effective_config["max_depth"] = int(effective_max_depth)
    metadata = {
        "samples": config.samples,
        "seed": config.seed,
        "stream_count": config.sample_streams,
        "sample_streams": config.sample_streams,
        "mis": config.mis,
        "power_heuristic_beta": config.power_heuristic_beta,
        # ADR-019: which combine domain produced the component powers. "power"
        # is the default incoherent per-path accumulation; "coherent" sums the
        # enumerated delta/UTD complex field per (tx, rx, component).
        "combine_domain": "coherent" if config.coherent else "power",
        "coherent": bool(config.coherent),
        "max_depth": config.max_depth,
        "max_light_depth": config.max_light_depth,
        "max_diffraction_order": config.max_diffraction_order,
        "path_counts_by_strategy": path_counts_by_strategy,
        "valid_contribution_count": valid_contribution_count,
        "components": component_status(
            config=config,
            reflection_available=reflection_available,
            diffraction_available=diffraction_available,
        ),
        "native_capabilities": {
            "cuda": bool(cuda_available),
            "raydn": bool(reflection_available or diffraction_available),
            "reflection": bool(reflection_available),
            "diffraction": bool(diffraction_available),
            "optix": bool(optix_available),
        },
        "raydn": {
            "reflection": bool(reflection_available),
            "diffraction": bool(diffraction_available),
        },
        "launch_count": max(1, int(launch_count)),
        "accumulation_strategy": selected_accumulation_strategy,
        "workspace_bytes": int(workspace_bytes),
        "variance": bool(variance_enabled),
        "throughput_domain": "complex3_jones_coherent_events",
        # ADR-021 D4: BDPT multi-order diffuse scattering. Order 1 (default)
        # keeps the single-bounce terminal rule (a scattered subpath connects
        # via NEE and terminates); order > 1 lets a scattered subpath continue
        # and scatter again up to the cap, emitting an NEE row at every scatter
        # vertex (power domain, excluded from the coherent combine).
        "max_scattering_order": int(config.max_scattering_order),
        "scattering_depth_rule": (
            "single_bounce_terminal"
            if int(config.max_scattering_order) <= 1
            else "multi_order_continuation"
        ),
        "field_transport": {
            "authoritative_carrier": "complex3_jones",
            "scalar_throughput_role": "sampling_probability_proxy_only",
            "local_frame": "interaction_local_s_p_recomputed_per_event",
            "scattering": "incoherent_power_only_no_complex_field",
            "sensor_depth": "receiver_endpoint_only_always_zero",
        },
        "pdf_domain": "proposal_density_excludes_geometry_jacobian",
        "event_classification": {
            "endpoint": 0,
            "delta_specular_reflection": 1,
            "delta_specular_transmission": 2,
        },
        "component_mask_bits": dict(_COMPONENT_MASK_BITS),
        "delta_strategy": "canonical_enumeration_unit_bidirectional_mass",
        "sampled_delta_mass": "event_selection_probability_in_forward_reverse_pdf",
        "mis_capabilities": {
            "delta_specular_classification": True,
            "continuous_diffraction_strategies": True,
            "reflection_diffraction_coupled_bidirectional_pdf": True,
            "coupled_pdf_domain": "enumerated_bidirectional_discrete_mass",
        },
        "ad_status": config.ad_mode if ad_active else "none",
        # ADR-022: geometry gradients exist only for the enumerated discrete
        # blocks (fixed-winner endpoints / mesh vertices); the stochastic
        # sampler's hit geometry stays frozen in v1. Reported loudly so a caller
        # never mistakes a zero geometry grad through the sampler for a bug.
        "ad_geometry": _AD_GEOMETRY_SCOPE,
        "ad_differentiable_parameters": list(_AD_DIFFERENTIABLE_PARAMETERS),
        "kernel": kernel_metadata,
    }
    metadata.update(
        config_metadata(
            requested=requested_config,
            effective=effective_config,
            component_max_depth=_component_max_depth(config, effective_max_depth),
        )
    )
    metadata["semantic_capabilities"] = capabilities()["solvers"]["montecarlo_bdpt"]
    return metadata
