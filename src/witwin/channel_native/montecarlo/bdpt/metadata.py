from __future__ import annotations

from typing import Any

from witwin.channel_native.core.kernels.metadata import make_metadata
from witwin.channel_native.capabilities import capabilities, config_metadata, serialize_config

from .config import Config


_KERNEL_ACCUMULATION = {
    "atomic": "atomic_add",
    "staged": "cell_reduce",
    "compact": "compact_atomic_add",
}


def select_accumulation_strategy(config: Config, *, grid_cells: int, estimated_valid_ratio: float) -> str:
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
    status = {
        "los": "enabled" if "los" in config.components else "not_requested",
        "reflection": "not_requested",
        "diffraction": "not_requested",
    }
    if "reflection" in config.components:
        if not reflection_available:
            raise RuntimeError("BDPT reflection requires RayDN native capability")
        status["reflection"] = "enabled"
    if "diffraction" in config.components:
        if not diffraction_available:
            raise RuntimeError("BDPT diffraction requires RayDN native capability")
        status["diffraction"] = "enabled"
    return status


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
) -> dict[str, Any]:
    raydn_component_enabled = (
        ("reflection" in config.components and reflection_available)
        or ("diffraction" in config.components and diffraction_available)
    )
    kernel_metadata = make_metadata(
        primitive="montecarlo_bdpt_primal",
        forward_launch_count=max(1, int(launch_count)),
        fused_stages=1 if raydn_component_enabled else 0,
        intermediate_bytes=int(workspace_bytes),
        accumulation_strategy=_KERNEL_ACCUMULATION[selected_accumulation_strategy],
        scheduling_strategy="native_fused" if raydn_component_enabled else "native_cuda",
        raydn_native=reflection_available or diffraction_available,
        ad_status="none",
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
        "max_depth": config.max_depth,
        "max_light_depth": config.max_light_depth,
        "max_sensor_depth": config.max_sensor_depth,
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
        "throughput_domain": "unit_excitation_complex3",
        "pdf_domain": "cumulative_non_delta_proposal_density",
        "event_classification": {
            "endpoint": 0,
            "delta_specular_reflection": 1,
        },
        "delta_strategy": "canonical_enumeration_unit_bidirectional_mass",
        "mis_capabilities": {
            "delta_specular_classification": True,
            "continuous_diffraction_strategies": True,
            "reflection_diffraction_coupled_bidirectional_pdf": True,
            "coupled_pdf_domain": "enumerated_bidirectional_discrete_mass",
        },
        "ad_status": "none",
        "kernel": kernel_metadata,
    }
    metadata.update(
        config_metadata(
            requested=requested_config,
            effective=effective_config,
            component_max_depth={
                "los": 0 if "los" in config.components else -1,
                "reflection": int(effective_max_depth) if "reflection" in config.components else -1,
                "diffraction": min(1, int(effective_max_depth)) if "diffraction" in config.components else -1,
            },
        )
    )
    metadata["semantic_capabilities"] = capabilities()["solvers"]["montecarlo_bdpt"]
    return metadata
