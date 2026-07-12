from __future__ import annotations

from typing import Any

from witwin.channel_native.core.kernels.metadata import make_metadata
from witwin.channel_native.capabilities import capabilities, config_metadata, serialize_config

from .config import Config


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
            raise RuntimeError("reflection requires RayDN native capability")
        status["reflection"] = "enabled"
    if "diffraction" in config.components:
        if not diffraction_available:
            raise RuntimeError("diffraction requires RayDN native capability")
        status["diffraction"] = "enabled"
    # transmission carries real straight-penetration physics; scattering
    # deposits the Kirchhoff diffuse radiomap from area-sampled rough faces.
    status["transmission"] = (
        "enabled" if "transmission" in config.components else "not_requested"
    )
    status["scattering"] = (
        "enabled" if "scattering" in config.components else "not_requested"
    )
    return status


def make_solver_metadata(
    *,
    config: Config,
    path_count: int,
    valid_contribution_count: int,
    reflection_available: bool,
    diffraction_available: bool,
) -> dict[str, Any]:
    forward_launch_count = 1 if valid_contribution_count else 0
    backward_launch_count = 1 if config.ad_mode == "vjp" and valid_contribution_count else 0
    jvp_launch_count = 1 if config.ad_mode == "jvp" and valid_contribution_count else 0
    tape_bytes = valid_contribution_count * 16 if config.ad_mode in {"vjp", "jvp"} else 0
    raydn_component_enabled = (
        ("reflection" in config.components and reflection_available)
        or ("diffraction" in config.components and diffraction_available)
    )
    kernel_metadata = make_metadata(
        primitive="montecarlo_basic_primal",
        forward_launch_count=forward_launch_count,
        backward_launch_count=backward_launch_count,
        jvp_launch_count=jvp_launch_count,
        tape_bytes=tape_bytes,
        fused_stages=1 if raydn_component_enabled else 0,
        accumulation_strategy=config.accumulation_strategy,
        scheduling_strategy="native_fused" if raydn_component_enabled else "native_cuda",
        raydn_native=reflection_available or diffraction_available,
        ad_status=config.ad_mode if config.ad_mode != "none" else "none",
    )
    requested_config = serialize_config(config)
    effective_config = dict(requested_config)
    metadata = {
        "seed": config.seed,
        "samples": config.samples,
        "max_depth": config.max_depth,
        "ad_mode": config.ad_mode,
        "fixed_topology": config.fixed_topology,
        "requires_fixed_seed": config.requires_fixed_seed,
        "reflection_accumulation_strategy": config.reflection_accumulation_strategy,
        "reflection_compact_min_samples": config.reflection_compact_min_samples,
        "reflection_staged_min_samples_per_cell": config.reflection_staged_min_samples_per_cell,
        "path_count": path_count,
        "valid_contribution_count": valid_contribution_count,
        "components": component_status(
            config=config,
            reflection_available=reflection_available,
            diffraction_available=diffraction_available,
        ),
        "raydn": {
            "reflection": reflection_available,
            "diffraction": diffraction_available,
        },
        "kernel": kernel_metadata,
    }
    metadata.update(
        config_metadata(
            requested=requested_config,
            effective=effective_config,
            component_max_depth={
                "los": 0 if "los" in config.components else -1,
                "reflection": config.max_depth if "reflection" in config.components else -1,
                "diffraction": 1 if "diffraction" in config.components else -1,
                # transmission chains are capped like reflection; scattering
                # is single-bounce in v1.
                "transmission": config.max_depth if "transmission" in config.components else -1,
                "scattering": 1 if "scattering" in config.components else -1,
            },
        )
    )
    metadata["semantic_capabilities"] = capabilities()["solvers"]["montecarlo_basic"]
    return metadata
