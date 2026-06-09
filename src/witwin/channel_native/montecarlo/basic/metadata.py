from __future__ import annotations

from typing import Any

from witwin.channel_native.core.kernels.metadata import make_metadata

from .config import Config


def component_status(
    *,
    config: Config,
    reflection_available: bool,
    diffraction_available: bool,
) -> dict[str, str]:
    status = {
        "los": "enabled" if "los" in config.components else "disabled",
        "reflection": "disabled",
        "diffraction": "disabled",
    }
    if "reflection" in config.components:
        status["reflection"] = "enabled" if reflection_available else "capability-disabled"
    if "diffraction" in config.components:
        status["diffraction"] = "enabled" if diffraction_available else "capability-disabled"
    return status


def make_solver_metadata(
    *,
    config: Config,
    path_count: int,
    valid_contribution_count: int,
    reflection_available: bool,
    diffraction_available: bool,
) -> dict[str, Any]:
    kernel_metadata = make_metadata(
        primitive="montecarlo_basic_primal",
        forward_launch_count=1 if valid_contribution_count else 0,
        accumulation_strategy=config.accumulation_strategy,
        scheduling_strategy="torch_cuda",
        raydn_native=reflection_available or diffraction_available,
        ad_status="unsupported",
    )
    return {
        "seed": config.seed,
        "samples": config.samples,
        "max_depth": config.max_depth,
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
