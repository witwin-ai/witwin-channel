from __future__ import annotations

from typing import Any

# One AdLaunchLedger shape for every solver (plan 07 AD-4): montecarlo.basic
# counts one companion per LoS matrix, per grid-map layout Function, per
# transmitter for the reflection/diffraction accumulators and per layer-stack
# evaluation inside the transmission chain march; the finalize sum registers
# no native companion (its cotangent is a view).
from witwin.channel_native.core.kernels.metadata import AdLaunchLedger, make_metadata
from witwin.channel_native.capabilities import capabilities, config_metadata, serialize_config
from witwin.channel_native.core.components import component_availability_status

from .config import Config


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
        reflection_error="reflection requires RayD native capability",
        diffraction_error="diffraction requires RayD native capability",
    )


def make_solver_metadata(
    *,
    config: Config,
    path_count: int,
    valid_contribution_count: int,
    reflection_available: bool,
    diffraction_available: bool,
    ad_ledger: AdLaunchLedger | None = None,
) -> dict[str, Any]:
    forward_launch_count = 1 if valid_contribution_count else 0
    # Plan 07 AD-3: report the companion launches this solve actually
    # registered (see AdLaunchLedger), not the pre-design fused-launch
    # placeholder. ad_mode="none" wires no companions and retains no tape.
    ledger = ad_ledger if ad_ledger is not None else AdLaunchLedger()
    backward_launch_count = ledger.launches if config.ad_mode == "vjp" else 0
    jvp_launch_count = ledger.launches if config.ad_mode == "jvp" else 0
    tape_bytes = ledger.tape_bytes if config.ad_mode == "vjp" else 0
    rayd_component_enabled = (
        ("reflection" in config.components and reflection_available)
        or ("diffraction" in config.components and diffraction_available)
    )
    kernel_metadata = make_metadata(
        primitive="montecarlo_basic_primal",
        forward_launch_count=forward_launch_count,
        backward_launch_count=backward_launch_count,
        jvp_launch_count=jvp_launch_count,
        tape_bytes=tape_bytes,
        fused_stages=1 if rayd_component_enabled else 0,
        accumulation_strategy="atomic_add",
        scheduling_strategy="native_fused" if rayd_component_enabled else "native_cuda",
        rayd_native=reflection_available or diffraction_available,
        ad_status=config.ad_mode if config.ad_mode != "none" else "none",
    )
    requested_config = serialize_config(config)
    effective_config = dict(requested_config)
    metadata = {
        "seed": config.seed,
        "samples": config.samples,
        "max_depth": config.max_depth,
        "ad_mode": config.ad_mode,
        "path_count": path_count,
        "valid_contribution_count": valid_contribution_count,
        "components": component_status(
            config=config,
            reflection_available=reflection_available,
            diffraction_available=diffraction_available,
        ),
        "rayd": {
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
