from __future__ import annotations

from typing import Any

from witwin.channel.constants import UNIT_EXCITATION_PHASE_CONVENTION
from witwin.channel.runtime.kernel_metadata import make_metadata

from .config import Config


def _component_status(
    *,
    config: Config,
    reflection_available: bool,
    diffraction_available: bool,
    transmission_path_count: int,
    scattering_path_count: int = 0,
) -> dict[str, str]:
    status = {
        "los": "enabled" if "los" in config.components else "not_requested",
        "reflection": "not_requested",
        "diffraction": "not_requested",
    }
    if "reflection" in config.components:
        if not reflection_available:
            raise RuntimeError("reflection paths require RayD native capability")
        if config.max_depth < 1:
            raise RuntimeError("reflection paths require max_depth >= 1")
        status["reflection"] = "enabled"
    if "diffraction" in config.components:
        if not diffraction_available:
            raise RuntimeError("diffraction paths require RayD native capability")
        if config.max_depth < 1:
            raise RuntimeError("diffraction paths require max_depth >= 1")
        status["diffraction"] = "enabled"
    # transmission (wave 2) and scattering (wave 3) export real paths; the
    # truthful requested-but-empty status remains when no wall penetration /
    # rough surface produced a path.
    if "transmission" in config.components:
        status["transmission"] = (
            "enabled" if transmission_path_count > 0 else "enabled_no_paths"
        )
    else:
        status["transmission"] = "not_requested"
    if "scattering" in config.components:
        status["scattering"] = (
            "enabled" if scattering_path_count > 0 else "enabled_no_paths"
        )
    else:
        status["scattering"] = "not_requested"
    return status


def _metadata(
    *,
    config: Config,
    path_count: int,
    reflection_available: bool,
    diffraction_available: bool,
    path_native_available: bool,
    transmission_path_count: int = 0,
    scattering_path_count: int = 0,
    ad_companion_launches: int = 0,
    ad_tape_bytes: int = 0,
    forward_time_ms: float = 0.0,
    peak_memory_bytes: int = 0,
    scattering_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Plan 07 AD-4: the real registered-companion accounting. vjp retains
    # tape and schedules its companions on the user's later backward; jvp
    # runs its dual companions inside this forward and retains no tape.
    kernel = make_metadata(
        primitive="path_solver",
        forward_launch_count=1 if path_count else 0,
        backward_launch_count=(
            ad_companion_launches if config.ad_mode == "vjp" else 0
        ),
        jvp_launch_count=(
            ad_companion_launches if config.ad_mode == "jvp" else 0
        ),
        tape_bytes=ad_tape_bytes if config.ad_mode == "vjp" else 0,
        accumulation_strategy="none",
        scheduling_strategy="native_cuda",
        rayd_native=reflection_available or diffraction_available,
        ad_status=config.ad_mode,
        forward_time_ms=forward_time_ms,
        peak_memory_bytes=peak_memory_bytes,
    )
    capability = {
        "path_native": path_native_available,
        "rayd_native": reflection_available or diffraction_available,
        "reflection": reflection_available,
        "diffraction": diffraction_available,
    }
    reflection_depth = config.max_depth if "reflection" in config.components else -1
    diffraction_depth = (
        2 if config.coupled_paths else (1 if "diffraction" in config.components else -1)
    )
    effective_max_depth = max(
        0 if "los" in config.components else -1, reflection_depth, diffraction_depth
    )
    metadata = {
        "solver": "path",
        "device": "cuda",
        "path_count": path_count,
        "effective_max_depth": effective_max_depth,
        "component_max_depth": {
            "los": 0 if "los" in config.components else -1,
            "reflection": reflection_depth,
            "diffraction": diffraction_depth,
            "transmission": config.max_depth
            if "transmission" in config.components
            else -1,
            "scattering": 1 if "scattering" in config.components else -1,
        },
        "max_paths_per_pair": config.max_paths,
        "native_capabilities": capability,
        "components": _component_status(
            config=config,
            reflection_available=reflection_available,
            diffraction_available=diffraction_available,
            transmission_path_count=transmission_path_count,
            scattering_path_count=scattering_path_count,
        ),
        "kernel": kernel,
        "field_abi": "complex3_v1",
        "phase_convention": dict(UNIT_EXCITATION_PHASE_CONVENTION),
        "coefficient_semantics": "unit_excitation_dimensionless_receiver_projection",
        "coupled_paths": {
            "requested": config.coupled_paths,
            "geometry": "native_1r1d_reciprocal"
            if config.coupled_paths
            else "not_requested",
            "coefficient": "unified_complex3_jones"
            if config.coupled_paths
            else "not_requested",
        },
    }
    if "transmission" in config.components:
        # Endpoint-connection thin_sheet contract (plan 05 section 4).
        metadata["transmission"] = {
            "thin_sheet_straight_path_approximation": True,
            "group_delay": "geometric",
        }
    if scattering_info is not None:
        # Incoherent Kirchhoff patch quadrature (plan 05 wave 3); the flag
        # documents that per-path phases are NOT physical for ensemble rows.
        metadata["scattering"] = dict(scattering_info)
    return metadata
