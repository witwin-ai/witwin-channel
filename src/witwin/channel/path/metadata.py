from __future__ import annotations

from typing import Any

from witwin.channel.components import (
    apply_exported_path_counts,
    component_availability_status,
    component_max_depth,
)
from witwin.channel.constants import UNIT_EXCITATION_PHASE_CONVENTION
from witwin.channel.runtime import make_metadata

from .config import Config


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
    component_depths = component_max_depth(
        config.components, chain_depth=config.max_depth, single_bounce_depth=1
    )
    if config.coupled_paths:
        # The coupled 1R1D/1D1R family reaches depth 2 whenever it is requested,
        # including when plain diffraction is not in the component set.
        component_depths["diffraction"] = 2
    effective_max_depth = max(
        component_depths["los"],
        component_depths["reflection"],
        component_depths["diffraction"],
    )
    components = component_availability_status(
        config.components,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        reflection_error="reflection paths require RayD native capability",
        diffraction_error="diffraction paths require RayD native capability",
        depth_available=config.max_depth >= 1,
        reflection_depth_error="reflection paths require max_depth >= 1",
        diffraction_depth_error="diffraction paths require max_depth >= 1",
    )
    apply_exported_path_counts(
        components,
        config.components,
        transmission_path_count=transmission_path_count,
        scattering_path_count=scattering_path_count,
    )
    metadata = {
        "solver": "path",
        "device": "cuda",
        "path_count": path_count,
        "effective_max_depth": effective_max_depth,
        "component_max_depth": component_depths,
        "max_paths_per_pair": config.max_paths,
        "native_capabilities": capability,
        "components": components,
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
