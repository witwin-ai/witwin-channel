from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from witwin.channel_native import Scene
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.core.kernels.metadata import make_metadata
from witwin.channel_native.core.field_state import PHASE_CONVENTION
from witwin.channel_native.core.path_topology import export_topology
from witwin.channel_native.core.scene_tensors import (
    receiver_positions as _shared_receiver_positions,
    transmitter_positions as _shared_transmitter_positions,
)
from witwin.channel_native.propagation.enumerated import append_scattering_paths

from .config import Config
from .arrays import (
    explicit_array_scene,
    pack_explicit_arrays,
    pack_synthetic_arrays,
    validate_synthetic_array_scene,
)
from .result import PathResult, from_topology_result


_COMPONENT_ID = {
    "los": 0,
    "reflection": 1,
    "diffraction": 2,
    "reflection_diffraction": 3,
    "diffraction_reflection": 4,
    # transmission exports specular wall-penetration paths since wave 2;
    # scattering exports incoherent Kirchhoff patch paths since wave 3.
    "transmission": 5,
    "scattering": 6,
}


def _transmitter_tensors(scene: Scene) -> tuple[torch.Tensor, torch.Tensor]:
    return _shared_transmitter_positions(scene, device=torch.device("cuda"))


def _receiver_positions(scene: Scene, *, reference: torch.Tensor) -> torch.Tensor:
    return _shared_receiver_positions(
        scene, device=reference.device, reference=reference
    )


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
            raise RuntimeError("reflection paths require RayDN native capability")
        if config.max_depth < 1:
            raise RuntimeError("reflection paths require max_depth >= 1")
        status["reflection"] = "enabled"
    if "diffraction" in config.components:
        if not diffraction_available:
            raise RuntimeError("diffraction paths require RayDN native capability")
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
        raydn_native=reflection_available or diffraction_available,
        ad_status=config.ad_mode,
        forward_time_ms=forward_time_ms,
        peak_memory_bytes=peak_memory_bytes,
    )
    capability = {
        "path_native": path_native_available,
        "raydn_native": reflection_available or diffraction_available,
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
        "phase_convention": dict(PHASE_CONVENTION),
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


def _validate_runtime(config: Config) -> tuple[bool, bool, bool]:
    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel_native.path solver requires CUDA")
    if config.ad_mode != "none" and "scattering" in config.components:
        # Kirchhoff patch paths bypass the shared field seam; their fields are
        # not differentiable yet (plan 07 AD-4). Fail before any launch.
        raise RuntimeError(
            f"path ad_mode='{config.ad_mode}' does not support the scattering "
            "component yet"
        )
    info = build_info()
    reflection_available = bool(info["uses_raydn_native"])
    diffraction_available = bool(info["uses_raydn_native"])
    path_native_available = bool(info.get("uses_path_native", False))
    if not path_native_available:
        raise RuntimeError(
            "path solver requires _channel_native path native CUDA kernels"
        )
    if "reflection" in config.components and not reflection_available:
        raise RuntimeError("reflection paths require RayDN native capability")
    if "diffraction" in config.components and not diffraction_available:
        raise RuntimeError("diffraction paths require RayDN native capability")
    if config.max_depth < 1 and (
        "reflection" in config.components or "diffraction" in config.components
    ):
        raise RuntimeError("requested scattering paths require max_depth >= 1")
    return reflection_available, diffraction_available, path_native_available


def _solve_base(scene: Scene, config: Config) -> PathResult:
    reflection_available, diffraction_available, path_native_available = (
        _validate_runtime(config)
    )
    # Solve-level wall time and CUDA high-water-mark delta for the kernel
    # metadata (plan 07 AD-4). AD instrumentation only: the syncs would break
    # host/device overlap for a caller looping over ad_mode="none" solves, so
    # none-mode reports zeros and takes no sync (zero-overhead primal contract).
    ad_instrumented = config.ad_mode != "none"
    solve_start = 0.0
    peak_before = 0
    if ad_instrumented:
        torch.cuda.synchronize()
        solve_start = perf_counter()
        peak_before = torch.cuda.max_memory_allocated()
    topology = export_topology(scene, config)
    scattering_info = None
    if "scattering" in config.components:
        topology, scattering_info = append_scattering_paths(scene, config, topology)
    path_count = int(topology.valid.numel())
    if ad_instrumented:
        torch.cuda.synchronize()
        forward_time_ms = (perf_counter() - solve_start) * 1.0e3
        peak_memory_bytes = max(0, torch.cuda.max_memory_allocated() - peak_before)
    else:
        forward_time_ms = 0.0
        peak_memory_bytes = 0
    metadata = _metadata(
        config=config,
        path_count=path_count,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        path_native_available=path_native_available,
        transmission_path_count=int(
            (topology.component_id == _COMPONENT_ID["transmission"]).sum().item()
        ),
        scattering_path_count=int(
            (topology.component_id == _COMPONENT_ID["scattering"]).sum().item()
        ),
        ad_companion_launches=topology.ad_companion_launches,
        ad_tape_bytes=topology.ad_tape_bytes,
        forward_time_ms=forward_time_ms,
        peak_memory_bytes=peak_memory_bytes,
        scattering_info=scattering_info,
    )
    tx_positions, _tx_power = _transmitter_tensors(scene)
    rx_positions = _receiver_positions(scene, reference=tx_positions)
    result = from_topology_result(
        topology,
        num_rx=int(rx_positions.shape[0]),
        num_tx=int(tx_positions.shape[0]),
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        metadata=metadata,
    )
    return result


def solve(scene: Scene, config: Config) -> PathResult:
    """Solve canonical paths and pack synthetic or explicit antenna arrays."""

    endpoints = [*scene.transmitters, *scene.receivers]
    if any(not endpoint.synthetic_array for endpoint in endpoints):
        expanded_scene, num_rx_ant, num_tx_ant = explicit_array_scene(scene)
        expanded = _solve_base(expanded_scene, config)
        return pack_explicit_arrays(
            expanded,
            scene=scene,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
        )
    validate_synthetic_array_scene(scene)
    result = _solve_base(scene, config)
    return pack_synthetic_arrays(
        result,
        frequency_hz=scene.frequency,
        transmitters=scene.transmitters,
        receivers=scene.receivers,
    )
