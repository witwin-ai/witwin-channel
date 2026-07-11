from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from witwin.channel_native import ReceiverGrid, ReceiverPoint, Scene
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.core.kernels.metadata import make_metadata
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.core.path_topology import export_topology
from witwin.channel_native.capabilities import (
    capabilities,
    config_metadata,
    serialize_config,
)

from .config import Config
from .result import Result
from .result_v2 import PathResultV2, from_topology_result


_COMPONENT_ID = {
    "los": 0,
    "reflection": 1,
    "diffraction": 2,
    "reflection_diffraction": 3,
    "diffraction_reflection": 4,
}


def _vector3_tuple(value: torch.Tensor) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _transmitter_tensors(scene: Scene) -> tuple[torch.Tensor, torch.Tensor]:
    flat_positions = tuple(
        component
        for transmitter in scene.transmitters
        for component in _vector3_tuple(transmitter.position)
    )
    powers = tuple(float(transmitter.power_w) for transmitter in scene.transmitters)
    exported = ops.mc_transmitter_tensors(flat_positions, powers)
    return exported["positions"], exported["power"]


def _host_vec3_tensor(flat_positions: tuple[float, ...]) -> torch.Tensor:
    powers = tuple(1.0 for _ in range(len(flat_positions) // 3))
    return ops.mc_transmitter_tensors(flat_positions, powers)["positions"]


def _receiver_positions(scene: Scene, *, reference: torch.Tensor) -> torch.Tensor:
    blocks: list[torch.Tensor] = []
    point_positions: list[float] = []

    def flush_points() -> None:
        nonlocal point_positions
        if point_positions:
            blocks.append(_host_vec3_tensor(tuple(point_positions)))
            point_positions = []

    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverPoint):
            point_positions.extend(_vector3_tuple(receiver.position))
        elif isinstance(receiver, ReceiverGrid):
            flush_points()
            blocks.append(
                ops.mc_receiver_grid_points(
                    reference,
                    origin=_vector3_tuple(receiver.origin),
                    x_axis=_vector3_tuple(receiver.x_axis),
                    y_axis=_vector3_tuple(receiver.y_axis),
                    shape=receiver.shape,
                    spacing=receiver.spacing,
                )
            )
        else:
            raise TypeError(f"receiver type is not accepted: {type(receiver).__name__}")
    flush_points()
    if not blocks:
        return _host_vec3_tensor(())
    if len(blocks) == 1:
        return blocks[0]
    return ops.path_concat_vec3(blocks)


def _component_status(
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
    return status


def _empty_result(
    *,
    device: torch.device,
    config: Config,
    reflection_available: bool,
    diffraction_available: bool,
    path_native_available: bool,
    diagnostics: dict[str, Any] | None = None,
) -> Result:
    metadata = _metadata(
        config=config,
        path_count=0,
        valid_contribution_count=0,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        path_native_available=path_native_available,
    )
    return Result(
        valid=torch.empty((0,), device=device, dtype=torch.bool),
        tx_id=torch.empty((0,), device=device, dtype=torch.int32),
        rx_id=torch.empty((0,), device=device, dtype=torch.int32),
        depth=torch.empty((0,), device=device, dtype=torch.int32),
        component_id=torch.empty((0,), device=device, dtype=torch.int32),
        primitive_id=torch.empty((0,), device=device, dtype=torch.int32),
        edge_id=torch.empty((0,), device=device, dtype=torch.int32),
        path_length_m=torch.empty((0,), device=device, dtype=torch.float32),
        delay_s=torch.empty((0,), device=device, dtype=torch.float32),
        path_gain=torch.empty((0,), device=device, dtype=torch.float32),
        metadata=metadata,
        diagnostics=diagnostics,
    )


def _metadata(
    *,
    config: Config,
    path_count: int,
    valid_contribution_count: int,
    reflection_available: bool,
    diffraction_available: bool,
    path_native_available: bool,
) -> dict[str, Any]:
    kernel = make_metadata(
        primitive="path_solver",
        forward_launch_count=1 if path_count else 0,
        accumulation_strategy="none",
        scheduling_strategy="native_cuda",
        raydn_native=reflection_available or diffraction_available,
        ad_status="none",
    )
    capability = {
        "path_native": path_native_available,
        "raydn_native": reflection_available or diffraction_available,
        "reflection": reflection_available,
        "diffraction": diffraction_available,
    }
    requested_config = serialize_config(config)
    reflection_depth = config.max_depth if "reflection" in config.components else -1
    diffraction_depth = (
        2 if config.coupled_paths else (1 if "diffraction" in config.components else -1)
    )
    effective_max_depth = max(
        0 if "los" in config.components else -1, reflection_depth, diffraction_depth
    )
    effective_config = dict(requested_config)
    effective_config["max_depth"] = effective_max_depth
    metadata = {
        "max_depth": config.max_depth,
        "max_paths": config.max_paths,
        "max_paths_scope": config.max_paths_scope,
        "sort_key": config.sort_key,
        "path_count": path_count,
        "valid_contribution_count": valid_contribution_count,
        "counts": {
            "path_count": path_count,
            "valid_path_count": valid_contribution_count,
        },
        "capability": capability,
        "components": _component_status(
            config=config,
            reflection_available=reflection_available,
            diffraction_available=diffraction_available,
        ),
        "raydn": {
            "reflection": reflection_available,
            "diffraction": diffraction_available,
        },
        "kernel": kernel,
        "coupled_paths": {
            "requested": config.coupled_paths,
            "geometry": "native_1r1d_reciprocal"
            if config.coupled_paths
            else "not_requested",
            "coefficient": "unavailable_until_phase_3"
            if config.coupled_paths
            else "not_requested",
        },
    }
    metadata.update(
        config_metadata(
            requested=requested_config,
            effective=effective_config,
            component_max_depth={
                "los": 0 if "los" in config.components else -1,
                "reflection": reflection_depth,
                "diffraction": diffraction_depth,
            },
        )
    )
    metadata["semantic_capabilities"] = capabilities()["solvers"]["path"]
    return metadata


def _validate_runtime(config: Config) -> tuple[bool, bool, bool]:
    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel_native.path solver requires CUDA")
    if config.ad_mode != "none":
        raise RuntimeError("path topology AD is not enabled")
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


def solve(scene: Scene, config: Config) -> Result:
    if config.coupled_paths:
        raise RuntimeError(
            "coupled_paths require solve_v2 until Phase 3 provides physical complex coefficients"
        )
    reflection_available, diffraction_available, path_native_available = (
        _validate_runtime(config)
    )

    if not scene.transmitters:
        device = torch.device("cuda")
        return _empty_result(
            device=device,
            config=config,
            reflection_available=reflection_available,
            diffraction_available=diffraction_available,
            path_native_available=path_native_available,
        )

    exported_paths = export_topology(scene, config)
    device = exported_paths.valid.device
    path_count = int(exported_paths.path_gain.shape[0])
    if path_count == 0:
        return _empty_result(
            device=device,
            config=config,
            reflection_available=reflection_available,
            diffraction_available=diffraction_available,
            path_native_available=path_native_available,
        )
    metadata = _metadata(
        config=config,
        path_count=path_count,
        valid_contribution_count=path_count,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        path_native_available=path_native_available,
    )
    diagnostics = None
    if config.diagnostics:
        diagnostics = {
            "device": str(device),
            "path_count": path_count,
            "component_order": dict(_COMPONENT_ID),
        }
    return Result(
        valid=exported_paths.valid,
        tx_id=exported_paths.tx_id,
        rx_id=exported_paths.rx_id,
        depth=exported_paths.depth,
        component_id=exported_paths.component_id,
        primitive_id=exported_paths.primitive_id,
        edge_id=exported_paths.edge_id,
        path_length_m=exported_paths.path_length_m,
        delay_s=exported_paths.delay_s,
        path_gain=exported_paths.path_gain,
        metadata=metadata,
        diagnostics=diagnostics,
    )


def solve_v2(scene: Scene, config: Config) -> PathResultV2:
    """Solve and pack the shared canonical topology into PathResultV2."""

    reflection_available, diffraction_available, path_native_available = (
        _validate_runtime(config)
    )
    topology = export_topology(scene, config)
    path_count = int(topology.valid.numel())
    metadata = _metadata(
        config=config,
        path_count=path_count,
        valid_contribution_count=path_count,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        path_native_available=path_native_available,
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
    metadata = dict(result.metadata)
    metadata["requested_max_paths_per_pair"] = config.max_paths
    if config.coupled_paths:
        metadata["coefficient_semantics"] = (
            "native_scalar_complex_field_with_nan_for_coupled_geometry"
        )
    return replace(result, metadata=metadata)
