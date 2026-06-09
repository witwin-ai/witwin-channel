from __future__ import annotations

from typing import Any

import torch

from witwin.channel_native import ReceiverGrid, ReceiverPoint, Scene
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.core.kernels.metadata import make_metadata
from witwin.channel_native.core.kernels import ops

from .config import Config
from .result import Result


_COMPONENT_ID = {"los": 0, "reflection": 1, "diffraction": 2}


def _receiver_positions(scene: Scene, *, device: torch.device) -> torch.Tensor:
    blocks = []
    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverPoint):
            blocks.append(receiver.position.reshape(1, 3))
        elif isinstance(receiver, ReceiverGrid):
            blocks.append(receiver.points())
        else:
            raise TypeError(f"unsupported receiver type: {type(receiver).__name__}")
    return torch.cat(blocks, dim=0).to(device=device, dtype=torch.float32).contiguous()


def _component_status(
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
        forward_launch_count=1 if path_count and path_native_available else 0,
        accumulation_strategy="none",
        scheduling_strategy="torch_cuda",
        raydn_native=reflection_available or diffraction_available,
        fusion_debt=not path_native_available,
        ad_status="unsupported",
    )
    capability = {
        "path_native": path_native_available,
        "raydn_native": reflection_available or diffraction_available,
        "reflection": reflection_available,
        "diffraction": diffraction_available,
    }
    return {
        "max_depth": config.max_depth,
        "max_paths": config.max_paths,
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
    }


def solve(scene: Scene, config: Config) -> Result:
    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel_native.path solver requires CUDA")
    if config.ad_mode != "none":
        raise RuntimeError("path topology AD is unsupported")

    info = build_info()
    reflection_available = bool(info["uses_raydn_native"])
    diffraction_available = bool(info["uses_raydn_native"])
    path_native_available = bool(info.get("uses_path_native", False))
    if config.require_reflection and "reflection" in config.components and not reflection_available:
        raise RuntimeError("reflection paths require RayDN native capability")
    if config.require_diffraction and "diffraction" in config.components and not diffraction_available:
        raise RuntimeError("diffraction paths require RayDN native capability")

    device = torch.device("cuda")
    if "los" not in config.components or not scene.transmitters:
        return _empty_result(
            device=device,
            config=config,
            reflection_available=reflection_available,
            diffraction_available=diffraction_available,
            path_native_available=path_native_available,
        )

    tx_positions = torch.stack([tx.position for tx in scene.transmitters], dim=0).to(
        device=device, dtype=torch.float32
    )
    tx_power = torch.tensor([tx.power_w for tx in scene.transmitters], device=device, dtype=torch.float32)
    rx_positions = _receiver_positions(scene, device=device)
    exported = ops.path_los_export(
        tx_positions,
        tx_power,
        rx_positions,
        frequency_hz=scene.frequency,
    )
    tx_id = exported["tx_id"]
    rx_id = exported["rx_id"]
    path_length = exported["path_length_m"]
    delay = exported["delay_s"]
    path_gain = exported["path_gain"]
    path_count = int(path_gain.numel())
    if config.max_paths is not None:
        path_count = min(path_count, config.max_paths)

    sl = slice(0, path_count)
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
        valid=torch.ones((path_count,), device=device, dtype=torch.bool),
        tx_id=tx_id[sl].contiguous(),
        rx_id=rx_id[sl].contiguous(),
        depth=torch.zeros((path_count,), device=device, dtype=torch.int32),
        component_id=torch.full((path_count,), _COMPONENT_ID["los"], device=device, dtype=torch.int32),
        primitive_id=torch.full((path_count,), -1, device=device, dtype=torch.int32),
        edge_id=torch.full((path_count,), -1, device=device, dtype=torch.int32),
        path_length_m=path_length[sl].to(dtype=torch.float32).contiguous(),
        delay_s=delay[sl].to(dtype=torch.float32).contiguous(),
        path_gain=path_gain[sl].to(dtype=torch.float32).contiguous(),
        metadata=metadata,
        diagnostics=diagnostics,
    )
