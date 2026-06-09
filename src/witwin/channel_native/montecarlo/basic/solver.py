from __future__ import annotations

from typing import Any

import torch

from witwin.channel_native import Scene
from witwin.channel_native.core.kernels.extension import build_info

from .backend import los_path_gain
from .config import Config
from .metadata import make_solver_metadata
from .result import Result
from .sampling import make_cuda_generator


def solve(scene: Scene, config: Config) -> Result:
    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel_native.montecarlo.basic requires CUDA")

    info = build_info()
    reflection_available = bool(info["uses_raydn_native"])
    diffraction_available = bool(info["uses_raydn_native"])
    if config.require_reflection and "reflection" in config.components and not reflection_available:
        raise RuntimeError("reflection requires RayDN native capability")
    if config.require_diffraction and "diffraction" in config.components and not diffraction_available:
        raise RuntimeError("diffraction requires RayDN native capability")

    device = torch.device("cuda")
    make_cuda_generator(config.seed)
    if "los" in config.components:
        los = los_path_gain(scene, device=device)
        path_count = config.samples
        valid_contribution_count = config.samples
    else:
        tx_count = len(scene.transmitters)
        rx_count = sum(
            receiver.points().shape[0] if hasattr(receiver, "points") else 1
            for receiver in scene.receivers
        )
        los = torch.zeros((tx_count, rx_count), device=device, dtype=torch.float32)
        path_count = 0
        valid_contribution_count = 0

    zero = torch.zeros((), device=device, dtype=torch.float32)
    component_power = {
        "los": los.sum(),
        "reflection": zero.clone(),
        "diffraction": zero.clone(),
    }
    metadata = make_solver_metadata(
        config=config,
        path_count=path_count,
        valid_contribution_count=valid_contribution_count,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
    )
    diagnostics: dict[str, Any] | None = None
    if config.diagnostics:
        diagnostics = {
            "path_gain_shape": tuple(los.shape),
            "device": str(los.device),
            "component_power_keys": tuple(component_power),
        }
    return Result(
        path_gain=los,
        component_power=component_power,
        metadata=metadata,
        diagnostics=diagnostics,
    )
