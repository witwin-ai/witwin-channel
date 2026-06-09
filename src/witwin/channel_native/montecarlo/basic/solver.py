from __future__ import annotations

from typing import Any

import torch

from witwin.channel_native import Scene
from witwin.channel_native.core.kernels.extension import build_info

from .backend import los_path_gain
from .config import Config
from .metadata import make_solver_metadata
from .raydn_components import (
    component_grid_shape,
    diffraction_component_map,
    first_receiver_grid,
    los_component_map,
    reflection_component_maps_with_wedges,
)
from .result import Result
from .sampling import make_cuda_generator


def _validate_ad_config(config: Config) -> None:
    if config.ad_mode == "none":
        return
    if not config.fixed_topology:
        raise RuntimeError("fixed_topology=True is required for MC basic AD")
    if not config.requires_fixed_seed:
        raise RuntimeError("MC basic AD requires a fixed seed")
    if config.components != {"los"}:
        raise RuntimeError("MC basic fixed-topology AD is LoS-only")


def solve(scene: Scene, config: Config) -> Result:
    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel_native.montecarlo.basic requires CUDA")
    _validate_ad_config(config)

    info = build_info()
    reflection_available = bool(info["uses_raydn_native"])
    diffraction_available = bool(info["uses_raydn_native"])
    if config.require_reflection and "reflection" in config.components and not reflection_available:
        raise RuntimeError("reflection requires RayDN native capability")
    if config.require_diffraction and "diffraction" in config.components and not diffraction_available:
        raise RuntimeError("diffraction requires RayDN native capability")

    device = torch.device("cuda")
    make_cuda_generator(config.seed)
    compiled = scene.compile()
    grid = first_receiver_grid(scene)
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
    component_maps: dict[str, torch.Tensor] | None = None
    if grid is not None:
        component_maps = {}
        if "los" in config.components:
            component_maps["los"] = los_component_map(scene, compiled.raydn, grid, device=device)
        else:
            component_maps["los"] = torch.zeros(
                (len(scene.transmitters), *component_grid_shape(grid)),
                device=device,
                dtype=torch.float32,
            )
        needs_reflection_launch = (
            reflection_available
            and (("reflection" in config.components) or ("diffraction" in config.components and diffraction_available))
        )
        reflection_result = None
        if needs_reflection_launch:
            reflection_result = reflection_component_maps_with_wedges(
                scene,
                compiled.raydn,
                grid,
                samples=config.samples,
                max_depth=config.max_depth,
                seed=config.seed,
                device=device,
                collect_wedges=("diffraction" in config.components and diffraction_available),
            )
        if "reflection" in config.components and reflection_available and reflection_result is not None:
            component_maps["reflection"] = reflection_result.maps
            path_count += config.samples
            valid_contribution_count += int(component_maps["reflection"].numel())
        else:
            component_maps["reflection"] = torch.zeros_like(component_maps["los"])
        if "diffraction" in config.components and diffraction_available:
            component_maps["diffraction"] = diffraction_component_map(
                scene,
                compiled.raydn,
                grid,
                samples=config.samples,
                seed=config.seed,
                device=device,
                wedge_events=None if reflection_result is None else reflection_result.wedge_events,
            )
            path_count += config.samples
            valid_contribution_count += int(component_maps["diffraction"].numel())
        else:
            component_maps["diffraction"] = torch.zeros_like(component_maps["los"])

    component_power = {
        "los": (component_maps["los"].sum() if component_maps is not None else los.sum()),
        "reflection": (
            component_maps["reflection"].sum()
            if component_maps is not None
            else zero.clone()
        ),
        "diffraction": (
            component_maps["diffraction"].sum()
            if component_maps is not None
            else zero.clone()
        ),
    }
    path_gain = los
    if component_maps is not None:
        total_map = component_maps["los"] + component_maps["reflection"] + component_maps["diffraction"]
        path_gain = total_map.reshape(len(scene.transmitters), -1).contiguous()
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
            "component_map_shapes": (
                None if component_maps is None else {key: tuple(value.shape) for key, value in component_maps.items()}
            ),
        }
    return Result(
        path_gain=path_gain,
        component_power=component_power,
        metadata=metadata,
        diagnostics=diagnostics,
        component_maps=component_maps,
    )
