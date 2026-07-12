from __future__ import annotations

from typing import Any

import torch

from witwin.channel_native import ReceiverGrid, Scene
from witwin.channel_native.core.edge_selection import resolve_scene_edge_policy
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.core.materials import effective_sigma_e
from witwin.channel_native.core.memory_budget import (
    enforce_memory_budget,
    estimate_monte_carlo_memory,
)
from witwin.channel_native.core.kernels.ops import (
    bdpt_face_material_tensors_from_host,
    mc_component_map_buffer,
    mc_finalize_component_maps,
    mc_point_component_power,
    mc_zero_matrix,
)

from .backend import apply_point_los_visibility, los_path_gain
from .config import Config
from .metadata import make_solver_metadata
from .raydn_components import (
    component_grid_shape,
    diffraction_component_map,
    first_receiver_grid,
    los_component_map,
    reflection_component_maps_with_wedges,
    transmission_component_map,
)
from .result import Result
from .sampling import make_cuda_generator


def _validate_ad_config(config: Config) -> None:
    if config.ad_mode != "none":
        raise RuntimeError("MC basic ad_mode must be 'none'")


def _receiver_count(scene: Scene) -> int:
    return sum(
        int(receiver.shape[0]) * int(receiver.shape[1])
        if isinstance(receiver, ReceiverGrid)
        else 1
        for receiver in scene.receivers
    )


def _enforce_workspace_budget(scene: Scene, config: Config) -> None:
    if config.workspace_limit_bytes is None:
        return
    estimate = estimate_monte_carlo_memory(
        samples=config.samples,
        transmitters=len(scene.transmitters),
        receivers=_receiver_count(scene),
        depth=config.max_depth,
    )
    enforce_memory_budget(
        estimate,
        budget_bytes=config.workspace_limit_bytes,
        workload="workspace for Monte Carlo basic",
    )


def _host_material_tensors(scene: Scene) -> tuple[torch.Tensor, ...]:
    material_eps_r: list[float] = []
    material_sigma_e: list[float] = []
    material_mu_r: list[float] = []
    face_material_id: list[int] = []
    for material_id, structure in enumerate(scene.structures):
        params = structure.material.parameters(scene.frequency)
        material_eps_r.append(float(params["eps_r"]))
        material_sigma_e.append(effective_sigma_e(params))
        material_mu_r.append(float(params["mu_r"]))
        face_material_id.extend([material_id] * int(structure.faces.shape[0]))
    if not material_eps_r:
        material_eps_r = [1.0]
        material_sigma_e = [0.0]
        material_mu_r = [1.0]
    exported = bdpt_face_material_tensors_from_host(
        tuple(material_eps_r),
        tuple(material_sigma_e),
        tuple(material_mu_r),
        tuple(face_material_id),
    )
    return (
        exported["eps_r"],
        exported["sigma_e"],
        exported["mu_r"],
        exported["gain"],
        exported["valid"],
    )


def solve(scene: Scene, config: Config) -> Result:
    _enforce_workspace_budget(scene, config)
    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel_native.montecarlo.basic requires CUDA")
    _validate_ad_config(config)

    info = build_info()
    reflection_available = bool(info["uses_raydn_native"])
    diffraction_available = bool(info["uses_raydn_native"])
    transmission_available = bool(info["uses_raydn_native"])
    if "reflection" in config.components and not reflection_available:
        raise RuntimeError("reflection requires RayDN native capability")
    if "diffraction" in config.components and not diffraction_available:
        raise RuntimeError("diffraction requires RayDN native capability")
    if "transmission" in config.components and not transmission_available:
        raise RuntimeError("transmission requires RayDN native capability")

    device = torch.device("cuda")
    make_cuda_generator(config.seed)
    raydn = scene.raydn_scene()
    grid = first_receiver_grid(scene)
    if "los" in config.components:
        los = los_path_gain(scene, device=device)
        if grid is None:
            los = apply_point_los_visibility(scene, raydn, los, device=device)
        path_count = config.samples
        valid_contribution_count = config.samples
    else:
        tx_count = len(scene.transmitters)
        rx_count = sum(
            receiver.points().shape[0] if hasattr(receiver, "points") else 1
            for receiver in scene.receivers
        )
        reference = torch.empty((1, 1), device=device, dtype=torch.float32)
        los = mc_zero_matrix(reference, rows=tx_count, cols=rx_count)
        path_count = 0
        valid_contribution_count = 0

    component_maps: dict[str, torch.Tensor] | None = None
    if grid is not None:
        component_maps = {}
        grid_dim0, grid_dim1 = component_grid_shape(grid)
        streaming_planar = config.reflection_accumulation_strategy == "streaming_planar"

        def zero_component_map() -> torch.Tensor:
            return mc_component_map_buffer(
                los,
                tx_count=len(scene.transmitters),
                dim0=grid_dim0,
                dim1=grid_dim1,
        )

        if "los" in config.components and not streaming_planar:
            component_maps["los"] = los_component_map(scene, raydn, grid, device=device, los=los)
        else:
            component_maps["los"] = zero_component_map()
        needs_reflection_launch = (
            reflection_available
            and (("reflection" in config.components) or ("diffraction" in config.components and diffraction_available))
        )
        material_tensors = None
        if needs_reflection_launch or ("diffraction" in config.components and diffraction_available):
            material_tensors = _host_material_tensors(scene)
        reflection_result = None
        collect_diffraction_wedges = "diffraction" in config.components and diffraction_available
        if needs_reflection_launch:
            if material_tensors is None:
                raise RuntimeError("material tensors are required for native reflection")
            reflection_result = reflection_component_maps_with_wedges(
                scene,
                raydn,
                grid,
                samples=config.samples,
                max_depth=config.max_depth,
                seed=config.seed,
                device=device,
                material_tensors=material_tensors,
                collect_wedges=collect_diffraction_wedges,
                reflection_accumulation_strategy=config.reflection_accumulation_strategy,
                reflection_compact_min_samples=config.reflection_compact_min_samples,
                reflection_staged_min_samples_per_cell=config.reflection_staged_min_samples_per_cell,
                streaming_los_enabled=("los" in config.components),
            )
        if "reflection" in config.components and reflection_available and reflection_result is not None:
            component_maps["reflection"] = reflection_result.maps
            path_count += config.samples
            valid_contribution_count += int(component_maps["reflection"].numel())
        else:
            component_maps["reflection"] = zero_component_map()
        if "diffraction" in config.components and diffraction_available:
            if material_tensors is None:
                raise RuntimeError("material tensors are required for native diffraction")
            component_maps["diffraction"] = diffraction_component_map(
                scene,
                raydn,
                grid,
                samples=config.samples,
                seed=config.seed,
                device=device,
                material_tensors=material_tensors,
                wedge_events=(
                    reflection_result.wedge_events
                    if reflection_result is not None and collect_diffraction_wedges
                    else None
                ),
            )
            path_count += config.samples
            valid_contribution_count += int(component_maps["diffraction"].numel())
        else:
            component_maps["diffraction"] = zero_component_map()
        if "transmission" in config.components and scene.structures:
            component_maps["transmission"] = transmission_component_map(
                scene,
                raydn,
                grid,
                max_depth=config.max_depth,
                device=device,
                los=los if "los" in config.components else None,
            )
            path_count += len(scene.transmitters) * grid_dim0 * grid_dim1
            valid_contribution_count += int(
                torch.count_nonzero(component_maps["transmission"])
            )
        elif "transmission" in config.components:
            component_maps["transmission"] = zero_component_map()
        # scattering is accepted plumbing in v1: emit zero grid maps when
        # requested until the physics lands.
        if "scattering" in config.components:
            component_maps["scattering"] = zero_component_map()

    path_gain = los
    if component_maps is not None:
        finalized = mc_finalize_component_maps(
            component_maps["los"],
            component_maps["reflection"],
            component_maps["diffraction"],
            component_maps.get("transmission", zero_component_map()),
        )
        path_gain = finalized["path_gain"]
        component_power = {
            "los": finalized["los_power"],
            "reflection": finalized["reflection_power"],
            "diffraction": finalized["diffraction_power"],
        }
        if "transmission" in config.components:
            component_power["transmission"] = finalized["transmission_power"]
    else:
        component_power = mc_point_component_power(los, include_los=("los" in config.components))
        # Point receivers carry no transmission map in MC basic; report zero.
        if "transmission" in config.components:
            component_power["transmission"] = torch.zeros_like(component_power["los"])
    # Zero point-power for the accepted-but-empty v1 scattering when requested.
    if "scattering" in config.components:
        component_power["scattering"] = torch.zeros_like(component_power["los"])
    metadata = make_solver_metadata(
        config=config,
        path_count=path_count,
        valid_contribution_count=valid_contribution_count,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
    )
    edge_policy = resolve_scene_edge_policy(scene)
    metadata["edge_policy"] = {
        "edge_selection_mode": edge_policy.edge_selection_mode,
        "edge_diffraction": bool(edge_policy.edge_diffraction),
        "boundary_edge_policy": edge_policy.boundary_edge_policy,
    }
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
