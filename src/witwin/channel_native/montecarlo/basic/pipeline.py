from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from witwin.channel_native import build_info
from witwin.channel_native.scene.models import ReceiverGrid
from witwin.channel_native.core.antenna import validate_scalar_endpoint_features
from witwin.channel_native.core.edge_selection import resolve_scene_edge_policy
from witwin.channel_native.materials.encoding import face_material_field_bundle
from witwin.channel_native.core.memory_budget import (
    enforce_memory_budget,
    estimate_monte_carlo_memory,
)
from witwin.channel_native.materials.evaluation import (
    _require_frequency_ad_constant_materials,
)
from witwin.channel_native.core.receiver_geometry import first_receiver_grid
from witwin.channel_native.montecarlo.basic.kernels.maps import (
    mc_component_map_buffer,
    mc_finalize_component_maps,
    mc_finalize_component_maps_ad,
    mc_point_component_power,
    mc_reflection_ad_max_depth,
    mc_zero_matrix,
)
from .backend import apply_point_los_visibility, los_path_gain
from .config import Config
from .metadata import AdLaunchLedger, make_solver_metadata
from .raydn_components import (
    component_grid_shape,
    diffraction_component_map,
    los_component_map,
    reflection_component_maps_with_wedges,
    scattering_component_map,
    transmission_component_map,
)

from .result import Result
from .sampling import make_cuda_generator

if TYPE_CHECKING:
    from witwin.channel_native.scene.models import Scene


# Components whose Monte Carlo power maps have no AD companions yet: the
# Kirchhoff scattering map (deferred past plan 07 AD-4). Fail before any
# launch instead of returning silently detached maps.
_AD_PENDING_COMPONENTS = ("scattering",)


def _validate_ad_config(config: Config) -> None:
    if config.ad_mode == "none":
        return
    for name in _AD_PENDING_COMPONENTS:
        if name in config.components:
            raise RuntimeError(
                f"MC basic ad_mode='{config.ad_mode}' does not support the "
                f"{name} component yet (plan 07 AD-4)"
            )
    if "reflection" in config.components:
        depth_cap = mc_reflection_ad_max_depth()
        if config.max_depth > depth_cap:
            raise RuntimeError(
                f"MC basic ad_mode='{config.ad_mode}' supports the reflection "
                f"component only up to max_depth={depth_cap} (native "
                f"reflection AD depth cap); got max_depth={config.max_depth}"
            )


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


def _face_material_tensors(
    scene: Scene, *, device: torch.device, ad: bool
) -> tuple[torch.Tensor, ...]:
    """Per-face material tensors from the compiled material store.

    One material source for both ad_mode="none" and the AD modes (plan 07
    AD-3): the store leaves (``scene.compile().materials``) are the values the
    kernels see, so a finite difference taken on the store measures the same
    function the AD modes differentiate. The primal path reads under no_grad
    so it never builds a graph.
    """

    def build() -> tuple[torch.Tensor, ...]:
        bundle = face_material_field_bundle(scene, device=device)
        return (
            bundle["eps_r"],
            bundle["sigma_e"],
            bundle["mu_r"],
            bundle["gain"],
            bundle["valid"],
            bundle["thickness"],
        )

    if ad:
        return build()
    with torch.no_grad():
        return build()


def solve_pipeline(
    scene: Scene,
    config: Config,
    *,
    build_info_fn=build_info,
    make_cuda_generator_fn=make_cuda_generator,
    validate_scalar_endpoint_features_fn=validate_scalar_endpoint_features,
    require_frequency_ad_constant_materials_fn=(
        _require_frequency_ad_constant_materials
    ),
) -> Result:
    validate_scalar_endpoint_features_fn(
        scene.transmitters, scene.receivers, solver="Monte Carlo basic"
    )
    _enforce_workspace_budget(scene, config)
    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel_native.montecarlo.basic requires CUDA")
    _validate_ad_config(config)
    ad = config.ad_mode != "none"
    if ad:
        require_frequency_ad_constant_materials_fn(
            scene, scene.compile(), ad_mode=config.ad_mode
        )

    info = build_info_fn()
    reflection_available = bool(info["uses_raydn_native"])
    diffraction_available = bool(info["uses_raydn_native"])
    transmission_available = bool(info["uses_raydn_native"])
    scattering_available = bool(info["uses_raydn_native"])
    if "reflection" in config.components and not reflection_available:
        raise RuntimeError("reflection requires RayDN native capability")
    if "diffraction" in config.components and not diffraction_available:
        raise RuntimeError("diffraction requires RayDN native capability")
    if "transmission" in config.components and not transmission_available:
        raise RuntimeError("transmission requires RayDN native capability")
    if "scattering" in config.components and not scattering_available:
        raise RuntimeError("scattering requires RayDN native capability")

    device = torch.device("cuda")
    make_cuda_generator_fn(config.seed)
    raydn = scene.raydn_scene()
    grid = first_receiver_grid(scene)
    ledger = AdLaunchLedger()
    if "los" in config.components:
        los = los_path_gain(
            scene, device=device, ad=ad, ledger=ledger if ad else None
        )
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
    scattering_stats: dict[str, int] | None = None
    if grid is not None:
        component_maps = {}
        grid_dim0, grid_dim1 = component_grid_shape(grid)
        def zero_component_map() -> torch.Tensor:
            return mc_component_map_buffer(
                los,
                tx_count=len(scene.transmitters),
                dim0=grid_dim0,
                dim1=grid_dim1,
        )

        if "los" in config.components:
            component_maps["los"] = los_component_map(
                scene,
                raydn,
                grid,
                device=device,
                los=los,
                ad=ad,
                ledger=ledger if ad else None,
            )
        else:
            component_maps["los"] = zero_component_map()
        needs_reflection_launch = (
            reflection_available
            and (("reflection" in config.components) or ("diffraction" in config.components and diffraction_available))
        )
        material_tensors = None
        if needs_reflection_launch or ("diffraction" in config.components and diffraction_available):
            material_tensors = _face_material_tensors(scene, device=device, ad=ad)
        reflection_result = None
        collect_diffraction_wedges = "diffraction" in config.components and diffraction_available
        if needs_reflection_launch:
            if material_tensors is None:
                raise RuntimeError("material tensors are required for native reflection")
            # A diffraction-only AD solve still needs the reflection launch
            # for wedge discovery, but its (discarded) reflection map stays
            # primal so no spurious companions are registered.
            reflection_ad = ad and "reflection" in config.components
            reflection_result = reflection_component_maps_with_wedges(
                scene,
                raydn,
                grid,
                samples=config.samples,
                max_depth=config.max_depth,
                device=device,
                material_tensors=material_tensors,
                collect_wedges=collect_diffraction_wedges,
                ad=reflection_ad,
                ledger=ledger if reflection_ad else None,
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
                ad=ad,
                ledger=ledger if ad else None,
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
                ad=ad,
                ledger=ledger if ad else None,
            )
            path_count += len(scene.transmitters) * grid_dim0 * grid_dim1
            valid_contribution_count += int(
                torch.count_nonzero(component_maps["transmission"])
            )
        elif "transmission" in config.components:
            component_maps["transmission"] = zero_component_map()
        if "scattering" in config.components and scene.structures:
            component_maps["scattering"], scattering_stats = scattering_component_map(
                scene,
                raydn,
                grid,
                samples=config.samples,
                seed=config.seed,
                device=device,
            )
            path_count += scattering_stats["sample_count"]
            valid_contribution_count += int(
                torch.count_nonzero(component_maps["scattering"])
            )
        elif "scattering" in config.components:
            component_maps["scattering"] = zero_component_map()

    path_gain = los
    if component_maps is not None:
        finalize = mc_finalize_component_maps_ad if ad else mc_finalize_component_maps
        finalized = finalize(
            component_maps["los"],
            component_maps["reflection"],
            component_maps["diffraction"],
            component_maps.get("transmission", zero_component_map()),
            component_maps.get("scattering", zero_component_map()),
        )
        path_gain = finalized["path_gain"]
        component_power = {
            "los": finalized["los_power"],
            "reflection": finalized["reflection_power"],
            "diffraction": finalized["diffraction_power"],
        }
        if "transmission" in config.components:
            component_power["transmission"] = finalized["transmission_power"]
        if "scattering" in config.components:
            component_power["scattering"] = finalized["scattering_power"]
    else:
        component_power = mc_point_component_power(
            los.detach() if ad else los, include_los=("los" in config.components)
        )
        # Point receivers carry no transmission or scattering map in MC
        # basic; report zero (grid receivers carry the real physics).
        if "transmission" in config.components:
            component_power["transmission"] = torch.zeros_like(component_power["los"])
        if "scattering" in config.components:
            component_power["scattering"] = torch.zeros_like(component_power["los"])
    metadata = make_solver_metadata(
        config=config,
        path_count=path_count,
        valid_contribution_count=valid_contribution_count,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        ad_ledger=ledger,
    )
    if "scattering" in config.components:
        metadata["scattering"] = {
            "component_mask_bit": 16,
            "max_scattering_order": 1,
            **(scattering_stats or {}),
        }
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
