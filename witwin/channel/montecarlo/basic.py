# Copyright Xingyu Chen.
# Monte Carlo basic primal solver.

"""Monte Carlo basic primal solver."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, Any

import torch

from witwin.core import Scene, SceneSnapshot

from witwin.channel import build_info
from witwin.channel.capabilities import (
    capabilities,
    config_metadata,
    serialize_config,
)
from witwin.channel.components import (
    AD_MODES as _VALID_AD_MODES,
    DEFAULT_COMPONENTS as _DEFAULT_COMPONENTS,
    component_availability_status,
    component_max_depth,
    validate_bounce_depth,
    validate_max_depth,
    validate_samples,
    validate_seed,
    validate_workspace_limit_bytes,
    validated_components,
)
from witwin.channel.kernels import geometry as geometry_kernels
from witwin.channel.kernels import montecarlo as montecarlo_kernels
from witwin.channel.kernels.montecarlo import (
    mc_apply_los_visibility,
    mc_capacity_failure_component_maps_sanitize,
    mc_component_map_buffer,
    mc_finalize_component_maps,
    mc_finalize_component_maps_ad,
    mc_los_component_maps_from_matrix,
    mc_los_grid_maps_ad,
    mc_los_path_gain_ad,
    mc_los_visibility_inputs,
    mc_point_component_power,
    mc_reflection_ad_max_depth,
    mc_slab_reflection_accumulate,
    mc_slab_reflection_accumulate_ad,
    mc_store_component_map,
    mc_store_scaled_component_map,
    mc_transmission_wall_product,
    mc_transmission_wall_product_ad,
    mc_utd_diffraction_tape_accumulate,
    mc_utd_diffraction_tape_accumulate_ad,
    mc_zero_matrix,
)
from witwin.channel.kernels.topology import (
    deterministic_diffraction_state_pack,
    deterministic_diffraction_state_pack_selected,
    path_los_export,
)
from witwin.channel.materials import (
    _require_frequency_ad_constant_materials,
    face_material_field_bundle,
)
from witwin.channel.interactions.scattering import scattering_map_matrix
from witwin.channel.interactions.transmission import (
    straight_transmission_chains,
)
from witwin.channel.propagation.geometry import (
    cached_diffraction_edge_geometry as _cached_diffraction_edge_geometry,
)
from witwin.channel.runtime import (
    AdLaunchLedger,
    CapacityFailureState,
    SolveCapacityTransaction,
    create_solve_capacity_transaction,
    enforce_memory_budget,
    estimate_monte_carlo_memory,
    make_metadata,
)
from witwin.channel.scene.compiler import (
    LIGHT_SPEED_M_PER_S as _LIGHT_SPEED_M_PER_S,
    compile as compile_scene,
    receiver_grid_points,
    receiver_positions,
    transmitter_polarizations_as_stored,
    transmitter_positions,
)
from witwin.channel.scene.endpoints import (
    ReceiverGrid,
    _endpoint_views,
    _validate_scalar_endpoint_boundary,
    axis_aligned_grid_spec as grid_spec,
    bind_solver_scene,
    component_grid_shape,
    first_receiver_grid,
    receiver_positions_ad,
    require_compiled,
    scene_vertex_table,
    transmitter_positions_ad,
    validate_scalar_endpoint_features,
)
from witwin.channel.scene.resources import RayDSceneResource, resolve_scene_edge_policy

if TYPE_CHECKING:
    from witwin.channel.scene.endpoints import SolverScene


# --- Configuration --------------------------------------------------------

# Public component set. transmission traces straight penetration chains
# through up to max_depth walls (grid radiomaps only); scattering is accepted
# plumbing that emits zero maps until its wave lands. Both are surface events
# that require at least one bounce (max_depth >= 1).
# Default component set is unchanged: the new components are strictly opt-in.
@dataclass(frozen=True, slots=True)
class Config:
    samples: int = 4096
    max_depth: int = 1
    seed: int = 0
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str] = _DEFAULT_COMPONENTS
    diagnostics: bool = False
    ad_mode: str = "none"
    workspace_limit_bytes: int | None = 1 << 30

    def __post_init__(self) -> None:
        validate_samples(self.samples)
        validate_max_depth(self.max_depth)
        validate_seed(self.seed)
        components = validated_components(
            self.components, error_message="components must be a subset of {valid}"
        )
        validate_bounce_depth(
            self.max_depth,
            components,
            error_message="MC basic scattering requires max_depth >= 1",
        )
        if self.ad_mode not in _VALID_AD_MODES:
            raise ValueError(
                "montecarlo_basic ad_mode must be one of "
                f"{sorted(_VALID_AD_MODES)}"
            )
        validate_workspace_limit_bytes(self.workspace_limit_bytes)
        object.__setattr__(self, "components", components)


# --- Result ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Result:
    path_gain: torch.Tensor
    component_power: dict[str, torch.Tensor]
    metadata: dict[str, Any]
    diagnostics: dict[str, Any] | None = None
    component_maps: dict[str, torch.Tensor] | None = None


# --- Sampling -------------------------------------------------------------


def make_cuda_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return generator


# --- Solver metadata ------------------------------------------------------

# One AdLaunchLedger shape for every solver (diffraction AD): montecarlo.basic
# counts one companion per LoS matrix, per grid-map layout Function, per
# transmitter for the reflection/diffraction accumulators. Straight
# transmission registers one flattened RayD geometry companion, one native
# wall-product companion, and one final capacity-map sanitizer companion; the
# finalize sum registers no native companion (its cotangent is a view).
def make_solver_metadata(
    *,
    config: Config,
    path_count: int,
    contribution_capacity: int,
    reflection_available: bool,
    diffraction_available: bool,
    ad_ledger: AdLaunchLedger | None = None,
) -> dict[str, Any]:
    forward_launch_count = 1 if contribution_capacity else 0
    # solver derivatives: report the companion launches this solve actually
    # registered (see AdLaunchLedger), not the pre-design fused-launch
    # placeholder. ad_mode="none" wires no companions and retains no tape.
    ledger = ad_ledger if ad_ledger is not None else AdLaunchLedger()
    backward_launch_count = ledger.launches if config.ad_mode == "vjp" else 0
    jvp_launch_count = ledger.launches if config.ad_mode == "jvp" else 0
    tape_bytes = ledger.tape_bytes if config.ad_mode == "vjp" else 0
    rayd_component_enabled = (
        "reflection" in config.components and reflection_available
    ) or ("diffraction" in config.components and diffraction_available)
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
        "contribution_capacity": contribution_capacity,
        "components": component_availability_status(
            config.components,
            reflection_available=reflection_available,
            diffraction_available=diffraction_available,
            reflection_error="reflection requires RayD native capability",
            diffraction_error="diffraction requires RayD native capability",
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
            component_max_depth=component_max_depth(
                config.components,
                chain_depth=config.max_depth,
                single_bounce_depth=1,
            ),
        )
    )
    metadata["semantic_capabilities"] = capabilities()["solvers"]["montecarlo_basic"]
    return metadata


# --- LoS backend ----------------------------------------------------------


def los_path_gain(
    scene: SolverScene,
    *,
    device: torch.device,
    ad: bool = False,
    ledger: object | None = None,
) -> torch.Tensor:
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    rx_pos = receiver_positions(scene, device=device, reference=tx_pos)
    if tx_pos.shape[0] == 0 or rx_pos.shape[0] == 0:
        return mc_zero_matrix(tx_pos, rows=tx_pos.shape[0], cols=rx_pos.shape[0])
    # the true per-transmitter polarization drives the LoS dipole sin^2
    # pattern (frozen winner of AD; the pattern moves through the endpoints).
    tx_pol = transmitter_polarizations_as_stored(scene, device=device)

    if ad:
        # solver derivatives: swap the host-float endpoint tensors for the live
        # scene leaves (same float32 values) and route through the LoS AD
        # Function so tx/rx position and frequency gradients survive. Grid
        # receiver points stay native: a grid exposes no position leaf.
        tx_live = transmitter_positions_ad(scene, tx_pos, device=device)
        rx_live = receiver_positions_ad(scene, rx_pos, device=device)
        if ledger is not None:
            ledger.add(tx_live, tx_power, rx_live)  # type: ignore[attr-defined]
        return mc_los_path_gain_ad(
            tx_live, tx_power, rx_live, tx_pol, frequency=scene.frequency
        )
    exported = path_los_export(
        tx_pos,
        tx_power,
        rx_pos,
        tx_pol,
        frequency_hz=float(scene.frequency),
    )
    return exported["path_gain_matrix"]


def apply_point_los_visibility(scene: SolverScene, rayd: object, los: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    """Zero occluded (tx, rx) entries of a point-receiver LoS matrix."""

    if not scene.structures or los.numel() == 0:
        return los
    if not rayd.available:  # type: ignore[attr-defined]
        raise RuntimeError("LoS visibility requires RayD native scene capability")
    handle = rayd.require_resource()  # type: ignore[attr-defined]
    tx_pos, _ = transmitter_positions(scene, device=device)
    rx_pos = receiver_positions(scene, device=device, reference=tx_pos)
    masks: list[torch.Tensor] = []
    for tx_index in range(int(tx_pos.shape[0])):
        inputs = mc_los_visibility_inputs(tx_pos, tx_index=tx_index, rx_count=int(rx_pos.shape[0]))
        masks.append(
            geometry_kernels.rayd_visibility_forward(
                handle, inputs["start"], rx_pos, inputs["active"]
            )[0]
        )
    return los * torch.stack(masks, dim=0).to(dtype=los.dtype)


# --- RayD component maps --------------------------------------------------


def _frequency_scalar(scene: SolverScene) -> float:
    """Detached scalar carrier read (one host sync per solve at most)."""

    frequency = scene.frequency
    if isinstance(frequency, torch.Tensor):
        return float(frequency.detach())
    return float(frequency)


@dataclass(frozen=True, slots=True)
class WedgeEventBatch:
    tx_pos: torch.Tensor
    event_count: torch.Tensor | None
    ray_dir: torch.Tensor
    prim_id: torch.Tensor
    hit_p: torch.Tensor
    hit_n: torch.Tensor
    hit_geo_n: torch.Tensor
    bounce_depth: torch.Tensor


@dataclass(frozen=True, slots=True)
class ReflectionComponentResult:
    maps: torch.Tensor
    wedge_events: tuple[WedgeEventBatch, ...]


# Per-face material tensors from the compiled store (one source for both
# ad_mode="none" and the AD modes): eps_r, sigma_e, mu_r, gain, valid,
# thickness.
MaterialTensors = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _grid_los_matrix(
    scene: SolverScene,
    grid: ReceiverGrid,
    *,
    device: torch.device,
    los: torch.Tensor | None = None,
    ad: bool = False,
    ledger: object | None = None,
) -> torch.Tensor:
    if los is not None and len(scene.receivers) == 1 and scene.receivers[0] is grid:
        return los
    grid_scene = replace(scene, receivers=(grid,))
    return los_path_gain(grid_scene, device=device, ad=ad, ledger=ledger)


def _grid_visibility_masks(
    scene: SolverScene,
    rayd: RayDSceneResource,
    grid: ReceiverGrid,
    *,
    device: torch.device,
) -> torch.Tensor:
    handle = rayd.require_resource()
    tx_pos, _ = transmitter_positions(scene, device=device)
    rx_pos = receiver_grid_points(grid, reference=tx_pos)
    masks: list[torch.Tensor] = []
    for tx_index in range(tx_pos.shape[0]):
        inputs = mc_los_visibility_inputs(
            tx_pos, tx_index=tx_index, rx_count=rx_pos.shape[0]
        )
        masks.append(
            geometry_kernels.rayd_visibility_forward(
                handle, inputs["start"], rx_pos, inputs["active"]
            )[0]
        )
    return torch.stack(masks, dim=0)


def los_component_map(
    scene: SolverScene,
    rayd: RayDSceneResource,
    grid: ReceiverGrid,
    *,
    device: torch.device,
    los: torch.Tensor | None = None,
    ad: bool = False,
    ledger: object | None = None,
) -> torch.Tensor:
    los = _grid_los_matrix(scene, grid, device=device, los=los, ad=ad, ledger=ledger)
    if ad:
        # Same layout and visibility kernels behind one autograd Function:
        # the visibility mask is a frozen winner, the matrix keeps its graph.
        visible = None
        if scene.structures:
            if not rayd.available:
                raise RuntimeError(
                    "LoS visibility requires RayD native scene capability"
                )
            visible = _grid_visibility_masks(scene, rayd, grid, device=device)
        if ledger is not None:
            ledger.add(visible)  # type: ignore[attr-defined]
        return mc_los_grid_maps_ad(los, visible, rows=grid.shape[0], cols=grid.shape[1])
    maps = mc_los_component_maps_from_matrix(
        los, rows=grid.shape[0], cols=grid.shape[1]
    )
    if not scene.structures:
        return maps
    if not rayd.available:
        raise RuntimeError("LoS visibility requires RayD native scene capability")
    handle = rayd.require_resource()
    tx_pos, _ = transmitter_positions(scene, device=device)
    rx_pos = receiver_grid_points(grid, reference=tx_pos)
    for tx_index in range(tx_pos.shape[0]):
        inputs = mc_los_visibility_inputs(
            tx_pos, tx_index=tx_index, rx_count=rx_pos.shape[0]
        )
        visible = geometry_kernels.rayd_visibility_forward(
            handle, inputs["start"], rx_pos, inputs["active"]
        )[0]
        mc_apply_los_visibility(maps, los, visible, tx_index=tx_index)
    return maps


def _sample_directions(count: int, *, reference: torch.Tensor) -> torch.Tensor:
    return montecarlo_kernels.mc_sample_directions(count, reference)


def transmission_component_map(
    scene: SolverScene,
    rayd: RayDSceneResource,
    grid: ReceiverGrid,
    *,
    max_depth: int,
    device: torch.device,
    failure_state: CapacityFailureState,
    los: torch.Tensor | None = None,
    ad: bool = False,
    ledger: object | None = None,
) -> torch.Tensor:
    """Straight-penetration transmission radiomap (the transmission behavior,
 endpoint-connection context).

 Mirrors the LoS map's geometric convention exactly: the analytic per-cell
 Friis gain along the straight tx->cell segment, with the binary LoS
 visibility mask replaced by the native transmission polarization incident-polarized TE/TM
 wall product, evaluated in ascending resident-hit order. Cells whose
 segment crosses no wall belong to the exclusive los
 path class and stay zero here, so los + transmission never double counts.
 A single eps_r=1 vacuum wall has unit power transmittance, which makes
 this map reproduce the unobstructed LoS map exactly (acceptance test);
 the mandatory ``max_depth + 1`` probe makes over-capacity chains poison the
 shared solve transaction instead of returning a truncated map.
 """

    if not scene.structures:
        tx_pos, _ = transmitter_positions(scene, device=device)
        dim0, dim1 = component_grid_shape(grid)
        return mc_component_map_buffer(
            tx_pos, tx_count=len(scene.transmitters), dim0=dim0, dim1=dim1
        )
    if not rayd.available:
        raise RuntimeError("transmission requires RayD native scene capability")
    los_matrix = _grid_los_matrix(
        scene, grid, device=device, los=los, ad=ad, ledger=ledger
    )
    tx_pos, _ = transmitter_positions(scene, device=device)
    # Live transmitter origins in AD mode: the straight-line incidence cosine
    # (and with it every per-wall transmittance) moves with the transmitter,
    # so the chain march must see the graph, not the detached native table
    # (the derivative capability matrix TX x transmission for M).
    tx_march = transmitter_positions_ad(scene, tx_pos, device=device) if ad else tx_pos
    rx_pos = receiver_grid_points(grid, reference=tx_pos)
    if ad:
        bundle = face_material_field_bundle(scene, device=device)
    else:
        with torch.no_grad():
            bundle = face_material_field_bundle(scene, device=device)
    compiled = require_compiled(scene)
    rx_count = int(rx_pos.shape[0])
    tx_count = int(tx_pos.shape[0])
    origins = tx_march.repeat_interleave(rx_count, dim=0)
    targets = rx_pos.repeat(tx_count, 1)
    frequency_value = _frequency_scalar(scene)
    tx_pol = transmitter_polarizations_as_stored(scene, device=device)
    pair_polarization = tx_pol.repeat_interleave(rx_count, dim=0)
    base_power = los_matrix.view(-1)
    vertices = scene_vertex_table(scene, compiled) if ad else None
    if ad and ledger is not None:
        ledger.add(vertices, origins, targets)  # type: ignore[attr-defined]
    penetration = straight_transmission_chains(
        rayd,
        origins,
        targets,
        vertices=vertices,
        max_depth=int(max_depth),
        scene_diagonal=compiled.montecarlo_penetration_scene_diagonal_m,
        failure_state=failure_state,
        ad=ad,
    )
    wall_product_args = (
        penetration.valid,
        penetration.num_hits,
        penetration.reached_target,
        penetration.direction,
        penetration.normal,
        penetration.global_primitive_id,
        bundle["material_id"],
        bundle["geometry_mode_id"],
        bundle["layer_offset"],
        bundle["layer_count"],
        bundle["layer_thickness_m"],
        bundle["layer_eps_r"],
        bundle["layer_sigma_e"],
        bundle["layer_mu_r"],
        pair_polarization,
        base_power,
    )
    if ad:
        if ledger is not None:
            ledger.add(  # type: ignore[attr-defined]
                penetration.direction,
                penetration.normal,
                bundle["layer_thickness_m"],
                bundle["layer_eps_r"],
                bundle["layer_sigma_e"],
                base_power,
            )
        product = mc_transmission_wall_product_ad(
            *wall_product_args,
            scene.frequency,
            failure_state,
            frequency_value=frequency_value,
        )
    else:
        product = mc_transmission_wall_product(
            *wall_product_args,
            failure_state,
            frequency_hz=frequency_value,
        )
    matrix = product.scaled_power.view(tx_count, rx_count)
    if ad:
        if ledger is not None:
            ledger.add()  # type: ignore[attr-defined]
        return mc_los_grid_maps_ad(matrix, None, rows=grid.shape[0], cols=grid.shape[1])
    return mc_los_component_maps_from_matrix(
        matrix, rows=grid.shape[0], cols=grid.shape[1]
    )


def scattering_component_map(
    scene: SolverScene,
    rayd: RayDSceneResource,
    grid: ReceiverGrid,
    *,
    samples: int,
    seed: int,
    device: torch.device,
    ad: bool = False,
    ledger: object | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Kirchhoff diffuse scattering radiomap from area-sampled rough faces.

 Thin grid wrapper around:func:`witwin.channel.interactions.scattering.scattering_map_matrix`
 (which documents the estimator and its v1 simplifications): the matrix
 holds the per-cell scattering PATH GAIN at the cell center times the
 transmitter power, mirroring the LoS / transmission map conventions, so
 component_power equals the map sum.

 Under ``ad`` the matrix keeps its graph (table values, frequency and tx
 power gradients, scattering AD) and the grid layout runs behind the same
 ``mc_los_grid_maps_ad`` autograd Function the LoS/transmission maps use;
 the area-sample set, both visibility masks and the incidence gates stay
 frozen winners (they are folded into the matrix before layout, so the layout
 carries no separate visibility mask).
 """

    tx_pos, tx_power = transmitter_positions(scene, device=device)
    dim0, dim1 = component_grid_shape(grid)
    if not scene.structures:
        maps = mc_component_map_buffer(
            tx_pos, tx_count=len(scene.transmitters), dim0=dim0, dim1=dim1
        )
        return maps, {
            "sample_count": 0,
            "rough_face_count": 0,
            "tx_visible_samples": 0,
            "deposited_rows": 0,
        }
    if not rayd.available:
        raise RuntimeError("scattering requires RayD native scene capability")
    rx_pos = receiver_grid_points(grid, reference=tx_pos)
    matrix, stats = scattering_map_matrix(
        scene,
        rayd,
        tx_pos,
        tx_power,
        rx_pos,
        samples=samples,
        seed=seed,
        device=device,
        ad=ad,
        ledger=ledger,
    )
    if ad:
        if ledger is not None:
            ledger.add()  # type: ignore[attr-defined]
        return (
            mc_los_grid_maps_ad(matrix, None, rows=grid.shape[0], cols=grid.shape[1]),
            stats,
        )
    return (
        mc_los_component_maps_from_matrix(
            matrix, rows=grid.shape[0], cols=grid.shape[1]
        ),
        stats,
    )


def reflection_component_maps_with_wedges(
    scene: SolverScene,
    rayd: RayDSceneResource,
    grid: ReceiverGrid,
    *,
    samples: int,
    max_depth: int,
    device: torch.device,
    material_tensors: MaterialTensors,
    collect_wedges: bool = False,
    ad: bool = False,
    ledger: object | None = None,
) -> ReflectionComponentResult:
    if not scene.structures:
        tx_pos, _ = transmitter_positions(scene, device=device)
        dim0, dim1 = component_grid_shape(grid)
        maps = mc_component_map_buffer(
            tx_pos, tx_count=len(scene.transmitters), dim0=dim0, dim1=dim1
        )
        return ReflectionComponentResult(
            maps=maps,
            wedge_events=(),
        )
    if not rayd.available:
        raise RuntimeError("reflection requires RayD native scene capability")
    spec = grid_spec(grid)
    handle = rayd.require_resource()
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    tx_live = transmitter_positions_ad(scene, tx_pos, device=device) if ad else tx_pos
    # per-transmitter polarization seeds the reflection field's unnormalized
    # transverse projection (short-dipole sin(theta) pattern).
    tx_pol = transmitter_polarizations_as_stored(scene, device=device)
    wavelength = _LIGHT_SPEED_M_PER_S / _frequency_scalar(scene)
    (
        material_eta_r,
        material_sigma,
        _material_mu_r,
        material_gain,
        material_valid,
        material_thickness,
    ) = material_tensors
    face_normals = rayd.edge_records().face_normals
    solid_angle_per_ray = float(4.0 * math.pi / max(1, int(samples)))
    dim0, dim1 = component_grid_shape(grid)
    ad_maps: list[torch.Tensor] = []
    maps = None
    if not ad:
        maps = mc_component_map_buffer(
            tx_pos, tx_count=tx_pos.shape[0], dim0=dim0, dim1=dim1
        )
    wedge_batches: list[WedgeEventBatch] = []
    for tx_index, tx in enumerate(tx_pos):
        ray_d = _sample_directions(samples, reference=tx_pos)
        launch_inputs = montecarlo_kernels.mc_reflection_launch_inputs(
            tx_pos, tx_index=tx_index, sample_count=samples
        )
        ray_o = launch_inputs["ray_o"]
        ray_tmax = launch_inputs["ray_tmax"]
        active = launch_inputs["active"]
        trace = geometry_kernels.rayd_trace_reflections_forward(
            handle,
            ray_o,
            ray_d,
            ray_tmax,
            active,
            max(1, int(max_depth) + 1),
        )
        if ad:
            # Same accumulate kernel behind the autograd Function; the
            # tx_power scale and the per-tx stack below replicate the
            # mc_store_scaled_component_map arithmetic value for value.
            if ledger is not None:
                ledger.add(  # type: ignore[attr-defined]
                    ray_o,
                    ray_d,
                    trace[0],
                    trace[1],
                    trace[2],
                    face_normals,
                    material_eta_r,
                    material_sigma,
                    material_gain,
                    material_valid,
                    material_thickness,
                )
            reflection_map = mc_slab_reflection_accumulate_ad(
                tx_live,
                material_eta_r,
                material_sigma,
                material_gain,
                material_thickness,
                scene.frequency,
                ray_o,
                ray_d,
                trace[0],
                trace[1],
                trace[2],
                face_normals,
                material_valid,
                tx_pol=tx_pol[tx_index],
                contribution_depth=int(max_depth),
                grid_axis=int(spec.axis),
                grid_position=float(spec.position),
                grid_coord0_min=float(spec.coord0_min),
                grid_coord0_max=float(spec.coord0_max),
                grid_coord1_min=float(spec.coord1_min),
                grid_coord1_max=float(spec.coord1_max),
                grid_resolution0=int(spec.resolution0),
                grid_resolution1=int(spec.resolution1),
                wavelength=float(wavelength),
                solid_angle_per_ray=solid_angle_per_ray,
                grid_cell_area=float(spec.cell_area),
            )
            ad_maps.append(reflection_map * tx_power[tx_index])
        else:
            reflection_map = mc_slab_reflection_accumulate(
                ray_o,
                ray_d,
                trace[0],
                trace[1],
                trace[2],
                face_normals,
                material_eta_r,
                material_sigma,
                material_gain,
                material_valid,
                material_thickness,
                tx_pol=tx_pol[tx_index],
                contribution_depth=int(max_depth),
                grid_axis=int(spec.axis),
                grid_position=float(spec.position),
                grid_coord0_min=float(spec.coord0_min),
                grid_coord0_max=float(spec.coord0_max),
                grid_coord1_min=float(spec.coord1_min),
                grid_coord1_max=float(spec.coord1_max),
                grid_resolution0=int(spec.resolution0),
                grid_resolution1=int(spec.resolution1),
                wavelength=float(wavelength),
                solid_angle_per_ray=solid_angle_per_ray,
                grid_cell_area=float(spec.cell_area),
            )
            mc_store_scaled_component_map(
                maps,
                reflection_map,
                tx_power,
                tx_index=tx_index,
                scale_index=tx_index,
            )
        if collect_wedges:
            # Winner-event extraction on detached trace outputs (frozen
            # discovery bookkeeping in both primal and AD modes).
            valid_indices = torch.nonzero(trace[0][:, 0], as_tuple=False).flatten()
            prim_id = trace[2][:, 0].index_select(0, valid_indices)
            ray_dir = ray_d.index_select(0, valid_indices)
            hit_p = (
                ray_o.index_select(0, valid_indices)
                + trace[1][:, 0].index_select(0, valid_indices)[:, None] * ray_dir
            )
            hit_n = face_normals.detach().index_select(0, prim_id.to(dtype=torch.int64))
            wedge_batches.append(
                WedgeEventBatch(
                    tx_pos=tx,
                    event_count=None,
                    ray_dir=ray_dir,
                    prim_id=prim_id,
                    hit_p=hit_p,
                    hit_n=hit_n,
                    hit_geo_n=hit_n,
                    bounce_depth=torch.zeros_like(prim_id),
                )
            )
    if ad:
        maps = torch.stack(ad_maps, dim=0)
    return ReflectionComponentResult(maps=maps, wedge_events=tuple(wedge_batches))


def _native_surface_group_edge_candidates(  # type: ignore[no-untyped-def]
    records, selected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return geometry_kernels.mc_surface_group_edge_candidates(  # type: ignore[no-any-return]
        records.vertices,
        records.faces,
        records.face_normals,
        records.edge_v0,
        records.edge_v1,
        records.face0,
        records.face1,
        selected,
        plane_tol=1.0e-5,
    )


def _cached_primitive_edge_candidates(
    rayd: RayDSceneResource,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (int(selected.data_ptr()), int(selected.numel()))
    cache = rayd.runtime_cache
    cached = cache.get("mc_primitive_edge_candidates")
    if cached is not None:
        cached_key, cached_candidates = cached  # type: ignore[misc]
        if cached_key == key:
            return cached_candidates  # type: ignore[return-value, no-any-return]
    candidates = _native_surface_group_edge_candidates(rayd.edge_records(), selected)
    cache["mc_primitive_edge_candidates"] = (key, candidates)
    return candidates


def _discover_diffraction_edges_from_wedges(
    rayd: RayDSceneResource,
    wedges: WedgeEventBatch,
    edge_geometry: tuple[torch.Tensor, ...] | None = None,
    edge_candidates: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    (
        selected,
        edge_pos,
        edge_dir,
        _lengths,
        line_min,
        line_max,
        n0,
        n1,
        _face0,
        face1,
        _exterior_angle,
    ) = (
        edge_geometry
        if edge_geometry is not None
        else _cached_diffraction_edge_geometry(rayd)
    )
    if edge_candidates is None:
        edge_candidates = _cached_primitive_edge_candidates(rayd, selected)
    triangle_edge_count, triangle_edge_indices = edge_candidates
    if wedges.event_count is not None:
        return montecarlo_kernels.mc_diffraction_discover_edges_counted(
            wedges.tx_pos,
            wedges.ray_dir,
            wedges.prim_id,
            wedges.hit_p,
            wedges.hit_n,
            wedges.hit_geo_n,
            wedges.event_count,
            triangle_edge_count,
            triangle_edge_indices,
            edge_pos,
            edge_dir,
            n0,
            n1,
            line_min,
            line_max,
            face1,
        )
    return montecarlo_kernels.mc_diffraction_discover_edges(
        wedges.tx_pos,
        wedges.ray_dir,
        wedges.prim_id,
        wedges.hit_p,
        wedges.hit_n,
        wedges.hit_geo_n,
        triangle_edge_count,
        triangle_edge_indices,
        edge_pos,
        edge_dir,
        n0,
        n1,
        line_min,
        line_max,
        face1,
    )


def _diffraction_states_from_edge_indices(
    rayd: RayDSceneResource,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    tx_power_index: int,
    edge_indices: torch.Tensor,
    edge_geometry: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    (
        _selected,
        edge_pos,
        edge_dir,
        _lengths,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
    ) = (
        edge_geometry
        if edge_geometry is not None
        else _cached_diffraction_edge_geometry(rayd)
    )
    return deterministic_diffraction_state_pack(  # type: ignore[no-any-return]
        edge_indices,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        int(tx_power_index),
    )


def _diffraction_states(
    scene: SolverScene,
    rayd: RayDSceneResource,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    tx_power_index: int,
    edge_geometry: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    del scene
    geometry = (
        edge_geometry
        if edge_geometry is not None
        else _cached_diffraction_edge_geometry(rayd)
    )
    (
        selected,
        edge_pos,
        edge_dir,
        _lengths,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
    ) = geometry
    return deterministic_diffraction_state_pack_selected(  # type: ignore[no-any-return]
        selected,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        tx_power_index,
    )


def diffraction_component_map(
    scene: SolverScene,
    rayd: RayDSceneResource,
    grid: ReceiverGrid,
    *,
    samples: int,
    seed: int,
    device: torch.device,
    material_tensors: MaterialTensors,
    wedge_events: tuple[WedgeEventBatch, ...] | None = None,
    ad: bool = False,
    ledger: object | None = None,
) -> torch.Tensor:
    if not scene.structures:
        tx_pos, _ = transmitter_positions(scene, device=device)
        dim0, dim1 = component_grid_shape(grid)
        return mc_component_map_buffer(
            tx_pos, tx_count=len(scene.transmitters), dim0=dim0, dim1=dim1
        )
    if not rayd.available:
        raise RuntimeError("diffraction requires RayD native scene capability")
    spec = grid_spec(grid)
    handle = rayd.require_resource()
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    tx_live = transmitter_positions_ad(scene, tx_pos, device=device) if ad else tx_pos
    # per-transmitter polarization fed into direct_source_vector's incident
    # basis (replaces the fabricated z-axis).
    tx_pol = transmitter_polarizations_as_stored(scene, device=device)
    (
        material_eta_r,
        material_sigma,
        material_mu_r,
        material_gain,
        material_valid,
        material_thickness,
    ) = material_tensors
    wavelength = _LIGHT_SPEED_M_PER_S / _frequency_scalar(scene)
    dim0, dim1 = component_grid_shape(grid)
    ad_maps: list[torch.Tensor] = []
    maps = None
    if not ad:
        maps = mc_component_map_buffer(
            tx_pos, tx_count=tx_pos.shape[0], dim0=dim0, dim1=dim1
        )
    edge_geometry: tuple[torch.Tensor, ...] | None = None
    edge_candidates: tuple[torch.Tensor, torch.Tensor] | None = None
    mitsuba_metadata = scene.metadata.get("mitsuba", {})
    preserve_imported_edges = isinstance(mitsuba_metadata, dict) and bool(
        mitsuba_metadata.get("merge_shapes", False)
    )

    def get_edge_geometry() -> tuple[torch.Tensor, ...]:
        nonlocal edge_geometry
        if edge_geometry is None:
            edge_geometry = _cached_diffraction_edge_geometry(
                rayd,
                preserve_imported_edges=preserve_imported_edges,
            )
        return edge_geometry

    def get_edge_candidates(
        geometry: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal edge_candidates
        if edge_candidates is None:
            edge_candidates = _cached_primitive_edge_candidates(rayd, geometry[0])
        return edge_candidates

    def zero_map() -> torch.Tensor:
        return mc_component_map_buffer(tx_pos, tx_count=1, dim0=dim0, dim1=dim1)[0]

    for tx_index, tx in enumerate(tx_pos):
        if wedge_events is not None:
            if int(wedge_events[tx_index].prim_id.numel()) == 0:
                if ad:
                    ad_maps.append(zero_map())
                continue
            geometry = get_edge_geometry()
            edge_indices = _discover_diffraction_edges_from_wedges(
                rayd,
                wedge_events[tx_index],
                edge_geometry=geometry,
                edge_candidates=get_edge_candidates(geometry),
            )
            if int(edge_indices.numel()) == 0:
                if ad:
                    ad_maps.append(zero_map())
                continue
            states = _diffraction_states_from_edge_indices(
                rayd,
                tx,
                tx_power,
                tx_index,
                edge_indices,
                edge_geometry=geometry,
            )
        else:
            states = _diffraction_states(
                scene,
                rayd,
                tx,
                tx_power,
                tx_index,
                edge_geometry=get_edge_geometry(),
            )
        state_count = int(states[0].shape[0])
        if state_count <= 0:
            if ad:
                ad_maps.append(zero_map())
            continue
        edge_lengths = (states[4] - states[3]).clamp_min(0.0)
        # One intentional device-to-host sync per transmitter. The host scalar
        # gates the per-tx early-out below and sets the float32 edge-weight
        # fill through a float64 divide (total_edge_length / samples). Keeping
        # the reduction on the host preserves that double-rounded fill value; a
        # GPU float32 scalar divide would single-round and is not bitwise
        # identical, so this sync is deliberate rather than an oversight.
        total_edge_length = float(edge_lengths.sum().item())
        if not total_edge_length > 0.0:
            if ad:
                ad_maps.append(zero_map())
            continue
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        sample_state_index = torch.multinomial(
            edge_lengths,
            int(samples),
            replacement=True,
            generator=generator,
        ).to(dtype=torch.int32)
        sample_edge_weight = torch.full(
            (int(samples),),
            total_edge_length / float(samples),
            device=device,
            dtype=torch.float32,
        )
        state_wi = montecarlo_kernels.mc_diffraction_state_wi(states[1], states[10])
        sampled = geometry_kernels.rayd_diffraction_sample_tape_forward(
            handle,
            None,
            *states,
            state_wi,
            state_wi,
            material_eta_r,
            material_sigma,
            material_mu_r,
            material_gain,
            material_valid,
            state_count,
            int(spec.axis),
            float(spec.position),
            float(spec.coord0_min),
            float(spec.coord0_max),
            float(spec.coord1_min),
            float(spec.coord1_max),
            int(spec.resolution0),
            int(spec.resolution1),
            float(spec.cell_area),
            float(wavelength),
            0,
            int(samples),
            0,
            int(seed),
            1,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            sample_state_index,
            sample_edge_weight,
        )
        if ad:
            # Same tape-accumulate kernel behind the autograd Function
            # (diffraction AD): the RayD sampling tape and the packed edge
            # states are frozen winners; the anchor keeps the transmitter
            # graph, the store leaves keep the material graph and the live
            # frequency carries the carrier graph.
            tape_tensors = (sampled[14], sampled[15], sampled[16], sampled[18])
            state_tensors = tuple(states[1:12])
            if ledger is not None:
                ledger.add(  # type: ignore[attr-defined]
                    *tape_tensors,
                    *state_tensors,
                    material_eta_r,
                    material_sigma,
                    material_mu_r,
                    material_gain,
                    material_valid,
                    material_thickness,
                )
            diffraction_map = mc_utd_diffraction_tape_accumulate_ad(
                tx_live[tx_index],
                material_eta_r,
                material_sigma,
                material_gain,
                material_thickness,
                scene.frequency,
                tape_tensors,
                state_tensors,
                material_mu_r,
                material_valid,
                tx_pol=tx_pol[tx_index],
                grid_axis=int(spec.axis),
                grid_position=float(spec.position),
                grid_coord0_min=float(spec.coord0_min),
                grid_coord0_max=float(spec.coord0_max),
                grid_coord1_min=float(spec.coord1_min),
                grid_coord1_max=float(spec.coord1_max),
                grid_resolution0=int(spec.resolution0),
                grid_resolution1=int(spec.resolution1),
                wavelength=float(wavelength),
                grid_cell_area=float(spec.cell_area),
                seed=int(seed),
                total_edge_length=total_edge_length,
            )
            ad_maps.append(diffraction_map)
            continue
        diffraction_map = mc_utd_diffraction_tape_accumulate(
            sampled[14],
            sampled[15],
            sampled[16],
            sampled[18],
            states[1],
            states[2],
            states[3],
            states[4],
            states[5],
            states[6],
            states[7],
            states[8],
            states[9],
            states[10],
            states[11],
            material_eta_r,
            material_sigma,
            material_mu_r,
            material_gain,
            material_valid,
            material_thickness,
            int(spec.axis),
            float(spec.position),
            float(spec.coord0_min),
            float(spec.coord0_max),
            float(spec.coord1_min),
            float(spec.coord1_max),
            int(spec.resolution0),
            int(spec.resolution1),
            float(wavelength),
            float(spec.cell_area),
            int(seed),
            total_edge_length,
            tx_pol[tx_index],
        )
        mc_store_component_map(maps, diffraction_map, tx_index=tx_index)
    if ad:
        maps = torch.stack(ad_maps, dim=0)
    return maps


# --- Shared solve pipeline ------------------------------------------------


def _validate_ad_config(config: Config) -> None:
    if config.ad_mode == "none":
        return
    if "reflection" in config.components:
        depth_cap = mc_reflection_ad_max_depth()
        if config.max_depth > depth_cap:
            raise RuntimeError(
                f"MC basic ad_mode='{config.ad_mode}' supports the reflection "
                f"component only up to max_depth={depth_cap} (native "
                f"reflection AD depth cap); got max_depth={config.max_depth}"
            )


def _receiver_count(scene: SolverScene) -> int:
    return sum(
        int(receiver.shape[0]) * int(receiver.shape[1])
        if isinstance(receiver, ReceiverGrid)
        else 1
        for receiver in scene.receivers
    )


def _enforce_workspace_budget(scene: SolverScene, config: Config) -> None:
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
    scene: SolverScene, *, device: torch.device, ad: bool
) -> tuple[torch.Tensor, ...]:
    """Per-face material tensors from the compiled material store.

 One material source for both ad_mode="none" and the AD modes (solver derivatives): the compiled store leaves (``scene.compiled.materials``) are the values the
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


def _mc_scattering_component(  # type: ignore[no-untyped-def]
    scene: SolverScene,
    rayd,
    grid,
    config: Config,
    *,
    device: torch.device,
    ad: bool,
    ledger,
    zero_component_map,
) -> tuple[torch.Tensor, dict[str, int] | None, int, int]:
    """Scattering component map and (path, capacity) row-count deltas.

 Grid receivers with structures carry the native scattering map; otherwise the
 component is a zero map with no row-count contribution. Preserves the exact
 call semantics and ad/ledger threading of the inline dispatch it replaces.
 """

    if scene.structures:
        component_map, stats = scattering_component_map(
            scene,
            rayd,
            grid,
            samples=config.samples,
            seed=config.seed,
            device=device,
            ad=ad,
            ledger=ledger if ad else None,
        )
        path_delta = stats["sample_count"]
        capacity_delta = int(component_map.numel())
        return component_map, stats, path_delta, capacity_delta
    return zero_component_map(), None, 0, 0


def _validate_native_component_capabilities(
    config: Config,
    *,
    reflection_available: bool,
    diffraction_available: bool,
    transmission_available: bool,
    scattering_available: bool,
) -> None:
    if "reflection" in config.components and not reflection_available:
        raise RuntimeError("reflection requires RayD native capability")
    if "diffraction" in config.components and not diffraction_available:
        raise RuntimeError("diffraction requires RayD native capability")
    if "transmission" in config.components and not transmission_available:
        raise RuntimeError("transmission requires RayD native capability")
    if "scattering" in config.components and not scattering_available:
        raise RuntimeError("scattering requires RayD native capability")


def _initial_los_state(
    scene: SolverScene,
    config: Config,
    *,
    device: torch.device,
    rayd: object,
    grid: ReceiverGrid | None,
    ad: bool,
    ledger: AdLaunchLedger,
) -> tuple[torch.Tensor, int, int]:
    if "los" in config.components:
        los = los_path_gain(scene, device=device, ad=ad, ledger=ledger if ad else None)
        if grid is None:
            los = apply_point_los_visibility(scene, rayd, los, device=device)
        return los, config.samples, config.samples

    tx_count = len(scene.transmitters)
    rx_count = sum(
        int(receiver.shape[0]) * int(receiver.shape[1])
        if isinstance(receiver, ReceiverGrid)
        else 1
        for receiver in scene.receivers
    )
    reference = torch.empty((1, 1), device=device, dtype=torch.float32)
    los = mc_zero_matrix(reference, rows=tx_count, cols=rx_count)
    return los, 0, 0


def solve_pipeline(  # type: ignore[no-untyped-def]
    scene: SolverScene,
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
        raise RuntimeError("witwin.channel.montecarlo.basic requires CUDA")
    _validate_ad_config(config)
    ad = config.ad_mode != "none"
    if ad:
        require_frequency_ad_constant_materials_fn(
            scene, require_compiled(scene), ad_mode=config.ad_mode
        )

    info = build_info_fn()
    reflection_available = bool(info["uses_rayd_native"])
    diffraction_available = bool(info["uses_rayd_native"])
    transmission_available = bool(info["uses_rayd_native"])
    scattering_available = bool(info["uses_rayd_native"])
    _validate_native_component_capabilities(
        config,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        transmission_available=transmission_available,
        scattering_available=scattering_available,
    )

    device = torch.device("cuda")
    make_cuda_generator_fn(config.seed)
    rayd = require_compiled(scene).rayd
    grid = first_receiver_grid(scene)
    ledger = AdLaunchLedger()
    los, path_count, contribution_capacity = _initial_los_state(
        scene,
        config,
        device=device,
        rayd=rayd,
        grid=grid,
        ad=ad,
        ledger=ledger,
    )

    component_maps: dict[str, torch.Tensor] | None = None
    scattering_stats: dict[str, int] | None = None
    capacity_transaction: SolveCapacityTransaction | None = None
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
                rayd,
                grid,
                device=device,
                los=los,
                ad=ad,
                ledger=ledger if ad else None,
            )
        else:
            component_maps["los"] = zero_component_map()
        needs_reflection_launch = reflection_available and (
            ("reflection" in config.components)
            or ("diffraction" in config.components and diffraction_available)
        )
        material_tensors = None
        if needs_reflection_launch or (
            "diffraction" in config.components and diffraction_available
        ):
            material_tensors = _face_material_tensors(scene, device=device, ad=ad)
        reflection_result = None
        collect_diffraction_wedges = (
            "diffraction" in config.components and diffraction_available
        )
        if needs_reflection_launch:
            if material_tensors is None:
                raise RuntimeError(
                    "material tensors are required for native reflection"
                )
            # A diffraction-only AD solve still needs the reflection launch
            # for wedge discovery, but its (discarded) reflection map stays
            # primal so no spurious companions are registered.
            reflection_ad = ad and "reflection" in config.components
            reflection_result = reflection_component_maps_with_wedges(
                scene,
                rayd,
                grid,
                samples=config.samples,
                max_depth=config.max_depth,
                device=device,
                material_tensors=material_tensors,
                collect_wedges=collect_diffraction_wedges,
                ad=reflection_ad,
                ledger=ledger if reflection_ad else None,
            )
        if (
            "reflection" in config.components
            and reflection_available
            and reflection_result is not None
        ):
            component_maps["reflection"] = reflection_result.maps
            path_count += config.samples
            contribution_capacity += int(component_maps["reflection"].numel())
        else:
            component_maps["reflection"] = zero_component_map()
        if "diffraction" in config.components and diffraction_available:
            if material_tensors is None:
                raise RuntimeError(
                    "material tensors are required for native diffraction"
                )
            component_maps["diffraction"] = diffraction_component_map(
                scene,
                rayd,
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
            contribution_capacity += int(component_maps["diffraction"].numel())
        else:
            component_maps["diffraction"] = zero_component_map()
        transmission_has_rows = (
            "transmission" in config.components
            and bool(scene.structures)
            and len(scene.transmitters) > 0
            and grid_dim0 * grid_dim1 > 0
        )
        if transmission_has_rows:
            capacity_transaction = create_solve_capacity_transaction(los)
            component_maps["transmission"] = transmission_component_map(
                scene,
                rayd,
                grid,
                max_depth=config.max_depth,
                device=device,
                failure_state=capacity_transaction.failure_state,
                los=los if "los" in config.components else None,
                ad=ad,
                ledger=ledger if ad else None,
            )
            path_count += len(scene.transmitters) * grid_dim0 * grid_dim1
            contribution_capacity += int(component_maps["transmission"].numel())
        elif "transmission" in config.components:
            component_maps["transmission"] = zero_component_map()
        if "scattering" in config.components:
            (
                component_maps["scattering"],
                scattering_stats,
                scattering_path_delta,
                scattering_capacity_delta,
            ) = _mc_scattering_component(
                scene,
                rayd,
                grid,
                config,
                device=device,
                ad=ad,
                ledger=ledger,
                zero_component_map=zero_component_map,
            )
            path_count += scattering_path_delta
            contribution_capacity += scattering_capacity_delta

    path_gain = los
    if component_maps is not None:
        transmission_map = (
            component_maps["transmission"]
            if "transmission" in component_maps
            else zero_component_map()
        )
        scattering_map = (
            component_maps["scattering"]
            if "scattering" in component_maps
            else zero_component_map()
        )
        final_component_maps = {
            "los": component_maps["los"],
            "reflection": component_maps["reflection"],
            "diffraction": component_maps["diffraction"],
            "transmission": transmission_map,
            "scattering": scattering_map,
        }
        if capacity_transaction is not None:
            if ad:
                ledger.add(*final_component_maps.values())
            final_component_maps = mc_capacity_failure_component_maps_sanitize(
                final_component_maps["los"],
                final_component_maps["reflection"],
                final_component_maps["diffraction"],
                final_component_maps["transmission"],
                final_component_maps["scattering"],
                failure_state=capacity_transaction.failure_state,
            )
            for name in tuple(component_maps):
                component_maps[name] = final_component_maps[name]
        finalize = mc_finalize_component_maps_ad if ad else mc_finalize_component_maps
        finalized = finalize(
            final_component_maps["los"],
            final_component_maps["reflection"],
            final_component_maps["diffraction"],
            final_component_maps["transmission"],
            final_component_maps["scattering"],
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
        contribution_capacity=contribution_capacity,
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
                None
                if component_maps is None
                else {key: tuple(value.shape) for key, value in component_maps.items()}
            ),
        }
    result = Result(
        path_gain=path_gain,
        component_power=component_power,
        metadata=metadata,
        diagnostics=diagnostics,
        component_maps=component_maps,
    )
    if capacity_transaction is not None:
        capacity_transaction.terminal_check()
    return result


# --- Public entry point ---------------------------------------------------


def solve(  # type: ignore[no-untyped-def]
    scene: Scene | SceneSnapshot,
    config: Config,
    *,
    reference_frequency_hz,
) -> Result:
    """Run the Monte Carlo Basic solver pipeline."""

    endpoint_views = _endpoint_views(scene)
    _validate_scalar_endpoint_boundary(endpoint_views)
    validate_scalar_endpoint_features(
        tuple(view for view in endpoint_views if view.source.role == "tx"),
        tuple(view for view in endpoint_views if view.source.role == "rx"),
        solver="Monte Carlo basic",
    )
    compiled = compile_scene(
        scene, reference_frequency_hz=reference_frequency_hz
    )
    return solve_pipeline(
        bind_solver_scene(compiled),
        config,
        build_info_fn=build_info,
        make_cuda_generator_fn=make_cuda_generator,
        validate_scalar_endpoint_features_fn=validate_scalar_endpoint_features,
        require_frequency_ad_constant_materials_fn=(
            _require_frequency_ad_constant_materials
        ),
    )


__all__ = ["Config", "Result", "solve"]