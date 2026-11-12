# Copyright Xingyu Chen.
# Stochastic radiomap estimator: Monte Carlo ray tubes deposited per receiver.

"""Stochastic radiomap estimator: Monte Carlo ray tubes deposited per receiver."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, Any, cast

import torch

from witwin.core import Scene, SceneSnapshot

from witwin.channel import build_info
from witwin.channel.capabilities import (
    ConfigMetadata,
    ResultSemantics,
    capabilities,
    component_quantities,
    config_metadata,
    estimator_metadata,
    serialize_config,
)
from witwin.channel.components import (
    AD_MODES as _VALID_AD_MODES,
    DEFAULT_COMPONENTS as _DEFAULT_COMPONENTS,
    component_availability_status,
    component_max_depth,
    require_config_type,
    validate_bounce_depth,
    validate_config_fields,
    validate_max_depth,
    validate_samples,
    validate_seed,
    validate_vocabulary,
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
    KernelMetadata,
    SolveCapacityTransaction,
    _participates_in_ad,
    create_solve_capacity_transaction,
    enforce_memory_budget,
    estimate_monte_carlo_memory,
    make_metadata,
)
from witwin.channel.scene.compiler import (
    LIGHT_SPEED_M_PER_S as _LIGHT_SPEED_M_PER_S,
    compile as compile_scene,
    receiver_grid_points,
    record_dropped_faces,
    receiver_positions,
    transmitter_polarizations_as_stored,
    transmitter_positions,
)
from witwin.channel.scene.endpoints import (
    ReceiverGrid,
    ReceiverPoint,
    _endpoint_views,
    _validate_scalar_endpoint_boundary,
    axis_aligned_grid_spec as grid_spec,
    bind_solver_scene,
    component_grid_shape,
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
class Stochastic:
    samples: int = 4096
    max_depth: int = 1
    seed: int = 0
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str] = _DEFAULT_COMPONENTS
    diagnostics: bool = False
    ad_mode: str = "none"
    workspace_limit_bytes: int | None = 1 << 30
    receiver_radius_m: float | None = None
    scattering_samples: int | None = None

    def __post_init__(self) -> None:
        validate_config_fields(self)
        validate_samples(self.samples)
        if self.scattering_samples is not None:
            validate_samples(self.scattering_samples)
        if self.receiver_radius_m is not None and not (
            isinstance(self.receiver_radius_m, (int, float))
            and not isinstance(self.receiver_radius_m, bool)
            and math.isfinite(float(self.receiver_radius_m))
            and float(self.receiver_radius_m) > 0.0
        ):
            raise ValueError(
                "receiver_radius_m must be None or a finite positive float, got "
                f"{self.receiver_radius_m!r}"
            )
        validate_max_depth(self.max_depth)
        validate_seed(self.seed)
        components = validated_components(
            self.components, error_message="components must be a non-empty subset of {valid}"
        )
        validate_bounce_depth(
            self.max_depth,
            components,
            error_message=(
                "radiomap.Stochastic requires max_depth >= 1 for components "
                f"{sorted(components - {'los'})}, got max_depth={self.max_depth}"
            ),
        )
        validate_vocabulary("radiomap.Stochastic ad_mode", self.ad_mode, _VALID_AD_MODES)
        validate_workspace_limit_bytes(self.workspace_limit_bytes)
        object.__setattr__(self, "components", components)

    @property
    def scattering_sample_count(self) -> int:
        """Area samples of the scattering estimator (``scattering_samples``, else ``samples``)."""

        return int(self.scattering_samples if self.scattering_samples is not None else self.samples)


# --- RadioMap ---------------------------------------------------------------


class _MetadataKeys(ResultSemantics, ConfigMetadata):
    seed: int
    samples: int
    max_depth: int
    ad_mode: str
    path_count: int
    contribution_capacity: int
    components: dict[str, str]
    rayd: dict[str, bool]
    kernel: KernelMetadata
    semantic_capabilities: dict[str, Any]
    edge_policy: dict[str, Any]


class RadioMapMetadata(_MetadataKeys, total=False):
    """The stable keys of ``RadioMap.metadata``; a plain ``dict`` at run time."""

    # Present when the transmission component was requested.
    transmission: dict[str, Any]
    # Present when the scattering component was requested.
    scattering: dict[str, Any]
    # Present when compilation removed degenerate or duplicate faces.
    dropped_faces: dict[str, int]


@dataclass(frozen=True, slots=True)
class RadioMap:
    path_gain: torch.Tensor
    component_power: dict[str, torch.Tensor]
    metadata: RadioMapMetadata
    diagnostics: dict[str, Any] | None = None
    component_maps: dict[str, torch.Tensor] | None = None


# --- Sampling -------------------------------------------------------------


def make_cuda_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return generator


# --- Solver metadata ------------------------------------------------------

# One AdLaunchLedger shape for every solver (diffraction AD): radiomap.Stochastic
# counts one companion per LoS matrix, per grid-map layout Function, per
# transmitter for the reflection/diffraction accumulators. Straight
# transmission registers one flattened RayD geometry companion, one native
# wall-product companion, and one final capacity-map sanitizer companion; the
# finalize sum registers no native companion (its cotangent is a view).
def make_solver_metadata(
    *, config: Stochastic, path_count: int, contribution_capacity: int, reflection_available: bool,
    diffraction_available: bool, ad_ledger: AdLaunchLedger | None = None,
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
        **estimator_metadata(
            estimator="stochastic",
            # LoS and transmission are evaluated at the receiver; the ray estimators deposit
            # over the cell or the point receiver's capture sphere.
            quantity=component_quantities(
                config.components, cell_average=("reflection", "diffraction", "scattering")
            ),
            combine="power",
            complete=False,
        ),
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
    scene: SolverScene, *, device: torch.device, ad: bool = False,
    ledger: AdLaunchLedger | None = None,
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
            ledger.add(tx_live, tx_power, rx_live)
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


def apply_point_los_visibility(
    scene: SolverScene, rayd: RayDSceneResource, los: torch.Tensor, *, device: torch.device,
) -> torch.Tensor:
    """Zero occluded (tx, rx) entries of a point-receiver LoS matrix."""

    if not scene.structures or los.numel() == 0:
        return los
    if not rayd.available:
        raise RuntimeError("LoS visibility requires RayD native scene capability")
    handle = rayd.require_resource()
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


# Point receivers capture reflected rays in a sphere of this fraction of their straight
# transmitter distance unless Stochastic.receiver_radius_m fixes the radius.
_RELATIVE_POINT_RADIUS = 0.01

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
    scene: SolverScene, grid: ReceiverGrid, *, device: torch.device,
    los: torch.Tensor | None = None, ad: bool = False, ledger: AdLaunchLedger | None = None,
) -> torch.Tensor:
    if los is not None and len(scene.receivers) == 1 and scene.receivers[0] is grid:
        return los
    grid_scene = replace(scene, receivers=(grid,))
    return los_path_gain(grid_scene, device=device, ad=ad, ledger=ledger)


def _grid_visibility_masks(
    scene: SolverScene, rayd: RayDSceneResource, grid: ReceiverGrid, *, device: torch.device,
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
    scene: SolverScene, rayd: RayDSceneResource, grid: ReceiverGrid, *, device: torch.device,
    los: torch.Tensor | None = None, ad: bool = False, ledger: AdLaunchLedger | None = None,
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
            ledger.add(visible)
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


def _transmission_capacity_metadata(
    config: Stochastic, capacity_transaction: SolveCapacityTransaction | None,
) -> dict[str, Any]:
    """Observe the solve's failure record once and report the dropped transmission cells.

    A failed solve raises ``CapacityError`` here and returns no result.
    """

    dropped = capacity_transaction.terminal_check() if capacity_transaction is not None else 0
    if "transmission" not in config.components:
        return {}
    # Grid cells whose straight tx->cell segment crosses more walls than max_depth.
    return {"transmission": {"dropped_rows_over_max_depth": int(dropped)}}


def transmission_component_map(
    scene: SolverScene, rayd: RayDSceneResource, grid: ReceiverGrid, *, max_depth: int,
    device: torch.device, failure_state: CapacityFailureState, los: torch.Tensor | None = None,
    ad: bool = False, ledger: AdLaunchLedger | None = None,
) -> torch.Tensor:
    """Straight-penetration transmission radiomap ``[T, rows, cols]`` of one receiver grid."""

    tx_pos, _ = transmitter_positions(scene, device=device)
    if not scene.structures:
        dim0, dim1 = component_grid_shape(grid)
        return mc_component_map_buffer(
            tx_pos, tx_count=len(scene.transmitters), dim0=dim0, dim1=dim1
        )
    los_matrix = _grid_los_matrix(
        scene, grid, device=device, los=los, ad=ad, ledger=ledger
    )
    matrix = transmission_matrix(
        scene,
        rayd,
        receiver_grid_points(grid, reference=tx_pos),
        max_depth=max_depth,
        device=device,
        failure_state=failure_state,
        base_power=los_matrix,
        ad=ad,
        ledger=ledger,
    )
    if ad:
        if ledger is not None:
            ledger.add()
        return mc_los_grid_maps_ad(matrix, None, rows=grid.shape[0], cols=grid.shape[1])
    return mc_los_component_maps_from_matrix(
        matrix, rows=grid.shape[0], cols=grid.shape[1]
    )


def transmission_matrix(
    scene: SolverScene, rayd: RayDSceneResource, rx_pos: torch.Tensor, *, max_depth: int,
    device: torch.device, failure_state: CapacityFailureState,
    base_power: torch.Tensor | None = None, ad: bool = False, ledger: AdLaunchLedger | None = None,
) -> torch.Tensor:
    """Straight-penetration transmission path gain ``[T, R]`` at receiver positions ``rx_pos``.

    Mirrors the LoS geometric convention exactly: the analytic Friis gain along the straight
    tx->rx segment, with the binary LoS visibility mask replaced by the native incident-polarized
    TE/TM wall product, evaluated in ascending resident-hit order. Pairs whose segment crosses no
    wall belong to the exclusive los path class and stay zero here, so los + transmission never
    double counts. A single eps_r=1 vacuum wall has unit power transmittance, which makes this
    reproduce the unobstructed LoS gain exactly; a segment crossing more than ``max_depth`` walls
    is dropped (zero) and counted in the solve's failure record, never truncated.
    ``base_power`` is the free-space LoS gain ``[T, R]``; when omitted, ``rx_pos`` must be every
    scene receiver in order and the gain is computed here.
    """

    if not rayd.available:
        raise RuntimeError("transmission requires RayD native scene capability")
    tx_pos, _ = transmitter_positions(scene, device=device)
    if base_power is None:
        # rx_pos is every scene receiver: the free-space LoS gain of the same pairs.
        base_power = los_path_gain(scene, device=device, ad=ad, ledger=ledger)
    # Live transmitter origins in AD mode: the straight-line incidence cosine
    # (and with it every per-wall transmittance) moves with the transmitter,
    # so the chain march must see the graph, not the detached native table
    # (AD capability matrix cell TX x transmission for M).
    tx_march = transmitter_positions_ad(scene, tx_pos, device=device) if ad else tx_pos
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
    flat_base_power = base_power.view(-1)
    vertices = scene_vertex_table(scene, compiled) if ad else None
    if ad and ledger is not None:
        ledger.add(vertices, origins, targets)
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
        flat_base_power,
    )
    if ad:
        if ledger is not None:
            ledger.add(
                penetration.direction,
                penetration.normal,
                bundle["layer_thickness_m"],
                bundle["layer_eps_r"],
                bundle["layer_sigma_e"],
                flat_base_power,
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
    return product.scaled_power.view(tx_count, rx_count)


def scattering_component_map(
    scene: SolverScene, rayd: RayDSceneResource, grid: ReceiverGrid, *, samples: int, seed: int,
    device: torch.device, ad: bool = False, ledger: AdLaunchLedger | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Kirchhoff diffuse scattering radiomap from area-sampled rough faces.

    Thin grid wrapper around :func:`witwin.channel.interactions.scattering.scattering_map_matrix`
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
            ledger.add()
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


@dataclass(frozen=True, slots=True)
class ReceiverBlock:
    """A run of receivers one native deposit launch covers: one grid, or consecutive points."""

    grid: ReceiverGrid | None
    receivers: tuple[ReceiverPoint | ReceiverGrid, ...]
    column: int
    count: int


def receiver_blocks(scene: SolverScene) -> tuple[ReceiverBlock, ...]:
    """Split the scene receivers into grids and runs of point receivers, in column order."""

    blocks: list[ReceiverBlock] = []
    column = 0
    run: list[ReceiverPoint | ReceiverGrid] = []

    def close_run() -> None:
        nonlocal column
        if run:
            blocks.append(ReceiverBlock(None, tuple(run), column, len(run)))
            column += len(run)
            run.clear()

    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverGrid):
            close_run()
            count = int(receiver.shape[0]) * int(receiver.shape[1])
            blocks.append(ReceiverBlock(receiver, (receiver,), column, count))
            column += count
        else:
            run.append(receiver)
    close_run()
    return tuple(blocks)


def _block_deposit_params(
    scene: SolverScene, block: ReceiverBlock, *, reference: torch.Tensor,
    receiver_radius_m: float | None,
) -> dict[str, object]:
    """Native deposit arguments of one receiver block (grid geometry, or point rows)."""

    if block.grid is not None:
        spec = grid_spec(block.grid)
        return {
            "grid_axis": int(spec.axis),
            "grid_position": float(spec.position),
            "grid_coord0_min": float(spec.coord0_min),
            "grid_coord0_max": float(spec.coord0_max),
            "grid_coord1_min": float(spec.coord1_min),
            "grid_coord1_max": float(spec.coord1_max),
            "grid_resolution0": int(spec.resolution0),
            "grid_resolution1": int(spec.resolution1),
            "grid_cell_area": float(spec.cell_area),
            "points": None,
            "point_radius": -_RELATIVE_POINT_RADIUS,
        }
    points = receiver_positions(
        replace(scene, receivers=block.receivers), device=reference.device, reference=reference
    )
    return {
        "grid_axis": 2,
        "grid_position": 0.0,
        "grid_coord0_min": 0.0,
        "grid_coord0_max": 0.0,
        "grid_coord1_min": 0.0,
        "grid_coord1_max": 0.0,
        "grid_resolution0": int(block.count),
        "grid_resolution1": 1,
        "grid_cell_area": 1.0,
        "points": points,
        "point_radius": (
            float(receiver_radius_m)
            if receiver_radius_m is not None
            else -_RELATIVE_POINT_RADIUS
        ),
    }


def _scene_bounds(rayd: RayDSceneResource) -> torch.Tensor:
    vertices = rayd.edge_records().vertices
    key = (int(vertices.data_ptr()), int(vertices.shape[0]))
    cache = rayd.runtime_cache
    cached = cast("tuple[object, torch.Tensor] | None", cache.get("mc_scene_bounds"))
    if cached is not None and cached[0] == key:
        return cached[1]
    bounds = montecarlo_kernels.mc_scene_bounds(vertices.detach())
    cache["mc_scene_bounds"] = (key, bounds)
    return bounds


@dataclass(frozen=True, slots=True)
class ReflectionBlocksResult:
    outputs: tuple[torch.Tensor, ...]
    wedge_events: tuple[WedgeEventBatch, ...]


def _block_map_shape(block: ReceiverBlock) -> tuple[int, int]:
    return component_grid_shape(block.grid) if block.grid is not None else (1, block.count)


def _bounded_reflection_trace(
    handle: object, ray_o: torch.Tensor, ray_d: torch.Tensor, active: torch.Tensor, *,
    params: tuple[dict[str, object], ...], face_normals: torch.Tensor, material_valid: torch.Tensor,
    wavelength: float, max_depth: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``max_depth`` closest hits plus the final segment traced up to its farthest deposit."""

    trace = geometry_kernels.rayd_trace_reflections_forward(
        handle, ray_o, ray_d, ray_o.new_empty((0,)), active, int(max_depth)
    )
    segments = None
    for block_params in params:
        segments = montecarlo_kernels.mc_reflection_final_segment(
            ray_o,
            ray_d,
            trace,
            face_normals,
            material_valid,
            receiver=block_params,
            wavelength=float(wavelength),
            segments=segments,
        )
    assert segments is not None
    final = geometry_kernels.rayd_trace_reflections_forward(
        handle, segments.origin, segments.direction, segments.tmax, segments.active, 1
    )
    return trace, montecarlo_kernels.mc_reflection_append_final(trace, final, segments)


def _first_hit_wedges(
    tx: torch.Tensor, ray_o: torch.Tensor, ray_d: torch.Tensor, trace: tuple[torch.Tensor, ...],
    face_normals: torch.Tensor,
) -> WedgeEventBatch:
    """Winner-event extraction on detached trace outputs (frozen discovery bookkeeping)."""

    valid_indices = torch.nonzero(trace[0][:, 0], as_tuple=False).flatten()
    prim_id = trace[2][:, 0].index_select(0, valid_indices)
    ray_dir = ray_d.index_select(0, valid_indices)
    hit_p = (
        ray_o.index_select(0, valid_indices)
        + trace[1][:, 0].index_select(0, valid_indices)[:, None] * ray_dir
    )
    hit_n = face_normals.detach().index_select(0, prim_id.to(dtype=torch.int64))
    return WedgeEventBatch(
        tx_pos=tx,
        event_count=None,
        ray_dir=ray_dir,
        prim_id=prim_id,
        hit_p=hit_p,
        hit_n=hit_n,
        hit_geo_n=hit_n,
        bounce_depth=torch.zeros_like(prim_id),
    )


def _reflection_layer_bundle(
    scene: SolverScene, device: torch.device,
) -> tuple[tuple[torch.Tensor, ...] | None, bool]:
    """Per-face material ids and CSR layer store, so multilayer faces reflect as their stack.

    Returns ``(None, False)`` when every material has a single layer: the slab law then needs no
    layer store.
    """

    from witwin.channel.materials import face_material_field_bundle, has_multilayer_material

    if not has_multilayer_material(scene):
        return None, False
    bundle = face_material_field_bundle(scene, device=device)
    return (
        bundle["material_id"], bundle["layer_offset"], bundle["layer_count"],
        bundle["layer_thickness_m"], bundle["layer_eps_r"], bundle["layer_sigma_e"],
        bundle["layer_mu_r"],
    ), True


def _accumulate_reflection_block(
    scene: SolverScene, block_params: dict[str, object], *, ray_o: torch.Tensor,
    ray_d: torch.Tensor, full_trace: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    face_normals: torch.Tensor, material_tensors: MaterialTensors, tx_live: torch.Tensor,
    tx_pol: torch.Tensor, solid_angle: torch.Tensor, max_depth: int, wavelength: float, ad: bool,
    ledger: AdLaunchLedger | None,
) -> torch.Tensor:
    eta_r, sigma, _mu_r, gain, valid, thickness = material_tensors
    deposit = {key: value for key, value in block_params.items() if key.startswith("grid_")}
    common = {
        "tx_pol": tx_pol,
        "contribution_depth": int(max_depth),
        "wavelength": float(wavelength),
        "solid_angle_per_ray": solid_angle,
        "points": block_params["points"],
        "point_radius": float(cast(float, block_params["point_radius"])),
        **deposit,
    }
    layers, multilayer = _reflection_layer_bundle(scene, ray_o.device)
    if not ad:
        return mc_slab_reflection_accumulate(
            ray_o, ray_d, *full_trace, face_normals, eta_r, sigma, gain, valid, thickness,
            layers=layers, **common,
        )
    # Same accumulate kernel behind the autograd Function; the caller scales by the tx power.
    if ledger is not None:
        ledger.add(
            ray_o, ray_d, *full_trace, face_normals, eta_r, sigma, gain, valid, thickness
        )
    return mc_slab_reflection_accumulate_ad(
        tx_live, eta_r, sigma, gain, thickness, scene.frequency, ray_o, ray_d, *full_trace,
        face_normals, valid, layers=layers, multilayer=multilayer, **common,
    )


def reflection_block_outputs(
    scene: SolverScene, rayd: RayDSceneResource, blocks: tuple[ReceiverBlock, ...], *, samples: int,
    max_depth: int, seed: int, receiver_radius_m: float | None, device: torch.device,
    material_tensors: MaterialTensors, collect_wedges: bool = False, ad: bool = False,
    ledger: AdLaunchLedger | None = None,
) -> ReflectionBlocksResult:
    """Specular reflection deposits of every receiver block, one ``[T, ...]`` tensor per block.

    One launch generates the seeded, bounds-folded lattice rays of every transmitter; per
    transmitter RayD traces ``max_depth`` closest hits, then one bounded trace of the final
    reflected segment up to the farthest deposit any block needs. Grid blocks give
    ``[T, rows, cols]`` maps, point blocks ``[T, 1, P]``.
    """

    tx_pos, tx_power = transmitter_positions(scene, device=device)
    tx_count = int(tx_pos.shape[0])
    if not scene.structures or tx_count == 0:
        empty = tuple(
            mc_component_map_buffer(tx_pos, tx_count=tx_count, dim0=shape[0], dim1=shape[1])
            for shape in map(_block_map_shape, blocks)
        )
        return ReflectionBlocksResult(outputs=empty, wedge_events=())
    if not rayd.available:
        raise RuntimeError("reflection requires RayD native scene capability")
    handle = rayd.require_resource()
    tx_live = transmitter_positions_ad(scene, tx_pos, device=device) if ad else tx_pos
    # per-transmitter polarization seeds the reflection field's unnormalized
    # transverse projection (short-dipole sin(theta) pattern).
    tx_pol = transmitter_polarizations_as_stored(scene, device=device)
    wavelength = _LIGHT_SPEED_M_PER_S / _frequency_scalar(scene)
    face_normals = rayd.edge_records().face_normals
    params = tuple(
        _block_deposit_params(scene, block, reference=tx_pos, receiver_radius_m=receiver_radius_m)
        for block in blocks
    )
    samples = int(samples)
    rays = montecarlo_kernels.mc_reflection_rays(
        tx_pos, bounds=_scene_bounds(rayd), sample_count=samples, seed=int(seed)
    )
    per_block: list[list[torch.Tensor]] = [[] for _ in blocks]
    wedge_batches: list[WedgeEventBatch] = []
    for tx_index in range(tx_count):
        rows = slice(tx_index * samples, (tx_index + 1) * samples)
        ray_o, ray_d = rays["ray_o"][rows], rays["ray_d"][rows]
        trace, full_trace = _bounded_reflection_trace(
            handle, ray_o, ray_d, rays["active"][rows], params=params,
            face_normals=face_normals, material_valid=material_tensors[4],
            wavelength=wavelength, max_depth=max_depth,
        )
        for block_index, block_params in enumerate(params):
            output = _accumulate_reflection_block(
                scene, block_params, ray_o=ray_o, ray_d=ray_d, full_trace=full_trace,
                face_normals=face_normals, material_tensors=material_tensors, tx_live=tx_live,
                tx_pol=tx_pol[tx_index], solid_angle=rays["solid_angle"][tx_index : tx_index + 1],
                max_depth=max_depth, wavelength=wavelength, ad=ad, ledger=ledger,
            )
            per_block[block_index].append(output * tx_power[tx_index] if ad else output)
        if collect_wedges:
            wedge_batches.append(
                _first_hit_wedges(tx_pos[tx_index], ray_o, ray_d, trace, face_normals)
            )
    outputs = tuple(
        _stack_block_outputs(per_block[index], block, tx_pos, tx_power, ad=ad)
        for index, block in enumerate(blocks)
    )
    return ReflectionBlocksResult(outputs=outputs, wedge_events=tuple(wedge_batches))


def _stack_block_outputs(
    outputs: list[torch.Tensor], block: ReceiverBlock, tx_pos: torch.Tensor, tx_power: torch.Tensor,
    *, ad: bool,
) -> torch.Tensor:
    if ad:
        return torch.stack(outputs, dim=0)
    dim0, dim1 = _block_map_shape(block)
    maps = mc_component_map_buffer(tx_pos, tx_count=int(tx_pos.shape[0]), dim0=dim0, dim1=dim1)
    for tx_index, output in enumerate(outputs):
        mc_store_scaled_component_map(maps, output, tx_power, tx_index=tx_index, scale_index=tx_index)
    return maps


def reflection_component_maps_with_wedges(
    scene: SolverScene, rayd: RayDSceneResource, grid: ReceiverGrid, *, samples: int,
    max_depth: int, device: torch.device, material_tensors: MaterialTensors, seed: int = 0,
    collect_wedges: bool = False, ad: bool = False, ledger: AdLaunchLedger | None = None,
) -> ReflectionComponentResult:
    """Reflection maps ``[T, rows, cols]`` of one receiver grid plus the diffraction wedge events."""

    block = ReceiverBlock(grid, (grid,), 0, int(grid.shape[0]) * int(grid.shape[1]))
    result = reflection_block_outputs(
        scene,
        rayd,
        (block,),
        samples=samples,
        max_depth=max_depth,
        seed=seed,
        receiver_radius_m=None,
        device=device,
        material_tensors=material_tensors,
        collect_wedges=collect_wedges,
        ad=ad,
        ledger=ledger,
    )
    return ReflectionComponentResult(maps=result.outputs[0], wedge_events=result.wedge_events)


def _native_surface_group_edge_candidates(
    records: Any, selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return geometry_kernels.mc_surface_group_edge_candidates(
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
    rayd: RayDSceneResource, selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (int(selected.data_ptr()), int(selected.numel()))
    cache = rayd.runtime_cache
    cached = cache.get("mc_primitive_edge_candidates")
    if cached is not None:
        cached_key, cached_candidates = cast(
            "tuple[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]", cached
        )
        if cached_key == key:
            return cached_candidates
    candidates = _native_surface_group_edge_candidates(rayd.edge_records(), selected)
    cache["mc_primitive_edge_candidates"] = (key, candidates)
    return candidates


def _discover_diffraction_edges_from_wedges(
    rayd: RayDSceneResource, wedges: WedgeEventBatch,
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
    rayd: RayDSceneResource, tx: torch.Tensor, tx_power: torch.Tensor, tx_power_index: int,
    edge_indices: torch.Tensor, edge_geometry: tuple[torch.Tensor, ...] | None = None,
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
    return deterministic_diffraction_state_pack(
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
    scene: SolverScene, rayd: RayDSceneResource, tx: torch.Tensor, tx_power: torch.Tensor,
    tx_power_index: int, edge_geometry: tuple[torch.Tensor, ...] | None = None,
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
    return deterministic_diffraction_state_pack_selected(
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


def _tx_diffraction_states(
    scene: SolverScene, rayd: RayDSceneResource, tx: torch.Tensor, tx_power: torch.Tensor,
    tx_index: int, *, wedge_events: Any, get_edge_geometry: Any, get_edge_candidates: Any,
) -> tuple[torch.Tensor, ...] | None:
    """Packed edge states of one transmitter (from its wedge events when given), or None."""

    if wedge_events is None:
        return _diffraction_states(
            scene, rayd, tx, tx_power, tx_index, edge_geometry=get_edge_geometry()
        )
    if int(wedge_events[tx_index].prim_id.numel()) == 0:
        return None
    geometry = get_edge_geometry()
    edge_indices = _discover_diffraction_edges_from_wedges(
        rayd,
        wedge_events[tx_index],
        edge_geometry=geometry,
        edge_candidates=get_edge_candidates(geometry),
    )
    if int(edge_indices.numel()) == 0:
        return None
    return _diffraction_states_from_edge_indices(
        rayd, tx, tx_power, tx_index, edge_indices, edge_geometry=geometry
    )


# Tape entries (lane, point receiver) per point diffraction launch.
_POINT_DIFFRACTION_ENTRIES = 1 << 22


def _point_diffraction_map(
    handle: Any, states: tuple[torch.Tensor, ...], sample_state_index: torch.Tensor,
    points: torch.Tensor, *, material_tensors: MaterialTensors, tx_anchor: torch.Tensor,
    tx_pol: torch.Tensor, frequency: Any, wavelength: float, samples: int, seed: int,
    total_edge_length: float, point_radius: float, ad: bool, ledger: AdLaunchLedger | None,
) -> torch.Tensor:
    """Capture-sphere diffraction ``[1, P]`` of one transmitter at point receivers.

    Every sampling lane draws its edge point and Keller-cone ray as the grid tape does; a ray
    passing within a receiver's capture radius, with both visibility legs clear, deposits the
    flux through the sphere's cross-section, evaluated by the same tape accumulate kernels.
    """

    eta_r, sigma, mu_r, gain, valid, thickness = material_tensors
    state_tensors = tuple(states[1:12])
    point_count = int(points.shape[0])
    chunk = max(1, _POINT_DIFFRACTION_ENTRIES // max(samples, 1))
    total: torch.Tensor | None = None
    for begin in range(0, point_count, chunk):
        count = min(chunk, point_count - begin)
        tape = montecarlo_kernels.mc_point_diffraction_tape(
            sample_state_index, state_tensors, points, point_begin=begin, point_count=count,
            seed=seed, point_radius=point_radius,
        )
        source_clear = geometry_kernels.rayd_visibility_forward(
            handle, tape.leg_source, tape.leg_edge, tape.active
        )[0]
        receiver_clear = geometry_kernels.rayd_visibility_forward(
            handle, tape.leg_edge, tape.leg_receiver, tape.active
        )[0]
        active = tape.active & source_clear & receiver_clear
        tape_tensors = (active, tape.state, tape.cell, tape.u)
        point_tape = (tape.lane, points, float(point_radius), int(samples))
        grid_args: dict[str, Any] = {
            "grid_axis": 2,
            "grid_position": 0.0,
            "grid_coord0_min": 0.0,
            "grid_coord0_max": 0.0,
            "grid_coord1_min": 0.0,
            "grid_coord1_max": 0.0,
            "grid_resolution0": point_count,
            "grid_resolution1": 1,
            "wavelength": float(wavelength),
            "grid_cell_area": 1.0,
            "seed": int(seed),
            "total_edge_length": total_edge_length,
        }
        if ad:
            if ledger is not None:
                ledger.add(*tape_tensors, *state_tensors, eta_r, sigma, mu_r, gain, valid, thickness)
            part = mc_utd_diffraction_tape_accumulate_ad(
                tx_anchor, eta_r, sigma, gain, thickness, frequency, tape_tensors, state_tensors,
                mu_r, valid, tx_pol=tx_pol, point_tape=point_tape, **grid_args,
            )
        else:
            part = mc_utd_diffraction_tape_accumulate(
                *tape_tensors, *state_tensors, eta_r, sigma, mu_r, gain, valid, thickness,
                *list(grid_args.values())[:8], grid_args["wavelength"],
                grid_args["grid_cell_area"], grid_args["seed"], total_edge_length, tx_pol,
                point_tape=point_tape,
            )
        total = part if total is None else total + part
    assert total is not None
    return total


def diffraction_component_map(
    scene: SolverScene, rayd: RayDSceneResource, grid: ReceiverGrid | None, *, samples: int,
    seed: int, device: torch.device, material_tensors: MaterialTensors,
    wedge_events: tuple[WedgeEventBatch, ...] | None = None, ad: bool = False,
    ledger: AdLaunchLedger | None = None, points: torch.Tensor | None = None,
    point_radius: float = -_RELATIVE_POINT_RADIUS,
) -> torch.Tensor:
    """Keller-cone diffraction deposits ``[T, rows, cols]`` on ``grid``, or ``[T, 1, P]`` at
    point receivers ``points`` (capture spheres of ``point_radius``) when ``grid`` is None."""

    def map_shape() -> tuple[int, int]:
        if grid is not None:
            dim0, dim1 = component_grid_shape(grid)
            return int(dim0), int(dim1)
        assert points is not None
        return (1, int(points.shape[0]))

    if not scene.structures:
        tx_pos, _ = transmitter_positions(scene, device=device)
        dim0, dim1 = map_shape()
        return mc_component_map_buffer(
            tx_pos, tx_count=len(scene.transmitters), dim0=dim0, dim1=dim1
        )
    if not rayd.available:
        raise RuntimeError("diffraction requires RayD native scene capability")
    spec = grid_spec(grid) if grid is not None else None
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
    dim0, dim1 = map_shape()
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
        states = _tx_diffraction_states(
            scene, rayd, tx, tx_power, tx_index, wedge_events=wedge_events,
            get_edge_geometry=get_edge_geometry, get_edge_candidates=get_edge_candidates,
        )
        if states is None:
            if ad:
                ad_maps.append(zero_map())
            continue
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
        if spec is None:
            assert points is not None
            point_map = _point_diffraction_map(
                handle, states, sample_state_index, points, material_tensors=material_tensors,
                tx_anchor=tx_live[tx_index], tx_pol=tx_pol[tx_index], frequency=scene.frequency,
                wavelength=wavelength, samples=int(samples), seed=int(seed),
                total_edge_length=total_edge_length, point_radius=point_radius, ad=ad,
                ledger=ledger,
            )
            if ad:
                ad_maps.append(point_map)
            else:
                mc_store_component_map(maps, point_map, tx_index=tx_index)
            continue
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
                ledger.add(
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


def _validate_ad_config(config: Stochastic) -> None:
    if config.ad_mode == "none":
        return
    if "reflection" in config.components:
        depth_cap = mc_reflection_ad_max_depth()
        if config.max_depth > depth_cap:
            raise RuntimeError(
                f"radiomap.Stochastic ad_mode='{config.ad_mode}' supports the reflection "
                f"component only up to max_depth={depth_cap} (native "
                f"reflection AD depth cap); got max_depth={config.max_depth}"
            )


def _reject_vertex_leaves(scene: SolverScene, config: Stochastic) -> None:
    """Refuse mesh-vertex AD leaves for the blocks that do not differentiate them.

    Only the straight transmission chain carries mesh-vertex derivatives; the
    reflection, diffraction and scattering maps would silently drop them.
    """

    if config.ad_mode == "none":
        return
    blocks = sorted(set(config.components) & {"reflection", "diffraction", "scattering"})
    if not blocks:
        return
    live = [
        f"structure[{index}].vertices"
        for index, structure in enumerate(scene.structures)
        if _participates_in_ad(structure.vertices)
    ]
    if live:
        raise NotImplementedError(
            f"witwin.channel.radiomap.Stochastic ad_mode={config.ad_mode!r} does not "
            f"differentiate mesh vertices through {blocks}; {', '.join(live)} carries "
            "a gradient or tangent. Detach it, restrict components to "
            "{'los', 'transmission'}, or use witwin.channel.radiomap.Coherent or "
            "witwin.channel.path for vertex derivatives"
        )


def _receiver_count(scene: SolverScene) -> int:
    return sum(
        int(receiver.shape[0]) * int(receiver.shape[1])
        if isinstance(receiver, ReceiverGrid)
        else 1
        for receiver in scene.receivers
    )


def _enforce_workspace_budget(scene: SolverScene, config: Stochastic) -> None:
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
        workload="workspace for radiomap.Stochastic",
    )


def _face_material_tensors(
    scene: SolverScene, *, device: torch.device, ad: bool,
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
    # The primal tensors are a pure function of the compiled store and face assignment, so they
    # are kept on the RayD scene resource and reused while every source tensor is unchanged
    # (same storage, same in-place version): a warm solve uploads no material table.
    compiled = require_compiled(scene)
    sources = [
        value
        for value in (
            *(getattr(compiled.materials, name) for name in compiled.materials.__dataclass_fields__),
            compiled.assignments.face_material_id,
        )
        if isinstance(value, torch.Tensor)
    ]
    key = (
        id(compiled.materials),
        str(device),
        tuple((int(t.data_ptr()), int(t._version), tuple(t.shape)) for t in sources),
    )
    cache = compiled.rayd.runtime_cache
    cached = cast("tuple[object, MaterialTensors] | None", cache.get("mc_face_material_tensors"))
    if cached is not None and cached[0] == key:
        return cached[1]
    with torch.no_grad():
        tensors = build()
    cache["mc_face_material_tensors"] = (key, tensors)
    return tensors


def _mc_scattering_component(
    scene: SolverScene, rayd: RayDSceneResource, grid: Any, config: Stochastic, *,
    device: torch.device, ad: bool, ledger: Any, zero_component_map: Any,
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
            samples=config.scattering_sample_count,
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
    config: Stochastic, *, reflection_available: bool, diffraction_available: bool,
    transmission_available: bool, scattering_available: bool,
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
    scene: SolverScene, config: Stochastic, *, device: torch.device, rayd: RayDSceneResource,
    grid: ReceiverGrid | None, ad: bool, ledger: AdLaunchLedger,
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


def _matrix_columns(output: torch.Tensor, block: ReceiverBlock) -> torch.Tensor:
    """``[T, count]`` receiver-order columns of one block output (grid maps are row-major)."""

    return output.view(output.shape[0], block.count)


def _public_layout(
    path_gain: torch.Tensor, component_maps: dict[str, torch.Tensor], grid: ReceiverGrid | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Public result layout: ``[T, R]`` receiver columns, or ``[T, rows, cols]`` for one grid."""

    if grid is None:
        return path_gain, {
            name: value.view(value.shape[0], value.shape[2])
            for name, value in component_maps.items()
        }
    # The native maps are already receiver row-major [T, rows, cols], like every other solver.
    rows, cols = (int(size) for size in grid.shape)
    return path_gain.view(path_gain.shape[0], rows, cols), component_maps


def _layout_maps(columns: list[torch.Tensor]) -> torch.Tensor:
    matrix = columns[0] if len(columns) == 1 else torch.cat(columns, dim=1)
    return matrix.view(matrix.shape[0], 1, matrix.shape[1])


def _layout_specular_maps(
    scene: SolverScene, config: Stochastic, blocks: tuple[ReceiverBlock, ...], *,
    rayd: RayDSceneResource, device: torch.device, ad: bool, ledger: AdLaunchLedger,
    available: bool,
) -> dict[str, torch.Tensor]:
    """Reflection and diffraction ``[T, 1, R]`` maps; absent when nothing is traced."""

    wants_diffraction = "diffraction" in config.components and available
    if "reflection" not in config.components and not wants_diffraction:
        return {}
    material_tensors = _face_material_tensors(scene, device=device, ad=ad)
    # A diffraction-only AD solve still traces for wedge discovery, but its
    # discarded reflection deposits stay primal so no companions are registered.
    reflection_ad = ad and "reflection" in config.components
    reflection = reflection_block_outputs(
        scene,
        rayd,
        blocks,
        samples=config.samples,
        max_depth=config.max_depth,
        seed=config.seed,
        receiver_radius_m=config.receiver_radius_m,
        device=device,
        material_tensors=material_tensors,
        collect_wedges=wants_diffraction,
        ad=reflection_ad,
        ledger=ledger if reflection_ad else None,
    )
    maps: dict[str, torch.Tensor] = {}
    if "reflection" in config.components:
        maps["reflection"] = _layout_maps(
            [
                _matrix_columns(out, block)
                for out, block in zip(reflection.outputs, blocks)
            ]
        )
    if wants_diffraction:
        columns = []
        tx_pos, _ = transmitter_positions(scene, device=device)
        for block in blocks:
            params = _block_deposit_params(
                scene, block, reference=tx_pos, receiver_radius_m=config.receiver_radius_m
            )
            block_map = diffraction_component_map(
                scene,
                rayd,
                block.grid,
                samples=config.samples,
                seed=config.seed,
                device=device,
                material_tensors=material_tensors,
                wedge_events=reflection.wedge_events,
                ad=ad,
                ledger=ledger if ad else None,
                points=params["points"],
                point_radius=float(cast(float, params["point_radius"])),
            )
            columns.append(_matrix_columns(block_map, block))
        maps["diffraction"] = _layout_maps(columns)
    return maps


def _layout_scattering_map(
    scene: SolverScene, config: Stochastic, *, rayd: RayDSceneResource, device: torch.device,
    ad: bool, ledger: AdLaunchLedger,
) -> tuple[torch.Tensor, dict[str, int]]:
    if not rayd.available:
        raise RuntimeError("scattering requires RayD native scene capability")
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    matrix, stats = scattering_map_matrix(
        scene,
        rayd,
        tx_pos,
        tx_power,
        receiver_positions(scene, device=device, reference=tx_pos),
        samples=config.scattering_sample_count,
        seed=config.seed,
        device=device,
        ad=ad,
        ledger=ledger if ad else None,
    )
    return matrix.view(matrix.shape[0], 1, matrix.shape[1]), stats


def _receiver_layout_component_maps(
    scene: SolverScene, config: Stochastic, *, rayd: RayDSceneResource, los: torch.Tensor,
    device: torch.device, ad: bool, ledger: AdLaunchLedger, available: bool, path_count: int,
    contribution_capacity: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int] | None, SolveCapacityTransaction | None, int, int]:
    """Component maps ``[T, 1, R]`` for any receiver layout, columns in receiver order.

    Point receivers take reflection in capture spheres; grids keep their cell estimator. LoS,
    transmission and scattering are evaluated per receiver position for every block.
    """

    tx_count = len(scene.transmitters)
    rx_count = int(los.shape[1])
    has_rows = bool(scene.structures) and tx_count > 0 and rx_count > 0
    maps: dict[str, torch.Tensor] = {}
    if "los" in config.components:
        maps["los"] = los.view(tx_count, 1, rx_count)
    if has_rows:
        maps.update(
            _layout_specular_maps(
                scene, config, receiver_blocks(scene), rayd=rayd, device=device, ad=ad,
                ledger=ledger, available=available,
            )
        )
    for name in ("reflection", "diffraction"):
        if name in maps:
            path_count += config.samples
            contribution_capacity += int(maps[name].numel())
    capacity_transaction = None
    if "transmission" in config.components and has_rows:
        capacity_transaction = create_solve_capacity_transaction(los)
        tx_pos, _ = transmitter_positions(scene, device=device)
        matrix = transmission_matrix(
            scene,
            rayd,
            receiver_positions(scene, device=device, reference=tx_pos),
            max_depth=config.max_depth,
            device=device,
            failure_state=capacity_transaction.failure_state,
            ad=ad,
            ledger=ledger if ad else None,
        )
        maps["transmission"] = matrix.view(tx_count, 1, rx_count)
        path_count += tx_count * rx_count
        contribution_capacity += int(matrix.numel())
    scattering_stats = None
    if "scattering" in config.components and has_rows:
        maps["scattering"], scattering_stats = _layout_scattering_map(
            scene, config, rayd=rayd, device=device, ad=ad, ledger=ledger
        )
        path_count += scattering_stats["sample_count"]
        contribution_capacity += int(maps["scattering"].numel())
    for name in ("los", "reflection", "diffraction", "transmission", "scattering"):
        if name not in maps:
            maps[name] = mc_component_map_buffer(los, tx_count=tx_count, dim0=1, dim1=rx_count)
    return maps, scattering_stats, capacity_transaction, path_count, contribution_capacity


def solve_pipeline(
    scene: SolverScene, config: Stochastic, *, build_info_fn: Any=build_info,
    make_cuda_generator_fn: Any=make_cuda_generator,
    validate_scalar_endpoint_features_fn: Any=validate_scalar_endpoint_features,
    require_frequency_ad_constant_materials_fn: Any=_require_frequency_ad_constant_materials,
) -> RadioMap:
    validate_scalar_endpoint_features_fn(
        scene.transmitters, scene.receivers, solver="radiomap.Stochastic"
    )
    _enforce_workspace_budget(scene, config)
    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel.radiomap requires CUDA")
    _validate_ad_config(config)
    _reject_vertex_leaves(scene, config)
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
    single_grid = len(scene.receivers) == 1 and isinstance(scene.receivers[0], ReceiverGrid)
    grid = cast("ReceiverGrid | None", scene.receivers[0] if single_grid else None)
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
    if grid is None:
        (
            component_maps,
            scattering_stats,
            capacity_transaction,
            path_count,
            contribution_capacity,
        ) = _receiver_layout_component_maps(
            scene,
            config,
            rayd=rayd,
            los=los,
            device=device,
            ad=ad,
            ledger=ledger,
            available=reflection_available,
            path_count=path_count,
            contribution_capacity=contribution_capacity,
        )

        def zero_component_map() -> torch.Tensor:
            return mc_component_map_buffer(
                los, tx_count=len(scene.transmitters), dim0=1, dim1=int(los.shape[1])
            )
    else:
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
                seed=config.seed,
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

    assert component_maps is not None
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
    path_gain, component_maps = _public_layout(path_gain, component_maps, grid)
    transmission_metadata = _transmission_capacity_metadata(config, capacity_transaction)
    metadata = make_solver_metadata(
        config=config,
        path_count=path_count,
        contribution_capacity=contribution_capacity,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        ad_ledger=ledger,
    )
    metadata.update(transmission_metadata)
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
    result = RadioMap(
        path_gain=path_gain,
        component_power=component_power,
        metadata=cast(RadioMapMetadata, metadata),
        diagnostics=diagnostics,
        component_maps=component_maps,
    )
    return result


# --- Public entry point ---------------------------------------------------


def solve(
    scene: Scene | SceneSnapshot, config: Stochastic, *,
    reference_frequency_hz: float | torch.Tensor,
) -> RadioMap:
    """Run the Monte Carlo Basic solver pipeline."""

    require_config_type(config, Stochastic, "witwin.channel.radiomap")
    endpoint_views = _endpoint_views(scene)
    _validate_scalar_endpoint_boundary(endpoint_views)
    validate_scalar_endpoint_features(
        tuple(view for view in endpoint_views if view.source.role == "tx"),
        tuple(view for view in endpoint_views if view.source.role == "rx"),
        solver="radiomap.Stochastic",
    )
    compiled = compile_scene(
        scene, reference_frequency_hz=reference_frequency_hz
    )
    result = solve_pipeline(
        bind_solver_scene(compiled),
        config,
        build_info_fn=build_info,
        make_cuda_generator_fn=make_cuda_generator,
        validate_scalar_endpoint_features_fn=validate_scalar_endpoint_features,
        require_frequency_ad_constant_materials_fn=(
            _require_frequency_ad_constant_materials
        ),
    )
    record_dropped_faces(result, compiled)
    return result


# The public home of these types is ``witwin.channel.radiomap``.
Stochastic.__module__ = "witwin.channel.radiomap"
RadioMap.__module__ = "witwin.channel.radiomap"
RadioMapMetadata.__module__ = "witwin.channel.radiomap"

__all__ = ["RadioMap", "RadioMapMetadata", "Stochastic", "solve"]
