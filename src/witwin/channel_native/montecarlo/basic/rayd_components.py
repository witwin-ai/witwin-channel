from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from witwin.channel_native.scene.models import ReceiverGrid
from witwin.channel_native.core.ad_geometry import transmitter_positions_ad
from typing import TYPE_CHECKING
from witwin.channel_native.montecarlo.basic.kernels.maps import (
    mc_apply_los_visibility,
    mc_component_map_buffer,
    mc_los_component_maps_from_matrix,
    mc_los_grid_maps_ad,
    mc_los_visibility_inputs,
    mc_sionna_diffraction_tape_accumulate,
    mc_sionna_diffraction_tape_accumulate_ad,
    mc_sionna_reflection_accumulate,
    mc_sionna_reflection_accumulate_ad,
    mc_store_component_map,
    mc_store_scaled_component_map,
)
from witwin.channel_native.montecarlo.basic.kernels import sampling as sampling_kernels
from witwin.channel_native.core.diffraction_geometry import (
    cached_diffraction_edge_geometry as _cached_diffraction_edge_geometry,
    diffraction_edge_geometry as _diffraction_edge_geometry,
)
from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge
from witwin.channel_native.propagation.geometry.kernels import (
    primitives as geometry_primitives,
)
from witwin.channel_native.propagation.topology.kernels.primitives import (
    deterministic_diffraction_state_pack,
    deterministic_diffraction_state_pack_selected,
)

from witwin.channel_native.core.receiver_geometry import (
    axis_aligned_grid_spec as grid_spec,
    component_grid_shape,
)
from witwin.channel_native.materials.encoding import face_material_field_bundle
from witwin.channel_native.scene.kernels.rayd_scene import RayDSceneResource
from witwin.channel_native.montecarlo.events.scattering import scattering_map_matrix
from witwin.channel_native.montecarlo.events.transmission import (
    layer_csr_view,
    scene_diagonal_m,
    straight_transmission_chains,
)

from .backend import _LIGHT_SPEED_M_PER_S, los_path_gain, receiver_grid_points, transmitter_positions
from witwin.channel_native.scene.tensors import transmitter_polarizations

if TYPE_CHECKING:
    from witwin.channel_native.scene.models import Scene

__all__ = ["_diffraction_edge_geometry"]


def _frequency_scalar(scene: Scene) -> float:
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
    scene: Scene,
    grid: ReceiverGrid,
    *,
    device: torch.device,
    los: torch.Tensor | None = None,
    ad: bool = False,
    ledger: object | None = None,
) -> torch.Tensor:
    from witwin.channel_native.scene.models import Scene

    if los is not None and len(scene.receivers) == 1 and scene.receivers[0] is grid:
        return los
    grid_scene = Scene(
        structures=scene.structures,
        transmitters=scene.transmitters,
        receivers=[grid],
        frequency=scene.frequency,
        metadata=scene.metadata,
    )
    return los_path_gain(grid_scene, device=device, ad=ad, ledger=ledger)


def _grid_visibility_masks(
    scene: Scene,
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
        inputs = mc_los_visibility_inputs(tx_pos, tx_index=tx_index, rx_count=rx_pos.shape[0])
        masks.append(
            geometry_bridge.rayd_visibility_forward(
                handle, inputs["start"], rx_pos, inputs["active"]
            )[0]
        )
    return torch.stack(masks, dim=0)


def los_component_map(
    scene: Scene,
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
                raise RuntimeError("LoS visibility requires RayD native scene capability")
            visible = _grid_visibility_masks(scene, rayd, grid, device=device)
        if ledger is not None:
            ledger.add(visible)
        return mc_los_grid_maps_ad(
            los, visible, rows=grid.shape[0], cols=grid.shape[1]
        )
    maps = mc_los_component_maps_from_matrix(los, rows=grid.shape[0], cols=grid.shape[1])
    if not scene.structures:
        return maps
    if not rayd.available:
        raise RuntimeError("LoS visibility requires RayD native scene capability")
    handle = rayd.require_resource()
    tx_pos, _ = transmitter_positions(scene, device=device)
    rx_pos = receiver_grid_points(grid, reference=tx_pos)
    for tx_index in range(tx_pos.shape[0]):
        inputs = mc_los_visibility_inputs(tx_pos, tx_index=tx_index, rx_count=rx_pos.shape[0])
        visible = geometry_bridge.rayd_visibility_forward(
            handle, inputs["start"], rx_pos, inputs["active"]
        )[0]
        mc_apply_los_visibility(maps, los, visible, tx_index=tx_index)
    return maps


def _sample_directions(count: int, *, reference: torch.Tensor) -> torch.Tensor:
    return sampling_kernels.mc_sample_directions(count, reference)


def transmission_component_map(
    scene: Scene,
    rayd: RayDSceneResource,
    grid: ReceiverGrid,
    *,
    max_depth: int,
    device: torch.device,
    los: torch.Tensor | None = None,
    ad: bool = False,
    ledger: object | None = None,
) -> torch.Tensor:
    """Straight-penetration transmission radiomap (contract section 4,
    endpoint-connection context).

    Mirrors the LoS map's geometric convention exactly: the analytic per-cell
    Friis gain along the straight tx->cell segment, with the binary LoS
    visibility mask replaced by the through-wall power transmittance product
    (unpolarized TE/TM mean per wall, evaluated at the straight-line incidence
    angle). Cells whose segment crosses no wall belong to the exclusive los
    path class and stay zero here, so los + transmission never double counts.
    A single eps_r=1 vacuum wall has unit power transmittance, which makes
    this map reproduce the unobstructed LoS map exactly (acceptance test);
    chains needing more than ``max_depth`` penetrations are truthfully zero.
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
    # (plan 07 section 9.3 TX x transmission for M).
    tx_march = (
        transmitter_positions_ad(scene, tx_pos, device=device) if ad else tx_pos
    )
    rx_pos = receiver_grid_points(grid, reference=tx_pos)
    if ad:
        bundle = face_material_field_bundle(scene, device=device)
    else:
        with torch.no_grad():
            bundle = face_material_field_bundle(scene, device=device)
    layer_csr = layer_csr_view(bundle)
    diagonal = scene_diagonal_m(scene)
    rx_count = int(rx_pos.shape[0])
    # One host read of a tensor frequency for every per-tx chain march below
    # (audit M3); the AD march additionally keeps the live tensor so the
    # carrier gradient survives.
    frequency_value = _frequency_scalar(scene)
    frequency_hz = scene.frequency if ad else frequency_value
    # ADR-020: per-tx incident polarization drives the polarized wall
    # transmittance (frozen physical vector; a detached AD winner).
    tx_pol = transmitter_polarizations(scene, device=device)
    gains = []
    for tx_index in range(int(tx_pos.shape[0])):
        origins = tx_march[tx_index].unsqueeze(0).repeat(rx_count, 1)
        chain = straight_transmission_chains(
            rayd,
            origins,
            rx_pos,
            face_material_id=bundle["material_id"],
            layer_csr=layer_csr,
            polarization=tx_pol[tx_index],
            frequency_hz=frequency_hz,
            frequency_value=frequency_value if ad else None,
            max_depth=int(max_depth),
            scene_diagonal=diagonal,
            ad=ad,
            ledger=ledger,
        )
        gains.append(
            torch.where(
                chain["penetrated"],
                chain["transmittance"],
                torch.zeros_like(chain["transmittance"]),
            )
        )
    matrix = los_matrix * torch.stack(gains, dim=0)
    if ad:
        if ledger is not None:
            ledger.add()
        return mc_los_grid_maps_ad(
            matrix, None, rows=grid.shape[0], cols=grid.shape[1]
        )
    return mc_los_component_maps_from_matrix(
        matrix, rows=grid.shape[0], cols=grid.shape[1]
    )


def scattering_component_map(
    scene: Scene,
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

    Thin grid wrapper around
    :func:`witwin.channel_native.montecarlo.events.scattering.scattering_map_matrix`
    (which documents the estimator and its v1 simplifications): the matrix
    holds the per-cell scattering PATH GAIN at the cell center times the
    transmitter power, mirroring the LoS / transmission map conventions, so
    component_power equals the map sum.

    Under ``ad`` the matrix keeps its graph (table values, frequency and tx
    power gradients, ADR-015 op 1) and the grid layout runs behind the same
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
        return maps, {"sample_count": 0, "rough_face_count": 0, "tx_visible_samples": 0, "deposited_rows": 0}
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
            mc_los_grid_maps_ad(
                matrix, None, rows=grid.shape[0], cols=grid.shape[1]
            ),
            stats,
        )
    return (
        mc_los_component_maps_from_matrix(
            matrix, rows=grid.shape[0], cols=grid.shape[1]
        ),
        stats,
    )


def reflection_component_maps_with_wedges(
    scene: Scene,
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
        maps = mc_component_map_buffer(tx_pos, tx_count=len(scene.transmitters), dim0=dim0, dim1=dim1)
        return ReflectionComponentResult(
            maps=maps,
            wedge_events=(),
        )
    if not rayd.available:
        raise RuntimeError("reflection requires RayD native scene capability")
    spec = grid_spec(grid)
    handle = rayd.require_resource()
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    tx_live = (
        transmitter_positions_ad(scene, tx_pos, device=device) if ad else tx_pos
    )
    # R5: per-transmitter polarization seeds the reflection field's unnormalized
    # transverse projection (short-dipole sin(theta) pattern).
    tx_pol = transmitter_polarizations(scene, device=device)
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
        maps = mc_component_map_buffer(tx_pos, tx_count=tx_pos.shape[0], dim0=dim0, dim1=dim1)
    wedge_batches: list[WedgeEventBatch] = []
    for tx_index, tx in enumerate(tx_pos):
        ray_d = _sample_directions(samples, reference=tx_pos)
        launch_inputs = sampling_kernels.mc_reflection_launch_inputs(
            tx_pos, tx_index=tx_index, sample_count=samples
        )
        ray_o = launch_inputs["ray_o"]
        ray_tmax = launch_inputs["ray_tmax"]
        active = launch_inputs["active"]
        trace = geometry_bridge.rayd_trace_reflections_forward(
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
                ledger.add(
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
            reflection_map = mc_sionna_reflection_accumulate_ad(
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
            reflection_map = mc_sionna_reflection_accumulate(
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
            hit_n = face_normals.detach().index_select(
                0, prim_id.to(dtype=torch.int64)
            )
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


def _native_surface_group_edge_candidates(records, selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return geometry_primitives.mc_surface_group_edge_candidates(
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
            return cached_candidates  # type: ignore[return-value]
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
    ) = edge_geometry if edge_geometry is not None else _cached_diffraction_edge_geometry(rayd)
    if edge_candidates is None:
        edge_candidates = _cached_primitive_edge_candidates(rayd, selected)
    triangle_edge_count, triangle_edge_indices = edge_candidates
    if wedges.event_count is not None:
        return sampling_kernels.mc_diffraction_discover_edges_counted(
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
    return sampling_kernels.mc_diffraction_discover_edges(
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
    ) = edge_geometry if edge_geometry is not None else _cached_diffraction_edge_geometry(rayd)
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
    scene: Scene,
    rayd: RayDSceneResource,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    tx_power_index: int,
    edge_geometry: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    del scene
    geometry = edge_geometry if edge_geometry is not None else _cached_diffraction_edge_geometry(rayd)
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


def diffraction_component_map(
    scene: Scene,
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
        return mc_component_map_buffer(tx_pos, tx_count=len(scene.transmitters), dim0=dim0, dim1=dim1)
    if not rayd.available:
        raise RuntimeError("diffraction requires RayD native scene capability")
    spec = grid_spec(grid)
    handle = rayd.require_resource()
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    tx_live = (
        transmitter_positions_ad(scene, tx_pos, device=device) if ad else tx_pos
    )
    # R5: per-transmitter polarization fed into direct_source_vector's incident
    # basis (replaces the fabricated z-axis).
    tx_pol = transmitter_polarizations(scene, device=device)
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
        maps = mc_component_map_buffer(tx_pos, tx_count=tx_pos.shape[0], dim0=dim0, dim1=dim1)
    edge_geometry: tuple[torch.Tensor, ...] | None = None
    edge_candidates: tuple[torch.Tensor, torch.Tensor] | None = None
    mitsuba_metadata = scene.metadata.get("mitsuba", {})
    preserve_imported_edges = (
        isinstance(mitsuba_metadata, dict)
        and bool(mitsuba_metadata.get("merge_shapes", False))
    )

    def get_edge_geometry() -> tuple[torch.Tensor, ...]:
        nonlocal edge_geometry
        if edge_geometry is None:
            edge_geometry = _cached_diffraction_edge_geometry(
                rayd,
                preserve_imported_edges=preserve_imported_edges,
            )
        return edge_geometry

    def get_edge_candidates(geometry: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
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
        state_wi = sampling_kernels.mc_diffraction_state_wi(states[1], states[10])
        sampled = geometry_bridge.rayd_diffraction_sample_tape_forward(
            handle, None, *states, state_wi, state_wi,
            material_eta_r, material_sigma, material_mu_r, material_gain, material_valid,
            state_count, int(spec.axis), float(spec.position), float(spec.coord0_min),
            float(spec.coord0_max), float(spec.coord1_min), float(spec.coord1_max),
            int(spec.resolution0), int(spec.resolution1), float(spec.cell_area), float(wavelength),
            0, int(samples), 0, int(seed), 1, 0,
            None, None, None, None, None, None, None, None, None, None, None,
            1, sample_state_index, sample_edge_weight,
        )
        if ad:
            # Same tape-accumulate kernel behind the autograd Function
            # (plan 07 AD-4): the RayD sampling tape and the packed edge
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
            diffraction_map = mc_sionna_diffraction_tape_accumulate_ad(
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
        diffraction_map = mc_sionna_diffraction_tape_accumulate(
            sampled[14], sampled[15], sampled[16], sampled[18],
            states[1], states[2], states[3], states[4], states[5], states[6],
            states[7], states[8], states[9], states[10], states[11],
            material_eta_r, material_sigma, material_mu_r, material_gain, material_valid, material_thickness,
            int(spec.axis), float(spec.position), float(spec.coord0_min), float(spec.coord0_max),
            float(spec.coord1_min), float(spec.coord1_max), int(spec.resolution0),
            int(spec.resolution1), float(wavelength), float(spec.cell_area), int(seed), total_edge_length,
            tx_pol[tx_index],
        )
        mc_store_component_map(maps, diffraction_map, tx_index=tx_index)
    if ad:
        maps = torch.stack(ad_maps, dim=0)
    return maps
