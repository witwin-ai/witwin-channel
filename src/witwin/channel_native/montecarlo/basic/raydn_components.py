from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from witwin.channel_native import ReceiverGrid, Scene
from witwin.channel_native.core.kernels.ops import (
    mc_apply_los_visibility,
    mc_component_map_buffer,
    mc_diffraction_edge_geometry,
    mc_diffraction_state_pack,
    mc_diffraction_state_wi,
    mc_los_component_maps,
    mc_los_visibility_inputs,
    mc_reflection_launch_inputs,
    mc_sample_directions,
    mc_selected_edge_indices,
    mc_store_component_map,
    mc_store_scaled_component_map,
    mc_surface_group_edge_candidates,
)
from witwin.channel_native.core.material_runtime import face_material_tensors
from witwin.channel_native.core.scene import _RAYD_EDGE_INFO_PLANE_TOL
from witwin.channel_native.core.runtime.raydn import RayDNScene

from .backend import _LIGHT_SPEED_M_PER_S, los_path_gain, receiver_grid_points, transmitter_positions


@dataclass(frozen=True, slots=True)
class GridSpec:
    grid: ReceiverGrid
    axis: int
    position: float
    coord0_min: float
    coord0_max: float
    coord1_min: float
    coord1_max: float
    resolution0: int
    resolution1: int
    cell_area: float


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


MaterialTensors = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def first_receiver_grid(scene: Scene) -> ReceiverGrid | None:
    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverGrid):
            return receiver
    return None


def component_grid_shape(grid: ReceiverGrid) -> tuple[int, int]:
    return (grid.shape[1], grid.shape[0])


def _vector3_values(vector: torch.Tensor) -> tuple[float, float, float]:
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _axis_index(values: tuple[float, float, float], *, name: str) -> tuple[int, float]:
    nonzero = [idx for idx, value in enumerate(values) if abs(value) > 1.0e-6]
    if len(nonzero) != 1:
        raise ValueError(f"{name} must be axis-aligned")
    index = nonzero[0]
    value = values[index]
    sign = 1.0 if value > 0.0 else -1.0
    if abs(abs(value) - 1.0) > 1.0e-5:
        raise ValueError(f"{name} must be a unit axis vector")
    return index, sign


def grid_spec(grid: ReceiverGrid) -> GridSpec:
    rows, cols = grid.shape
    origin = _vector3_values(grid.origin)
    axis0, sign0 = _axis_index(_vector3_values(grid.x_axis), name="ReceiverGrid.x_axis")
    axis1, sign1 = _axis_index(_vector3_values(grid.y_axis), name="ReceiverGrid.y_axis")
    if axis0 == axis1:
        raise ValueError("ReceiverGrid axes must be orthogonal")
    axis = ({0, 1, 2} - {axis0, axis1}).pop()
    if axis == 0:
        expected = (1, 2)
    elif axis == 1:
        expected = (0, 2)
    else:
        expected = (0, 1)
    if (axis0, axis1) != expected:
        raise ValueError("ReceiverGrid axes must match RayDN grid coordinate order")

    step0 = float(grid.spacing[0]) * sign0
    step1 = float(grid.spacing[1]) * sign1
    first0 = origin[axis0]
    first1 = origin[axis1]
    last0 = first0 + step0 * float(rows - 1)
    last1 = first1 + step1 * float(cols - 1)
    half0 = abs(float(grid.spacing[0])) * 0.5
    half1 = abs(float(grid.spacing[1])) * 0.5
    coord0_min = min(first0, last0) - half0
    coord0_max = max(first0, last0) + half0
    coord1_min = min(first1, last1) - half1
    coord1_max = max(first1, last1) + half1
    return GridSpec(
        grid=grid,
        axis=axis,
        position=origin[axis],
        coord0_min=coord0_min,
        coord0_max=coord0_max,
        coord1_min=coord1_min,
        coord1_max=coord1_max,
        resolution0=rows,
        resolution1=cols,
        cell_area=abs((coord0_max - coord0_min) * (coord1_max - coord1_min)) / float(rows * cols),
    )


def _grid_los_gain(
    scene: Scene,
    grid: ReceiverGrid,
    *,
    device: torch.device,
    los: torch.Tensor | None = None,
) -> torch.Tensor:
    if los is not None and len(scene.receivers) == 1 and scene.receivers[0] is grid:
        return los.reshape(len(scene.transmitters), *grid.shape)
    grid_scene = Scene(
        structures=scene.structures,
        transmitters=scene.transmitters,
        receivers=[grid],
        frequency=scene.frequency,
        metadata=scene.metadata,
    )
    return los_path_gain(grid_scene, device=device).reshape(len(scene.transmitters), *grid.shape)


def los_component_map(
    scene: Scene,
    raydn: RayDNScene,
    grid: ReceiverGrid,
    *,
    device: torch.device,
    los: torch.Tensor | None = None,
) -> torch.Tensor:
    los = _grid_los_gain(scene, grid, device=device, los=los)
    maps = mc_los_component_maps(los)
    if not raydn.available or not scene.structures:
        return maps
    handle = raydn.require_handle()
    tx_pos, _ = transmitter_positions(scene, device=device)
    rx_pos = receiver_grid_points(grid, reference=tx_pos)
    for tx_index in range(tx_pos.shape[0]):
        inputs = mc_los_visibility_inputs(tx_pos, tx_index=tx_index, rx_count=rx_pos.shape[0])
        visible = torch.ops.raydn.visibility_forward(handle, inputs["start"], rx_pos, inputs["active"])[0]
        mc_apply_los_visibility(maps, los, visible, tx_index=tx_index)
    return maps


def _sample_directions(count: int, *, reference: torch.Tensor) -> torch.Tensor:
    return mc_sample_directions(count, reference)


def _empty_wedge_events(tx: torch.Tensor) -> WedgeEventBatch:
    device = tx.device
    return WedgeEventBatch(
        tx_pos=tx,
        event_count=None,
        ray_dir=torch.empty((0, 3), device=device, dtype=torch.float32),
        prim_id=torch.empty((0,), device=device, dtype=torch.int32),
        hit_p=torch.empty((0, 3), device=device, dtype=torch.float32),
        hit_n=torch.empty((0, 3), device=device, dtype=torch.float32),
        hit_geo_n=torch.empty((0, 3), device=device, dtype=torch.float32),
        bounce_depth=torch.empty((0,), device=device, dtype=torch.int32),
    )


def reflection_component_maps_with_wedges(
    scene: Scene,
    raydn: RayDNScene,
    grid: ReceiverGrid,
    *,
    samples: int,
    max_depth: int,
    seed: int,
    device: torch.device,
    material_tensors: MaterialTensors,
    collect_wedges: bool = False,
    reflection_accumulation_strategy: str = "auto",
    reflection_compact_min_samples: int = 262_144,
    reflection_staged_min_samples_per_cell: int = 64,
) -> ReflectionComponentResult:
    strategy_id = {
        "auto": 0,
        "atomic": 1,
        "staged": 2,
        "compact": 3,
    }[reflection_accumulation_strategy]
    if not raydn.available or not scene.structures:
        tx_pos, _ = transmitter_positions(scene, device=device)
        dim0, dim1 = component_grid_shape(grid)
        maps = mc_component_map_buffer(tx_pos, tx_count=len(scene.transmitters), dim0=dim0, dim1=dim1)
        return ReflectionComponentResult(
            maps=maps,
            wedge_events=tuple(_empty_wedge_events(tx) for tx in tx_pos),
        )
    spec = grid_spec(grid)
    handle = raydn.require_handle()
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    wavelength = _LIGHT_SPEED_M_PER_S / float(scene.frequency)
    material_eta_r, material_sigma, material_mu_r, material_gain, material_valid = material_tensors
    solid_angle_per_ray = float(4.0 * math.pi / max(1, int(samples)))
    dim0, dim1 = component_grid_shape(grid)
    maps = mc_component_map_buffer(tx_pos, tx_count=tx_pos.shape[0], dim0=dim0, dim1=dim1)
    wedge_batches: list[WedgeEventBatch] = []
    for tx_index, tx in enumerate(tx_pos):
        ray_d = _sample_directions(samples, reference=tx_pos)
        launch_inputs = mc_reflection_launch_inputs(tx_pos, tx_index=tx_index, sample_count=samples)
        ray_o = launch_inputs["ray_o"]
        ray_tmax = launch_inputs["ray_tmax"]
        active = launch_inputs["active"]
        tx_batch = ray_o
        tx_pol = launch_inputs["tx_pol"]
        out = torch.ops.raydn.reflection_accumulation_forward(
            handle,
            ray_o,
            ray_d,
            ray_tmax,
            active,
            tx_batch,
            tx_pol,
            material_eta_r,
            material_sigma,
            material_mu_r,
            material_gain,
            material_valid,
            int(max_depth),
            int(spec.axis),
            float(spec.position),
            float(spec.coord0_min),
            float(spec.coord0_max),
            float(spec.coord1_min),
            float(spec.coord1_max),
            int(spec.resolution0),
            int(spec.resolution1),
            float(wavelength),
            solid_angle_per_ray,
            bool(collect_wedges),
            False,
            int(samples) if collect_wedges else 0,
            1,
            strategy_id,
            int(reflection_compact_min_samples),
            int(reflection_staged_min_samples_per_cell),
        )
        mc_store_scaled_component_map(
            maps,
            out[0].contiguous(),
            tx_power,
            tx_index=tx_index,
            scale_index=tx_index,
        )
        if collect_wedges:
            wedge_batches.append(
                WedgeEventBatch(
                    tx_pos=tx,
                    event_count=out[8].contiguous(),
                    ray_dir=out[13],
                    prim_id=out[12],
                    hit_p=out[10],
                    hit_n=out[11],
                    hit_geo_n=out[11],
                    bounce_depth=out[17],
                )
            )
        else:
            wedge_batches.append(_empty_wedge_events(tx))
    return ReflectionComponentResult(maps=maps, wedge_events=tuple(wedge_batches))


def reflection_component_map(
    scene: Scene,
    raydn: RayDNScene,
    grid: ReceiverGrid,
    *,
    samples: int,
    max_depth: int,
    seed: int,
    device: torch.device,
    material_tensors: MaterialTensors,
) -> torch.Tensor:
    return reflection_component_maps_with_wedges(
        scene,
        raydn,
        grid,
        samples=samples,
        max_depth=max_depth,
        seed=seed,
        device=device,
        material_tensors=material_tensors,
        collect_wedges=False,
    ).maps


def _diffraction_edge_geometry(records) -> tuple[torch.Tensor, ...]:
    return mc_diffraction_edge_geometry(
        records.vertices,
        records.faces,
        records.face_normals,
        records.edge_v0,
        records.edge_v1,
        records.face0,
        records.face1,
        plane_tol=_RAYD_EDGE_INFO_PLANE_TOL,
    )


def _cached_diffraction_edge_geometry(raydn: RayDNScene) -> tuple[torch.Tensor, ...]:
    cache = raydn.runtime_cache
    cached = cache.get("mc_diffraction_edge_geometry")
    if cached is not None:
        return cached  # type: ignore[return-value]
    geometry = _diffraction_edge_geometry(raydn.edge_records())
    cache["mc_diffraction_edge_geometry"] = geometry
    return geometry


def _native_surface_group_edge_candidates(records, selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return mc_surface_group_edge_candidates(
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


def _cached_surface_group_edge_candidates(
    raydn: RayDNScene,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (int(selected.data_ptr()), int(selected.numel()))
    cache = raydn.runtime_cache
    cached = cache.get("mc_surface_group_edge_candidates")
    if cached is not None:
        cached_key, cached_candidates = cached  # type: ignore[misc]
        if cached_key == key:
            return cached_candidates  # type: ignore[return-value]
    candidates = _native_surface_group_edge_candidates(raydn.edge_records(), selected)
    cache["mc_surface_group_edge_candidates"] = (key, candidates)
    return candidates


def _discover_diffraction_edges_from_wedges(
    raydn: RayDNScene,
    wedges: WedgeEventBatch,
    edge_geometry: tuple[torch.Tensor, ...] | None = None,
    edge_candidates: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if int(wedges.prim_id.numel()) == 0:
        return torch.empty((0,), device=wedges.tx_pos.device, dtype=torch.int32)
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
    ) = edge_geometry if edge_geometry is not None else _cached_diffraction_edge_geometry(raydn)
    if edge_candidates is None:
        edge_candidates = _cached_surface_group_edge_candidates(raydn, selected)
    triangle_edge_count, triangle_edge_indices = edge_candidates
    if wedges.event_count is not None:
        return torch.ops.raydn.diffraction_discover_edges_counted(
            wedges.tx_pos.contiguous(),
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
    return torch.ops.raydn.diffraction_discover_edges(
        wedges.tx_pos.contiguous(),
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
    raydn: RayDNScene,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    edge_indices: torch.Tensor,
    edge_geometry: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    records = raydn.edge_records()
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
    ) = edge_geometry if edge_geometry is not None else _cached_diffraction_edge_geometry(raydn)
    return mc_diffraction_state_pack(
        edge_indices.to(device=records.vertices.device, dtype=torch.int32).contiguous(),
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
    )


def _diffraction_states(
    scene: Scene,
    raydn: RayDNScene,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    edge_geometry: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    del scene
    geometry = edge_geometry if edge_geometry is not None else _cached_diffraction_edge_geometry(raydn)
    selected = geometry[0]
    return _diffraction_states_from_edge_indices(
        raydn,
        tx,
        tx_power,
        mc_selected_edge_indices(selected),
        edge_geometry=geometry,
    )


def diffraction_component_map(
    scene: Scene,
    raydn: RayDNScene,
    grid: ReceiverGrid,
    *,
    samples: int,
    seed: int,
    device: torch.device,
    material_tensors: MaterialTensors,
    wedge_events: tuple[WedgeEventBatch, ...] | None = None,
) -> torch.Tensor:
    if not raydn.available or not scene.structures:
        tx_pos, _ = transmitter_positions(scene, device=device)
        dim0, dim1 = component_grid_shape(grid)
        return mc_component_map_buffer(tx_pos, tx_count=len(scene.transmitters), dim0=dim0, dim1=dim1)
    spec = grid_spec(grid)
    handle = raydn.require_handle()
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    material_eta_r, material_sigma, material_mu_r, material_gain, material_valid = material_tensors
    wavelength = _LIGHT_SPEED_M_PER_S / float(scene.frequency)
    dim0, dim1 = component_grid_shape(grid)
    maps = mc_component_map_buffer(tx_pos, tx_count=tx_pos.shape[0], dim0=dim0, dim1=dim1)
    edge_geometry: tuple[torch.Tensor, ...] | None = None
    edge_candidates: tuple[torch.Tensor, torch.Tensor] | None = None

    def get_edge_geometry() -> tuple[torch.Tensor, ...]:
        nonlocal edge_geometry
        if edge_geometry is None:
            edge_geometry = _cached_diffraction_edge_geometry(raydn)
        return edge_geometry

    def get_edge_candidates(geometry: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal edge_candidates
        if edge_candidates is None:
            edge_candidates = _cached_surface_group_edge_candidates(raydn, geometry[0])
        return edge_candidates

    for tx_index, tx in enumerate(tx_pos):
        if wedge_events is not None:
            if int(wedge_events[tx_index].prim_id.numel()) == 0:
                continue
            geometry = get_edge_geometry()
            edge_indices = _discover_diffraction_edges_from_wedges(
                raydn,
                wedge_events[tx_index],
                edge_geometry=geometry,
                edge_candidates=get_edge_candidates(geometry),
            )
            if int(edge_indices.numel()) == 0:
                continue
            states = _diffraction_states_from_edge_indices(
                raydn,
                tx,
                tx_power[tx_index],
                edge_indices,
                edge_geometry=geometry,
            )
        else:
            states = _diffraction_states(
                scene,
                raydn,
                tx,
                tx_power[tx_index],
                edge_geometry=get_edge_geometry(),
            )
        state_count = int(states[0].shape[0])
        if state_count <= 0:
            continue
        state_wi = mc_diffraction_state_wi(states[1], states[10])
        out = torch.ops.raydn.diffraction_accumulation_forward(
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
            0,
            None,
            None,
        )
        mc_store_component_map(maps, out[0].contiguous(), tx_index=tx_index)
    return maps
