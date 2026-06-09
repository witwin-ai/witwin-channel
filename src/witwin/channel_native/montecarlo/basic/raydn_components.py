from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from witwin.channel_native import ReceiverGrid, Scene
from witwin.channel_native.core.material_runtime import face_material_tensors
from witwin.channel_native.core.edge_policy import EdgePolicy
from witwin.channel_native.core.scene import (
    _RAYD_EDGE_INFO_PLANE_TOL,
    _RAYD_NORMAL_COS_TOL,
    _selected_diffraction_edges,
)
from witwin.channel_native.core.runtime.raydn import RayDNScene

from .backend import _LIGHT_SPEED_M_PER_S, los_path_gain, transmitter_positions


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


def first_receiver_grid(scene: Scene) -> ReceiverGrid | None:
    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverGrid):
            return receiver
    return None


def component_grid_shape(grid: ReceiverGrid) -> tuple[int, int]:
    return (grid.shape[1], grid.shape[0])


def _axis_index(vector: torch.Tensor, *, name: str) -> tuple[int, float]:
    values = vector.detach().cpu()
    nonzero = (values.abs() > 1.0e-6).nonzero(as_tuple=False).flatten()
    if int(nonzero.numel()) != 1:
        raise ValueError(f"{name} must be axis-aligned")
    index = int(nonzero[0].item())
    sign = float(torch.sign(values[index]).item())
    if abs(abs(float(values[index].item())) - 1.0) > 1.0e-5:
        raise ValueError(f"{name} must be a unit axis vector")
    return index, sign


def grid_spec(grid: ReceiverGrid) -> GridSpec:
    rows, cols = grid.shape
    axis0, sign0 = _axis_index(grid.x_axis, name="ReceiverGrid.x_axis")
    axis1, sign1 = _axis_index(grid.y_axis, name="ReceiverGrid.y_axis")
    if axis0 == axis1:
        raise ValueError("ReceiverGrid axes must be orthogonal")
    axis = ({0, 1, 2} - {axis0, axis1}).pop()
    origin = grid.origin.detach().cpu()
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
    first0 = float(origin[axis0].item())
    first1 = float(origin[axis1].item())
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
        position=float(origin[axis].item()),
        coord0_min=coord0_min,
        coord0_max=coord0_max,
        coord1_min=coord1_min,
        coord1_max=coord1_max,
        resolution0=rows,
        resolution1=cols,
        cell_area=abs((coord0_max - coord0_min) * (coord1_max - coord1_min)) / float(rows * cols),
    )


def _grid_los_gain(scene: Scene, grid: ReceiverGrid, *, device: torch.device) -> torch.Tensor:
    grid_scene = Scene(
        structures=scene.structures,
        transmitters=scene.transmitters,
        receivers=[grid],
        frequency=scene.frequency,
        metadata=scene.metadata,
    )
    return los_path_gain(grid_scene, device=device).reshape(len(scene.transmitters), *grid.shape)


def los_component_map(scene: Scene, raydn: RayDNScene, grid: ReceiverGrid, *, device: torch.device) -> torch.Tensor:
    los = _grid_los_gain(scene, grid, device=device)
    if not raydn.available or not scene.structures:
        return los.transpose(1, 2).contiguous()
    handle = raydn.require_handle()
    tx_pos, _ = transmitter_positions(scene, device=device)
    rx_pos = grid.points().to(device=device, dtype=torch.float32).contiguous()
    maps = []
    for tx_index in range(tx_pos.shape[0]):
        start = tx_pos[tx_index].expand(rx_pos.shape[0], 3).contiguous()
        active = torch.ones((rx_pos.shape[0],), device=device, dtype=torch.bool)
        visible = torch.ops.raydn.visibility_forward(handle, start, rx_pos, active)[0]
        internal = (los[tx_index].reshape(-1) * visible.to(dtype=los.dtype)).reshape(*grid.shape)
        maps.append(internal.transpose(0, 1).contiguous())
    return torch.stack(maps, dim=0) if maps else los.transpose(1, 2).contiguous()


def _sample_directions(count: int, *, device: torch.device) -> torch.Tensor:
    if count <= 0:
        return torch.empty((0, 3), device=device, dtype=torch.float32)
    indices = torch.arange(count, device=device, dtype=torch.float64)
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    azimuth_u = torch.frac(indices / golden_ratio)
    if count == 1:
        elevation_v = torch.zeros((1,), device=device, dtype=torch.float64)
    else:
        elevation_v = indices / float(count - 1)
    phi = 2.0 * math.pi * azimuth_u
    z = 1.0 - 2.0 * elevation_v
    radial = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    directions = torch.stack((radial * torch.cos(phi), radial * torch.sin(phi), z), dim=1)
    return directions.to(dtype=torch.float32).contiguous()


def _u32(value: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_and(value, 0xFFFFFFFF)


def _hash_uniform(index: torch.Tensor, *, stream: int, seed: int) -> torch.Tensor:
    idx = index.to(dtype=torch.int64)
    resolved_seed = int(seed) & 0xFFFFFFFF
    stream_value = int(stream) + 1
    value = (
        idx * 747796405
        + (resolved_seed + 1) * 2891336453
        + stream_value * 277803737
    )
    value = _u32(value)
    value = _u32(torch.bitwise_xor(value, value >> 16) * 2246822519)
    value = _u32(torch.bitwise_xor(value, value >> 13) * 3266489917)
    value = _u32(torch.bitwise_xor(value, value >> 16))
    mantissa = torch.bitwise_and(value, 0x00FFFFFF).to(dtype=torch.float32)
    return mantissa * (1.0 / 16777216.0)


def _diffraction_sample_slots(
    edge_length: torch.Tensor,
    *,
    samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = max(0, int(samples))
    device = edge_length.device
    if count == 0 or int(edge_length.numel()) == 0:
        return (
            torch.empty((0,), device=device, dtype=torch.int32),
            torch.empty((0,), device=device, dtype=torch.float32),
        )
    positive_length = edge_length.to(dtype=torch.float32).clamp_min(0.0)
    cdf = torch.cumsum(positive_length, dim=0)
    total_length = float(cdf[-1].detach().cpu().item())
    if total_length <= 0.0:
        return (
            torch.empty((0,), device=device, dtype=torch.int32),
            torch.empty((0,), device=device, dtype=torch.float32),
        )
    sample_index = torch.arange(count, device=device, dtype=torch.int64)
    sample_u = _hash_uniform(sample_index, stream=601, seed=seed) * total_length
    slots = torch.searchsorted(cdf, sample_u, right=False).to(dtype=torch.int32)
    slots = torch.clamp(slots, 0, int(edge_length.numel()) - 1)
    weight = torch.full(
        (count,),
        total_length / float(max(1, count)),
        device=device,
        dtype=torch.float32,
    )
    return slots.contiguous(), weight


def _empty_wedge_events(tx: torch.Tensor) -> WedgeEventBatch:
    device = tx.device
    return WedgeEventBatch(
        tx_pos=tx,
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
    collect_wedges: bool = False,
) -> ReflectionComponentResult:
    if not raydn.available or not scene.structures:
        tx_pos, _ = transmitter_positions(scene, device=device)
        maps = torch.zeros((len(scene.transmitters), *component_grid_shape(grid)), device=device, dtype=torch.float32)
        return ReflectionComponentResult(
            maps=maps,
            wedge_events=tuple(_empty_wedge_events(tx) for tx in tx_pos),
        )
    spec = grid_spec(grid)
    handle = raydn.require_handle()
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    wavelength = _LIGHT_SPEED_M_PER_S / float(scene.frequency)
    material_eta_r, material_sigma, material_mu_r, material_gain, material_valid = face_material_tensors(
        scene,
        device=device,
    )
    material_gain = torch.ones_like(material_gain)
    solid_angle_per_ray = float(4.0 * math.pi / max(1, int(samples)))
    maps = []
    wedge_batches: list[WedgeEventBatch] = []
    for tx_index, tx in enumerate(tx_pos):
        ray_d = _sample_directions(samples, device=device)
        ray_o = tx.expand(samples, 3).contiguous()
        ray_tmax = torch.empty((0,), device=device, dtype=torch.float32)
        active = torch.ones((samples,), device=device, dtype=torch.bool)
        tx_batch = ray_o
        tx_pol = torch.zeros_like(ray_o)
        tx_pol[:, 0] = 1.0
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
        )
        maps.append(out[0].contiguous() * tx_power[tx_index])
        if collect_wedges:
            count = int(out[8].detach().cpu().item())
            capacity = int(out[9].shape[0])
            if count > capacity:
                raise RuntimeError(
                    "RayDN reflection wedge event capacity was exceeded; "
                    "increase samples or wedge capacity before diffraction."
                )
            event_count = max(0, count)
            wedge_batches.append(
                WedgeEventBatch(
                    tx_pos=tx,
                    ray_dir=out[13][:event_count].contiguous(),
                    prim_id=out[12][:event_count].contiguous(),
                    hit_p=out[10][:event_count].contiguous(),
                    hit_n=out[11][:event_count].contiguous(),
                    hit_geo_n=out[11][:event_count].contiguous(),
                    bounce_depth=out[17][:event_count].contiguous(),
                )
            )
        else:
            wedge_batches.append(_empty_wedge_events(tx))
    stacked = torch.stack(maps, dim=0) if maps else torch.zeros((0, *component_grid_shape(grid)), device=device)
    return ReflectionComponentResult(maps=stacked, wedge_events=tuple(wedge_batches))


def reflection_component_map(
    scene: Scene,
    raydn: RayDNScene,
    grid: ReceiverGrid,
    *,
    samples: int,
    max_depth: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    return reflection_component_maps_with_wedges(
        scene,
        raydn,
        grid,
        samples=samples,
        max_depth=max_depth,
        seed=seed,
        device=device,
        collect_wedges=False,
    ).maps


def _safe_normalize_vectors(vectors: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    return torch.nn.functional.normalize(vectors, dim=1, eps=eps)


def _unsigned_angle(a: torch.Tensor, b: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    cross = torch.cross(a, b, dim=1)
    signed_norm = torch.sign((cross * axis).sum(dim=1)) * torch.linalg.vector_norm(cross, dim=1)
    angle = torch.atan2(signed_norm, (a * b).sum(dim=1))
    return torch.where(angle < 0.0, angle + 2.0 * torch.pi, angle)


def _diffraction_edge_geometry(records) -> tuple[torch.Tensor, ...]:
    selected = _selected_diffraction_edges(
        vertices=records.vertices,
        faces=records.faces,
        face_normals=records.face_normals,
        edge_v0=records.edge_v0,
        edge_v1=records.edge_v1,
        face0=records.face0,
        face1=records.face1,
        edge_policy=EdgePolicy(edge_selection_mode="all_edges", boundary_edge_policy="half_plane"),
        plane_tol=_RAYD_EDGE_INFO_PLANE_TOL,
    )
    edge_v0 = records.edge_v0.to(dtype=torch.long)
    edge_v1 = records.edge_v1.to(dtype=torch.long)
    vertices = records.vertices
    face0 = records.face0.to(dtype=torch.long)
    face1 = records.face1.to(dtype=torch.long)
    start = vertices[edge_v0]
    end = vertices[edge_v1]
    vectors = vertices[edge_v1] - vertices[edge_v0]
    lengths = torch.linalg.vector_norm(vectors, dim=1).clamp_min(1.0e-12)
    edge_dir = vectors / lengths[:, None]
    safe0 = face0.clamp_min(0)
    safe1 = face1.clamp_min(0)
    n0_cand = _safe_normalize_vectors(records.face_normals[safe0])
    n1_cand = _safe_normalize_vectors(records.face_normals[safe1])

    to1 = _safe_normalize_vectors(torch.cross(n0_cand, edge_dir, dim=1))
    tn1 = _safe_normalize_vectors(torch.cross(n1_cand, edge_dir, dim=1))
    to2 = _safe_normalize_vectors(torch.cross(n1_cand, edge_dir, dim=1))
    tn2 = _safe_normalize_vectors(torch.cross(n0_cand, edge_dir, dim=1))
    choose_first = _unsigned_angle(to1, tn1, edge_dir) < _unsigned_angle(to2, tn2, edge_dir)
    ordered_n0 = torch.where(choose_first[:, None], n0_cand, n1_cand)
    ordered_n1 = torch.where(choose_first[:, None], n1_cand, n0_cand)

    interior = (face0 >= 0) & (face1 >= 0)
    boundary = face1 < 0
    n0 = torch.where(interior[:, None], ordered_n0, n0_cand)
    n1 = torch.where(interior[:, None], ordered_n1, n1_cand)
    n1 = torch.where(boundary[:, None], -n0_cand, n1)
    normal_dot = (n0 * n1).sum(dim=1)
    interior_angle = torch.acos(torch.clamp(-normal_dot, -1.0, 1.0))
    exterior_angle = torch.where(
        interior,
        2.0 * torch.pi - interior_angle,
        torch.full_like(interior_angle, 2.0 * torch.pi),
    )
    return (
        selected,
        ((start + end) * 0.5).contiguous(),
        edge_dir.contiguous(),
        lengths.contiguous(),
        (-0.5 * lengths).contiguous(),
        (0.5 * lengths).contiguous(),
        n0.contiguous(),
        n1.contiguous(),
        face0,
        face1,
        exterior_angle.contiguous(),
    )


def _opposite_vertex_cpu(face: torch.Tensor, shared0: torch.Tensor, shared1: torch.Tensor) -> torch.Tensor:
    face = face.to(dtype=torch.long)
    x_other = (face[:, 0] != shared0) & (face[:, 0] != shared1)
    y_other = (face[:, 1] != shared0) & (face[:, 1] != shared1)
    return torch.where(x_other, face[:, 0], torch.where(y_other, face[:, 1], face[:, 2]))


def _surface_component_labels(records) -> tuple[torch.Tensor, int]:
    n_faces = int(records.faces.shape[0])
    if n_faces <= 0:
        return torch.empty((0,), dtype=torch.long), 0

    vertices = records.vertices.detach().cpu()
    faces = records.faces.detach().cpu().to(dtype=torch.long)
    face_normals = records.face_normals.detach().cpu()
    edge_v0 = records.edge_v0.detach().cpu().to(dtype=torch.long)
    edge_v1 = records.edge_v1.detach().cpu().to(dtype=torch.long)
    face0 = records.face0.detach().cpu().to(dtype=torch.long)
    face1 = records.face1.detach().cpu().to(dtype=torch.long)
    valid = (face0 >= 0) & (face1 >= 0)
    if not bool(valid.any()):
        labels = torch.arange(n_faces, dtype=torch.long)
        return labels, n_faces

    safe0 = face0.clamp_min(0)
    safe1 = face1.clamp_min(0)
    n0 = _safe_normalize_vectors(face_normals[safe0])
    n1 = _safe_normalize_vectors(face_normals[safe1])
    face_a = faces[safe0]
    face_b = faces[safe1]
    plane_point = vertices[edge_v0]
    point_a = vertices[_opposite_vertex_cpu(face_a, edge_v0, edge_v1)]
    point_b = vertices[_opposite_vertex_cpu(face_b, edge_v0, edge_v1)]
    normal_dot = (n0 * n1).sum(dim=1)
    aligned = normal_dot.abs() >= _RAYD_NORMAL_COS_TOL
    plane_dist_a = ((point_a - plane_point) * n0).sum(dim=1).abs()
    plane_dist_b = ((point_b - plane_point) * n0).sum(dim=1).abs()
    coplanar = (
        valid
        & aligned
        & (plane_dist_a <= 1.0e-5)
        & (plane_dist_b <= 1.0e-5)
    )

    parent = list(range(n_faces))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        low = min(root_a, root_b)
        high = max(root_a, root_b)
        parent[high] = low

    for edge_id in coplanar.nonzero(as_tuple=False).flatten().tolist():
        union(int(face0[edge_id].item()), int(face1[edge_id].item()))

    roots = [find(i) for i in range(n_faces)]
    root_to_group: dict[int, int] = {}
    group_ids = []
    for root in roots:
        group = root_to_group.setdefault(root, len(root_to_group))
        group_ids.append(group)
    return torch.tensor(group_ids, dtype=torch.long), len(root_to_group)


def _surface_group_edge_candidates(records, selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = records.vertices.device
    n_faces = int(records.faces.shape[0])
    if n_faces <= 0:
        return (
            torch.empty((0,), device=device, dtype=torch.int32),
            torch.empty((0, 0), device=device, dtype=torch.int32),
        )

    group_id, group_count = _surface_component_labels(records)
    if group_count <= 0:
        return (
            torch.zeros((n_faces,), device=device, dtype=torch.int32),
            torch.empty((n_faces, 0), device=device, dtype=torch.int32),
        )

    face0 = records.face0.detach().cpu().to(dtype=torch.long)
    face1 = records.face1.detach().cpu().to(dtype=torch.long)
    group_edges: list[list[int]] = [[] for _ in range(group_count)]
    for edge_idx in selected.detach().cpu().nonzero(as_tuple=False).flatten().tolist():
        f0 = int(face0[edge_idx].item())
        f1 = int(face1[edge_idx].item())
        valid0 = f0 >= 0
        valid1 = f1 >= 0
        group0 = int(group_id[f0].item()) if valid0 else -1
        if valid0:
            group_edges[group0].append(int(edge_idx))
        if valid1:
            group1 = int(group_id[f1].item())
            if (not valid0) or group1 != group0:
                group_edges[group1].append(int(edge_idx))

    max_edge_count = max((len(edges) for edges in group_edges), default=0)
    counts_cpu = torch.empty((n_faces,), dtype=torch.int32)
    indices_cpu = torch.full((n_faces, max_edge_count), -1, dtype=torch.int32)
    for face_idx in range(n_faces):
        edges = group_edges[int(group_id[face_idx].item())]
        counts_cpu[face_idx] = len(edges)
        if edges:
            indices_cpu[face_idx, : len(edges)] = torch.tensor(edges, dtype=torch.int32)
    return counts_cpu.to(device=device), indices_cpu.to(device=device)


def _triangle_edge_candidates(records, selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = records.vertices.device
    n_faces = int(records.faces.shape[0])
    if n_faces <= 0:
        return (
            torch.empty((0,), device=device, dtype=torch.int32),
            torch.empty((0, 3), device=device, dtype=torch.int32),
        )
    triangle_edge_count = torch.zeros((n_faces,), device=device, dtype=torch.int32)
    triangle_edge_indices = torch.full((n_faces, 3), -1, device=device, dtype=torch.int32)
    edge_idx = selected.nonzero(as_tuple=False).flatten().to(device=device, dtype=torch.int64)
    if int(edge_idx.numel()) == 0:
        return triangle_edge_count, triangle_edge_indices

    face0 = records.face0[edge_idx].to(dtype=torch.int64)
    face1 = records.face1[edge_idx].to(dtype=torch.int64)
    rows = torch.cat((face0, face1), dim=0)
    edges = torch.cat((edge_idx, edge_idx), dim=0)
    valid = rows >= 0
    rows = rows[valid]
    edges = edges[valid]
    if int(rows.numel()) == 0:
        return triangle_edge_count, triangle_edge_indices

    order = torch.argsort(rows)
    rows = rows[order]
    edges = edges[order]
    counts64 = torch.bincount(rows, minlength=n_faces).clamp_max(3).to(dtype=torch.int32)
    triangle_edge_count.copy_(counts64)

    new_group = torch.ones((rows.shape[0],), device=device, dtype=torch.bool)
    new_group[1:] = rows[1:] != rows[:-1]
    group_starts = new_group.nonzero(as_tuple=False).flatten()
    group_ends = torch.cat(
        (
            group_starts[1:],
            torch.tensor([rows.shape[0]], device=device, dtype=torch.long),
        ),
        dim=0,
    )
    group_lengths = group_ends - group_starts
    slot = torch.arange(rows.shape[0], device=device, dtype=torch.long) - torch.repeat_interleave(
        group_starts,
        group_lengths,
    )
    keep = slot < 3
    triangle_edge_indices[rows[keep], slot[keep]] = edges[keep].to(dtype=torch.int32)
    return triangle_edge_count, triangle_edge_indices


def _discover_diffraction_edges_from_wedges(raydn: RayDNScene, wedges: WedgeEventBatch) -> torch.Tensor:
    if int(wedges.prim_id.numel()) == 0:
        return torch.empty((0,), device=wedges.tx_pos.device, dtype=torch.int32)
    records = raydn.edge_records()
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
    ) = _diffraction_edge_geometry(records)
    if int(selected.sum().item()) == 0:
        return torch.empty((0,), device=wedges.tx_pos.device, dtype=torch.int32)
    triangle_edge_count, triangle_edge_indices = _surface_group_edge_candidates(records, selected)
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
        face1.to(device=wedges.tx_pos.device, dtype=torch.int32).contiguous(),
    )


def _diffraction_states_from_edge_indices(
    raydn: RayDNScene,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    edge_indices: torch.Tensor,
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
    ) = _diffraction_edge_geometry(records)
    idx = edge_indices.to(device=records.vertices.device, dtype=torch.long).contiguous()
    device = records.vertices.device
    count = int(idx.numel())
    src = tx.expand(count, 3).contiguous()
    src_power = tx_power.expand(count).contiguous()
    return (
        idx.to(device=device, dtype=torch.int32),
        edge_pos[idx].contiguous(),
        edge_dir[idx].contiguous(),
        line_min[idx].contiguous(),
        line_max[idx].contiguous(),
        n0[idx].contiguous(),
        n1[idx].contiguous(),
        face0[idx].to(device=device, dtype=torch.int32).contiguous(),
        face1[idx].to(device=device, dtype=torch.int32).contiguous(),
        exterior_angle[idx].contiguous(),
        src,
        src_power,
    )


def _diffraction_states(
    scene: Scene,
    raydn: RayDNScene,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    del scene
    records = raydn.edge_records()
    selected = _diffraction_edge_geometry(records)[0]
    return _diffraction_states_from_edge_indices(
        raydn,
        tx,
        tx_power,
        selected.nonzero(as_tuple=False).flatten().to(dtype=torch.int32),
    )


def diffraction_component_map(
    scene: Scene,
    raydn: RayDNScene,
    grid: ReceiverGrid,
    *,
    samples: int,
    seed: int,
    device: torch.device,
    wedge_events: tuple[WedgeEventBatch, ...] | None = None,
) -> torch.Tensor:
    if not raydn.available or not scene.structures:
        return torch.zeros((len(scene.transmitters), *component_grid_shape(grid)), device=device, dtype=torch.float32)
    spec = grid_spec(grid)
    handle = raydn.require_handle()
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    material_eta_r, material_sigma, material_mu_r, material_gain, material_valid = face_material_tensors(
        scene,
        device=device,
    )
    material_gain = torch.ones_like(material_gain)
    wavelength = _LIGHT_SPEED_M_PER_S / float(scene.frequency)
    maps = []
    for tx_index, tx in enumerate(tx_pos):
        if wedge_events is not None:
            edge_indices = _discover_diffraction_edges_from_wedges(raydn, wedge_events[tx_index])
            states = _diffraction_states_from_edge_indices(raydn, tx, tx_power[tx_index], edge_indices)
        else:
            states = _diffraction_states(scene, raydn, tx, tx_power[tx_index])
        state_count = int(states[0].shape[0])
        if state_count <= 0:
            maps.append(torch.zeros(component_grid_shape(grid), device=device, dtype=torch.float32))
            continue
        state_wi = torch.nn.functional.normalize(states[1] - states[10], dim=1, eps=1.0e-6).contiguous()
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
        maps.append(out[0].contiguous())
    return torch.stack(maps, dim=0) if maps else torch.zeros((0, *component_grid_shape(grid)), device=device)
