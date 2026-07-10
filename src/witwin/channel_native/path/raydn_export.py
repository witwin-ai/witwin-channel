from __future__ import annotations

import torch

from witwin.channel_native import Scene
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.core.edge_selection import refine_edge_geometry
from witwin.channel_native.core.runtime.raydn import RayDNScene
from witwin.channel_native.core.scene import _RAYD_EDGE_INFO_PLANE_TOL

MaterialTensors = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
_LIGHT_SPEED_M_PER_S = 299_792_458.0


def _empty_path_block(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "valid": torch.empty((0,), device=device, dtype=torch.bool),
        "tx_id": torch.empty((0,), device=device, dtype=torch.int32),
        "rx_id": torch.empty((0,), device=device, dtype=torch.int32),
        "depth": torch.empty((0,), device=device, dtype=torch.int32),
        "component_id": torch.empty((0,), device=device, dtype=torch.int32),
        "primitive_id": torch.empty((0,), device=device, dtype=torch.int32),
        "edge_id": torch.empty((0,), device=device, dtype=torch.int32),
        "path_length_m": torch.empty((0,), device=device, dtype=torch.float32),
        "delay_s": torch.empty((0,), device=device, dtype=torch.float32),
        "path_gain": torch.empty((0,), device=device, dtype=torch.float32),
    }


def empty_path_block(device: torch.device) -> dict[str, torch.Tensor]:
    return _empty_path_block(device)


def _diffraction_edge_geometry(raydn: RayDNScene) -> tuple[torch.Tensor, ...]:
    cached = raydn.runtime_cache.get("path_diffraction_edge_geometry")
    if cached is not None:
        return cached  # type: ignore[return-value]
    records = raydn.edge_records()
    geometry = refine_edge_geometry(
        raydn,
        ops.bdpt_diffraction_edge_geometry(
            records.vertices,
            records.faces,
            records.face_normals,
            records.edge_v0,
            records.edge_v1,
            records.face0,
            records.face1,
            plane_tol=_RAYD_EDGE_INFO_PLANE_TOL,
        ),
    )
    raydn.runtime_cache["path_diffraction_edge_geometry"] = geometry
    return geometry


def los_paths(
    scene: Scene,
    raydn: RayDNScene,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    exported = ops.path_los_export(
        tx_positions,
        tx_power,
        rx_positions,
        frequency_hz=frequency_hz,
    )
    if exported["tx_id"].shape[0] == 0:
        return _empty_path_block(tx_positions.device)
    visibility_inputs = ops.path_los_visibility_inputs(
        tx_positions,
        rx_positions,
        exported["tx_id"],
        exported["rx_id"],
    )
    if scene.structures:
        if not raydn.available:
            raise RuntimeError("LoS path export requires RayDN native scene capability")
        visible = ops.raydn_visibility_forward(
            raydn.require_handle(),
            visibility_inputs["start"],
            visibility_inputs["end"],
            visibility_inputs["active"],
        )[0]
    else:
        visible = visibility_inputs["active"]
    return ops.path_filter_los(
        exported["tx_id"],
        exported["rx_id"],
        exported["path_length_m"],
        exported["delay_s"],
        exported["path_gain"],
        visible,
    )


def reflection_paths_order1(
    scene: Scene,
    raydn: RayDNScene,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
    material_tensors: MaterialTensors,
) -> dict[str, torch.Tensor]:
    device = tx_positions.device
    if not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return _empty_path_block(device)
    if not raydn.available:
        raise RuntimeError("reflection paths require RayDN native scene capability")

    records = raydn.edge_records()
    if records.faces.shape[0] == 0:
        return _empty_path_block(device)
    face_gain = material_tensors[3]
    candidates = ops.path_reflection_candidates(
        records.vertices,
        records.faces,
        records.face_normals,
        face_gain,
        tx_positions,
        tx_power,
        rx_positions,
        frequency_hz=frequency_hz,
    )
    if candidates["valid"].shape[0] == 0:
        return _empty_path_block(device)

    handle = raydn.require_handle()
    visible0 = ops.raydn_visibility_forward(
        handle,
        candidates["seg0_start"],
        candidates["seg0_end"],
        candidates["active"],
    )[0]
    visible1 = ops.raydn_visibility_forward(
        handle,
        candidates["seg1_start"],
        candidates["seg1_end"],
        candidates["active"],
    )[0]
    block = ops.path_filter_block(candidates, visible0, visible1)
    return _dedupe_coplanar_reflections(block, records, rx_count=int(rx_positions.shape[0]))


def _dedupe_coplanar_reflections(
    block: dict[str, torch.Tensor],
    records: object,
    *,
    rx_count: int,
) -> dict[str, torch.Tensor]:
    """Keep one specular path per (tx, rx, coplanar face group).

    A wall meshed from several coplanar triangles yields one physical
    reflection; when the specular point lands on a shared edge the per-face
    candidate enumeration passes containment for every incident triangle.
    """

    valid = block["valid"]
    if int(valid.numel()) == 0 or int(valid.sum()) <= 1:
        return block
    faces = records.faces
    tri_a = ops.deterministic_face_anchor_points(records.vertices, faces)
    normals = ops.deterministic_normalize_vec3(records.face_normals, eps=1.0e-6)
    groups = ops.deterministic_face_groups(
        tri_a,
        normals,
        torch.zeros((int(faces.shape[0]),), device=faces.device, dtype=torch.long),
        quantization=1.0e-4,
    )
    face_group = groups["face_group_id"].to(dtype=torch.long)
    group_count = max(int(groups["group_count"]), 1)
    group = face_group[block["primitive_id"].to(dtype=torch.long).clamp_(min=0)]
    key = (block["tx_id"].to(dtype=torch.long) * rx_count + block["rx_id"].to(dtype=torch.long)) * group_count + group
    # Give invalid rows unique keys so they never absorb a valid row's slot.
    row = torch.arange(int(valid.numel()), device=valid.device, dtype=torch.long)
    key = torch.where(valid, key, key.max() + 1 + row)
    order = torch.argsort(key, stable=True)
    sorted_key = key[order]
    first = torch.ones_like(sorted_key, dtype=torch.bool)
    first[1:] = sorted_key[1:] != sorted_key[:-1]
    keep = torch.empty_like(first)
    keep[order] = first
    if bool(keep.all()):
        return block
    return {name: tensor[keep] for name, tensor in block.items()}


def diffraction_paths_order1(
    scene: Scene,
    raydn: RayDNScene,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
    material_tensors: MaterialTensors,
) -> dict[str, torch.Tensor]:
    device = tx_positions.device
    if not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return _empty_path_block(device)
    if not raydn.available:
        raise RuntimeError("diffraction paths require RayDN native scene capability")

    eps_r, sigma_e, mu_r, material_gain, material_valid = material_tensors
    wavelength = _LIGHT_SPEED_M_PER_S / float(frequency_hz)
    return ops.path_diffraction_paths_order1(
        raydn.require_handle(),
        tx_positions,
        tx_power,
        rx_positions,
        _diffraction_edge_geometry(raydn),
        eps_r,
        sigma_e,
        mu_r,
        material_gain,
        material_valid,
        wavelength=float(wavelength),
    )
