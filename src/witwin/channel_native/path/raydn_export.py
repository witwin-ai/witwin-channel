from __future__ import annotations

import math

import torch

from witwin.channel_native import Scene
from witwin.channel_native.core.material_runtime import face_material_tensors
from witwin.channel_native.core.runtime.raydn import RayDNScene
from witwin.channel_native.montecarlo.basic.backend import _LIGHT_SPEED_M_PER_S
from witwin.channel_native.montecarlo.basic.raydn_components import _diffraction_states


_COMPONENT_ID = {"los": 0, "reflection": 1, "diffraction": 2}
_REFLECTION_VIS_EPS = 1.0e-4
_REFLECTION_BARY_EPS = 1.0e-4


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


def _inside_triangle(points: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    v0 = b - a
    v1 = c - a
    v2 = points - a
    d00 = (v0 * v0).sum(dim=1)
    d01 = (v0 * v1).sum(dim=1)
    d11 = (v1 * v1).sum(dim=1)
    d20 = (v2 * v0).sum(dim=1)
    d21 = (v2 * v1).sum(dim=1)
    denom = d00 * d11 - d01 * d01
    safe = denom.abs() > 1.0e-12
    inv = torch.where(safe, denom.reciprocal(), torch.zeros_like(denom))
    v = (d11 * d20 - d01 * d21) * inv
    w = (d00 * d21 - d01 * d20) * inv
    u = 1.0 - v - w
    eps = _REFLECTION_BARY_EPS
    return safe & (u >= -eps) & (v >= -eps) & (w >= -eps) & (u <= 1.0 + eps) & (v <= 1.0 + eps) & (w <= 1.0 + eps)


def _segment_visibility(
    raydn: RayDNScene,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    if start.shape[0] == 0:
        return torch.empty((0,), device=start.device, dtype=torch.bool)
    active = torch.ones((start.shape[0],), device=start.device, dtype=torch.bool)
    return torch.ops.raydn.visibility_forward(raydn.require_handle(), start.contiguous(), end.contiguous(), active)[0]


def los_visibility_mask(
    raydn: RayDNScene,
    tx_for_path: torch.Tensor,
    rx_for_path: torch.Tensor,
    *,
    has_structures: bool,
) -> torch.Tensor:
    if not has_structures or not raydn.available or tx_for_path.shape[0] == 0:
        return torch.ones((tx_for_path.shape[0],), device=tx_for_path.device, dtype=torch.bool)
    return _segment_visibility(raydn, tx_for_path, rx_for_path)


def reflection_paths_order1(
    scene: Scene,
    raydn: RayDNScene,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    device = tx_positions.device
    if not raydn.available or not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return _empty_path_block(device)

    records = raydn.edge_records()
    vertices = records.vertices
    faces = records.faces.to(dtype=torch.long)
    normals = torch.nn.functional.normalize(records.face_normals, dim=1, eps=1.0e-6)
    if faces.shape[0] == 0:
        return _empty_path_block(device)

    tri_a = vertices[faces[:, 0]]
    tri_b = vertices[faces[:, 1]]
    tri_c = vertices[faces[:, 2]]
    face_gain = face_material_tensors(scene, device=device)[3]
    wavelength = _LIGHT_SPEED_M_PER_S / float(frequency_hz)

    blocks: list[dict[str, torch.Tensor]] = []
    face_count = int(faces.shape[0])
    face_ids = torch.arange(face_count, device=device, dtype=torch.int32)
    for rx_index, rx in enumerate(rx_positions):
        for tx_index, tx in enumerate(tx_positions):
            tx_to_plane = ((tx.reshape(1, 3) - tri_a) * normals).sum(dim=1)
            image = tx.reshape(1, 3) - 2.0 * tx_to_plane[:, None] * normals
            line = rx.reshape(1, 3) - image
            denom = (line * normals).sum(dim=1)
            valid = denom.abs() > 1.0e-8
            s = ((tri_a - image) * normals).sum(dim=1) / torch.where(valid, denom, torch.ones_like(denom))
            valid = valid & (s > 1.0e-6) & (s < 1.0 - 1.0e-6)
            points = image + s[:, None] * line
            valid = valid & _inside_triangle(points, tri_a, tri_b, tri_c)
            idx = valid.nonzero(as_tuple=False).flatten()
            if int(idx.numel()) == 0:
                continue

            p = points[idx]
            tx_batch = tx.expand(p.shape[0], 3)
            rx_batch = rx.expand(p.shape[0], 3)
            d0 = torch.nn.functional.normalize(p - tx_batch, dim=1, eps=1.0e-6)
            d1 = torch.nn.functional.normalize(rx_batch - p, dim=1, eps=1.0e-6)
            seg0_start = tx_batch + d0 * _REFLECTION_VIS_EPS
            seg0_end = p - d0 * _REFLECTION_VIS_EPS
            seg1_start = p + d1 * _REFLECTION_VIS_EPS
            seg1_end = rx_batch - d1 * _REFLECTION_VIS_EPS
            visible = _segment_visibility(raydn, seg0_start, seg0_end) & _segment_visibility(raydn, seg1_start, seg1_end)
            keep = visible.nonzero(as_tuple=False).flatten()
            if int(keep.numel()) == 0:
                continue

            selected_faces = idx[keep]
            selected_points = p[keep]
            tx_keep = tx.expand(keep.shape[0], 3)
            rx_keep = rx.expand(keep.shape[0], 3)
            path_length = (
                torch.linalg.vector_norm(selected_points - tx_keep, dim=1)
                + torch.linalg.vector_norm(rx_keep - selected_points, dim=1)
            ).clamp_min(1.0e-6)
            gain = face_gain[selected_faces.to(dtype=torch.long)].clamp_min(0.0)
            path_gain = tx_power[tx_index] * gain * torch.square(wavelength / (4.0 * math.pi * path_length))
            count = int(path_length.shape[0])
            blocks.append(
                {
                    "valid": torch.ones((count,), device=device, dtype=torch.bool),
                    "tx_id": torch.full((count,), tx_index, device=device, dtype=torch.int32),
                    "rx_id": torch.full((count,), rx_index, device=device, dtype=torch.int32),
                    "depth": torch.ones((count,), device=device, dtype=torch.int32),
                    "component_id": torch.full((count,), _COMPONENT_ID["reflection"], device=device, dtype=torch.int32),
                    "primitive_id": face_ids[selected_faces].contiguous(),
                    "edge_id": torch.full((count,), -1, device=device, dtype=torch.int32),
                    "path_length_m": path_length.to(dtype=torch.float32).contiguous(),
                    "delay_s": (path_length / _LIGHT_SPEED_M_PER_S).to(dtype=torch.float32).contiguous(),
                    "path_gain": path_gain.to(dtype=torch.float32).contiguous(),
                }
            )

    return concatenate_path_blocks(blocks, device=device)


def diffraction_paths_order1(
    scene: Scene,
    raydn: RayDNScene,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    device = tx_positions.device
    if not raydn.available or not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return _empty_path_block(device)

    material_gain = face_material_tensors(scene, device=device)[3]
    material_valid = torch.ones_like(material_gain, dtype=torch.bool)
    wavelength = _LIGHT_SPEED_M_PER_S / float(frequency_hz)
    handle = raydn.require_handle()
    blocks: list[dict[str, torch.Tensor]] = []
    for tx_index, tx in enumerate(tx_positions):
        states = _diffraction_states(scene, raydn, tx, tx_power[tx_index])
        state_count = int(states[0].shape[0])
        if state_count <= 0:
            continue
        capacity = int(rx_positions.shape[0]) * state_count
        out = torch.ops.raydn.diffraction_paths_order1_forward(
            handle,
            tx.reshape(1, 3).contiguous(),
            rx_positions.contiguous(),
            None,
            *states,
            material_gain,
            material_valid,
            state_count,
            capacity,
            float(wavelength),
        )
        count = max(0, min(capacity, int(out[0].detach().cpu().item())))
        if count <= 0:
            continue
        valid = out[1][:count]
        idx = valid.nonzero(as_tuple=False).flatten()
        if int(idx.numel()) == 0:
            continue
        field_power = (
            out[9][idx].square()
            + out[10][idx].square()
            + out[11][idx].square()
            + out[12][idx].square()
            + out[13][idx].square()
            + out[14][idx].square()
        )
        delay = out[8][idx].to(dtype=torch.float32).contiguous()
        path_length = (delay * _LIGHT_SPEED_M_PER_S).to(dtype=torch.float32).contiguous()
        path_count = int(idx.numel())
        blocks.append(
            {
                "valid": torch.ones((path_count,), device=device, dtype=torch.bool),
                "tx_id": torch.full((path_count,), tx_index, device=device, dtype=torch.int32),
                "rx_id": out[3][idx].to(dtype=torch.int32).contiguous(),
                "depth": out[4][idx].to(dtype=torch.int32).contiguous(),
                "component_id": torch.full((path_count,), _COMPONENT_ID["diffraction"], device=device, dtype=torch.int32),
                "primitive_id": torch.full((path_count,), -1, device=device, dtype=torch.int32),
                "edge_id": out[5][idx].to(dtype=torch.int32).contiguous(),
                "path_length_m": path_length,
                "delay_s": delay,
                "path_gain": field_power.to(dtype=torch.float32).contiguous(),
            }
        )
    return concatenate_path_blocks(blocks, device=device)


def concatenate_path_blocks(blocks: list[dict[str, torch.Tensor]], *, device: torch.device) -> dict[str, torch.Tensor]:
    nonempty = [block for block in blocks if int(block["valid"].numel()) > 0]
    if not nonempty:
        return _empty_path_block(device)
    return {key: torch.cat([block[key] for block in nonempty], dim=0).contiguous() for key in nonempty[0]}
