"""Enumerated straight-segment transmission topology discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from witwin.channel_native.core.scene_tensors import (
    LIGHT_SPEED_M_PER_S as _LIGHT_SPEED_M_PER_S,
)
from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge
from witwin.channel_native.propagation.geometry.kernels import (
    primitives as geometry_primitives,
)
from witwin.channel_native.propagation.topology.concatenate import _empty_path_block
from witwin.channel_native.propagation.topology.export import _ensure_topology_fields

if TYPE_CHECKING:
    from witwin.channel_native.core.scene import Scene


def _transmission_topology(
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    max_depth: int,
) -> tuple[dict[str, torch.Tensor], int, int, int]:
    """Straight-segment specular transmission topology (contract section 4).

    For every (tx, rx) pair, march the direct segment through the scene with
    batched closest-hit queries: each hit past the previous one is a wall
    penetration event. Pairs whose direct segment is clear keep their LoS path
    and get no transmission path; pairs with 1..max_depth penetrations through
    valid thin_sheet materials become one component_id=5 path. Deeper chains
    and invalid materials produce no path and are counted as guardrails.

    Returns ``(block, launch_count, candidate_count, guardrail_count)``.
    """

    device = tx_positions.device
    raydn = compiled.raydn
    if (
        not scene.structures
        or tx_positions.numel() == 0
        or rx_positions.numel() == 0
        or max_depth < 1
    ):
        return _ensure_topology_fields(_empty_path_block(device)), 0, 0, 0
    if not raydn.available:
        raise RuntimeError(
            "deterministic transmission requires RayDN native scene capability"
        )

    handle = raydn.require_handle()
    records = raydn.edge_records()
    vertices = records.vertices
    scene_diagonal = (
        vertices.max(dim=0).values - vertices.min(dim=0).values
    ).norm()
    face_material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int64
    ).contiguous()
    geometry_mode_id = compiled.materials.geometry_mode_id.to(
        device=device, dtype=torch.int64
    ).contiguous()

    tx_count = int(tx_positions.shape[0])
    rx_count = int(rx_positions.shape[0])
    pair_count = tx_count * rx_count
    tx_index = torch.arange(
        tx_count, device=device, dtype=torch.int64
    ).repeat_interleave(rx_count)
    rx_index = torch.arange(rx_count, device=device, dtype=torch.int64).repeat(
        tx_count
    )
    source = tx_positions[tx_index].contiguous()
    target = rx_positions[rx_index].contiguous()
    offset = target - source
    total_length = offset.norm(dim=-1)
    direction = geometry_primitives.deterministic_normalize_vec3(
        offset.contiguous(), eps=1.0e-9
    )

    positions = torch.zeros(
        (pair_count, max_depth, 3), device=device, dtype=torch.float32
    )
    normals = torch.zeros_like(positions)
    prims = torch.full((pair_count, max_depth), -1, device=device, dtype=torch.int64)
    depth_count = torch.zeros((pair_count,), device=device, dtype=torch.int64)
    invalid = torch.zeros((pair_count,), device=device, dtype=torch.bool)
    # Degenerate zero-length segments carry no transmission path.
    done = total_length <= 1.0e-9
    origin = source.clone()
    traveled = torch.zeros_like(total_length)
    launch_count = 0

    # max_depth events plus one probe that distinguishes "clear behind the
    # last wall" from "more than max_depth penetrations" (depth cap stays
    # truthful: deeper chains are dropped, never truncated).
    for _step in range(max_depth + 1):
        rows = torch.nonzero(~done & ~invalid, as_tuple=False).reshape(-1)
        if int(rows.shape[0]) == 0:
            break
        remaining = (total_length[rows] - traveled[rows]).clamp_min(0.0)
        hit = geometry_bridge.bdpt_intersect_forward(
            handle,
            origin[rows].contiguous(),
            direction[rows].contiguous(),
            remaining.contiguous(),
            None,
            flags=7,
        )
        launch_count += 1
        hit_t = hit["t"]
        hit_prim = hit["global_prim_id"].to(dtype=torch.int64)
        blocked = (hit_prim >= 0) & torch.isfinite(hit_t) & (hit_t < remaining)
        done[rows[~blocked]] = True
        hit_rows = rows[blocked]
        if int(hit_rows.shape[0]) == 0:
            continue
        overflow = depth_count[hit_rows] >= max_depth
        invalid[hit_rows[overflow]] = True
        keep = hit_rows[~overflow]
        if int(keep.shape[0]) == 0:
            continue
        kept = ~overflow
        hit_position = hit["p"][blocked][kept]
        hit_normal = geometry_primitives.deterministic_normalize_vec3(
            hit["geo_n"][blocked][kept].contiguous(), eps=1.0e-9
        )
        kept_prim = hit_prim[blocked][kept]
        kept_t = hit_t[blocked][kept]
        slot = depth_count[keep]
        positions[keep, slot] = hit_position
        normals[keep, slot] = hit_normal
        prims[keep, slot] = kept_prim
        depth_count[keep] += 1
        # Scale-aware epsilon (contract section 4): advance the segment start
        # past the hit so the march never re-hits the same coplanar sheet.
        epsilon = torch.maximum(
            hit_position.norm(dim=-1) * 1.0e-6, scene_diagonal * 1.0e-6
        ).clamp_min(1.0e-6)
        origin[keep] = hit_position + direction[keep] * epsilon[:, None]
        traveled[keep] = traveled[keep] + kept_t + epsilon

    mats = torch.where(prims >= 0, face_material_id[prims.clamp_min(0)], prims)
    # Every penetrated face must resolve to a valid thin_sheet material.
    event_active = torch.arange(max_depth, device=device).reshape(
        1, -1
    ) < depth_count.reshape(-1, 1)
    bad_material = (
        event_active & ((mats < 0) | (geometry_mode_id[mats.clamp_min(0)] != 0))
    ).any(dim=-1)
    penetrated = done & ~invalid & (depth_count >= 1)
    selected = penetrated & ~bad_material
    candidate_count = int((invalid | penetrated).sum())
    guardrail_count = int((invalid | (penetrated & bad_material)).sum())

    chosen = torch.nonzero(selected, as_tuple=False).reshape(-1)
    count = int(chosen.shape[0])
    if count == 0:
        return (
            _ensure_topology_fields(_empty_path_block(device)),
            launch_count,
            candidate_count,
            guardrail_count,
        )
    path_length = total_length[chosen].to(dtype=torch.float32).contiguous()
    nan = torch.full((count,), float("nan"), device=device, dtype=torch.float32)
    block = _ensure_topology_fields(
        {
            "valid": torch.ones((count,), device=device, dtype=torch.bool),
            "tx_id": tx_index[chosen].to(dtype=torch.int32).contiguous(),
            "rx_id": rx_index[chosen].to(dtype=torch.int32).contiguous(),
            "depth": depth_count[chosen].to(dtype=torch.int32).contiguous(),
            "component_id": torch.full(
                (count,), 5, device=device, dtype=torch.int32
            ),
            "primitive_id": prims[chosen, 0].to(dtype=torch.int32).contiguous(),
            "edge_id": torch.full((count,), -1, device=device, dtype=torch.int32),
            "path_length_m": path_length,
            "delay_s": (path_length / _LIGHT_SPEED_M_PER_S).contiguous(),
            # Physical coefficients come from the shared complex3 evaluation
            # (field_transmission_sequence); the topology exports geometry only.
            "path_gain": nan,
        },
        interaction_position=positions[chosen, 0],
        interaction_normal=normals[chosen, 0],
        material_id=mats[chosen, 0].to(dtype=torch.int32),
        path_field=torch.complex(nan, nan),
        primitive_sequence=prims[chosen].to(dtype=torch.int32),
        material_sequence=mats[chosen].to(dtype=torch.int32),
        interaction_positions=positions[chosen],
        interaction_normals=normals[chosen],
    )
    return block, launch_count, candidate_count, guardrail_count
