from __future__ import annotations

import torch

from witwin.channel_native.propagation.topology.kernels import blocks as topology_blocks
from witwin.channel_native.propagation.topology.kernels import (
    compaction as topology_compaction,
)
from witwin.channel_native.propagation.topology.kernels import (
    construction as topology_construction,
)


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


def _block_sequence_width(block: dict[str, torch.Tensor]) -> int:
    sequence_field = block.get("primitive_sequence")
    return (
        int(sequence_field.shape[1]) if isinstance(sequence_field, torch.Tensor) else 0
    )


def concatenate_path_blocks(
    blocks: list[dict[str, torch.Tensor]], *, device: torch.device
) -> dict[str, torch.Tensor]:
    nonempty = [block for block in blocks if int(block["valid"].numel()) > 0]
    if not nonempty:
        return _empty_path_block(device)
    # Blocks from different bounce depths carry different sequence widths
    # (e.g. depth-2 and depth-3 multibounce blocks); pad to the widest before
    # the native concat, which requires a uniform width.
    sequence_width = max(_block_sequence_width(block) for block in nonempty)
    nonempty = [
        block
        if _block_sequence_width(block) == sequence_width
        else _pad_topology_sequences(block, width=sequence_width)
        for block in nonempty
    ]
    return topology_blocks.deterministic_concat_topology_blocks(
        nonempty, sequence_width=sequence_width
    )


def _sort_order(
    paths: dict[str, torch.Tensor], *, tx_count: int, max_depth: int
) -> torch.Tensor:
    del tx_count, max_depth
    sequence = paths.get("primitive_sequence")
    if sequence is None or sequence.dim() != 2:
        sequence = torch.empty(
            (paths["valid"].numel(), 0), device=paths["valid"].device, dtype=torch.int32
        )
    return topology_compaction.deterministic_sort_order(
        paths["valid"],
        paths["tx_id"],
        paths["rx_id"],
        paths["depth"],
        paths["component_id"],
        paths["primitive_id"],
        paths["edge_id"],
        sequence.to(dtype=torch.int32).contiguous(),
    )


def _interaction_type_sequence(
    *, component_id: torch.Tensor, depth: torch.Tensor, width: int
) -> torch.Tensor:
    count = int(component_id.numel())
    result = torch.zeros((count, width), device=component_id.device, dtype=torch.int32)
    if count == 0 or width == 0:
        return result
    slots = torch.arange(width, device=component_id.device).reshape(1, -1)
    active = slots < depth.to(dtype=torch.int64).reshape(-1, 1)
    result[active & (component_id.reshape(-1, 1) == 1)] = 1
    diffraction = (component_id == 2) & (depth > 0)
    result[diffraction, 0] = 2
    # component_id 5=transmission, 6=scattering map to the InteractionType flags
    # TRANSMISSION=4 and SCATTERING=8. A transmission chain penetrates `depth`
    # walls, so every active slot is a TRANSMISSION event (like reflection);
    # scattering is single-bounce in v1.
    result[active & (component_id.reshape(-1, 1) == 5)] = 4
    scattering = (component_id == 6) & (depth > 0)
    result[scattering, 0] = 8
    if width >= 2:
        reflection_diffraction = (component_id == 3) & (depth >= 2)
        result[reflection_diffraction, 0] = 1
        result[reflection_diffraction, 1] = 2
        diffraction_reflection = (component_id == 4) & (depth >= 2)
        result[diffraction_reflection, 0] = 2
        result[diffraction_reflection, 1] = 1
    return result.contiguous()


def canonical_sequence_key(paths: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return canonical ``(event_type, object_id)`` columns for each path."""

    sequence = paths.get("primitive_sequence")
    if sequence is None or sequence.dim() != 2:
        sequence = torch.empty(
            (paths["valid"].numel(), 0),
            device=paths["valid"].device,
            dtype=torch.int32,
        )
    sequence = sequence.to(dtype=torch.int32).contiguous()
    width = int(sequence.shape[1])
    interaction_type = _interaction_type_sequence(
        component_id=paths["component_id"], depth=paths["depth"], width=width
    )
    object_id = sequence.clone()
    if width:
        diffraction = (paths["component_id"] == 2) & (paths["depth"] > 0)
        object_id[diffraction, 0] = paths["edge_id"][diffraction]
        object_id[interaction_type == 0] = -1
    return torch.stack((interaction_type, object_id), dim=-1).contiguous()


def _canonical_selection_order(
    paths: dict[str, torch.Tensor],
    *,
    tx_count: int,
    max_depth: int,
    max_paths: int | None,
    max_paths_scope: str,
) -> torch.Tensor:
    order = _sort_order(paths, tx_count=tx_count, max_depth=max_depth)
    count = int(order.numel())
    if count > 1:
        key = canonical_sequence_key(paths)[order].reshape(count, -1)
        tx_id = paths["tx_id"][order]
        rx_id = paths["rx_id"][order]
        same = (tx_id[1:] == tx_id[:-1]) & (rx_id[1:] == rx_id[:-1])
        if int(key.shape[1]) > 0:
            same &= (key[1:] == key[:-1]).all(dim=1)
        group_start = torch.ones((count,), device=order.device, dtype=torch.bool)
        group_start[1:] = ~same
        group_id = group_start.cumsum(dim=0, dtype=torch.int64) - 1
        length = paths["path_length_m"][order]
        minimum = torch.full(
            (count,), float("inf"), device=order.device, dtype=length.dtype
        )
        minimum.scatter_reduce_(0, group_id, length, reduce="amin", include_self=True)
        shortest = length == minimum[group_id]
        shortest_group = group_id[shortest]
        unique_shortest = torch.ones_like(shortest_group, dtype=torch.bool)
        unique_shortest[1:] = shortest_group[1:] != shortest_group[:-1]
        unique = torch.zeros((count,), device=order.device, dtype=torch.bool)
        unique[torch.nonzero(shortest, as_tuple=False).reshape(-1)[unique_shortest]] = (
            True
        )
        order = order[unique]

    if max_paths is not None and max_paths_scope == "global":
        order = order[: int(max_paths)]
    elif max_paths is not None and int(order.numel()) > 0:
        tx_id = paths["tx_id"][order].to(dtype=torch.int64)
        rx_id = paths["rx_id"][order].to(dtype=torch.int64)
        pair = rx_id * max(int(tx_count), 1) + tx_id
        row = torch.arange(int(order.numel()), device=order.device, dtype=torch.int64)
        first = torch.ones_like(pair, dtype=torch.bool)
        first[1:] = pair[1:] != pair[:-1]
        starts = torch.where(first, row, torch.zeros_like(row)).cummax(dim=0).values
        order = order[(row - starts) < int(max_paths)]
    return order.contiguous()


def _pad_topology_sequences(
    block: dict[str, torch.Tensor], *, width: int
) -> dict[str, torch.Tensor]:
    if width < 0:
        raise ValueError("sequence width must be non-negative")
    count = int(block["valid"].numel())
    device = block["valid"].device
    empty_i32 = torch.empty((count, 0), device=device, dtype=torch.int32)
    empty_vec3 = torch.empty((count, 0, 3), device=device, dtype=torch.float32)
    sequences = topology_construction.deterministic_pad_topology_sequences(
        depth=block["depth"].to(dtype=torch.int32).contiguous(),
        primitive_id=block["primitive_id"].to(dtype=torch.int32).contiguous(),
        material_id=block["material_id"].to(dtype=torch.int32).contiguous(),
        interaction_position=block["interaction_position"]
        .to(dtype=torch.float32)
        .contiguous(),
        interaction_normal=block["interaction_normal"]
        .to(dtype=torch.float32)
        .contiguous(),
        primitive_sequence=block.get("primitive_sequence", empty_i32)
        .to(dtype=torch.int32)
        .contiguous(),
        material_sequence=block.get("material_sequence", empty_i32)
        .to(dtype=torch.int32)
        .contiguous(),
        interaction_positions=block.get("interaction_positions", empty_vec3)
        .to(dtype=torch.float32)
        .contiguous(),
        interaction_normals=block.get("interaction_normals", empty_vec3)
        .to(dtype=torch.float32)
        .contiguous(),
        width=int(width),
    )

    padded = dict(block)
    padded["primitive_sequence"] = sequences["primitive_sequence"]
    padded["material_sequence"] = sequences["material_sequence"]
    padded["interaction_positions"] = sequences["interaction_positions"]
    padded["interaction_normals"] = sequences["interaction_normals"]
    return padded


__all__ = ["canonical_sequence_key", "concatenate_path_blocks"]
