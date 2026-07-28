"""Discrete propagation topology ownership boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch

from witwin.channel.kernels.topology import (
    mc_sample_directions as mc_sample_directions,
    path_los_export as path_los_export,
)
from witwin.channel.kernels import topology as topology_kernels
from witwin.channel.propagation.rows import (
    EvaluatedPaths,
    PathFields,
    PathGeometry,
    PathTopology,
)
from witwin.channel.runtime import CapacityExecutionCounts, SolveCapacityTransaction


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
    return topology_kernels.deterministic_concat_topology_blocks(
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
    return topology_kernels.deterministic_sort_order(
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
        # component_id 7 = double diffraction: two sequential edge (DIFFRACTION)
        # events. Both slots carry the edge object ids in primitive_sequence, so
        # the canonical dedup key keeps distinct edge pairs distinct.
        double_diffraction = (component_id == 7) & (depth >= 2)
        result[double_diffraction, 0] = 2
        result[double_diffraction, 1] = 2
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
    sequences = topology_kernels.deterministic_pad_topology_sequences(
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


@dataclass(frozen=True, slots=True)
class PathExecutionStats:
    """Execution metadata kept outside row-aligned propagation contracts."""

    launch_count: int
    visibility_rejection_count: int
    selected_edge_count: int
    candidate_count: int
    guardrail_count: int
    ad_companion_launches: int
    ad_tape_bytes: int


@dataclass(frozen=True, slots=True)
class EvaluatedPathSidecars:
    """Non-row and optional data separated from ``EvaluatedPaths``."""

    execution: PathExecutionStats
    diffraction_vector_field: torch.Tensor | None
    capacity_execution: CapacityExecutionCounts | None = None
    capacity_transaction: SolveCapacityTransaction | None = None
    compact_metadata: topology_kernels.ExactPairMetadata | None = None


def evaluated_paths_from_result(
    paths: object,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    """Normalize one canonical result directly into split typed contracts."""

    path_count = int(paths.valid.numel())
    device = paths.valid.device
    path_gain = paths.path_gain.to(dtype=torch.float32).contiguous()
    defaults: dict[str, torch.Tensor] | None = None

    def topology_defaults() -> dict[str, torch.Tensor]:
        nonlocal defaults
        if defaults is None:
            defaults = topology_kernels.deterministic_topology_default_fields(
                path_gain
            )
        return defaults

    path_field = getattr(paths, "path_field", None)
    if path_field is None:
        path_field = topology_defaults()["path_field"]
    field_xyz = getattr(paths, "field_xyz", None)
    if field_xyz is None:
        field_xyz = torch.zeros((path_count, 3), device=device, dtype=torch.complex64)
    coefficient = getattr(paths, "coefficient", path_field)
    field_direction = getattr(paths, "field_direction", None)
    if field_direction is None:
        field_direction = torch.zeros(
            (path_count, 3), device=device, dtype=torch.float32
        )
    interaction_position = getattr(paths, "interaction_position", None)
    if interaction_position is None:
        interaction_position = topology_defaults()["interaction_position"]
    interaction_normal = getattr(paths, "interaction_normal", None)
    if interaction_normal is None:
        interaction_normal = topology_defaults()["interaction_normal"]
    material_id = getattr(paths, "material_id", None)
    if material_id is None:
        material_id = topology_defaults()["material_id"]
    primitive_sequence = (
        getattr(
            paths,
            "primitive_sequence",
            torch.empty((path_count, 0), device=device, dtype=torch.int32),
        )
        .to(dtype=torch.int32)
        .contiguous()
    )

    # Preserve the established argument evaluation order exactly.
    valid_value = paths.valid.contiguous()
    tx_id_value = paths.tx_id.to(dtype=torch.int32).contiguous()
    rx_id_value = paths.rx_id.to(dtype=torch.int32).contiguous()
    depth_value = paths.depth.to(dtype=torch.int32).contiguous()
    component_id_value = paths.component_id.to(dtype=torch.int32).contiguous()
    primitive_id_value = paths.primitive_id.to(dtype=torch.int32).contiguous()
    edge_id_value = paths.edge_id.to(dtype=torch.int32).contiguous()
    path_length_value = paths.path_length_m.to(dtype=torch.float32).contiguous()
    delay_value = paths.delay_s.to(dtype=torch.float32).contiguous()
    path_field_value = path_field.to(dtype=torch.complex64).contiguous()
    field_xyz_value = field_xyz.to(dtype=torch.complex64).contiguous()
    coefficient_value = coefficient.to(dtype=torch.complex64).contiguous()
    field_direction_value = field_direction.to(dtype=torch.float32).contiguous()
    interaction_position_value = interaction_position.to(
        dtype=torch.float32
    ).contiguous()
    interaction_normal_value = interaction_normal.to(dtype=torch.float32).contiguous()
    material_id_value = material_id.to(dtype=torch.int32).contiguous()
    material_sequence_value = (
        getattr(
            paths,
            "material_sequence",
            torch.empty((path_count, 0), device=device, dtype=torch.int32),
        )
        .to(dtype=torch.int32)
        .contiguous()
    )
    interaction_type_value = _interaction_type_sequence(
        component_id=paths.component_id,
        depth=paths.depth,
        width=int(primitive_sequence.shape[1]),
    )
    interaction_positions_value = (
        getattr(
            paths,
            "interaction_positions",
            torch.empty((path_count, 0, 3), device=device, dtype=torch.float32),
        )
        .to(dtype=torch.float32)
        .contiguous()
    )
    interaction_normals_value = (
        getattr(
            paths,
            "interaction_normals",
            torch.empty((path_count, 0, 3), device=device, dtype=torch.float32),
        )
        .to(dtype=torch.float32)
        .contiguous()
    )
    launch_count = int(getattr(paths, "launch_count", 0))
    visibility_rejection_count = int(getattr(paths, "visibility_rejection_count", 0))
    selected_edge_count = int(getattr(paths, "selected_edge_count", 0))
    candidate_count = int(getattr(paths, "candidate_count", path_count))
    guardrail_count = int(getattr(paths, "guardrail_count", 0))

    topology = PathTopology(
        valid=valid_value,
        tx_id=tx_id_value,
        rx_id=rx_id_value,
        depth=depth_value,
        component_id=component_id_value,
        primitive_id=primitive_id_value,
        edge_id=edge_id_value,
        material_id=material_id_value,
        primitive_sequence=primitive_sequence,
        material_sequence=material_sequence_value,
        interaction_type=interaction_type_value,
    )
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        path_length_m=path_length_value,
        delay_s=delay_value,
        field_direction=field_direction_value,
        interaction_position=interaction_position_value,
        interaction_normal=interaction_normal_value,
        interaction_positions=interaction_positions_value,
        interaction_normals=interaction_normals_value,
    )
    fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=path_gain,
        path_field=path_field_value,
        field_xyz=field_xyz_value,
        coefficient=coefficient_value,
    )
    evaluated = EvaluatedPaths(topology=topology, geometry=geometry, fields=fields)
    sidecars = EvaluatedPathSidecars(
        execution=PathExecutionStats(
            launch_count=launch_count,
            visibility_rejection_count=visibility_rejection_count,
            selected_edge_count=selected_edge_count,
            candidate_count=candidate_count,
            guardrail_count=guardrail_count,
            ad_companion_launches=0,
            ad_tape_bytes=0,
        ),
        diffraction_vector_field=None,
    )
    return evaluated, sidecars


def evaluated_paths_from_block(
    paths: dict[str, torch.Tensor],
    *,
    max_paths: int | None,
    max_paths_scope: str,
    tx_count: int,
    rx_count: int,
    max_depth: int,
    launch_count: int,
    visibility_rejection_count: int = 0,
    selected_edge_count: int = 0,
    candidate_count: int | None = None,
    guardrail_count: int = 0,
    source_stable_ids: torch.Tensor | None = None,
    sink_stable_ids: torch.Tensor | None = None,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    """Select a canonical path block and construct split typed contracts."""

    if "primitive_sequence" not in paths and int(paths["valid"].shape[0]) == 0:
        device = paths["valid"].device
        empty_i32 = torch.empty((0, max_depth), device=device, dtype=torch.int32)
        empty_vec3 = torch.empty(
            (0, max_depth, 3), device=device, dtype=torch.float32
        )
        paths = _ensure_topology_fields(
            paths,
            primitive_sequence=empty_i32,
            material_sequence=empty_i32,
            interaction_positions=empty_vec3,
            interaction_normals=empty_vec3,
        )
    compact = topology_kernels.enumerated_canonical_compact(
        paths,
        pair_count=tx_count * rx_count,
        num_tx=tx_count,
        num_rx=rx_count,
        max_paths=max_paths,
        max_paths_scope=max_paths_scope,
        sequence_width=max_depth,
        source_stable_ids=source_stable_ids,
        sink_stable_ids=sink_stable_ids,
    )
    evaluated, sidecars = evaluated_paths_from_result(
        SimpleNamespace(
            **compact.block,
            launch_count=launch_count,
            visibility_rejection_count=visibility_rejection_count,
            selected_edge_count=selected_edge_count,
            candidate_count=int(
                candidate_count
                if candidate_count is not None
                else paths["valid"].numel()
            ),
            guardrail_count=guardrail_count,
        )
    )
    return evaluated, replace(
        sidecars,
        compact_metadata=topology_kernels.ExactPairMetadata(
            pair_index=compact.pair_index,
            pair_offsets=compact.pair_offsets,
            source_id=compact.source_id,
            sink_id=compact.sink_id,
            path_count=compact.path_count,
            count_d2h_copies=compact.count_d2h_copies,
            count_d2h_bytes=compact.count_d2h_bytes,
            count_synchronizations=compact.count_synchronizations,
        ),
    )


def _ensure_topology_fields(
    block: dict[str, torch.Tensor],
    *,
    interaction_position: torch.Tensor | None = None,
    interaction_normal: torch.Tensor | None = None,
    material_id: torch.Tensor | None = None,
    path_field: torch.Tensor | None = None,
    field_xyz: torch.Tensor | None = None,
    coefficient: torch.Tensor | None = None,
    primitive_sequence: torch.Tensor | None = None,
    material_sequence: torch.Tensor | None = None,
    interaction_positions: torch.Tensor | None = None,
    interaction_normals: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    extended = dict(block)
    defaults: dict[str, torch.Tensor] | None = None

    def topology_defaults() -> dict[str, torch.Tensor]:
        nonlocal defaults
        if defaults is None:
            defaults = topology_kernels.deterministic_topology_default_fields(
                block["path_gain"].to(dtype=torch.float32).contiguous()
            )
        return defaults

    interaction_position_value = (
        interaction_position
        if interaction_position is not None
        else block.get("interaction_position")
    )
    if interaction_position_value is None:
        interaction_position_value = topology_defaults()["interaction_position"]
    interaction_normal_value = (
        interaction_normal
        if interaction_normal is not None
        else block.get("interaction_normal")
    )
    if interaction_normal_value is None:
        interaction_normal_value = topology_defaults()["interaction_normal"]
    material_id_value = (
        material_id if material_id is not None else block.get("material_id")
    )
    if material_id_value is None:
        material_id_value = topology_defaults()["material_id"]
    path_field_value = path_field if path_field is not None else block.get("path_field")
    if path_field_value is None:
        path_field_value = topology_defaults()["path_field"]
    field_xyz_value = field_xyz if field_xyz is not None else block.get("field_xyz")
    if field_xyz_value is None:
        field_xyz_value = torch.zeros(
            (int(block["valid"].shape[0]), 3),
            device=block["valid"].device,
            dtype=torch.complex64,
        )
    coefficient_value = (
        coefficient if coefficient is not None else block.get("coefficient")
    )
    if coefficient_value is None:
        coefficient_value = path_field_value
    extended["interaction_position"] = interaction_position_value.to(
        dtype=torch.float32
    ).contiguous()
    extended["interaction_normal"] = interaction_normal_value.to(
        dtype=torch.float32
    ).contiguous()
    extended["material_id"] = material_id_value.to(dtype=torch.int32).contiguous()
    extended["path_field"] = path_field_value.to(dtype=torch.complex64).contiguous()
    extended["field_xyz"] = field_xyz_value.to(dtype=torch.complex64).contiguous()
    extended["coefficient"] = coefficient_value.to(dtype=torch.complex64).contiguous()
    if primitive_sequence is not None:
        extended["primitive_sequence"] = primitive_sequence.to(
            dtype=torch.int32
        ).contiguous()
    if material_sequence is not None:
        extended["material_sequence"] = material_sequence.to(
            dtype=torch.int32
        ).contiguous()
    if interaction_positions is not None:
        extended["interaction_positions"] = interaction_positions.to(
            dtype=torch.float32
        ).contiguous()
    if interaction_normals is not None:
        extended["interaction_normals"] = interaction_normals.to(
            dtype=torch.float32
        ).contiguous()
    return extended


__all__ = ["mc_sample_directions"]
