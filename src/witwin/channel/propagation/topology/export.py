from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from witwin.channel.propagation.models.capacity import CapacityExecutionCounts
from witwin.channel.propagation.models.evaluated import EvaluatedPaths
from witwin.channel.propagation.models.fields import PathFields
from witwin.channel.propagation.models.geometry import PathGeometry
from witwin.channel.propagation.models.topology import PathTopology
from witwin.channel.propagation.topology.concatenate import (
    _canonical_selection_order,
    _interaction_type_sequence,
)
from witwin.channel.propagation.topology.kernels import (
    construction as topology_construction,
)
from witwin.channel.propagation.topology.kernels import blocks as topology_blocks
from witwin.channel.runtime.capacity import SolveCapacityTransaction


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
            defaults = topology_construction.deterministic_topology_default_fields(
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
    max_depth: int,
    launch_count: int,
    visibility_rejection_count: int = 0,
    selected_edge_count: int = 0,
    candidate_count: int | None = None,
    guardrail_count: int = 0,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    """Select a canonical path block and construct split typed contracts."""

    order = _canonical_selection_order(
        paths,
        tx_count=tx_count,
        max_depth=max_depth,
        max_paths=max_paths,
        max_paths_scope=max_paths_scope,
    )
    selected = topology_blocks.deterministic_gather_topology_block(
        paths,
        order,
        max_count=-1,
        sequence_width=max_depth,
    )
    return evaluated_paths_from_result(
        SimpleNamespace(
            **selected,
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
            defaults = topology_construction.deterministic_topology_default_fields(
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


__all__ = [
    "EvaluatedPathSidecars",
    "PathExecutionStats",
    "evaluated_paths_from_block",
    "evaluated_paths_from_result",
]
