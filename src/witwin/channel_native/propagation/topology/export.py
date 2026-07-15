from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel_native.propagation.models.contracts import EvaluatedRowsSource
from witwin.channel_native.propagation.models.evaluated import EvaluatedPaths
from witwin.channel_native.propagation.models.fields import PathFields
from witwin.channel_native.propagation.models.geometry import PathGeometry
from witwin.channel_native.propagation.models.topology import PathTopology
from witwin.channel_native.propagation.topology.kernels import (
    construction as topology_construction,
)


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


def export_evaluated_rows(
    source: EvaluatedRowsSource,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    """Export canonical rows as split contracts without transforming tensors.

    ``source`` is already selected and canonically ordered. This boundary only
    names its existing tensor objects; it must not gather, sort, clone, make
    tensors contiguous, or otherwise materialize row data.
    """

    topology = PathTopology(
        valid=source.valid,
        tx_id=source.tx_id,
        rx_id=source.rx_id,
        depth=source.depth,
        component_id=source.component_id,
        primitive_id=source.primitive_id,
        edge_id=source.edge_id,
        material_id=source.material_id,
        primitive_sequence=source.primitive_sequence,
        material_sequence=source.material_sequence,
        interaction_type=source.interaction_type,
    )
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        path_length_m=source.path_length_m,
        delay_s=source.delay_s,
        field_direction=source.field_direction,
        interaction_position=source.interaction_position,
        interaction_normal=source.interaction_normal,
        interaction_positions=source.interaction_positions,
        interaction_normals=source.interaction_normals,
    )
    fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=source.path_gain,
        path_field=source.path_field,
        field_xyz=source.field_xyz,
        coefficient=source.coefficient,
    )
    evaluated = EvaluatedPaths(topology=topology, geometry=geometry, fields=fields)
    sidecars = EvaluatedPathSidecars(
        execution=PathExecutionStats(
            launch_count=source.launch_count,
            visibility_rejection_count=source.visibility_rejection_count,
            selected_edge_count=source.selected_edge_count,
            candidate_count=source.candidate_count,
            guardrail_count=source.guardrail_count,
            ad_companion_launches=source.ad_companion_launches,
            ad_tape_bytes=source.ad_tape_bytes,
        ),
        diffraction_vector_field=source.diffraction_vector_field,
    )
    return evaluated, sidecars


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
    "export_evaluated_rows",
]
