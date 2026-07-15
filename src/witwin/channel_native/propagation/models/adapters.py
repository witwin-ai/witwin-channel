"""Zero-copy adapters for legacy propagation row tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .evaluated import EvaluatedPaths
from .fields import PathFields
from .geometry import PathGeometry
from .topology import PathTopology


class _TopologyBatchSource(Protocol):
    """Structural source contract for the legacy mixed row table."""

    valid: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    depth: torch.Tensor
    component_id: torch.Tensor
    primitive_id: torch.Tensor
    edge_id: torch.Tensor
    material_id: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    interaction_type: torch.Tensor
    path_length_m: torch.Tensor
    delay_s: torch.Tensor
    field_direction: torch.Tensor
    interaction_position: torch.Tensor
    interaction_normal: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    path_gain: torch.Tensor
    path_field: torch.Tensor
    field_xyz: torch.Tensor
    coefficient: torch.Tensor
    launch_count: int
    visibility_rejection_count: int
    selected_edge_count: int
    candidate_count: int
    guardrail_count: int
    ad_companion_launches: int
    ad_tape_bytes: int
    diffraction_vector_field: torch.Tensor | None


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
class TopologyBatchSidecars:
    """Non-row and optional data separated from ``EvaluatedPaths``."""

    execution: PathExecutionStats
    diffraction_vector_field: torch.Tensor | None


def evaluated_paths_from_topology_batch(
    source: _TopologyBatchSource,
) -> tuple[EvaluatedPaths, TopologyBatchSidecars]:
    """Expose a legacy mixed row table through the split propagation contracts."""

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
    sidecars = TopologyBatchSidecars(
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
