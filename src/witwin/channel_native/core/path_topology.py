from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from witwin.channel_native.materials.evaluation import (
    _require_frequency_ad_constant_materials,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.enumerated.coupled import (
    _COUPLED_CANDIDATE_CHUNK_SIZE,  # noqa: F401 - compatibility re-export
    _MAX_COUPLED_CANDIDATES,  # noqa: F401 - compatibility re-export
    _coupled_reflection_diffraction_topology_order2,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.enumerated.diffraction import (
    _DIFFRACTION_PREFILTER_EDGE_FRACTIONS,  # noqa: F401 - compatibility re-export
    _deterministic_diffraction_states,  # noqa: F401 - compatibility re-export
    _diffraction_topology_order1,  # noqa: F401 - compatibility re-export
    _tx_visible_diffraction_states,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.enumerated.engine import (
    _path_components,  # noqa: F401 - compatibility re-export
    evaluate_enumerated_paths,
)
from witwin.channel_native.propagation.enumerated.los import (  # noqa: F401 - compatibility re-export
    _los_topology,
)
from witwin.channel_native.propagation.enumerated.reflection import (
    _discovered_group_chains,  # noqa: F401 - compatibility re-export
    _reflection_topology_multibounce,  # noqa: F401 - compatibility re-export
    _reflection_topology_order1,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.enumerated.transmission import (
    _transmission_topology,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.fields.evaluation import (
    _evaluate_shared_fields,  # noqa: F401 - compatibility re-export
    _rough_reflection_factor,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.geometry.endpoints import (
    ReceiverLayout,  # noqa: F401 - compatibility re-export
    apply_receiver_layout,  # noqa: F401 - compatibility re-export
    receiver_positions_and_layout,  # noqa: F401 - compatibility re-export
    transmitter_tensors,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.geometry.reevaluate import (
    _PLANE_GROUP_QUANTIZATION,  # noqa: F401 - compatibility re-export
    _cached_coplanar_face_groups,  # noqa: F401 - compatibility re-export
    _coplanar_face_groups,  # noqa: F401 - compatibility re-export
    _geometry_participates_in_ad,  # noqa: F401 - compatibility re-export
    _opposite_vertex_ids,  # noqa: F401 - compatibility re-export
    _participates_in_ad,  # noqa: F401 - compatibility re-export
    _reflection_geometry_ad,  # noqa: F401 - compatibility re-export
    _reflect_points,  # noqa: F401 - compatibility re-export
    _vertices_participate_in_ad,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.geometry.visibility import (
    _los_visibility_mask,  # noqa: F401 - compatibility re-export
    _raydn_visibility_mask,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.models.contracts import TopologyConfig
from witwin.channel_native.propagation.models.evaluated import EvaluatedPaths
from witwin.channel_native.propagation.topology.concatenate import (
    _block_sequence_width,  # noqa: F401 - compatibility re-export
    _canonical_selection_order,  # noqa: F401 - compatibility re-export
    _empty_path_block,  # noqa: F401 - compatibility re-export
    _interaction_type_sequence,  # noqa: F401 - compatibility re-export
    _pad_topology_sequences,  # noqa: F401 - compatibility re-export
    _sort_order,  # noqa: F401 - compatibility re-export
    canonical_sequence_key,  # noqa: F401 - compatibility re-export
    concatenate_path_blocks,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.topology.discovery.reflection import (
    _MAX_MULTIBOUNCE_FACE_SEQUENCES,  # noqa: F401 - compatibility re-export
    _MULTIBOUNCE_DISCOVERY_RAYS,  # noqa: F401 - compatibility re-export
    _MULTIBOUNCE_PAIR_CHUNK_SIZE,  # noqa: F401 - compatibility re-export
    _MULTIBOUNCE_SEQUENCE_CHUNK_SIZE,  # noqa: F401 - compatibility re-export
    _ORDER1_EXHAUSTIVE_GROUP_LIMIT,  # noqa: F401 - compatibility re-export
    _face_sequence_chunks,  # noqa: F401 - compatibility re-export
    _face_sequence_count,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.topology.export import (
    EvaluatedPathSidecars,
    _ensure_topology_fields,  # noqa: F401 - compatibility re-export
    evaluated_paths_from_block,
    evaluated_paths_from_result,
)
from witwin.channel_native.propagation.topology.kernels import (
    compaction as topology_compaction,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.topology.kernels.sampling import (
    mc_sample_directions,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.runtime.autograd_contracts import (
    _frequency_participates_in_ad,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.materials.encoding import (  # noqa: F401
    face_material_tensors,
)
from witwin.channel_native.scene.tensors import (
    _frequency_scalar,  # noqa: F401 - compatibility re-export
)

if TYPE_CHECKING:
    from witwin.channel_native.core.scene import Scene


@dataclass(frozen=True, slots=True)
class TopologyBatch:
    valid: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    depth: torch.Tensor
    component_id: torch.Tensor
    primitive_id: torch.Tensor
    edge_id: torch.Tensor
    path_length_m: torch.Tensor
    delay_s: torch.Tensor
    path_gain: torch.Tensor
    path_field: torch.Tensor
    field_xyz: torch.Tensor
    coefficient: torch.Tensor
    field_direction: torch.Tensor
    interaction_position: torch.Tensor
    interaction_normal: torch.Tensor
    material_id: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    interaction_type: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    launch_count: int = 0
    visibility_rejection_count: int = 0
    selected_edge_count: int = 0
    candidate_count: int = 0
    guardrail_count: int = 0
    diffraction_vector_field: torch.Tensor | None = None
    # Plan 07 AD-4 metadata: companion launches one reverse (vjp) or
    # forward-dual (jvp) pass performs for the differentiable Functions this
    # export registered, and the bytes their reverse passes retain via
    # save_for_backward (zero under ad_mode="none").
    ad_companion_launches: int = 0
    ad_tape_bytes: int = 0


def _topology_batch_from_evaluated(
    evaluated: EvaluatedPaths,
    sidecars: EvaluatedPathSidecars,
) -> TopologyBatch:
    """Pack typed contracts into the legacy mixed table without tensor work."""

    topology = evaluated.topology
    geometry = evaluated.geometry
    fields = evaluated.fields
    execution = sidecars.execution
    return TopologyBatch(
        valid=topology.valid,
        tx_id=topology.tx_id,
        rx_id=topology.rx_id,
        depth=topology.depth,
        component_id=topology.component_id,
        primitive_id=topology.primitive_id,
        edge_id=topology.edge_id,
        path_length_m=geometry.path_length_m,
        delay_s=geometry.delay_s,
        path_gain=fields.path_gain,
        path_field=fields.path_field,
        field_xyz=fields.field_xyz,
        coefficient=fields.coefficient,
        field_direction=geometry.field_direction,
        interaction_position=geometry.interaction_position,
        interaction_normal=geometry.interaction_normal,
        material_id=topology.material_id,
        primitive_sequence=topology.primitive_sequence,
        material_sequence=topology.material_sequence,
        interaction_type=topology.interaction_type,
        interaction_positions=geometry.interaction_positions,
        interaction_normals=geometry.interaction_normals,
        launch_count=execution.launch_count,
        visibility_rejection_count=execution.visibility_rejection_count,
        selected_edge_count=execution.selected_edge_count,
        candidate_count=execution.candidate_count,
        guardrail_count=execution.guardrail_count,
        diffraction_vector_field=sidecars.diffraction_vector_field,
        ad_companion_launches=execution.ad_companion_launches,
        ad_tape_bytes=execution.ad_tape_bytes,
    )


def _from_path_result(paths: object) -> TopologyBatch:
    evaluated, sidecars = evaluated_paths_from_result(paths)
    return _topology_batch_from_evaluated(evaluated, sidecars)


def _from_path_block(
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
) -> TopologyBatch:
    evaluated, sidecars = evaluated_paths_from_block(
        paths,
        max_paths=max_paths,
        max_paths_scope=max_paths_scope,
        tx_count=tx_count,
        max_depth=max_depth,
        launch_count=launch_count,
        visibility_rejection_count=visibility_rejection_count,
        selected_edge_count=selected_edge_count,
        candidate_count=candidate_count,
        guardrail_count=guardrail_count,
    )
    return _topology_batch_from_evaluated(evaluated, sidecars)


def export_topology(
    scene: Scene,
    config: TopologyConfig,
    *,
    frequency_value: float | None = None,
) -> TopologyBatch:
    evaluated, sidecars = evaluate_enumerated_paths(
        scene,
        config,
        frequency_value=frequency_value,
    )
    return _topology_batch_from_evaluated(evaluated, sidecars)
