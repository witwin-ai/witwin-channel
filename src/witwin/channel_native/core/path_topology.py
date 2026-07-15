from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from witwin.channel_native.materials.evaluation import (
    _require_frequency_ad_constant_materials,
)
from witwin.channel_native.propagation.enumerated.reflection import (
    _discovered_group_chains,  # noqa: F401 - compatibility re-export
    _reflection_topology_order1,
    _reflection_topology_multibounce,
)
from witwin.channel_native.propagation.geometry.endpoints import (
    ReceiverLayout,  # noqa: F401 - compatibility re-export
    apply_receiver_layout,  # noqa: F401 - compatibility re-export
    receiver_positions_and_layout,
    transmitter_tensors,
)
from witwin.channel_native.propagation.fields.evaluation import (
    _evaluate_shared_fields,
    _rough_reflection_factor,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge
from witwin.channel_native.propagation.geometry.kernels import (
    primitives as geometry_primitives,
)
from witwin.channel_native.propagation.geometry.reevaluate import (
    _PLANE_GROUP_QUANTIZATION,  # noqa: F401 - compatibility re-export
    _cached_coplanar_face_groups,
    _coplanar_face_groups,  # noqa: F401 - compatibility re-export
    _geometry_participates_in_ad,  # noqa: F401 - compatibility re-export
    _opposite_vertex_ids,  # noqa: F401 - compatibility re-export
    _participates_in_ad,  # noqa: F401 - compatibility re-export
    _reflection_geometry_ad,  # noqa: F401 - compatibility re-export
    _vertices_participate_in_ad,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.models.contracts import TopologyConfig
from witwin.channel_native.propagation.topology.kernels import blocks as topology_blocks
from witwin.channel_native.propagation.topology.kernels import (
    compaction as topology_compaction,
)
from witwin.channel_native.propagation.topology.kernels import (
    construction as topology_construction,
)
from witwin.channel_native.propagation.topology.kernels import (
    primitives as topology_primitives,
)
from witwin.channel_native.propagation.topology.kernels.sampling import (
    mc_sample_directions,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.topology.concatenate import (
    _block_sequence_width,  # noqa: F401 - compatibility re-export
    _canonical_selection_order,
    _empty_path_block,
    _interaction_type_sequence,
    _pad_topology_sequences,
    _sort_order,  # noqa: F401 - compatibility re-export
    canonical_sequence_key,  # noqa: F401 - compatibility re-export
    concatenate_path_blocks,
)
from witwin.channel_native.propagation.topology.discovery.reflection import (
    _MAX_MULTIBOUNCE_FACE_SEQUENCES,  # noqa: F401 - compatibility re-export
    _MULTIBOUNCE_DISCOVERY_RAYS,  # noqa: F401 - compatibility re-export
    _MULTIBOUNCE_PAIR_CHUNK_SIZE,
    _MULTIBOUNCE_SEQUENCE_CHUNK_SIZE,  # noqa: F401 - compatibility re-export
    _ORDER1_EXHAUSTIVE_GROUP_LIMIT,  # noqa: F401 - compatibility re-export
    _face_sequence_chunks,  # noqa: F401 - compatibility re-export
    _face_sequence_count,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.runtime.autograd_contracts import (
    _frequency_participates_in_ad,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.topology.export import _ensure_topology_fields

if TYPE_CHECKING:
    from witwin.channel_native.core.scene import Scene
from witwin.channel_native.core.material_runtime import face_material_tensors
from witwin.channel_native.core.scene_tensors import (
    LIGHT_SPEED_M_PER_S as _LIGHT_SPEED_M_PER_S,
    _frequency_scalar,
)
from witwin.channel_native.core.diffraction_geometry import (
    cached_diffraction_edge_geometry as _cached_diffraction_edge_geometry,
    diffraction_edge_geometry as _diffraction_edge_geometry,
)




def _raydn_visibility_mask(
    raydn: object, start: torch.Tensor, end: torch.Tensor
) -> torch.Tensor:
    if start.shape[0] == 0:
        return torch.empty((0,), device=start.device, dtype=torch.bool)
    return geometry_bridge.raydn_visibility_forward(
        raydn.require_handle(), start.contiguous(), end.contiguous(), None
    )[0]


def _los_visibility_mask(
    raydn: object,
    tx_for_path: torch.Tensor,
    rx_for_path: torch.Tensor,
    *,
    has_structures: bool,
) -> torch.Tensor | None:
    if not has_structures or tx_for_path.shape[0] == 0:
        return None
    if not raydn.available:
        raise RuntimeError("LoS visibility requires RayDN native scene capability")
    return _raydn_visibility_mask(raydn, tx_for_path, rx_for_path)


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


def _path_components(config: TopologyConfig) -> set[str]:
    components = set(config.components)
    if config.max_depth == 0:
        # Every non-LoS component is a surface interaction that needs at least
        # one bounce. transmission is a wall penetration event and scattering is
        # a single-bounce rough-surface event, so both drop out at depth 0.
        components.discard("reflection")
        components.discard("diffraction")
        components.discard("transmission")
        components.discard("scattering")
    if int(getattr(config, "max_diffraction_order", 1)) == 0:
        components.discard("diffraction")
    return components


def _from_path_result(paths: object) -> TopologyBatch:
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
        field_xyz = torch.zeros(
            (path_count, 3), device=device, dtype=torch.complex64
        )
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
    return TopologyBatch(
        valid=paths.valid.contiguous(),
        tx_id=paths.tx_id.to(dtype=torch.int32).contiguous(),
        rx_id=paths.rx_id.to(dtype=torch.int32).contiguous(),
        depth=paths.depth.to(dtype=torch.int32).contiguous(),
        component_id=paths.component_id.to(dtype=torch.int32).contiguous(),
        primitive_id=paths.primitive_id.to(dtype=torch.int32).contiguous(),
        edge_id=paths.edge_id.to(dtype=torch.int32).contiguous(),
        path_length_m=paths.path_length_m.to(dtype=torch.float32).contiguous(),
        delay_s=paths.delay_s.to(dtype=torch.float32).contiguous(),
        path_gain=path_gain,
        path_field=path_field.to(dtype=torch.complex64).contiguous(),
        field_xyz=field_xyz.to(dtype=torch.complex64).contiguous(),
        coefficient=coefficient.to(dtype=torch.complex64).contiguous(),
        field_direction=field_direction.to(dtype=torch.float32).contiguous(),
        interaction_position=interaction_position.to(dtype=torch.float32).contiguous(),
        interaction_normal=interaction_normal.to(dtype=torch.float32).contiguous(),
        material_id=material_id.to(dtype=torch.int32).contiguous(),
        primitive_sequence=primitive_sequence,
        material_sequence=getattr(
            paths,
            "material_sequence",
            torch.empty((path_count, 0), device=device, dtype=torch.int32),
        )
        .to(dtype=torch.int32)
        .contiguous(),
        interaction_type=_interaction_type_sequence(
            component_id=paths.component_id,
            depth=paths.depth,
            width=int(primitive_sequence.shape[1]),
        ),
        interaction_positions=getattr(
            paths,
            "interaction_positions",
            torch.empty((path_count, 0, 3), device=device, dtype=torch.float32),
        )
        .to(dtype=torch.float32)
        .contiguous(),
        interaction_normals=getattr(
            paths,
            "interaction_normals",
            torch.empty((path_count, 0, 3), device=device, dtype=torch.float32),
        )
        .to(dtype=torch.float32)
        .contiguous(),
        launch_count=int(getattr(paths, "launch_count", 0)),
        visibility_rejection_count=int(getattr(paths, "visibility_rejection_count", 0)),
        selected_edge_count=int(getattr(paths, "selected_edge_count", 0)),
        candidate_count=int(getattr(paths, "candidate_count", path_count)),
        guardrail_count=int(getattr(paths, "guardrail_count", 0)),
    )


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
    return _from_path_result(
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


def _reflect_points(
    points: torch.Tensor, plane_points: torch.Tensor, normals: torch.Tensor
) -> torch.Tensor:
    return geometry_primitives.deterministic_reflect_points(
        points.contiguous(), plane_points.contiguous(), normals.contiguous()
    )


def _deterministic_diffraction_states(
    raydn: object,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    tx_index: int,
    *,
    preserve_imported_edges: bool = False,
) -> tuple[torch.Tensor, ...]:
    (
        selected,
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
    ) = (
        _diffraction_edge_geometry(raydn.edge_records())
        if preserve_imported_edges
        else _cached_diffraction_edge_geometry(raydn)
    )
    return topology_primitives.deterministic_diffraction_state_pack(
        topology_primitives.mc_selected_edge_indices(selected),
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        int(tx_index),
    )


_DIFFRACTION_PREFILTER_EDGE_FRACTIONS = (0.02, 1.0 / 3.0, 2.0 / 3.0, 0.98)


def _tx_visible_diffraction_states(
    raydn: object,
    states: tuple[torch.Tensor, ...],
    tx: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Drop edge states that are occluded from the transmitter.

    The UTD kernel checks visibility at the per-receiver stationary point, so
    a state is only culled when the transmitter cannot see the edge at any of
    several sample points along it. This shrinks the rx x state workspace and
    pair launches on city-scale scenes (mirrors the original tx_first
    pruning) while keeping states whose midpoint happens to be occluded.
    """

    state_count = int(states[0].shape[0])
    if state_count <= 0:
        return states
    edge_anchor = states[1]
    edge_dir = states[2]
    line_min = states[3]
    line_max = states[4]
    starts = tx.reshape(1, 3).expand(state_count, 3).contiguous()
    visible = torch.zeros((state_count,), device=edge_anchor.device, dtype=torch.bool)
    for fraction in _DIFFRACTION_PREFILTER_EDGE_FRACTIONS:
        t = line_min + fraction * (line_max - line_min)
        point = (edge_anchor + t.unsqueeze(1) * edge_dir).contiguous()
        visible |= geometry_bridge.raydn_visibility_forward(
            raydn.require_handle(), starts, point, None
        )[0]
    if bool(visible.all()):
        return states
    return tuple(
        tensor[visible] if tensor.shape[:1] == (state_count,) else tensor
        for tensor in states
    )


def _diffraction_topology_order1(
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
) -> tuple[dict[str, torch.Tensor], int, torch.Tensor]:
    from witwin.channel_native.propagation.fields.kernels import (
        deterministic as field_kernels,
    )

    device = tx_positions.device
    raydn = compiled.raydn
    if not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return (
            _ensure_topology_fields(
                {
                    "valid": torch.empty((0,), device=device, dtype=torch.bool),
                    "tx_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "rx_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "depth": torch.empty((0,), device=device, dtype=torch.int32),
                    "component_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "primitive_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "edge_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "path_length_m": torch.empty(
                        (0,), device=device, dtype=torch.float32
                    ),
                    "delay_s": torch.empty((0,), device=device, dtype=torch.float32),
                    "path_gain": torch.empty((0,), device=device, dtype=torch.float32),
                }
            ),
            0,
            torch.zeros(
                (int(tx_positions.shape[0]), int(rx_positions.shape[0]), 3),
                device=device,
                dtype=torch.complex64,
            ),
        )
    if not raydn.available:
        raise RuntimeError(
            "deterministic diffraction requires RayDN native scene capability"
        )

    face_eps_r, face_sigma_e, face_mu_r, material_gain, material_valid = (
        face_material_tensors(compiled, device=device)
    )
    wavelength = _LIGHT_SPEED_M_PER_S / float(frequency_hz)
    handle = raydn.require_handle()
    blocks: list[dict[str, torch.Tensor]] = []
    vector_field = torch.zeros(
        (int(tx_positions.shape[0]), int(rx_positions.shape[0]), 3),
        device=device,
        dtype=torch.complex64,
    )
    launch_count = 0
    rx_count = int(rx_positions.shape[0])
    mitsuba_metadata = scene.metadata.get("mitsuba", {})
    # Channel's merge_shapes import keeps the selected boundary-edge table
    # intact.  The synthetic-scene path instead merges coincident structure
    # boundaries into one physical wedge (the single-wedge test contract).
    preserve_imported_edges = isinstance(mitsuba_metadata, dict) and bool(
        mitsuba_metadata.get("merge_shapes", False)
    )
    for tx_index, tx in enumerate(tx_positions):
        states = _deterministic_diffraction_states(
            raydn,
            tx,
            tx_power,
            tx_index,
            preserve_imported_edges=preserve_imported_edges,
        )
        states = _tx_visible_diffraction_states(raydn, states, tx)
        state_count = int(states[0].shape[0])
        if state_count <= 0:
            continue
        # Chunk receivers so the rx x edge-state workspace stays bounded on
        # city-scale scenes (audit P-2); the reflection paths already chunk.
        rx_chunk_size = max(1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // state_count)
        for rx_start in range(0, rx_count, rx_chunk_size):
            rx_end = min(rx_start + rx_chunk_size, rx_count)
            rx_chunk = rx_positions[rx_start:rx_end].contiguous()
            capacity = int(rx_chunk.shape[0]) * state_count
            out = geometry_bridge.raydn_diffraction_paths_order1_forward(
                handle,
                tx.reshape(1, 3).contiguous(),
                rx_chunk,
                None,
                *states,
                face_eps_r,
                face_sigma_e,
                face_mu_r,
                material_gain,
                material_valid,
                state_count,
                capacity,
                float(wavelength),
            )
            launch_count += 1
            compacted = topology_compaction.deterministic_diffraction_order1_compact(
                valid=out[1],
                rx_id=out[3],
                depth=out[4],
                edge_id=out[5],
                delay_s=out[8],
                x_re=out[9],
                x_im=out[10],
                y_re=out[11],
                y_im=out[12],
                z_re=out[13],
                z_im=out[14],
                interaction_position=out[15],
            )
            if int(compacted["rx_id"].numel()) == 0:
                continue
            if rx_start > 0:
                compacted["rx_id"] = compacted["rx_id"] + rx_start
            # The path kernel has already evaluated the full UTD vector field.
            # Keep its xyz components until after summing all edges; reducing
            # each path to an equivalent scalar first loses vector coherence.
            receiver_index = compacted["rx_id"].to(dtype=torch.long)
            for axis, real_index in enumerate(("x_re", "y_re", "z_re")):
                imag_index = real_index.replace("_re", "_im")
                vector_field[tx_index, :, axis].index_add_(
                    0,
                    receiver_index,
                    torch.complex(compacted[real_index], compacted[imag_index]),
                )
            field_result = field_kernels.deterministic_diffraction_vector_field(
                x_re=compacted["x_re"],
                x_im=compacted["x_im"],
                y_re=compacted["y_re"],
                y_im=compacted["y_im"],
                z_re=compacted["z_re"],
                z_im=compacted["z_im"],
            )
            field_power = field_result["path_gain"]
            path_field = field_kernels.deterministic_pack_complex(
                field_result["field_real"], field_result["field_imag"]
            )
            field_xyz = torch.stack(
                (
                    torch.complex(compacted["x_re"], compacted["x_im"]),
                    torch.complex(compacted["y_re"], compacted["y_im"]),
                    torch.complex(compacted["z_re"], compacted["z_im"]),
                ),
                dim=1,
            ).contiguous()
            delay = compacted["delay_s"]
            path_length = field_kernels.deterministic_delay_to_path_length(delay)
            empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)
            blocks.append(
                _ensure_topology_fields(
                    topology_construction.deterministic_topology_base_fields(
                        rx_id=compacted["rx_id"],
                        path_length_m=path_length,
                        delay_s=delay,
                        path_gain=field_power,
                        tx_index=tx_index,
                        component_id=2,
                        depth_source=compacted["depth"],
                        depth_value=0,
                        primitive_source=empty_i32,
                        primitive_value=-1,
                        edge_source=compacted["edge_id"],
                        edge_value=-1,
                    ),
                    interaction_position=compacted["interaction_position"],
                    path_field=path_field,
                    field_xyz=field_xyz,
                )
            )
    return (
        _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)),
        launch_count,
        vector_field,
    )


_COUPLED_CANDIDATE_CHUNK_SIZE = 65_536
_MAX_COUPLED_CANDIDATES = 1_000_000


def _coupled_reflection_diffraction_topology_order2(
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    candidate_limit: int,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Construct bounded 1R+1D and reciprocal 1D+1R geometry.

    This phase deliberately exports no physical coefficient. Phase 3 applies
    the shared complex/Jones transport to these canonical event sequences.
    """

    device = tx_positions.device
    raydn = compiled.raydn
    if not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return _ensure_topology_fields(_empty_path_block(device)), 0, 0
    if not raydn.available:
        raise RuntimeError("coupled topology requires RayDN native scene capability")

    records = raydn.edge_records()
    faces = records.faces.contiguous()
    if int(faces.shape[0]) == 0:
        return _ensure_topology_fields(_empty_path_block(device)), 0, 0
    vertices = records.vertices.contiguous()
    normals = geometry_primitives.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    tri_a = topology_construction.deterministic_face_anchor_points(vertices, faces)
    groups = _cached_coplanar_face_groups(
        raydn,
        tri_a,
        normals,
        compiled.geometry.face_surface_id.to(
            device=device, dtype=torch.long
        ).contiguous(),
    )
    representative_faces = (
        groups["representative_faces"].to(dtype=torch.int32).contiguous()
    )
    group_count = int(representative_faces.shape[0])
    preserve_imported_edges = bool(
        isinstance(scene.metadata.get("mitsuba", {}), dict)
        and scene.metadata.get("mitsuba", {}).get("merge_shapes", False)
    )
    (
        selected,
        edge_pos,
        edge_dir,
        _edge_length,
        edge_t_min,
        edge_t_max,
        _n0,
        _n1,
        _face0,
        _face1,
        _exterior_angle,
    ) = (
        _diffraction_edge_geometry(records)
        if preserve_imported_edges
        else _cached_diffraction_edge_geometry(raydn)
    )
    selected_edges = topology_primitives.mc_selected_edge_indices(selected)
    edge_count = int(selected_edges.shape[0])
    candidates_per_pair = group_count * edge_count
    if candidates_per_pair == 0:
        return _ensure_topology_fields(_empty_path_block(device)), 0, 0
    theoretical_candidate_count = (
        int(tx_positions.shape[0])
        * int(rx_positions.shape[0])
        * candidates_per_pair
        * 2
    )
    effective_candidate_limit = min(candidate_limit, _MAX_COUPLED_CANDIDATES)
    if theoretical_candidate_count > effective_candidate_limit:
        raise RuntimeError(
            "coupled reflection-diffraction topology requires "
            f"{theoretical_candidate_count} candidates, exceeding "
            f"coupled_candidate_limit={effective_candidate_limit}"
        )

    face_material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int32
    ).contiguous()
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    candidate_count = 0
    rx_count = int(rx_positions.shape[0])
    base_candidate_count = theoretical_candidate_count // 2
    surface_group_id = groups["surface_group_id"].to(dtype=torch.int32).contiguous()
    surface_group_size = groups["surface_group_size"].to(dtype=torch.int32).contiguous()
    surface_group_members = (
        groups["surface_group_members"].to(dtype=torch.int32).contiguous()
    )
    for start in range(0, base_candidate_count, _COUPLED_CANDIDATE_CHUNK_SIZE):
        end = min(start + _COUPLED_CANDIDATE_CHUNK_SIZE, base_candidate_count)
        linear = torch.arange(start, end, device=device, dtype=torch.int64)
        pair_slot = torch.div(linear, candidates_per_pair, rounding_mode="floor")
        local_slot = torch.remainder(linear, candidates_per_pair)
        tx_slot = torch.div(pair_slot, rx_count, rounding_mode="floor")
        rx_slot = torch.remainder(pair_slot, rx_count)
        face_slot = torch.div(local_slot, edge_count, rounding_mode="floor")
        edge_slot = torch.remainder(local_slot, edge_count)
        face_id = representative_faces[face_slot]
        edge_id = selected_edges[edge_slot]
        edge_index = edge_id.to(dtype=torch.int64)
        count = int(linear.shape[0])
        common_args = (
            raydn.require_handle(),
            tx_positions[tx_slot].contiguous(),
            rx_positions[rx_slot].contiguous(),
            face_id,
            tri_a[face_id.to(dtype=torch.int64)].contiguous(),
            normals[face_id.to(dtype=torch.int64)].contiguous(),
            edge_id,
            edge_pos[edge_index].contiguous(),
            edge_dir[edge_index].contiguous(),
            edge_t_min[edge_index].contiguous(),
            edge_t_max[edge_index].contiguous(),
            surface_group_id,
            surface_group_size,
            surface_group_members,
        )
        for reverse, component_id in ((False, 3), (True, 4)):
            exported = geometry_bridge.raydn_coupled_rd_geometry_forward(
                *common_args, reverse
            )
            launch_count += 1
            candidate_count += count
            kept = torch.nonzero(exported["valid"], as_tuple=False).reshape(-1)
            kept_count = int(kept.shape[0])
            if kept_count == 0:
                continue
            interaction_type = exported["interaction_type_sequence"][kept]
            primitive_sequence = exported["primitive_sequence"][kept]
            edge_sequence = exported["edge_sequence"][kept]
            object_sequence = (
                torch.where(interaction_type == 2, edge_sequence, primitive_sequence)
                .to(dtype=torch.int32)
                .contiguous()
            )
            resolved_face = exported["face_id"][kept]
            resolved_edge = exported["edge_id"][kept]
            reflection_material = face_material_id[resolved_face.to(dtype=torch.int64)]
            material_sequence = (
                torch.where(
                    interaction_type == 1,
                    reflection_material.reshape(-1, 1),
                    torch.full_like(interaction_type, -1),
                )
                .to(dtype=torch.int32)
                .contiguous()
            )
            nan = torch.full(
                (kept_count,), float("nan"), device=device, dtype=torch.float32
            )
            blocks.append(
                _ensure_topology_fields(
                    {
                        "valid": torch.ones(
                            (kept_count,), device=device, dtype=torch.bool
                        ),
                        "tx_id": tx_slot[kept].to(dtype=torch.int32).contiguous(),
                        "rx_id": rx_slot[kept].to(dtype=torch.int32).contiguous(),
                        "depth": torch.full(
                            (kept_count,), 2, device=device, dtype=torch.int32
                        ),
                        "component_id": torch.full(
                            (kept_count,),
                            component_id,
                            device=device,
                            dtype=torch.int32,
                        ),
                        "primitive_id": resolved_face.to(dtype=torch.int32),
                        "edge_id": resolved_edge.to(dtype=torch.int32),
                        "path_length_m": exported["path_length_m"][kept],
                        "delay_s": exported["delay_s"][kept],
                        "path_gain": nan,
                    },
                    interaction_position=exported["interaction_positions"][kept, 0],
                    interaction_normal=exported["interaction_normals"][kept, 0],
                    material_id=reflection_material,
                    path_field=torch.complex(nan, nan),
                    primitive_sequence=object_sequence,
                    material_sequence=material_sequence,
                    interaction_positions=exported["interaction_positions"][kept],
                    interaction_normals=exported["interaction_normals"][kept],
                )
            )
    return (
        _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)),
        launch_count,
        candidate_count,
    )


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


def export_topology(
    scene: Scene,
    config: TopologyConfig,
    *,
    frequency_value: float | None = None,
) -> TopologyBatch:
    device = torch.device("cuda")
    tx_positions, tx_power = transmitter_tensors(scene, device=device)
    rx_positions, _ = receiver_positions_and_layout(scene, device=device)
    compiled = scene.compile()
    # One host read of a tensor frequency for the whole export: discovery and
    # the field seam below share this detached scalar (audit M3). Callers
    # that already read it (the solver seams) pass it in.
    frequency_hz = (
        _frequency_scalar(scene) if frequency_value is None else float(frequency_value)
    )
    ad_mode = str(getattr(config, "ad_mode", "none"))
    if ad_mode != "none":
        _require_frequency_ad_constant_materials(scene, compiled, ad_mode=ad_mode)
    components = _path_components(config)
    coupled_paths = bool(getattr(config, "coupled_paths", False))
    sequence_width = max(int(config.max_depth), 2 if coupled_paths else 0)
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    visibility_rejection_count = 0
    candidate_count = 0
    guardrail_count = 0
    diffraction_vector_field = None

    if "los" in components:
        exported = topology_blocks.path_los_export(
            tx_positions,
            tx_power,
            rx_positions,
            frequency_hz=frequency_hz,
        )
        launch_count += 1
        tx_id = exported["tx_id"]
        rx_id = exported["rx_id"]
        visible = None
        if bool(scene.structures) and int(tx_id.numel()) > 0:
            visibility_inputs = topology_blocks.path_los_visibility_inputs(
                tx_positions,
                rx_positions,
                tx_id.to(dtype=torch.int32).contiguous(),
                rx_id.to(dtype=torch.int32).contiguous(),
            )
            visible = geometry_bridge.raydn_visibility_forward(
                compiled.raydn.require_handle(),
                visibility_inputs["start"],
                visibility_inputs["end"],
                visibility_inputs["active"],
            )[0]
            launch_count += 1
        candidate_count += int(tx_id.numel())
        los_block = _ensure_topology_fields(
            topology_construction.deterministic_los_topology_block(
                tx_id.to(dtype=torch.int32).contiguous(),
                rx_id.to(dtype=torch.int32).contiguous(),
                exported["path_length_m"].to(dtype=torch.float32).contiguous(),
                exported["delay_s"].to(dtype=torch.float32).contiguous(),
                exported["path_gain"].to(dtype=torch.float32).contiguous(),
                visible,
                frequency_hz=frequency_hz,
                sequence_width=sequence_width,
            )
        )
        if visible is not None:
            visibility_rejection_count += int(tx_id.numel()) - int(
                los_block["valid"].numel()
            )
        blocks.append(los_block)
    if "reflection" in components and config.max_depth >= 1:
        block, reflection_launches = _reflection_topology_order1(
            scene,
            compiled,
            tx_positions,
            tx_power,
            rx_positions,
            frequency_hz=frequency_hz,
        )
        launch_count += reflection_launches
        candidate_count += int(block["valid"].numel())
        blocks.append(block)
    if "reflection" in components and config.max_depth >= 2:
        block, reflection_launches, reflection_candidates = (
            _reflection_topology_multibounce(
                scene,
                compiled,
                tx_positions,
                tx_power,
                rx_positions,
                frequency_hz=frequency_hz,
                min_depth=2,
                max_depth=int(config.max_depth),
                max_paths=config.max_paths,
            )
        )
        launch_count += reflection_launches
        candidate_count += int(reflection_candidates)
        blocks.append(block)
    if "diffraction" in components and config.max_depth >= 1:
        block, diffraction_launches, diffraction_vector_field = (
            _diffraction_topology_order1(
                scene,
                compiled,
                tx_positions,
                tx_power,
                rx_positions,
                frequency_hz=frequency_hz,
            )
        )
        launch_count += diffraction_launches
        candidate_count += int(block["valid"].numel())
        blocks.append(block)
    if "transmission" in components and config.max_depth >= 1:
        block, transmission_launches, transmission_candidates, transmission_guardrails = (
            _transmission_topology(
                scene,
                compiled,
                tx_positions,
                rx_positions,
                max_depth=int(config.max_depth),
            )
        )
        launch_count += transmission_launches
        candidate_count += transmission_candidates
        guardrail_count += transmission_guardrails
        blocks.append(block)
    if (
        coupled_paths
        and config.max_depth >= 2
        and {"reflection", "diffraction"}.issubset(components)
    ):
        block, coupled_launches, coupled_candidates = (
            _coupled_reflection_diffraction_topology_order2(
                scene,
                compiled,
                tx_positions,
                rx_positions,
                candidate_limit=int(
                    getattr(config, "coupled_candidate_limit", 1_000_000)
                ),
            )
        )
        launch_count += coupled_launches
        candidate_count += coupled_candidates
        blocks.append(block)
    if len(blocks) == 1 and components == {"los"} and config.max_paths is None:
        result = _from_path_result(
            SimpleNamespace(
                **blocks[0],
                launch_count=launch_count,
                visibility_rejection_count=visibility_rejection_count,
                selected_edge_count=0,
                candidate_count=candidate_count,
                guardrail_count=guardrail_count,
            )
        )
        return _evaluate_shared_fields(
            scene,
            compiled,
            result,
            tx_positions,
            tx_power,
            rx_positions,
            components=components,
            ad_mode=ad_mode,
            frequency_value=frequency_hz,
        )
    padded_blocks = [
        block
        if "primitive_sequence" in block
        and int(block["primitive_sequence"].shape[1]) == sequence_width
        else _pad_topology_sequences(block, width=sequence_width)
        for block in blocks
    ]
    paths = concatenate_path_blocks(padded_blocks, device=device)
    selected_edge_count = topology_primitives.deterministic_selected_edge_count(
        paths["edge_id"]
    )
    result = _from_path_block(
        paths,
        max_paths=config.max_paths,
        max_paths_scope=str(getattr(config, "max_paths_scope", "global")),
        tx_count=len(scene.transmitters),
        max_depth=config.max_depth,
        launch_count=launch_count,
        visibility_rejection_count=visibility_rejection_count,
        selected_edge_count=selected_edge_count,
        candidate_count=candidate_count,
        guardrail_count=guardrail_count,
    )
    if diffraction_vector_field is not None:
        result = replace(result, diffraction_vector_field=diffraction_vector_field)
    return _evaluate_shared_fields(
        scene,
        compiled,
        result,
        tx_positions,
        tx_power,
        rx_positions,
        components=components,
        ad_mode=ad_mode,
        frequency_value=frequency_hz,
    )
