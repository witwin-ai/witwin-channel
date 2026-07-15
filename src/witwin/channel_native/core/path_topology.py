from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from witwin.channel_native.materials.evaluation import (
    _require_frequency_ad_constant_materials,
)
from witwin.channel_native.propagation.enumerated.coupled import (
    _COUPLED_CANDIDATE_CHUNK_SIZE,  # noqa: F401 - compatibility re-export
    _MAX_COUPLED_CANDIDATES,  # noqa: F401 - compatibility re-export
    _coupled_reflection_diffraction_topology_order2,
)
from witwin.channel_native.propagation.enumerated.diffraction import (
    _DIFFRACTION_PREFILTER_EDGE_FRACTIONS,  # noqa: F401 - compatibility re-export
    _deterministic_diffraction_states,  # noqa: F401 - compatibility re-export
    _diffraction_topology_order1,
    _tx_visible_diffraction_states,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.enumerated.reflection import (
    _discovered_group_chains,  # noqa: F401 - compatibility re-export
    _reflection_topology_order1,
    _reflection_topology_multibounce,
)
from witwin.channel_native.propagation.enumerated.transmission import (
    _transmission_topology,
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
    compaction as topology_compaction,  # noqa: F401 - compatibility re-export
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
    _interaction_type_sequence,
    _pad_topology_sequences,
    _sort_order,  # noqa: F401 - compatibility re-export
    canonical_sequence_key,  # noqa: F401 - compatibility re-export
    concatenate_path_blocks,
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
from witwin.channel_native.runtime.autograd_contracts import (
    _frequency_participates_in_ad,  # noqa: F401 - compatibility re-export
)
from witwin.channel_native.propagation.topology.export import _ensure_topology_fields

if TYPE_CHECKING:
    from witwin.channel_native.core.scene import Scene
from witwin.channel_native.core.material_runtime import (  # noqa: F401
    face_material_tensors,
)
from witwin.channel_native.core.scene_tensors import (
    _frequency_scalar,
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
