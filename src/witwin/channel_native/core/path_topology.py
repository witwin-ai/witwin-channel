from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Protocol

import torch

from witwin.channel_native import ReceiverGrid, Scene
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.core.field_state import (
    receiver_polarizations,
    transmitter_polarizations,
)
from witwin.channel_native.core.material_runtime import (
    face_material_field_bundle,
    face_material_tensors,
)
from witwin.channel_native.montecarlo.basic.backend import (
    _LIGHT_SPEED_M_PER_S,
    receiver_positions as _native_receiver_positions,
    transmitter_positions as _native_transmitter_positions,
)
from witwin.channel_native.montecarlo.basic.raydn_components import (
    _cached_diffraction_edge_geometry,
    _diffraction_edge_geometry,
)


class TopologyConfig(Protocol):
    max_depth: int
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str]
    max_paths: int | None
    max_paths_scope: str


_MAX_MULTIBOUNCE_FACE_SEQUENCES = 100_000
_MULTIBOUNCE_SEQUENCE_CHUNK_SIZE = 65_536
_MULTIBOUNCE_PAIR_CHUNK_SIZE = 4_194_304
_ORDER1_EXHAUSTIVE_GROUP_LIMIT = 4096
_PLANE_GROUP_QUANTIZATION = 1.0e-4


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


def _raydn_visibility_mask(
    raydn: object, start: torch.Tensor, end: torch.Tensor
) -> torch.Tensor:
    if start.shape[0] == 0:
        return torch.empty((0,), device=start.device, dtype=torch.bool)
    return ops.raydn_visibility_forward(
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
    return ops.deterministic_concat_topology_blocks(
        nonempty, sequence_width=sequence_width
    )


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


@dataclass(frozen=True, slots=True)
class ReceiverLayout:
    """Maps flat receiver ids to the deterministic public result layout."""

    kind: str
    receiver_count: int
    grid_shape: tuple[int, int] | None = None

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        if self.kind == "grid":
            if self.grid_shape is None:
                raise ValueError("grid layout requires grid_shape")
            rows, cols = self.grid_shape
            return (
                values.reshape(values.shape[0], rows, cols).transpose(1, 2).contiguous()
            )
        if self.kind == "point":
            return values.contiguous()
        raise ValueError(f"receiver layout kind is not accepted: {self.kind}")


def transmitter_tensors(
    scene: Scene, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return _native_transmitter_positions(scene, device=device)


def receiver_positions_and_layout(
    scene: Scene, *, device: torch.device
) -> tuple[torch.Tensor, ReceiverLayout]:
    if not scene.receivers:
        return torch.empty((0, 3), device=device, dtype=torch.float32), ReceiverLayout(
            "point", 0
        )

    reference, _power = transmitter_tensors(scene, device=device)
    positions = _native_receiver_positions(scene, device=device, reference=reference)
    if len(scene.receivers) == 1 and isinstance(scene.receivers[0], ReceiverGrid):
        grid = scene.receivers[0]
        return positions, ReceiverLayout("grid", int(positions.shape[0]), grid.shape)

    return positions, ReceiverLayout("point", int(positions.shape[0]))


def _coplanar_face_groups(
    tri_a: torch.Tensor,
    normals: torch.Tensor,
    surface_ids: torch.Tensor,
    *,
    quantization: float = _PLANE_GROUP_QUANTIZATION,
) -> dict[str, torch.Tensor | int]:
    if tri_a.ndim != 2 or tri_a.shape[-1] != 3:
        raise ValueError("tri_a must have shape (face_count, 3)")
    if normals.shape != tri_a.shape:
        raise ValueError("normals must match tri_a shape")
    if surface_ids.ndim != 1 or surface_ids.shape[0] != tri_a.shape[0]:
        raise ValueError("surface_ids must have shape (face_count,)")

    return ops.deterministic_face_groups(
        tri_a.to(dtype=torch.float32).contiguous(),
        normals.to(dtype=torch.float32).contiguous(),
        surface_ids.to(device=tri_a.device, dtype=torch.long).contiguous(),
        quantization=float(quantization),
    )


def _cached_coplanar_face_groups(
    raydn: object,
    tri_a: torch.Tensor,
    normals: torch.Tensor,
    surface_ids: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    """Coplanar face groups are geometry-only; cache them per RayDN scene so
    the union-find does not rerun for every component of every solve."""

    cache = getattr(raydn, "runtime_cache", None)
    if cache is None:
        return _coplanar_face_groups(tri_a, normals, surface_ids)
    cached = cache.get("deterministic_coplanar_face_groups")
    if cached is not None:
        return cached  # type: ignore[return-value]
    groups = _coplanar_face_groups(tri_a, normals, surface_ids)
    cache["deterministic_coplanar_face_groups"] = groups
    return groups


def apply_receiver_layout(values: torch.Tensor, layout: ReceiverLayout) -> torch.Tensor:
    return layout.apply(values)


def _path_components(config: TopologyConfig) -> set[str]:
    components = set(config.components)
    if config.max_depth == 0:
        components.discard("reflection")
        components.discard("diffraction")
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
            defaults = ops.deterministic_topology_default_fields(path_gain)
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


def _sort_order(
    paths: dict[str, torch.Tensor], *, tx_count: int, max_depth: int
) -> torch.Tensor:
    del tx_count, max_depth
    sequence = paths.get("primitive_sequence")
    if sequence is None or sequence.dim() != 2:
        sequence = torch.empty(
            (paths["valid"].numel(), 0), device=paths["valid"].device, dtype=torch.int32
        )
    return ops.deterministic_sort_order(
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
        unique = torch.ones((count,), device=order.device, dtype=torch.bool)
        unique[1:] = ~same
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
            defaults = ops.deterministic_topology_default_fields(
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


def _pad_topology_sequences(
    block: dict[str, torch.Tensor], *, width: int
) -> dict[str, torch.Tensor]:
    if width < 0:
        raise ValueError("sequence width must be non-negative")
    count = int(block["valid"].numel())
    device = block["valid"].device
    empty_i32 = torch.empty((count, 0), device=device, dtype=torch.int32)
    empty_vec3 = torch.empty((count, 0, 3), device=device, dtype=torch.float32)
    sequences = ops.deterministic_pad_topology_sequences(
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
    selected = ops.deterministic_gather_topology_block(
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


def _evaluate_shared_fields(
    scene: Scene,
    compiled: object,
    topology: TopologyBatch,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
) -> TopologyBatch:
    """Evaluate selected canonical rows with the shared complex3 ABI."""

    count = int(topology.valid.shape[0])
    if count == 0:
        return topology
    device = topology.valid.device
    tx_id = topology.tx_id.to(dtype=torch.int64)
    rx_id = topology.rx_id.to(dtype=torch.int64)
    source = tx_positions[tx_id].contiguous()
    target = rx_positions[rx_id].contiguous()
    source_power = tx_power[tx_id].to(dtype=torch.float32).contiguous()
    tx_pol = transmitter_polarizations(scene, device=device)[tx_id].contiguous()
    rx_pol = receiver_polarizations(scene, device=device)[rx_id].contiguous()

    field_xyz = topology.field_xyz.clone()
    coefficient = topology.coefficient.clone()
    path_field = topology.path_field.clone()
    path_gain = topology.path_gain.clone()
    path_length = topology.path_length_m.clone()
    delay = topology.delay_s.clone()
    direction = topology.field_direction.clone()
    launch_count = topology.launch_count

    los_rows = torch.nonzero(topology.component_id == 0, as_tuple=False).reshape(-1)
    if int(los_rows.shape[0]) > 0:
        evaluated = ops.field_free_space(
            source[los_rows].contiguous(),
            target[los_rows].contiguous(),
            source_power[los_rows].contiguous(),
            tx_pol[los_rows].contiguous(),
            rx_pol[los_rows].contiguous(),
            frequency_hz=float(scene.frequency),
        )
        field_xyz.index_copy_(0, los_rows, evaluated["field_vector"])
        coefficient.index_copy_(0, los_rows, evaluated["coefficient"])
        path_field.index_copy_(0, los_rows, evaluated["path_field"])
        path_gain.index_copy_(0, los_rows, evaluated["path_gain"])
        path_length.index_copy_(0, los_rows, evaluated["path_length_m"])
        delay.index_copy_(0, los_rows, evaluated["delay_s"])
        direction.index_copy_(0, los_rows, evaluated["direction"])
        launch_count += 1

    material: dict[str, torch.Tensor] | None = None
    for depth_value in range(1, 6):
        rows = torch.nonzero(
            (topology.component_id == 1) & (topology.depth == depth_value),
            as_tuple=False,
        ).reshape(-1)
        if int(rows.shape[0]) == 0:
            continue
        if material is None:
            material = face_material_field_bundle(compiled, device=device)
        face_id = topology.primitive_sequence[rows, :depth_value].to(dtype=torch.int64)
        evaluated = ops.field_reflection_sequence(
            source[rows].contiguous(),
            target[rows].contiguous(),
            topology.interaction_positions[rows, :depth_value].contiguous(),
            topology.interaction_normals[rows, :depth_value].contiguous(),
            source_power[rows].contiguous(),
            tx_pol[rows].contiguous(),
            rx_pol[rows].contiguous(),
            material["eps_r"][face_id].contiguous(),
            material["sigma_e"][face_id].contiguous(),
            material["mu_r"][face_id].contiguous(),
            material["gain"][face_id].contiguous(),
            material["thickness"][face_id].contiguous(),
            frequency_hz=float(scene.frequency),
        )
        field_xyz.index_copy_(0, rows, evaluated["field_vector"])
        coefficient.index_copy_(0, rows, evaluated["coefficient"])
        path_field.index_copy_(0, rows, evaluated["path_field"])
        path_gain.index_copy_(0, rows, evaluated["path_gain"])
        path_length.index_copy_(0, rows, evaluated["path_length_m"])
        delay.index_copy_(0, rows, evaluated["delay_s"])
        direction.index_copy_(0, rows, evaluated["direction"])
        launch_count += 1

    diffraction_rows = torch.nonzero(
        topology.component_id == 2, as_tuple=False
    ).reshape(-1)
    if int(diffraction_rows.shape[0]) > 0:
        arrival = ops.deterministic_normalize_vec3(
            (
                target[diffraction_rows]
                - topology.interaction_positions[diffraction_rows, 0]
            ).contiguous(),
            eps=1.0e-6,
        )
        powered_xyz = topology.field_xyz[diffraction_rows].contiguous()
        projected = ops.field_project_complex3(
            powered_xyz,
            arrival,
            rx_pol[diffraction_rows].contiguous(),
        )
        amplitude = source_power[diffraction_rows].clamp_min(1.0e-30).sqrt()
        field_xyz.index_copy_(0, diffraction_rows, powered_xyz / amplitude[:, None])
        path_field.index_copy_(0, diffraction_rows, projected["coefficient"])
        coefficient.index_copy_(
            0, diffraction_rows, projected["coefficient"] / amplitude
        )
        path_gain.index_copy_(0, diffraction_rows, projected["path_gain"])
        direction.index_copy_(0, diffraction_rows, arrival)
        launch_count += 1

    coupled_rows = torch.nonzero(
        (topology.component_id == 3) | (topology.component_id == 4),
        as_tuple=False,
    ).reshape(-1)
    if int(coupled_rows.shape[0]) > 0:
        if material is None:
            material = face_material_field_bundle(compiled, device=device)
        raydn = compiled.raydn
        records = raydn.edge_records()
        preserve_imported_edges = bool(
            isinstance(scene.metadata.get("mitsuba", {}), dict)
            and scene.metadata.get("mitsuba", {}).get("merge_shapes", False)
        )
        edge_geometry = (
            _diffraction_edge_geometry(records)
            if preserve_imported_edges
            else _cached_diffraction_edge_geometry(raydn)
        )
        edge_n0 = edge_geometry[6]
        edge_n1 = edge_geometry[7]
        edge_face0 = edge_geometry[8].to(dtype=torch.int64)
        edge_face1 = edge_geometry[9].to(dtype=torch.int64)
        edge_exterior = edge_geometry[10]
        for component_id, reverse_order in ((3, False), (4, True)):
            rows = torch.nonzero(
                topology.component_id == component_id, as_tuple=False
            ).reshape(-1)
            if int(rows.shape[0]) == 0:
                continue
            edge_id = topology.edge_id[rows].to(dtype=torch.int64)
            reflection_face = topology.primitive_id[rows].to(dtype=torch.int64)
            face0 = edge_face0[edge_id]
            raw_face1 = edge_face1[edge_id]
            face1 = torch.where(raw_face1 >= 0, raw_face1, face0)
            reflection_slot = 1 if reverse_order else 0
            edge_slot = 0 if reverse_order else 1

            def material_tuple(face: torch.Tensor) -> tuple[torch.Tensor, ...]:
                return tuple(
                    material[name][face].contiguous()
                    for name in ("eps_r", "sigma_e", "mu_r", "gain", "thickness")
                )

            evaluated = ops.field_coupled_rd(
                source[rows].contiguous(),
                target[rows].contiguous(),
                topology.interaction_positions[
                    rows, reflection_slot
                ].contiguous(),
                topology.interaction_normals[
                    rows, reflection_slot
                ].contiguous(),
                topology.interaction_positions[rows, edge_slot].contiguous(),
                edge_geometry[2][edge_id].contiguous(),
                edge_n0[edge_id].contiguous(),
                edge_n1[edge_id].contiguous(),
                edge_exterior[edge_id].contiguous(),
                source_power[rows].contiguous(),
                tx_pol[rows].contiguous(),
                rx_pol[rows].contiguous(),
                material_tuple(reflection_face),
                material_tuple(face0),
                material_tuple(face1),
                frequency_hz=float(scene.frequency),
                reverse=reverse_order,
            )
            field_xyz.index_copy_(0, rows, evaluated["field_vector"])
            coefficient.index_copy_(0, rows, evaluated["coefficient"])
            path_field.index_copy_(0, rows, evaluated["path_field"])
            path_gain.index_copy_(0, rows, evaluated["path_gain"])
            direction.index_copy_(0, rows, evaluated["direction"])
            launch_count += 1

    return replace(
        topology,
        path_length_m=path_length,
        delay_s=delay,
        path_gain=path_gain,
        path_field=path_field,
        field_xyz=field_xyz,
        coefficient=coefficient,
        field_direction=direction,
        launch_count=launch_count,
    )


def _reflection_topology_order1(
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
) -> tuple[dict[str, torch.Tensor], int]:
    device = tx_positions.device
    raydn = compiled.raydn
    if not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return _ensure_topology_fields(
            {
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
        ), 0
    if not raydn.available:
        raise RuntimeError(
            "deterministic reflection requires RayDN native scene capability"
        )

    records = raydn.edge_records()
    vertices = records.vertices
    faces = records.faces.contiguous()
    normals = ops.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    if faces.shape[0] == 0:
        return _ensure_topology_fields(
            {
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
        ), 0

    tri_a = ops.deterministic_face_anchor_points(vertices.contiguous(), faces)
    face_eps_r, face_sigma_e, face_mu_r, face_gain, _face_valid = face_material_tensors(
        compiled, device=device
    )
    face_material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int32
    ).contiguous()
    empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)

    # Enumerate one candidate per coplanar face group so that a wall meshed
    # from several coplanar triangles yields exactly one specular path, and
    # every planar facade (not one representative per structure) is covered.
    # The EPC kernel resolves the actual containing triangle per path.
    groups = _cached_coplanar_face_groups(
        raydn,
        tri_a,
        normals,
        compiled.geometry.face_surface_id.to(
            device=device, dtype=torch.long
        ).contiguous(),
    )
    grouped_export = True
    group_count = int(groups["group_count"])
    representative_faces = groups["representative_faces"].contiguous()
    exhaustive = group_count <= _ORDER1_EXHAUSTIVE_GROUP_LIMIT
    base_sequences = (
        ops.deterministic_mapped_face_sequence_chunk(
            representative_faces,
            depth=1,
            start=0,
            end=group_count,
        )
        if exhaustive
        else None
    )
    face_group_id = (
        None
        if exhaustive
        else groups["face_group_id"].to(dtype=torch.long).contiguous()
    )

    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    rx_count = int(rx_positions.shape[0])
    if group_count <= 0 or rx_count <= 0:
        return _ensure_topology_fields(
            concatenate_path_blocks(blocks, device=device)
        ), launch_count

    for tx_index, tx in enumerate(tx_positions):
        if exhaustive:
            sequences = base_sequences
        else:
            # Large scenes: only transmitter-visible planes can host a valid
            # first-order specular path; discover them by tracing.
            chains = _discovered_group_chains(
                raydn, tx, face_group_id=face_group_id, max_depth=1
            )
            launch_count += 1
            first_groups = torch.unique(chains[chains[:, 0] >= 0][:, 0])
            if int(first_groups.numel()) == 0:
                continue
            sequences = representative_faces[first_groups].reshape(-1, 1).contiguous()
        sequence_count = int(sequences.shape[0])
        if sequence_count <= 0:
            continue
        rx_chunk_size = max(1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // sequence_count)
        for rx_start in range(0, rx_count, rx_chunk_size):
            rx_end = min(rx_start + rx_chunk_size, rx_count)
            epc_inputs = ops.deterministic_reflection_epc_input_batch(
                tx=tx,
                rx_positions=rx_positions.contiguous(),
                sequences=sequences.contiguous(),
                tri_a=tri_a.contiguous(),
                normals=normals.contiguous(),
                rx_start=rx_start,
                rx_end=rx_end,
            )
            epc = ops.raydn_reflection_epc_paths_forward(
                raydn.require_handle(),
                epc_inputs["tx_batch"],
                epc_inputs["rx_batch"],
                None,
                epc_inputs["sequence_batch"],
                epc_inputs["direct_plane_points"],
                epc_inputs["direct_plane_normals"],
                groups["surface_group_id"],
                groups["surface_group_size"],
                groups["surface_group_members"],
                1,
                1,
            )
            launch_count += 1
            selected = ops.deterministic_reflection_order1_compact(
                visible=epc[0],
                epc_faces=epc[2],
                epc_hits=epc[4],
                epc_normals=epc[5],
                sequence_batch=epc_inputs["sequence_batch"],
                rx_indices=epc_inputs["rx_indices"],
                tx=tx,
                rx_positions=rx_positions,
                tx_power=tx_power.to(dtype=torch.float32).contiguous(),
                tx_index=tx_index,
                face_eps_r=face_eps_r,
                face_sigma_e=face_sigma_e,
                face_mu_r=face_mu_r,
                face_gain=face_gain,
                face_material_id=face_material_id,
                grouped_export=grouped_export,
            )
            if int(selected["selected_faces"].numel()) == 0:
                continue

            field_result = ops.deterministic_reflection_field(
                tx_position=selected["tx_keep"],
                rx_position=selected["rx_keep"],
                hit_position=selected["selected_points"],
                normal=selected["selected_normals"],
                tx_power=selected["tx_power"],
                eps_r=selected["eps_r"],
                sigma_e=selected["sigma_e"],
                mu_r=selected["mu_r"],
                gain=selected["gain"],
                frequency_hz=frequency_hz,
            )
            path_gain = field_result["path_gain"].to(dtype=torch.float32).contiguous()
            path_field = ops.deterministic_pack_complex(
                field_result["field_real"], field_result["field_imag"]
            )
            path_length = (
                field_result["path_length_m"].to(dtype=torch.float32).contiguous()
            )
            delay = field_result["delay_s"].to(dtype=torch.float32).contiguous()
            blocks.append(
                _ensure_topology_fields(
                    ops.deterministic_topology_base_fields(
                        rx_id=selected["selected_rx_id"],
                        path_length_m=path_length.to(dtype=torch.float32).contiguous(),
                        delay_s=delay,
                        path_gain=path_gain.to(dtype=torch.float32).contiguous(),
                        tx_index=tx_index,
                        component_id=1,
                        depth_source=empty_i32,
                        depth_value=1,
                        primitive_source=selected["selected_faces"],
                        primitive_value=-1,
                        edge_source=empty_i32,
                        edge_value=-1,
                    ),
                    interaction_position=selected["selected_points"],
                    interaction_normal=selected["selected_normals"],
                    material_id=selected["material_id"],
                    path_field=path_field,
                )
            )
    return _ensure_topology_fields(
        concatenate_path_blocks(blocks, device=device)
    ), launch_count


def _reflect_points(
    points: torch.Tensor, plane_points: torch.Tensor, normals: torch.Tensor
) -> torch.Tensor:
    return ops.deterministic_reflect_points(
        points.contiguous(), plane_points.contiguous(), normals.contiguous()
    )


_MULTIBOUNCE_DISCOVERY_RAYS = 262_144


def _discovered_group_chains(
    raydn: object,
    tx: torch.Tensor,
    *,
    face_group_id: torch.Tensor,
    max_depth: int,
    ray_count: int = _MULTIBOUNCE_DISCOVERY_RAYS,
) -> torch.Tensor:
    """Trace specular chains from the transmitter and map them to plane groups.

    Returns an (N, max_depth) long tensor of plane-group ids per bounce with
    -1 past each ray's last hit. Only chains reachable from the transmitter
    can host a valid specular path, so validating the unique chains found here
    replaces the exhaustive plane-sequence product on large scenes.
    """

    device = face_group_id.device
    ray_o = tx.reshape(1, 3).expand(ray_count, 3).contiguous()
    ray_d = ops.mc_sample_directions(ray_count, tx.reshape(1, 3))
    ray_tmax = torch.empty((0,), device=device, dtype=torch.float32)
    out = ops.raydn_trace_reflections_forward(
        raydn.require_handle(),
        ray_o,
        ray_d,
        ray_tmax,
        None,
        int(max_depth),
    )
    prim_chain = out[2].to(dtype=torch.long).reshape(ray_count, int(max_depth))
    chains = torch.full_like(prim_chain, -1)
    hit = prim_chain >= 0
    chains[hit] = face_group_id[prim_chain[hit]]
    return chains


def _face_sequence_count(
    face_count: int, depth: int, *, adjacent_distinct: bool
) -> int:
    if adjacent_distinct and depth > 1:
        if face_count <= 1:
            return 0
        return int(face_count) * int(face_count - 1) ** int(depth - 1)
    return int(face_count) ** int(depth)


def _face_sequence_chunks(
    face_count: int,
    depth: int,
    *,
    chunk_size: int,
    reference: torch.Tensor,
    face_ids: torch.Tensor | None = None,
    adjacent_distinct: bool = False,
) -> object:
    total = _face_sequence_count(face_count, depth, adjacent_distinct=adjacent_distinct)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        if face_ids is None:
            sequences = ops.deterministic_face_sequence_chunk(
                reference,
                face_count=face_count,
                depth=depth,
                start=start,
                end=end,
                adjacent_distinct=adjacent_distinct,
            )
        else:
            sequences = ops.deterministic_mapped_face_sequence_chunk(
                face_ids,
                depth=depth,
                start=start,
                end=end,
                adjacent_distinct=adjacent_distinct,
            )
        if int(sequences.shape[0]) > 0:
            yield sequences


def _reflection_topology_multibounce(
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
    min_depth: int,
    max_depth: int,
    max_paths: int | None,
) -> tuple[dict[str, torch.Tensor], int, int]:
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
            0,
        )
    if not raydn.available:
        raise RuntimeError(
            "deterministic multibounce reflection requires RayDN native scene capability"
        )

    records = raydn.edge_records()
    vertices = records.vertices
    faces = records.faces.contiguous()
    normals = ops.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    face_count = int(faces.shape[0])
    if face_count == 0 or max_depth < min_depth:
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
            0,
        )

    tri_a = ops.deterministic_face_anchor_points(vertices.contiguous(), faces)
    face_eps_r, face_sigma_e, face_mu_r, face_gain, _face_valid = face_material_tensors(
        compiled, device=device
    )
    face_material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int32
    ).contiguous()
    face_group_source = compiled.geometry.face_surface_id.to(
        device=device, dtype=torch.long
    ).contiguous()
    planning_guard = _MAX_MULTIBOUNCE_FACE_SEQUENCES
    # Coplanar plane groups carry the specular semantics (dedup, adjacency,
    # visibility-ignore scope). When the exhaustive plane-sequence space fits
    # the planning guard, enumerate it exactly; otherwise discover reachable
    # plane chains by tracing rays from the transmitter and validate only
    # those, matching the original discovery-based implementation.
    groups = _cached_coplanar_face_groups(raydn, tri_a, normals, face_group_source)
    group_count = int(groups["group_count"])
    representative_faces = groups["representative_faces"].contiguous()
    surface_group_id = groups["surface_group_id"]
    surface_group_size = groups["surface_group_size"]
    surface_group_members = groups["surface_group_members"]
    exhaustive = all(
        _face_sequence_count(group_count, depth, adjacent_distinct=True)
        <= planning_guard
        for depth in range(min_depth, max_depth + 1)
    )
    tx_power_f32 = tx_power.to(dtype=torch.float32).contiguous()
    rx_count = int(rx_positions.shape[0])
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    theoretical_candidate_count = 0

    def emit_epc_chunk(
        sequences: torch.Tensor,
        depth: int,
        tx: torch.Tensor,
        tx_index: int,
        rx_start: int,
        rx_end: int,
    ) -> None:
        nonlocal launch_count
        epc_inputs = ops.deterministic_reflection_epc_input_batch(
            tx=tx,
            rx_positions=rx_positions.contiguous(),
            sequences=sequences.contiguous(),
            tri_a=tri_a.contiguous(),
            normals=normals.contiguous(),
            rx_start=rx_start,
            rx_end=rx_end,
        )
        epc = ops.raydn_reflection_epc_paths_forward(
            raydn.require_handle(),
            epc_inputs["tx_batch"],
            epc_inputs["rx_batch"],
            None,
            epc_inputs["sequence_batch"],
            epc_inputs["direct_plane_points"],
            epc_inputs["direct_plane_normals"],
            surface_group_id,
            surface_group_size,
            surface_group_members,
            int(depth),
            1,
        )
        launch_count += 1
        selected = ops.deterministic_reflection_sequence_compact(
            visible=epc[0],
            epc_sequences=epc[2],
            epc_hits=epc[4],
            epc_normals=epc[5],
            rx_indices=epc_inputs["rx_indices"],
            tx=tx,
            rx_positions=rx_positions,
            tx_power=tx_power_f32,
            tx_index=tx_index,
            face_eps_r=face_eps_r,
            face_sigma_e=face_sigma_e,
            face_mu_r=face_mu_r,
            face_gain=face_gain,
            face_material_id=face_material_id,
            max_count=-1,
        )
        count = int(selected["selected_sequences"].shape[0])
        if count == 0:
            return
        field_result = ops.deterministic_reflection_sequence_field(
            tx_position=selected["selected_tx"],
            rx_position=selected["selected_rx"],
            hit_positions=selected["selected_hits"],
            normals=selected["selected_normals"],
            tx_power=selected["tx_power"],
            eps_r=selected["eps_r"],
            sigma_e=selected["sigma_e"],
            mu_r=selected["mu_r"],
            gain=selected["gain"],
            frequency_hz=frequency_hz,
        )
        path_gain = field_result["path_gain"].to(dtype=torch.float32).contiguous()
        path_field = ops.deterministic_pack_complex(
            field_result["field_real"], field_result["field_imag"]
        )
        path_length = field_result["path_length_m"].to(dtype=torch.float32).contiguous()
        delay = field_result["delay_s"].to(dtype=torch.float32).contiguous()
        empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)
        blocks.append(
            _ensure_topology_fields(
                ops.deterministic_topology_base_fields(
                    rx_id=selected["selected_rx_id"],
                    path_length_m=path_length,
                    delay_s=delay,
                    path_gain=path_gain,
                    tx_index=tx_index,
                    component_id=1,
                    depth_source=empty_i32,
                    depth_value=depth,
                    primitive_source=selected["first_face"],
                    primitive_value=-1,
                    edge_source=empty_i32,
                    edge_value=-1,
                ),
                interaction_position=selected["first_hit"],
                interaction_normal=selected["first_normal"],
                material_id=selected["material_id"],
                path_field=path_field,
                primitive_sequence=selected["selected_sequences"],
                material_sequence=selected["material_sequence"],
                interaction_positions=selected["selected_hits"],
                interaction_normals=selected["selected_normals"],
            )
        )

    if exhaustive:
        for depth in range(min_depth, max_depth + 1):
            candidate_count = _face_sequence_count(
                group_count, depth, adjacent_distinct=True
            )
            theoretical_candidate_count += candidate_count
            chunk_size = min(_MULTIBOUNCE_SEQUENCE_CHUNK_SIZE, max(candidate_count, 1))
            for sequences in _face_sequence_chunks(
                group_count,
                depth,
                chunk_size=chunk_size,
                reference=tx_power_f32,
                face_ids=representative_faces,
                adjacent_distinct=True,
            ):
                sequence_count = int(sequences.shape[0])
                if sequence_count <= 0:
                    continue
                rx_chunk_size = max(1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // sequence_count)
                for rx_start in range(0, rx_count, rx_chunk_size):
                    rx_end = min(rx_start + rx_chunk_size, rx_count)
                    for tx_index, tx in enumerate(tx_positions):
                        emit_epc_chunk(sequences, depth, tx, tx_index, rx_start, rx_end)
    else:
        face_group_id = groups["face_group_id"].to(dtype=torch.long).contiguous()
        for tx_index, tx in enumerate(tx_positions):
            group_chains = _discovered_group_chains(
                raydn,
                tx,
                face_group_id=face_group_id,
                max_depth=max_depth,
            )
            launch_count += 1
            for depth in range(min_depth, max_depth + 1):
                reached = group_chains[:, depth - 1] >= 0
                if not bool(reached.any()):
                    continue
                unique_chains = torch.unique(group_chains[reached][:, :depth], dim=0)
                theoretical_candidate_count += int(unique_chains.shape[0])
                sequences_all = representative_faces[unique_chains].contiguous()
                for start in range(
                    0, int(sequences_all.shape[0]), _MULTIBOUNCE_SEQUENCE_CHUNK_SIZE
                ):
                    sequences = sequences_all[
                        start : start + _MULTIBOUNCE_SEQUENCE_CHUNK_SIZE
                    ].contiguous()
                    rx_chunk_size = max(
                        1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // int(sequences.shape[0])
                    )
                    for rx_start in range(0, rx_count, rx_chunk_size):
                        rx_end = min(rx_start + rx_chunk_size, rx_count)
                        emit_epc_chunk(sequences, depth, tx, tx_index, rx_start, rx_end)

    return (
        _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)),
        launch_count,
        theoretical_candidate_count,
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
    return ops.deterministic_diffraction_state_pack(
        ops.mc_selected_edge_indices(selected),
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
        visible |= ops.raydn_visibility_forward(
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
            out = ops.raydn_diffraction_paths_order1_forward(
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
            compacted = ops.deterministic_diffraction_order1_compact(
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
            field_result = ops.deterministic_diffraction_vector_field(
                x_re=compacted["x_re"],
                x_im=compacted["x_im"],
                y_re=compacted["y_re"],
                y_im=compacted["y_im"],
                z_re=compacted["z_re"],
                z_im=compacted["z_im"],
            )
            field_power = field_result["path_gain"]
            path_field = ops.deterministic_pack_complex(
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
            path_length = ops.deterministic_delay_to_path_length(delay)
            empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)
            blocks.append(
                _ensure_topology_fields(
                    ops.deterministic_topology_base_fields(
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
    normals = ops.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    tri_a = ops.deterministic_face_anchor_points(vertices, faces)
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
    selected_edges = ops.mc_selected_edge_indices(selected)
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
            exported = ops.raydn_coupled_rd_geometry_forward(*common_args, reverse)
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


def export_topology(scene: Scene, config: TopologyConfig) -> TopologyBatch:
    device = torch.device("cuda")
    tx_positions, tx_power = transmitter_tensors(scene, device=device)
    rx_positions, _ = receiver_positions_and_layout(scene, device=device)
    compiled = scene.compile()
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
        exported = ops.path_los_export(
            tx_positions,
            tx_power,
            rx_positions,
            frequency_hz=float(scene.frequency),
        )
        launch_count += 1
        tx_id = exported["tx_id"]
        rx_id = exported["rx_id"]
        visible = None
        if bool(scene.structures) and int(tx_id.numel()) > 0:
            visibility_inputs = ops.path_los_visibility_inputs(
                tx_positions,
                rx_positions,
                tx_id.to(dtype=torch.int32).contiguous(),
                rx_id.to(dtype=torch.int32).contiguous(),
            )
            visible = ops.raydn_visibility_forward(
                compiled.raydn.require_handle(),
                visibility_inputs["start"],
                visibility_inputs["end"],
                visibility_inputs["active"],
            )[0]
            launch_count += 1
        candidate_count += int(tx_id.numel())
        los_block = _ensure_topology_fields(
            ops.deterministic_los_topology_block(
                tx_id.to(dtype=torch.int32).contiguous(),
                rx_id.to(dtype=torch.int32).contiguous(),
                exported["path_length_m"].to(dtype=torch.float32).contiguous(),
                exported["delay_s"].to(dtype=torch.float32).contiguous(),
                exported["path_gain"].to(dtype=torch.float32).contiguous(),
                visible,
                frequency_hz=float(scene.frequency),
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
            frequency_hz=float(scene.frequency),
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
                frequency_hz=float(scene.frequency),
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
                frequency_hz=float(scene.frequency),
            )
        )
        launch_count += diffraction_launches
        candidate_count += int(block["valid"].numel())
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
            scene, compiled, result, tx_positions, tx_power, rx_positions
        )
    padded_blocks = [
        block
        if "primitive_sequence" in block
        and int(block["primitive_sequence"].shape[1]) == sequence_width
        else _pad_topology_sequences(block, width=sequence_width)
        for block in blocks
    ]
    paths = concatenate_path_blocks(padded_blocks, device=device)
    selected_edge_count = ops.deterministic_selected_edge_count(paths["edge_id"])
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
        scene, compiled, result, tx_positions, tx_power, rx_positions
    )
