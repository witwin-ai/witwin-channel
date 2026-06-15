from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace

import torch

from witwin.channel_native import ReceiverGrid, ReceiverPoint, Scene
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.core.material_runtime import face_material_tensors
from witwin.channel_native.montecarlo.basic.backend import (
    _LIGHT_SPEED_M_PER_S,
    receiver_positions as _native_receiver_positions,
    transmitter_positions as _native_transmitter_positions,
)
from witwin.channel_native.montecarlo.basic.raydn_components import _cached_diffraction_edge_geometry

from .config import Config

_MAX_MULTIBOUNCE_FACE_SEQUENCES = 100_000
_MULTIBOUNCE_SEQUENCE_CHUNK_SIZE = 8192
_MULTIBOUNCE_PAIR_CHUNK_SIZE = 262_144
_ORDER1_GROUPED_FACE_THRESHOLD = 4096
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


def _raydn_visibility_mask(raydn: object, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    if start.shape[0] == 0:
        return torch.empty((0,), device=start.device, dtype=torch.bool)
    return ops.raydn_visibility_forward(raydn.require_handle(), start.contiguous(), end.contiguous(), None)[0]


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


def concatenate_path_blocks(blocks: list[dict[str, torch.Tensor]], *, device: torch.device) -> dict[str, torch.Tensor]:
    nonempty = [block for block in blocks if int(block["valid"].numel()) > 0]
    if not nonempty:
        return _empty_path_block(device)
    sequence_field = nonempty[0].get("primitive_sequence")
    sequence_width = int(sequence_field.shape[1]) if isinstance(sequence_field, torch.Tensor) else 0
    return ops.deterministic_concat_topology_blocks(nonempty, sequence_width=sequence_width)


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
    interaction_position: torch.Tensor
    interaction_normal: torch.Tensor
    material_id: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    launch_count: int = 0
    visibility_rejection_count: int = 0
    selected_edge_count: int = 0
    candidate_count: int = 0
    guardrail_count: int = 0


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
            return values.reshape(values.shape[0], rows, cols).transpose(1, 2).contiguous()
        if self.kind == "point":
            return values.contiguous()
        raise ValueError(f"receiver layout kind is not accepted: {self.kind}")


def transmitter_tensors(scene: Scene, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return _native_transmitter_positions(scene, device=device)


def receiver_positions_and_layout(scene: Scene, *, device: torch.device) -> tuple[torch.Tensor, ReceiverLayout]:
    if not scene.receivers:
        return torch.empty((0, 3), device=device, dtype=torch.float32), ReceiverLayout("point", 0)

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


def _surface_face_groups(surface_ids: torch.Tensor) -> dict[str, torch.Tensor | int]:
    if surface_ids.ndim != 1:
        raise ValueError("surface_ids must have shape (face_count,)")
    return ops.deterministic_surface_face_groups(surface_ids.to(dtype=torch.long).contiguous())


def apply_receiver_layout(values: torch.Tensor, layout: ReceiverLayout) -> torch.Tensor:
    return layout.apply(values)


def _path_components(config: Config) -> set[str]:
    components = set(config.components)
    if config.max_depth == 0:
        components.discard("reflection")
        components.discard("diffraction")
    if config.max_diffraction_order == 0:
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
    interaction_position = getattr(paths, "interaction_position", None)
    if interaction_position is None:
        interaction_position = topology_defaults()["interaction_position"]
    interaction_normal = getattr(paths, "interaction_normal", None)
    if interaction_normal is None:
        interaction_normal = topology_defaults()["interaction_normal"]
    material_id = getattr(paths, "material_id", None)
    if material_id is None:
        material_id = topology_defaults()["material_id"]
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
        interaction_position=interaction_position.to(dtype=torch.float32).contiguous(),
        interaction_normal=interaction_normal.to(dtype=torch.float32).contiguous(),
        material_id=material_id.to(dtype=torch.int32).contiguous(),
        primitive_sequence=getattr(
            paths,
            "primitive_sequence",
            torch.empty((path_count, 0), device=device, dtype=torch.int32),
        ).to(dtype=torch.int32).contiguous(),
        material_sequence=getattr(
            paths,
            "material_sequence",
            torch.empty((path_count, 0), device=device, dtype=torch.int32),
        ).to(dtype=torch.int32).contiguous(),
        interaction_positions=getattr(
            paths,
            "interaction_positions",
            torch.empty((path_count, 0, 3), device=device, dtype=torch.float32),
        ).to(dtype=torch.float32).contiguous(),
        interaction_normals=getattr(
            paths,
            "interaction_normals",
            torch.empty((path_count, 0, 3), device=device, dtype=torch.float32),
        ).to(dtype=torch.float32).contiguous(),
        launch_count=int(getattr(paths, "launch_count", 0)),
        visibility_rejection_count=int(getattr(paths, "visibility_rejection_count", 0)),
        selected_edge_count=int(getattr(paths, "selected_edge_count", 0)),
        candidate_count=int(getattr(paths, "candidate_count", path_count)),
        guardrail_count=int(getattr(paths, "guardrail_count", 0)),
    )


def _sort_order(paths: dict[str, torch.Tensor], *, tx_count: int, max_depth: int) -> torch.Tensor:
    del tx_count, max_depth
    sequence = paths.get("primitive_sequence")
    if sequence is None or sequence.dim() != 2:
        sequence = torch.empty((paths["valid"].numel(), 0), device=paths["valid"].device, dtype=torch.int32)
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


def _ensure_topology_fields(
    block: dict[str, torch.Tensor],
    *,
    interaction_position: torch.Tensor | None = None,
    interaction_normal: torch.Tensor | None = None,
    material_id: torch.Tensor | None = None,
    path_field: torch.Tensor | None = None,
    primitive_sequence: torch.Tensor | None = None,
    material_sequence: torch.Tensor | None = None,
    interaction_positions: torch.Tensor | None = None,
    interaction_normals: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    count = int(block["valid"].numel())
    device = block["valid"].device
    extended = dict(block)
    defaults: dict[str, torch.Tensor] | None = None

    def topology_defaults() -> dict[str, torch.Tensor]:
        nonlocal defaults
        if defaults is None:
            defaults = ops.deterministic_topology_default_fields(block["path_gain"].to(dtype=torch.float32).contiguous())
        return defaults

    interaction_position_value = interaction_position if interaction_position is not None else block.get("interaction_position")
    if interaction_position_value is None:
        interaction_position_value = topology_defaults()["interaction_position"]
    interaction_normal_value = interaction_normal if interaction_normal is not None else block.get("interaction_normal")
    if interaction_normal_value is None:
        interaction_normal_value = topology_defaults()["interaction_normal"]
    material_id_value = material_id if material_id is not None else block.get("material_id")
    if material_id_value is None:
        material_id_value = topology_defaults()["material_id"]
    path_field_value = path_field if path_field is not None else block.get("path_field")
    if path_field_value is None:
        path_field_value = topology_defaults()["path_field"]
    extended["interaction_position"] = interaction_position_value.to(dtype=torch.float32).contiguous()
    extended["interaction_normal"] = interaction_normal_value.to(dtype=torch.float32).contiguous()
    extended["material_id"] = material_id_value.to(dtype=torch.int32).contiguous()
    extended["path_field"] = path_field_value.to(dtype=torch.complex64).contiguous()
    if primitive_sequence is not None:
        extended["primitive_sequence"] = primitive_sequence.to(dtype=torch.int32).contiguous()
    if material_sequence is not None:
        extended["material_sequence"] = material_sequence.to(dtype=torch.int32).contiguous()
    if interaction_positions is not None:
        extended["interaction_positions"] = interaction_positions.to(dtype=torch.float32).contiguous()
    if interaction_normals is not None:
        extended["interaction_normals"] = interaction_normals.to(dtype=torch.float32).contiguous()
    return extended


def _pad_topology_sequences(block: dict[str, torch.Tensor], *, width: int) -> dict[str, torch.Tensor]:
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
        interaction_position=block["interaction_position"].to(dtype=torch.float32).contiguous(),
        interaction_normal=block["interaction_normal"].to(dtype=torch.float32).contiguous(),
        primitive_sequence=block.get("primitive_sequence", empty_i32).to(dtype=torch.int32).contiguous(),
        material_sequence=block.get("material_sequence", empty_i32).to(dtype=torch.int32).contiguous(),
        interaction_positions=block.get("interaction_positions", empty_vec3).to(dtype=torch.float32).contiguous(),
        interaction_normals=block.get("interaction_normals", empty_vec3).to(dtype=torch.float32).contiguous(),
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
    tx_count: int,
    max_depth: int,
    launch_count: int,
    visibility_rejection_count: int = 0,
    selected_edge_count: int = 0,
    candidate_count: int | None = None,
    guardrail_count: int = 0,
) -> TopologyBatch:
    order = _sort_order(paths, tx_count=tx_count, max_depth=max_depth)
    selected = ops.deterministic_gather_topology_block(
        paths,
        order,
        max_count=-1 if max_paths is None else int(max_paths),
        sequence_width=max_depth,
    )
    return _from_path_result(
        SimpleNamespace(
            **selected,
            launch_count=launch_count,
            visibility_rejection_count=visibility_rejection_count,
            selected_edge_count=selected_edge_count,
            candidate_count=int(candidate_count if candidate_count is not None else paths["valid"].numel()),
            guardrail_count=guardrail_count,
        )
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
        raise RuntimeError("deterministic reflection requires RayDN native scene capability")

    records = raydn.edge_records()
    vertices = records.vertices
    faces = records.faces.contiguous()
    normals = ops.deterministic_normalize_vec3(records.face_normals.contiguous(), eps=1.0e-6)
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
    face_eps_r, face_sigma_e, face_mu_r, face_gain, _face_valid = face_material_tensors(compiled, device=device)
    face_material_id = compiled.assignments.face_material_id.to(device=device, dtype=torch.int32).contiguous()
    empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)

    grouped_export = int(faces.shape[0]) > _ORDER1_GROUPED_FACE_THRESHOLD
    groups = _surface_face_groups(compiled.geometry.face_surface_id.to(device=device, dtype=torch.long).contiguous())
    if grouped_export:
        sequences = ops.deterministic_mapped_face_sequence_chunk(
            groups["representative_faces"].contiguous(),
            depth=1,
            start=0,
            end=int(groups["representative_faces"].shape[0]),
        )
    else:
        sequences = ops.deterministic_face_sequence_chunk(
            tx_power.to(dtype=torch.float32).contiguous(),
            face_count=int(faces.shape[0]),
            depth=1,
            start=0,
            end=int(faces.shape[0]),
        )

    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    sequence_count = int(sequences.shape[0])
    rx_count = int(rx_positions.shape[0])
    if sequence_count <= 0 or rx_count <= 0:
        return _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)), launch_count

    rx_chunk_size = max(1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // sequence_count)
    for rx_start in range(0, rx_count, rx_chunk_size):
        rx_end = min(rx_start + rx_chunk_size, rx_count)
        for tx_index, tx in enumerate(tx_positions):
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
            path_field = ops.deterministic_pack_complex(field_result["field_real"], field_result["field_imag"])
            path_length = field_result["path_length_m"].to(dtype=torch.float32).contiguous()
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
    return _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)), launch_count


def _reflect_points(points: torch.Tensor, plane_points: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    return ops.deterministic_reflect_points(points.contiguous(), plane_points.contiguous(), normals.contiguous())


def _face_sequence_count(face_count: int, depth: int, *, adjacent_distinct: bool) -> int:
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
        ), 0, 0
    if not raydn.available:
        raise RuntimeError("deterministic multibounce reflection requires RayDN native scene capability")

    records = raydn.edge_records()
    vertices = records.vertices
    faces = records.faces.contiguous()
    normals = ops.deterministic_normalize_vec3(records.face_normals.contiguous(), eps=1.0e-6)
    face_count = int(faces.shape[0])
    if face_count == 0 or max_depth < min_depth:
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
        ), 0, 0

    tri_a = ops.deterministic_face_anchor_points(vertices.contiguous(), faces)
    face_eps_r, face_sigma_e, face_mu_r, face_gain, _face_valid = face_material_tensors(compiled, device=device)
    face_material_id = compiled.assignments.face_material_id.to(device=device, dtype=torch.int32).contiguous()
    face_group_source = compiled.geometry.face_surface_id.to(device=device, dtype=torch.long).contiguous()
    surface_groups = _surface_face_groups(face_group_source)
    planning_guard = (
        _MAX_MULTIBOUNCE_FACE_SEQUENCES
        if max_paths is None
        else min(_MAX_MULTIBOUNCE_FACE_SEQUENCES, max(int(max_paths) * 64, int(max_paths)))
    )
    if face_count > _ORDER1_GROUPED_FACE_THRESHOLD:
        groups = surface_groups
    else:
        plane_groups = _coplanar_face_groups(tri_a, normals, face_group_source)
        groups = (
            plane_groups
            if int(plane_groups["group_count"]) ** max_depth <= planning_guard
            else surface_groups
        )
    group_count = int(groups["group_count"])
    representative_faces = groups["representative_faces"]
    surface_group_id = groups["surface_group_id"]
    surface_group_size = groups["surface_group_size"]
    surface_group_members = groups["surface_group_members"]
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    theoretical_candidate_count = 0
    emitted_count = 0

    for depth in range(min_depth, max_depth + 1):
        candidate_count = _face_sequence_count(group_count, depth, adjacent_distinct=True)
        theoretical_candidate_count += candidate_count
        guard = (
            _MAX_MULTIBOUNCE_FACE_SEQUENCES
            if max_paths is None
            else min(_MAX_MULTIBOUNCE_FACE_SEQUENCES, max(int(max_paths) * 64, int(max_paths)))
        )
        if candidate_count > guard and max_paths is None:
            raise RuntimeError(
                "deterministic multi-bounce reflection candidate count exceeds the current native guardrail "
                f"({candidate_count} face sequences for depth={depth}, guard={guard}); use a smaller scene, "
                "lower max_depth, or wait for native sequence compaction"
            )
        chunk_size = min(_MULTIBOUNCE_SEQUENCE_CHUNK_SIZE, max(candidate_count, 1))
        for sequences in _face_sequence_chunks(
            group_count,
            depth,
            chunk_size=chunk_size,
            reference=tx_power.to(dtype=torch.float32).contiguous(),
            face_ids=representative_faces.contiguous(),
            adjacent_distinct=True,
        ):
            if max_paths is not None and emitted_count >= int(max_paths):
                break
            sequence_count = int(sequences.shape[0])
            if sequence_count <= 0:
                continue
            rx_chunk_size = max(1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // sequence_count)
            for rx_start in range(0, int(rx_positions.shape[0]), rx_chunk_size):
                if max_paths is not None and emitted_count >= int(max_paths):
                    break
                rx_end = min(rx_start + rx_chunk_size, int(rx_positions.shape[0]))
                for tx_index, tx in enumerate(tx_positions):
                    if max_paths is not None and emitted_count >= int(max_paths):
                        break
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
                    remaining = -1 if max_paths is None else int(max_paths) - emitted_count
                    selected = ops.deterministic_reflection_sequence_compact(
                        visible=epc[0],
                        epc_sequences=epc[2],
                        epc_hits=epc[4],
                        epc_normals=epc[5],
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
                        max_count=remaining,
                    )
                    count = int(selected["selected_sequences"].shape[0])
                    if count == 0:
                        continue
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
                    path_field = ops.deterministic_pack_complex(field_result["field_real"], field_result["field_imag"])
                    path_length = field_result["path_length_m"].to(dtype=torch.float32).contiguous()
                    delay = field_result["delay_s"].to(dtype=torch.float32).contiguous()
                    emitted_count += count
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

    return _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)), launch_count, theoretical_candidate_count


def _deterministic_diffraction_states(
    raydn: object,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    tx_index: int,
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
    ) = _cached_diffraction_edge_geometry(raydn)
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


def _diffraction_topology_order1(
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
        raise RuntimeError("deterministic diffraction requires RayDN native scene capability")

    _eps_r, _sigma_e, _mu_r, material_gain, material_valid = face_material_tensors(compiled, device=device)
    wavelength = _LIGHT_SPEED_M_PER_S / float(frequency_hz)
    handle = raydn.require_handle()
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    for tx_index, tx in enumerate(tx_positions):
        states = _deterministic_diffraction_states(raydn, tx, tx_power, tx_index)
        state_count = int(states[0].shape[0])
        if state_count <= 0:
            continue
        capacity = int(rx_positions.shape[0]) * state_count
        out = ops.raydn_diffraction_paths_order1_forward(
            handle,
            tx.reshape(1, 3).contiguous(),
            rx_positions.contiguous(),
            None,
            *states,
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
        field_result = ops.deterministic_diffraction_vector_field(
            x_re=compacted["x_re"],
            x_im=compacted["x_im"],
            y_re=compacted["y_re"],
            y_im=compacted["y_im"],
            z_re=compacted["z_re"],
            z_im=compacted["z_im"],
        )
        field_power = field_result["path_gain"]
        path_field = ops.deterministic_pack_complex(field_result["field_real"], field_result["field_imag"])
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
            )
        )
    return _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)), launch_count


def export_topology(scene: Scene, config: Config) -> TopologyBatch:
    device = torch.device("cuda")
    tx_positions, tx_power = transmitter_tensors(scene, device=device)
    rx_positions, _ = receiver_positions_and_layout(scene, device=device)
    compiled = scene.compile()
    components = _path_components(config)
    sequence_width = max(int(config.max_depth), 0)
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    visibility_rejection_count = 0
    candidate_count = 0
    guardrail_count = 0

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
        los_block = ops.deterministic_los_topology_block(
            tx_id.to(dtype=torch.int32).contiguous(),
            rx_id.to(dtype=torch.int32).contiguous(),
            exported["path_length_m"].to(dtype=torch.float32).contiguous(),
            exported["delay_s"].to(dtype=torch.float32).contiguous(),
            exported["path_gain"].to(dtype=torch.float32).contiguous(),
            visible,
            frequency_hz=float(scene.frequency),
            sequence_width=sequence_width,
        )
        if visible is not None:
            visibility_rejection_count += int(tx_id.numel()) - int(los_block["valid"].numel())
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
        blocks.append(
            block
        )
    if "reflection" in components and config.max_depth >= 2:
        block, reflection_launches, reflection_candidates = _reflection_topology_multibounce(
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
        launch_count += reflection_launches
        candidate_count += int(reflection_candidates)
        blocks.append(block)
    if "diffraction" in components and config.max_depth >= 1:
        block, diffraction_launches = _diffraction_topology_order1(
            scene,
            compiled,
            tx_positions,
            tx_power,
            rx_positions,
            frequency_hz=float(scene.frequency),
        )
        launch_count += diffraction_launches
        candidate_count += int(block["valid"].numel())
        blocks.append(
            block
        )
    if (
        len(blocks) == 1
        and components == {"los"}
        and config.max_paths is None
    ):
        return _from_path_result(
            SimpleNamespace(
                **blocks[0],
                launch_count=launch_count,
                visibility_rejection_count=visibility_rejection_count,
                selected_edge_count=0,
                candidate_count=candidate_count,
                guardrail_count=guardrail_count,
            )
        )
    padded_blocks = [
        block
        if "primitive_sequence" in block and int(block["primitive_sequence"].shape[1]) == sequence_width
        else _pad_topology_sequences(block, width=sequence_width)
        for block in blocks
    ]
    paths = concatenate_path_blocks(padded_blocks, device=device)
    selected_edge_count = ops.deterministic_selected_edge_count(paths["edge_id"])
    return _from_path_block(
        paths,
        max_paths=config.max_paths,
        tx_count=len(scene.transmitters),
        max_depth=config.max_depth,
        launch_count=launch_count,
        visibility_rejection_count=visibility_rejection_count,
        selected_edge_count=selected_edge_count,
        candidate_count=candidate_count,
        guardrail_count=guardrail_count,
    )
