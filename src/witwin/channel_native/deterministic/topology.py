from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace

import torch

from witwin.channel_native import ReceiverGrid, ReceiverPoint, Scene
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.core.material_runtime import face_material_tensors
from witwin.channel_native.montecarlo.basic.backend import _LIGHT_SPEED_M_PER_S
from witwin.channel_native.montecarlo.basic.raydn_components import _diffraction_states

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
    active = torch.ones((start.shape[0],), device=start.device, dtype=torch.bool)
    return torch.ops.raydn.visibility_forward(raydn.require_handle(), start.contiguous(), end.contiguous(), active)[0]


def _los_visibility_mask(
    raydn: object,
    tx_for_path: torch.Tensor,
    rx_for_path: torch.Tensor,
    *,
    has_structures: bool,
) -> torch.Tensor:
    if not has_structures or not raydn.available or tx_for_path.shape[0] == 0:
        return torch.ones((tx_for_path.shape[0],), device=tx_for_path.device, dtype=torch.bool)
    return _raydn_visibility_mask(raydn, tx_for_path, rx_for_path)


def concatenate_path_blocks(blocks: list[dict[str, torch.Tensor]], *, device: torch.device) -> dict[str, torch.Tensor]:
    nonempty = [block for block in blocks if int(block["valid"].numel()) > 0]
    if not nonempty:
        return _empty_path_block(device)
    return {key: torch.cat([block[key] for block in nonempty], dim=0).contiguous() for key in nonempty[0]}


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
        raise ValueError(f"unsupported receiver layout kind: {self.kind}")


def transmitter_tensors(scene: Scene, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not scene.transmitters:
        return (
            torch.empty((0, 3), device=device, dtype=torch.float32),
            torch.empty((0,), device=device, dtype=torch.float32),
        )
    positions = torch.stack([tx.position for tx in scene.transmitters], dim=0).to(
        device=device, dtype=torch.float32
    )
    power = torch.tensor([float(tx.power_w) for tx in scene.transmitters], device=device, dtype=torch.float32)
    return positions.contiguous(), power.contiguous()


def receiver_positions_and_layout(scene: Scene, *, device: torch.device) -> tuple[torch.Tensor, ReceiverLayout]:
    if not scene.receivers:
        return torch.empty((0, 3), device=device, dtype=torch.float32), ReceiverLayout("point", 0)

    if len(scene.receivers) == 1 and isinstance(scene.receivers[0], ReceiverGrid):
        grid = scene.receivers[0]
        points = grid.points().to(device=device, dtype=torch.float32).contiguous()
        return points, ReceiverLayout("grid", int(points.shape[0]), grid.shape)

    blocks: list[torch.Tensor] = []
    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverPoint):
            blocks.append(receiver.position.reshape(1, 3))
        elif isinstance(receiver, ReceiverGrid):
            blocks.append(receiver.points())
        else:
            raise TypeError(f"unsupported receiver type: {type(receiver).__name__}")
    positions = torch.cat(blocks, dim=0).to(device=device, dtype=torch.float32).contiguous()
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

    device = tri_a.device
    normals = torch.nn.functional.normalize(normals.to(dtype=torch.float32), dim=1, eps=1.0e-6)
    major_axis = normals.abs().argmax(dim=1, keepdim=True)
    major_component = normals.gather(1, major_axis).squeeze(1)
    sign = torch.where(major_component < 0.0, -torch.ones_like(major_component), torch.ones_like(major_component))
    canonical_normals = normals * sign.unsqueeze(1)
    plane_offsets = (canonical_normals * tri_a.to(dtype=torch.float32)).sum(dim=1, keepdim=True)
    scale = 1.0 / float(quantization)
    normal_keys = torch.round(canonical_normals * scale).to(dtype=torch.long)
    offset_keys = torch.round(plane_offsets * scale).to(dtype=torch.long)
    keys = torch.cat((surface_ids.to(device=device, dtype=torch.long).reshape(-1, 1), normal_keys, offset_keys), dim=1)
    return _face_groups_from_keys(keys)


def _face_groups_from_keys(keys: torch.Tensor) -> dict[str, torch.Tensor | int]:
    if keys.ndim != 2:
        raise ValueError("keys must have shape (face_count, key_width)")
    face_count = int(keys.shape[0])
    device = keys.device
    if face_count == 0:
        return {
            "face_group_id": torch.empty((0,), device=device, dtype=torch.int32),
            "representative_faces": torch.empty((0,), device=device, dtype=torch.long),
            "surface_group_id": torch.empty((0,), device=device, dtype=torch.int32),
            "surface_group_size": torch.empty((0,), device=device, dtype=torch.int32),
            "surface_group_members": torch.empty((0,), device=device, dtype=torch.int32),
            "group_count": 0,
        }

    _group_values, face_group_id_long = torch.unique(keys, sorted=True, return_inverse=True, dim=0)
    group_count = int(_group_values.shape[0])

    face_indices = torch.arange(face_count, device=device, dtype=torch.long)
    representative_faces = torch.full((group_count,), face_count, device=device, dtype=torch.long)
    representative_faces.scatter_reduce_(0, face_group_id_long, face_indices, reduce="amin", include_self=True)
    surface_group_id = face_group_id_long.to(dtype=torch.int32).contiguous()
    surface_group_size = torch.bincount(face_group_id_long, minlength=group_count).to(device=device, dtype=torch.int32)
    max_group_size = int(surface_group_size.max().item())
    surface_group_members_2d = torch.full((group_count, max_group_size), -1, device=device, dtype=torch.int32)
    for group in range(group_count):
        members = (face_group_id_long == group).nonzero(as_tuple=False).flatten().to(dtype=torch.int32)
        surface_group_members_2d[group, : int(members.numel())] = members

    return {
        "face_group_id": surface_group_id,
        "representative_faces": representative_faces.contiguous(),
        "surface_group_id": surface_group_id,
        "surface_group_size": surface_group_size.contiguous(),
        "surface_group_members": surface_group_members_2d.reshape(-1).contiguous(),
        "group_count": group_count,
    }


def _surface_face_groups(surface_ids: torch.Tensor) -> dict[str, torch.Tensor | int]:
    return _face_groups_from_keys(surface_ids.to(dtype=torch.long).reshape(-1, 1))


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
    zeros_vec = torch.zeros((path_count, 3), device=device, dtype=torch.float32)
    missing_i32 = torch.full((path_count,), -1, device=device, dtype=torch.int32)
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
        path_gain=paths.path_gain.to(dtype=torch.float32).contiguous(),
        path_field=getattr(
            paths,
            "path_field",
            torch.zeros(paths.path_gain.shape, device=device, dtype=torch.complex64),
        ).to(dtype=torch.complex64).contiguous(),
        interaction_position=getattr(paths, "interaction_position", zeros_vec).to(dtype=torch.float32).contiguous(),
        interaction_normal=getattr(paths, "interaction_normal", zeros_vec).to(dtype=torch.float32).contiguous(),
        material_id=getattr(paths, "material_id", missing_i32).to(dtype=torch.int32).contiguous(),
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
    order = torch.arange(paths["valid"].numel(), device=paths["valid"].device, dtype=torch.long)
    sequence = paths.get("primitive_sequence")
    sequence_width = 0 if sequence is None or sequence.dim() != 2 else int(sequence.shape[1])
    key_values = [paths["edge_id"], paths["primitive_id"]]
    if sequence_width > 0:
        key_values.extend(sequence[:, column] for column in reversed(range(sequence_width)))
    key_values.extend([paths["component_id"], paths["depth"], paths["tx_id"], paths["rx_id"]])
    for values_source in key_values:
        values = values_source.to(dtype=torch.long)[order]
        order = order[torch.argsort(values, stable=True)]
    return order


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
    extended["interaction_position"] = (
        interaction_position
        if interaction_position is not None
        else block.get("interaction_position", torch.zeros((count, 3), device=device, dtype=torch.float32))
    ).to(dtype=torch.float32).contiguous()
    extended["interaction_normal"] = (
        interaction_normal
        if interaction_normal is not None
        else block.get("interaction_normal", torch.zeros((count, 3), device=device, dtype=torch.float32))
    ).to(dtype=torch.float32).contiguous()
    extended["material_id"] = (
        material_id
        if material_id is not None
        else block.get("material_id", torch.full((count,), -1, device=device, dtype=torch.int32))
    ).to(dtype=torch.int32).contiguous()
    extended["path_field"] = (
        path_field
        if path_field is not None
        else block.get("path_field", torch.zeros((count,), device=device, dtype=torch.complex64))
    ).to(dtype=torch.complex64).contiguous()
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
    primitive = torch.full((count, width), -1, device=device, dtype=torch.int32)
    material = torch.full((count, width), -1, device=device, dtype=torch.int32)
    positions = torch.zeros((count, width, 3), device=device, dtype=torch.float32)
    normals = torch.zeros((count, width, 3), device=device, dtype=torch.float32)

    if "primitive_sequence" in block and width > 0:
        source = block["primitive_sequence"].to(device=device, dtype=torch.int32)
        cols = min(width, int(source.shape[1]))
        primitive[:, :cols] = source[:, :cols]
    elif width > 0:
        depth_one = block["depth"].to(dtype=torch.long) > 0
        primitive[depth_one, 0] = block["primitive_id"][depth_one].to(dtype=torch.int32)

    if "material_sequence" in block and width > 0:
        source = block["material_sequence"].to(device=device, dtype=torch.int32)
        cols = min(width, int(source.shape[1]))
        material[:, :cols] = source[:, :cols]
    elif width > 0:
        depth_one = block["depth"].to(dtype=torch.long) > 0
        material[depth_one, 0] = block["material_id"][depth_one].to(dtype=torch.int32)

    if "interaction_positions" in block and width > 0:
        source = block["interaction_positions"].to(device=device, dtype=torch.float32)
        cols = min(width, int(source.shape[1]))
        positions[:, :cols, :] = source[:, :cols, :]
    elif width > 0:
        depth_one = block["depth"].to(dtype=torch.long) > 0
        positions[depth_one, 0, :] = block["interaction_position"][depth_one].to(dtype=torch.float32)

    if "interaction_normals" in block and width > 0:
        source = block["interaction_normals"].to(device=device, dtype=torch.float32)
        cols = min(width, int(source.shape[1]))
        normals[:, :cols, :] = source[:, :cols, :]
    elif width > 0:
        depth_one = block["depth"].to(dtype=torch.long) > 0
        normals[depth_one, 0, :] = block["interaction_normal"][depth_one].to(dtype=torch.float32)

    padded = dict(block)
    padded["primitive_sequence"] = primitive.contiguous()
    padded["material_sequence"] = material.contiguous()
    padded["interaction_positions"] = positions.contiguous()
    padded["interaction_normals"] = normals.contiguous()
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
    if max_paths is not None:
        order = order[: int(max_paths)]
    selected = {key: value[order].contiguous() for key, value in paths.items()}
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
    if not raydn.available or not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
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

    records = raydn.edge_records()
    vertices = records.vertices
    faces = records.faces.to(dtype=torch.long)
    normals = torch.nn.functional.normalize(records.face_normals, dim=1, eps=1.0e-6)
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

    tri_a = vertices[faces[:, 0]]
    face_eps_r, face_sigma_e, face_mu_r, face_gain, _face_valid = face_material_tensors(compiled, device=device)
    face_material_id = compiled.assignments.face_material_id.to(device=device, dtype=torch.int32).contiguous()
    face_ids = torch.arange(int(faces.shape[0]), device=device, dtype=torch.int32)

    grouped_export = int(faces.shape[0]) > _ORDER1_GROUPED_FACE_THRESHOLD
    groups = _surface_face_groups(compiled.geometry.face_surface_id.to(device=device, dtype=torch.long).contiguous())
    if grouped_export:
        sequences = groups["representative_faces"].reshape(-1, 1).contiguous()
    else:
        sequences = torch.arange(int(faces.shape[0]), device=device, dtype=torch.long).reshape(-1, 1).contiguous()

    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    sequence_count = int(sequences.shape[0])
    rx_count = int(rx_positions.shape[0])
    if sequence_count <= 0 or rx_count <= 0:
        return _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)), launch_count

    rx_chunk_size = max(1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // sequence_count)
    for rx_start in range(0, rx_count, rx_chunk_size):
        rx_end = min(rx_start + rx_chunk_size, rx_count)
        rx_chunk = rx_positions[rx_start:rx_end]
        rx_chunk_count = int(rx_chunk.shape[0])
        pair_count = rx_chunk_count * sequence_count
        rx_indices = torch.arange(rx_start, rx_end, device=device, dtype=torch.int32).repeat_interleave(sequence_count)
        rx_batch = rx_chunk.repeat_interleave(sequence_count, dim=0).contiguous()
        sequence_batch = sequences.repeat(rx_chunk_count, 1).contiguous()
        for tx_index, tx in enumerate(tx_positions):
            tx_batch = tx.reshape(1, 3).expand(pair_count, 3).contiguous()
            epc = torch.ops.raydn.reflection_epc_paths_forward(
                raydn.require_handle(),
                tx_batch,
                rx_batch,
                None,
                sequence_batch.to(dtype=torch.int32).contiguous(),
                tri_a[sequence_batch].contiguous(),
                normals[sequence_batch].contiguous(),
                groups["surface_group_id"],
                groups["surface_group_size"],
                groups["surface_group_members"],
                1,
                1,
            )
            launch_count += 1
            keep = epc[0].nonzero(as_tuple=False).flatten()
            if int(keep.numel()) == 0:
                continue

            epc_faces = epc[2][keep, 0].to(dtype=torch.long)
            selected_faces = epc_faces if grouped_export else sequence_batch[keep, 0].to(dtype=torch.long)
            selected_points = epc[4][keep, 0, :].to(dtype=torch.float32).contiguous()
            selected_normals = epc[5][keep, 0, :].to(dtype=torch.float32).contiguous()
            selected_rx_id = rx_indices[keep].to(dtype=torch.int32).contiguous()
            tx_keep = tx.reshape(1, 3).expand(int(keep.numel()), 3).contiguous()
            rx_keep = rx_positions.index_select(0, selected_rx_id.to(dtype=torch.long)).contiguous()
            path_length = (
                torch.linalg.vector_norm(selected_points - tx_keep, dim=1)
                + torch.linalg.vector_norm(rx_keep - selected_points, dim=1)
            ).clamp_min(1.0e-6)
            field_result = ops.deterministic_reflection_field(
                tx_position=tx_keep,
                rx_position=rx_keep,
                hit_position=selected_points,
                normal=selected_normals,
                tx_power=tx_power[tx_index].expand_as(path_length).contiguous(),
                eps_r=face_eps_r[selected_faces].contiguous(),
                sigma_e=face_sigma_e[selected_faces].contiguous(),
                mu_r=face_mu_r[selected_faces].contiguous(),
                gain=face_gain[selected_faces].contiguous(),
                frequency_hz=frequency_hz,
            )
            path_gain = field_result["path_gain"].to(dtype=torch.float32).contiguous()
            path_field = torch.complex(field_result["field_real"], field_result["field_imag"]).to(torch.complex64)
            count = int(path_length.shape[0])
            blocks.append(
                _ensure_topology_fields(
                    {
                            "valid": torch.ones((count,), device=device, dtype=torch.bool),
                            "tx_id": torch.full((count,), tx_index, device=device, dtype=torch.int32),
                            "rx_id": selected_rx_id,
                            "depth": torch.ones((count,), device=device, dtype=torch.int32),
                            "component_id": torch.full((count,), 1, device=device, dtype=torch.int32),
                            "primitive_id": face_ids[selected_faces].contiguous(),
                        "edge_id": torch.full((count,), -1, device=device, dtype=torch.int32),
                        "path_length_m": path_length.to(dtype=torch.float32).contiguous(),
                        "delay_s": (path_length / _LIGHT_SPEED_M_PER_S).to(dtype=torch.float32).contiguous(),
                        "path_gain": path_gain.to(dtype=torch.float32).contiguous(),
                    },
                    interaction_position=selected_points,
                    interaction_normal=selected_normals,
                    material_id=face_material_id[selected_faces],
                    path_field=path_field,
                )
            )
    return _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)), launch_count


def _reflect_points(points: torch.Tensor, plane_points: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    distance = ((points - plane_points) * normals).sum(dim=1, keepdim=True)
    return points - 2.0 * distance * normals


def _face_sequence_chunks(face_count: int, depth: int, *, chunk_size: int, device: torch.device) -> object:
    total = face_count ** depth
    powers = torch.tensor(
        [face_count ** power for power in reversed(range(depth))],
        device=device,
        dtype=torch.long,
    )
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        flat = torch.arange(start, end, device=device, dtype=torch.long)
        sequences = torch.div(flat[:, None], powers[None, :], rounding_mode="floor") % face_count
        if depth > 1:
            adjacent_distinct = (sequences[:, 1:] != sequences[:, :-1]).all(dim=1)
            sequences = sequences[adjacent_distinct]
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
    if not raydn.available or not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
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

    records = raydn.edge_records()
    vertices = records.vertices
    faces = records.faces.to(dtype=torch.long)
    normals = torch.nn.functional.normalize(records.face_normals, dim=1, eps=1.0e-6)
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

    tri_a = vertices[faces[:, 0]]
    tri_b = vertices[faces[:, 1]]
    tri_c = vertices[faces[:, 2]]
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
        candidate_count = group_count ** depth
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
        for group_sequences in _face_sequence_chunks(group_count, depth, chunk_size=chunk_size, device=device):
            if max_paths is not None and emitted_count >= int(max_paths):
                break
            sequences = representative_faces[group_sequences]
            sequence_count = int(sequences.shape[0])
            if sequence_count <= 0:
                continue
            rx_chunk_size = max(1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // sequence_count)
            for rx_start in range(0, int(rx_positions.shape[0]), rx_chunk_size):
                if max_paths is not None and emitted_count >= int(max_paths):
                    break
                rx_end = min(rx_start + rx_chunk_size, int(rx_positions.shape[0]))
                rx_chunk = rx_positions[rx_start:rx_end]
                rx_chunk_count = int(rx_chunk.shape[0])
                pair_count = rx_chunk_count * sequence_count
                rx_indices = torch.arange(rx_start, rx_end, device=device, dtype=torch.int32).repeat_interleave(
                    sequence_count
                )
                rx_batch = rx_chunk.repeat_interleave(sequence_count, dim=0).contiguous()
                sequence_batch = sequences.repeat(rx_chunk_count, 1).contiguous()
                for tx_index, tx in enumerate(tx_positions):
                    if max_paths is not None and emitted_count >= int(max_paths):
                        break
                    tx_batch = tx.reshape(1, 3).expand(pair_count, 3).contiguous()
                    epc = torch.ops.raydn.reflection_epc_paths_forward(
                        raydn.require_handle(),
                        tx_batch,
                        rx_batch,
                        None,
                        sequence_batch.to(dtype=torch.int32).contiguous(),
                        tri_a[sequence_batch].contiguous(),
                        normals[sequence_batch].contiguous(),
                        surface_group_id,
                        surface_group_size,
                        surface_group_members,
                        int(depth),
                        1,
                    )
                    launch_count += 1
                    visible = epc[0]
                    keep = visible.nonzero(as_tuple=False).flatten()
                    if int(keep.numel()) == 0:
                        continue
                    if max_paths is not None:
                        remaining = int(max_paths) - emitted_count
                        keep = keep[:remaining]

                    selected_sequences = epc[2][keep].to(dtype=torch.long)
                    selected_hits = epc[4][keep].to(dtype=torch.float32).contiguous()
                    selected_normals = epc[5][keep].to(dtype=torch.float32).contiguous()
                    selected_tx = tx.reshape(1, 3).expand(int(keep.numel()), 3)
                    selected_rx_id = rx_indices[keep].to(dtype=torch.int32).contiguous()
                    selected_rx = rx_positions.index_select(0, selected_rx_id.to(dtype=torch.long))
                    field_result = ops.deterministic_reflection_sequence_field(
                        tx_position=selected_tx.contiguous(),
                        rx_position=selected_rx.contiguous(),
                        hit_positions=selected_hits.contiguous(),
                        normals=selected_normals.contiguous(),
                        tx_power=tx_power[tx_index].expand(int(keep.numel())).contiguous(),
                        eps_r=face_eps_r[selected_sequences].contiguous(),
                        sigma_e=face_sigma_e[selected_sequences].contiguous(),
                        mu_r=face_mu_r[selected_sequences].contiguous(),
                        gain=face_gain[selected_sequences].contiguous(),
                        frequency_hz=frequency_hz,
                    )
                    path_gain = field_result["path_gain"].to(dtype=torch.float32).contiguous()
                    path_field = torch.complex(field_result["field_real"], field_result["field_imag"]).to(torch.complex64)
                    path_length = field_result["path_length_m"].to(dtype=torch.float32).contiguous()
                    count = int(keep.numel())
                    emitted_count += count
                    first_face = selected_sequences[:, 0]
                    blocks.append(
                        _ensure_topology_fields(
                            {
                                "valid": torch.ones((count,), device=device, dtype=torch.bool),
                                "tx_id": torch.full((count,), tx_index, device=device, dtype=torch.int32),
                                "rx_id": selected_rx_id,
                                "depth": torch.full((count,), depth, device=device, dtype=torch.int32),
                                "component_id": torch.full((count,), 1, device=device, dtype=torch.int32),
                                "primitive_id": first_face.to(dtype=torch.int32).contiguous(),
                                "edge_id": torch.full((count,), -1, device=device, dtype=torch.int32),
                                "path_length_m": path_length,
                                "delay_s": (path_length / _LIGHT_SPEED_M_PER_S).to(dtype=torch.float32).contiguous(),
                                "path_gain": path_gain,
                            },
                            interaction_position=selected_hits[:, 0, :],
                            interaction_normal=selected_normals[:, 0, :],
                            material_id=face_material_id[first_face],
                            path_field=path_field,
                            primitive_sequence=selected_sequences.to(dtype=torch.int32),
                            material_sequence=face_material_id[selected_sequences],
                            interaction_positions=selected_hits,
                            interaction_normals=selected_normals,
                        )
                    )

    return _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)), launch_count, theoretical_candidate_count


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
    if not raydn.available or not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
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

    material_gain = face_material_tensors(compiled, device=device)[3]
    material_valid = torch.ones_like(material_gain, dtype=torch.bool)
    wavelength = _LIGHT_SPEED_M_PER_S / float(frequency_hz)
    handle = raydn.require_handle()
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    for tx_index, tx in enumerate(tx_positions):
        states = _diffraction_states(scene, raydn, tx, tx_power[tx_index])
        state_count = int(states[0].shape[0])
        if state_count <= 0:
            continue
        capacity = int(rx_positions.shape[0]) * state_count
        out = torch.ops.raydn.diffraction_paths_order1_forward(
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
        valid = out[1]
        idx = valid.nonzero(as_tuple=False).flatten()
        if int(idx.numel()) == 0:
            continue
        field_result = ops.deterministic_diffraction_vector_field(
            x_re=out[9][idx].to(dtype=torch.float32).contiguous(),
            x_im=out[10][idx].to(dtype=torch.float32).contiguous(),
            y_re=out[11][idx].to(dtype=torch.float32).contiguous(),
            y_im=out[12][idx].to(dtype=torch.float32).contiguous(),
            z_re=out[13][idx].to(dtype=torch.float32).contiguous(),
            z_im=out[14][idx].to(dtype=torch.float32).contiguous(),
        )
        field_power = field_result["path_gain"].to(dtype=torch.float32).contiguous()
        path_field = torch.complex(field_result["field_real"], field_result["field_imag"]).to(torch.complex64)
        delay = out[8][idx].to(dtype=torch.float32).contiguous()
        path_length = (delay * _LIGHT_SPEED_M_PER_S).to(dtype=torch.float32).contiguous()
        path_count = int(idx.numel())
        blocks.append(
            _ensure_topology_fields(
                {
                    "valid": torch.ones((path_count,), device=device, dtype=torch.bool),
                    "tx_id": torch.full((path_count,), tx_index, device=device, dtype=torch.int32),
                    "rx_id": out[3][idx].to(dtype=torch.int32).contiguous(),
                    "depth": out[4][idx].to(dtype=torch.int32).contiguous(),
                    "component_id": torch.full((path_count,), 2, device=device, dtype=torch.int32),
                    "primitive_id": torch.full((path_count,), -1, device=device, dtype=torch.int32),
                    "edge_id": out[5][idx].to(dtype=torch.int32).contiguous(),
                    "path_length_m": path_length,
                    "delay_s": delay,
                    "path_gain": field_power.to(dtype=torch.float32).contiguous(),
                },
                interaction_position=out[15][idx],
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
        tx_for_path = tx_positions.index_select(0, tx_id.to(dtype=torch.long))
        rx_for_path = rx_positions.index_select(0, rx_id.to(dtype=torch.long))
        visible = _los_visibility_mask(
            compiled.raydn,
            tx_for_path,
            rx_for_path,
            has_structures=bool(scene.structures),
        )
        if bool(scene.structures) and int(tx_id.numel()) > 0:
            launch_count += 1
        keep = visible.nonzero(as_tuple=False).flatten()
        candidate_count += int(tx_id.numel())
        visibility_rejection_count += int(tx_id.numel()) - int(keep.numel())
        los_path_gain = exported["path_gain"][keep].to(dtype=torch.float32).contiguous()
        los_path_length = exported["path_length_m"][keep].to(dtype=torch.float32).contiguous()
        field_result = ops.deterministic_los_field(
            path_gain=los_path_gain,
            path_length_m=los_path_length,
            frequency_hz=float(scene.frequency),
        )
        los_path_field = torch.complex(field_result["field_real"], field_result["field_imag"]).to(torch.complex64)
        blocks.append(
            _ensure_topology_fields(
                {
                    "valid": torch.ones((int(keep.numel()),), device=device, dtype=torch.bool),
                    "tx_id": tx_id[keep].to(dtype=torch.int32).contiguous(),
                    "rx_id": rx_id[keep].to(dtype=torch.int32).contiguous(),
                    "depth": torch.zeros((int(keep.numel()),), device=device, dtype=torch.int32),
                    "component_id": torch.zeros((int(keep.numel()),), device=device, dtype=torch.int32),
                    "primitive_id": torch.full((int(keep.numel()),), -1, device=device, dtype=torch.int32),
                    "edge_id": torch.full((int(keep.numel()),), -1, device=device, dtype=torch.int32),
                    "path_length_m": los_path_length,
                    "delay_s": exported["delay_s"][keep].to(dtype=torch.float32).contiguous(),
                    "path_gain": field_result["path_gain"].to(dtype=torch.float32).contiguous(),
                },
                path_field=los_path_field,
            )
        )
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
    sequence_width = max(int(config.max_depth), 0)
    padded_blocks = [_pad_topology_sequences(block, width=sequence_width) for block in blocks]
    paths = concatenate_path_blocks(padded_blocks, device=device)
    selected_edge_count = int(torch.unique(paths["edge_id"][paths["edge_id"] >= 0]).numel())
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
