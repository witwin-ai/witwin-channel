from __future__ import annotations

import torch

from witwin.channel.runtime.symbols import required_symbol as _required_native_op
from witwin.channel.runtime.tensor_contracts import validate_cuda_tensor

from .blocks import _validate_deterministic_topology_block, _validate_path_block


def deterministic_los_topology_block(
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    path_length_m: torch.Tensor,
    delay_s: torch.Tensor,
    path_gain: torch.Tensor,
    visible: torch.Tensor | None,
    *,
    frequency_hz: float,
    sequence_width: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx_id", tx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("path_length_m", path_length_m, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("delay_s", delay_s, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if sequence_width < 0:
        raise ValueError("sequence_width must be non-negative")
    for name, tensor in {
        "rx_id": rx_id,
        "path_length_m": path_length_m,
        "delay_s": delay_s,
        "path_gain": path_gain,
    }.items():
        if tensor.shape != tx_id.shape:
            raise ValueError(f"{name} must match tx_id")
    if visible is None:
        block = _required_native_op("deterministic_los_topology_block_all_visible")(
            tx_id,
            rx_id,
            path_length_m,
            delay_s,
            path_gain,
            float(frequency_hz),
            int(sequence_width),
        )
    else:
        validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
        if visible.shape != tx_id.shape:
            raise ValueError("visible must match tx_id")
        block = _required_native_op("deterministic_los_topology_block")(
            tx_id,
            rx_id,
            path_length_m,
            delay_s,
            path_gain,
            visible,
            float(frequency_hz),
            int(sequence_width),
        )
    if not isinstance(block, dict):
        raise TypeError(
            "_channel_native.deterministic_los_topology_block must return a dict"
        )
    _validate_deterministic_topology_block(
        "deterministic_los_topology_block", block, int(sequence_width)
    )
    return block


def deterministic_topology_default_fields(
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=1)
    exported = _required_native_op("deterministic_topology_default_fields")(reference)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_topology_default_fields must return a dict"
        )
    validate_cuda_tensor(
        "interaction_position",
        exported["interaction_position"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "interaction_normal",
        exported["interaction_normal"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "material_id", exported["material_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "path_field", exported["path_field"], dtype=torch.complex64, ndim=1
    )
    count = int(reference.shape[0])
    if (
        exported["interaction_position"].shape != (count, 3)
        or exported["interaction_normal"].shape != (count, 3)
        or exported["material_id"].shape != (count,)
        or exported["path_field"].shape != (count,)
    ):
        raise ValueError(
            "_channel_native.deterministic_topology_default_fields returned bad shape"
        )
    return exported


def deterministic_pad_topology_sequences(
    *,
    depth: torch.Tensor,
    primitive_id: torch.Tensor,
    material_id: torch.Tensor,
    interaction_position: torch.Tensor,
    interaction_normal: torch.Tensor,
    primitive_sequence: torch.Tensor,
    material_sequence: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    width: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("depth", depth, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("primitive_id", primitive_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("material_id", material_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "interaction_position",
        interaction_position,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "interaction_normal",
        interaction_normal,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "primitive_sequence", primitive_sequence, dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "material_sequence", material_sequence, dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "interaction_positions", interaction_positions, dtype=torch.float32, ndim=3
    )
    validate_cuda_tensor(
        "interaction_normals", interaction_normals, dtype=torch.float32, ndim=3
    )
    if width < 0:
        raise ValueError("width must be non-negative")
    count = int(depth.shape[0])
    for name, tensor in {
        "primitive_id": primitive_id,
        "material_id": material_id,
    }.items():
        if tensor.shape != (count,):
            raise ValueError(f"{name} must match depth")
    for name, tensor in {
        "interaction_position": interaction_position,
        "interaction_normal": interaction_normal,
    }.items():
        if tensor.shape != (count, 3):
            raise ValueError(f"{name} must have shape (N, 3)")
    for name, tensor in {
        "primitive_sequence": primitive_sequence,
        "material_sequence": material_sequence,
    }.items():
        if tensor.shape[0] != count:
            raise ValueError(f"{name} must share the path count")
    for name, tensor in {
        "interaction_positions": interaction_positions,
        "interaction_normals": interaction_normals,
    }.items():
        if tensor.shape[0] != count or tensor.shape[2] != 3:
            raise ValueError(f"{name} must have shape (N, D, 3)")
    exported = _required_native_op("deterministic_pad_topology_sequences")(
        depth,
        primitive_id,
        material_id,
        interaction_position,
        interaction_normal,
        primitive_sequence,
        material_sequence,
        interaction_positions,
        interaction_normals,
        int(width),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_pad_topology_sequences must return a dict"
        )
    validate_cuda_tensor(
        "primitive_sequence", exported["primitive_sequence"], dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "material_sequence", exported["material_sequence"], dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "interaction_positions",
        exported["interaction_positions"],
        dtype=torch.float32,
        ndim=3,
    )
    validate_cuda_tensor(
        "interaction_normals",
        exported["interaction_normals"],
        dtype=torch.float32,
        ndim=3,
    )
    expected_i32 = (count, int(width))
    expected_vec = (count, int(width), 3)
    if (
        exported["primitive_sequence"].shape != expected_i32
        or exported["material_sequence"].shape != expected_i32
    ):
        raise ValueError(
            "_channel_native.deterministic_pad_topology_sequences returned bad sequence shape"
        )
    if (
        exported["interaction_positions"].shape != expected_vec
        or exported["interaction_normals"].shape != expected_vec
    ):
        raise ValueError(
            "_channel_native.deterministic_pad_topology_sequences returned bad interaction shape"
        )
    return exported


def deterministic_topology_base_fields(
    *,
    rx_id: torch.Tensor,
    path_length_m: torch.Tensor,
    delay_s: torch.Tensor,
    path_gain: torch.Tensor,
    tx_index: int,
    component_id: int,
    depth_source: torch.Tensor,
    depth_value: int,
    primitive_source: torch.Tensor,
    primitive_value: int,
    edge_source: torch.Tensor,
    edge_value: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("path_length_m", path_length_m, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("delay_s", delay_s, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("depth_source", depth_source, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "primitive_source", primitive_source, dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor("edge_source", edge_source, dtype=torch.int32, ndim=1)
    count = int(rx_id.shape[0])
    for name, tensor in {
        "path_length_m": path_length_m,
        "delay_s": delay_s,
        "path_gain": path_gain,
    }.items():
        if tensor.shape != (count,):
            raise ValueError(f"{name} must match rx_id")
    for name, tensor in {
        "depth_source": depth_source,
        "primitive_source": primitive_source,
        "edge_source": edge_source,
    }.items():
        if tensor.shape not in {(0,), (count,)}:
            raise ValueError(f"{name} must be empty or match rx_id")
    block = _required_native_op("deterministic_topology_base_fields")(
        rx_id,
        path_length_m,
        delay_s,
        path_gain,
        int(tx_index),
        int(component_id),
        depth_source,
        int(depth_value),
        primitive_source,
        int(primitive_value),
        edge_source,
        int(edge_value),
    )
    if not isinstance(block, dict):
        raise TypeError(
            "_channel_native.deterministic_topology_base_fields must return a dict"
        )
    _validate_path_block("deterministic_topology_base_fields", block)
    return block


def deterministic_repeat_range(
    reference: torch.Tensor, *, start: int, end: int, repeats: int
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=1)
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    out = _required_native_op("deterministic_repeat_range")(
        reference, int(start), int(end), int(repeats)
    )
    validate_cuda_tensor("out", out, dtype=torch.int32, ndim=1)
    if out.shape != ((int(end) - int(start)) * int(repeats),):
        raise ValueError(
            "_channel_native.deterministic_repeat_range returned bad shape"
        )
    return out


def deterministic_face_anchor_points(
    vertices: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    validate_cuda_tensor(
        "vertices", vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("faces", faces, dtype=torch.int32, ndim=2, trailing_shape=(3,))
    if vertices.get_device() != faces.get_device():
        raise ValueError("vertices and faces must share a CUDA device")
    out = _required_native_op("deterministic_face_anchor_points")(vertices, faces)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.deterministic_face_anchor_points must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if out.shape != (faces.shape[0], 3):
        raise ValueError(
            "_channel_native.deterministic_face_anchor_points returned bad shape"
        )
    return out


def deterministic_reflection_epc_input_batch(
    *,
    tx: torch.Tensor,
    rx_positions: torch.Tensor,
    sequences: torch.Tensor,
    tri_a: torch.Tensor,
    normals: torch.Tensor,
    rx_start: int,
    rx_end: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    if tx.shape != (3,):
        raise ValueError("tx must have shape (3,)")
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "tri_a", tri_a, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "normals", normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if normals.shape != tri_a.shape:
        raise ValueError("normals must match tri_a")
    if sequences.dtype not in {torch.int32, torch.int64}:
        raise TypeError("sequences must have dtype torch.int32 or torch.int64")
    if not sequences.is_cuda:
        raise ValueError("sequences must be a CUDA tensor")
    if sequences.ndim != 2:
        raise ValueError("sequences must have shape (S, depth)")
    if not sequences.is_contiguous():
        raise ValueError("sequences must be contiguous")
    if rx_start < 0:
        raise ValueError("rx_start must be non-negative")
    if rx_end < rx_start:
        raise ValueError("rx_end must be greater than or equal to rx_start")
    if rx_end > int(rx_positions.shape[0]):
        raise ValueError("rx_end is out of range")
    exported = _required_native_op("deterministic_reflection_epc_input_batch")(
        tx,
        rx_positions,
        sequences,
        tri_a,
        normals,
        int(rx_start),
        int(rx_end),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_reflection_epc_input_batch must return a dict"
        )
    expected_fields = {
        "tx_batch",
        "rx_batch",
        "rx_indices",
        "sequence_batch",
        "direct_plane_points",
        "direct_plane_normals",
    }
    if set(exported) != expected_fields:
        raise ValueError(
            "_channel_native.deterministic_reflection_epc_input_batch returned unexpected fields"
        )
    pair_count = (int(rx_end) - int(rx_start)) * int(sequences.shape[0])
    depth = int(sequences.shape[1])
    validate_cuda_tensor(
        "tx_batch",
        exported["tx_batch"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "rx_batch",
        exported["rx_batch"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "rx_indices", exported["rx_indices"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "sequence_batch", exported["sequence_batch"], dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "direct_plane_points",
        exported["direct_plane_points"],
        dtype=torch.float32,
        ndim=3,
    )
    validate_cuda_tensor(
        "direct_plane_normals",
        exported["direct_plane_normals"],
        dtype=torch.float32,
        ndim=3,
    )
    if exported["tx_batch"].shape != (pair_count, 3):
        raise ValueError(
            "_channel_native.deterministic_reflection_epc_input_batch returned bad tx_batch shape"
        )
    if exported["rx_batch"].shape != (pair_count, 3):
        raise ValueError(
            "_channel_native.deterministic_reflection_epc_input_batch returned bad rx_batch shape"
        )
    if exported["rx_indices"].shape != (pair_count,):
        raise ValueError(
            "_channel_native.deterministic_reflection_epc_input_batch returned bad rx_indices shape"
        )
    if exported["sequence_batch"].shape != (pair_count, depth):
        raise ValueError(
            "_channel_native.deterministic_reflection_epc_input_batch returned bad sequence_batch shape"
        )
    if exported["direct_plane_points"].shape != (pair_count, depth, 3):
        raise ValueError(
            "_channel_native.deterministic_reflection_epc_input_batch returned bad direct_plane_points shape"
        )
    if exported["direct_plane_normals"].shape != (pair_count, depth, 3):
        raise ValueError(
            "_channel_native.deterministic_reflection_epc_input_batch returned bad direct_plane_normals shape"
        )
    return exported


def deterministic_face_sequence_chunk(
    reference: torch.Tensor,
    *,
    face_count: int,
    depth: int,
    start: int,
    end: int,
    adjacent_distinct: bool = False,
) -> torch.Tensor:
    if not isinstance(reference, torch.Tensor):
        raise TypeError("reference must be a torch.Tensor")
    if not reference.is_cuda:
        raise ValueError("reference must be a CUDA tensor")
    if face_count <= 0:
        raise ValueError("face_count must be positive")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    out = _required_native_op("deterministic_face_sequence_chunk")(
        reference,
        int(face_count),
        int(depth),
        int(start),
        int(end),
        bool(adjacent_distinct),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.deterministic_face_sequence_chunk must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.int32, ndim=2)
    if out.shape != (int(end) - int(start), int(depth)):
        raise ValueError(
            "_channel_native.deterministic_face_sequence_chunk returned bad shape"
        )
    return out


def deterministic_mapped_face_sequence_chunk(
    face_ids: torch.Tensor,
    *,
    depth: int,
    start: int,
    end: int,
    adjacent_distinct: bool = False,
) -> torch.Tensor:
    if not isinstance(face_ids, torch.Tensor):
        raise TypeError("face_ids must be a torch.Tensor")
    if face_ids.dtype not in {torch.int32, torch.int64}:
        raise TypeError("face_ids must have dtype torch.int32 or torch.int64")
    if not face_ids.is_cuda:
        raise ValueError("face_ids must be a CUDA tensor")
    if face_ids.ndim != 1:
        raise ValueError("face_ids must be one-dimensional")
    if not face_ids.is_contiguous():
        raise ValueError("face_ids must be contiguous")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    out = _required_native_op("deterministic_mapped_face_sequence_chunk")(
        face_ids,
        int(depth),
        int(start),
        int(end),
        bool(adjacent_distinct),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.deterministic_mapped_face_sequence_chunk must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.int32, ndim=2)
    if out.shape != (int(end) - int(start), int(depth)):
        raise ValueError(
            "_channel_native.deterministic_mapped_face_sequence_chunk returned bad shape"
        )
    return out
