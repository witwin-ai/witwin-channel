from __future__ import annotations

import torch

from witwin.channel.runtime import (
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)


def deterministic_reflection_order1_compact(
    *,
    visible: torch.Tensor,
    epc_faces: torch.Tensor,
    epc_hits: torch.Tensor,
    epc_normals: torch.Tensor,
    sequence_batch: torch.Tensor,
    rx_indices: torch.Tensor,
    tx: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    tx_index: int,
    face_eps_r: torch.Tensor,
    face_sigma_e: torch.Tensor,
    face_mu_r: torch.Tensor,
    face_gain: torch.Tensor,
    face_material_id: torch.Tensor,
    grouped_export: bool,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if epc_faces.dtype not in {torch.int32, torch.int64}:
        raise TypeError("epc_faces must have dtype torch.int32 or torch.int64")
    if not epc_faces.is_cuda or epc_faces.ndim != 2 or not epc_faces.is_contiguous():
        raise ValueError(
            "epc_faces must be a contiguous CUDA tensor with shape (N, depth)"
        )
    validate_cuda_tensor("epc_hits", epc_hits, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("epc_normals", epc_normals, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("sequence_batch", sequence_batch, dtype=torch.int32, ndim=2)
    validate_cuda_tensor("rx_indices", rx_indices, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    if tx.shape != (3,):
        raise ValueError("tx must have shape (3,)")
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_eps_r", face_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_sigma_e", face_sigma_e, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_mu_r", face_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_gain", face_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    count = int(visible.shape[0])
    if epc_faces.shape[0] != count or epc_faces.shape[1] < 1:
        raise ValueError("epc_faces must match visible and include the first bounce")
    if epc_hits.shape[0] != count or epc_hits.shape[1] < 1 or epc_hits.shape[2] != 3:
        raise ValueError("epc_hits must have shape (N, depth, 3)")
    if epc_normals.shape != epc_hits.shape:
        raise ValueError("epc_normals must match epc_hits")
    if sequence_batch.shape[0] != count or sequence_batch.shape[1] < 1:
        raise ValueError(
            "sequence_batch must match visible and include the first bounce"
        )
    if rx_indices.shape != (count,):
        raise ValueError("rx_indices must match visible")
    if not 0 <= int(tx_index) < int(tx_power.shape[0]):
        raise ValueError("tx_index is out of range")
    if (
        face_sigma_e.shape != face_eps_r.shape
        or face_mu_r.shape != face_eps_r.shape
        or face_gain.shape != face_eps_r.shape
        or face_material_id.shape != face_eps_r.shape
    ):
        raise ValueError("face material tensors must share shape")
    exported = _required_native_op("deterministic_reflection_order1_compact")(
        visible,
        epc_faces,
        epc_hits,
        epc_normals,
        sequence_batch,
        rx_indices,
        tx,
        rx_positions,
        tx_power,
        int(tx_index),
        face_eps_r,
        face_sigma_e,
        face_mu_r,
        face_gain,
        face_material_id,
        bool(grouped_export),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_reflection_order1_compact must return a dict"
        )
    expected_fields = {
        "selected_faces",
        "selected_points",
        "selected_normals",
        "selected_rx_id",
        "tx_keep",
        "rx_keep",
        "tx_power",
        "eps_r",
        "sigma_e",
        "mu_r",
        "gain",
        "material_id",
    }
    if set(exported) != expected_fields:
        raise ValueError(
            "_channel.deterministic_reflection_order1_compact returned unexpected fields"
        )
    selected_count = int(exported["selected_faces"].shape[0])
    validate_cuda_tensor(
        "selected_faces", exported["selected_faces"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "selected_points",
        exported["selected_points"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "selected_normals",
        exported["selected_normals"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "selected_rx_id", exported["selected_rx_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "tx_keep", exported["tx_keep"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_keep", exported["rx_keep"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", exported["tx_power"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("eps_r", exported["eps_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("sigma_e", exported["sigma_e"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("mu_r", exported["mu_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("gain", exported["gain"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "material_id", exported["material_id"], dtype=torch.int32, ndim=1
    )
    for key in expected_fields - {"selected_faces"}:
        if exported[key].shape[0] != selected_count:
            raise ValueError(
                f"_channel.deterministic_reflection_order1_compact returned bad {key} shape"
            )
    return exported


def deterministic_reflection_sequence_compact(
    *,
    visible: torch.Tensor,
    epc_sequences: torch.Tensor,
    epc_hits: torch.Tensor,
    epc_normals: torch.Tensor,
    rx_indices: torch.Tensor,
    tx: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    tx_index: int,
    face_eps_r: torch.Tensor,
    face_sigma_e: torch.Tensor,
    face_mu_r: torch.Tensor,
    face_gain: torch.Tensor,
    face_material_id: torch.Tensor,
    max_count: int = -1,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if epc_sequences.dtype not in {torch.int32, torch.int64}:
        raise TypeError("epc_sequences must have dtype torch.int32 or torch.int64")
    if (
        not epc_sequences.is_cuda
        or epc_sequences.ndim != 2
        or not epc_sequences.is_contiguous()
    ):
        raise ValueError(
            "epc_sequences must be a contiguous CUDA tensor with shape (N, depth)"
        )
    validate_cuda_tensor("epc_hits", epc_hits, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("epc_normals", epc_normals, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("rx_indices", rx_indices, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    if tx.shape != (3,):
        raise ValueError("tx must have shape (3,)")
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_eps_r", face_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_sigma_e", face_sigma_e, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_mu_r", face_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_gain", face_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    if max_count < -1:
        raise ValueError("max_count must be -1 or non-negative")
    count = int(visible.shape[0])
    depth = int(epc_sequences.shape[1])
    if depth <= 0 or epc_sequences.shape[0] != count:
        raise ValueError("epc_sequences must match visible and have positive depth")
    if epc_hits.shape != (count, depth, 3):
        raise ValueError("epc_hits must have shape (N, depth, 3)")
    if epc_normals.shape != epc_hits.shape:
        raise ValueError("epc_normals must match epc_hits")
    if rx_indices.shape != (count,):
        raise ValueError("rx_indices must match visible")
    if not 0 <= int(tx_index) < int(tx_power.shape[0]):
        raise ValueError("tx_index is out of range")
    if (
        face_sigma_e.shape != face_eps_r.shape
        or face_mu_r.shape != face_eps_r.shape
        or face_gain.shape != face_eps_r.shape
        or face_material_id.shape != face_eps_r.shape
    ):
        raise ValueError("face material tensors must share shape")
    exported = _required_native_op("deterministic_reflection_sequence_compact")(
        visible,
        epc_sequences,
        epc_hits,
        epc_normals,
        rx_indices,
        tx,
        rx_positions,
        tx_power,
        int(tx_index),
        face_eps_r,
        face_sigma_e,
        face_mu_r,
        face_gain,
        face_material_id,
        int(max_count),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_reflection_sequence_compact must return a dict"
        )
    expected_fields = {
        "selected_sequences",
        "selected_hits",
        "selected_normals",
        "selected_rx_id",
        "selected_tx",
        "selected_rx",
        "tx_power",
        "eps_r",
        "sigma_e",
        "mu_r",
        "gain",
        "first_face",
        "material_id",
        "material_sequence",
        "first_hit",
        "first_normal",
    }
    if set(exported) != expected_fields:
        raise ValueError(
            "_channel.deterministic_reflection_sequence_compact returned unexpected fields"
        )
    selected_count = int(exported["selected_sequences"].shape[0])
    validate_cuda_tensor(
        "selected_sequences", exported["selected_sequences"], dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "selected_hits", exported["selected_hits"], dtype=torch.float32, ndim=3
    )
    validate_cuda_tensor(
        "selected_normals", exported["selected_normals"], dtype=torch.float32, ndim=3
    )
    validate_cuda_tensor(
        "selected_rx_id", exported["selected_rx_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "selected_tx",
        exported["selected_tx"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "selected_rx",
        exported["selected_rx"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor("tx_power", exported["tx_power"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("eps_r", exported["eps_r"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("sigma_e", exported["sigma_e"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("mu_r", exported["mu_r"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("gain", exported["gain"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor(
        "first_face", exported["first_face"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "material_id", exported["material_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "material_sequence", exported["material_sequence"], dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "first_hit",
        exported["first_hit"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "first_normal",
        exported["first_normal"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    expected_shapes = {
        "selected_sequences": (selected_count, depth),
        "selected_hits": (selected_count, depth, 3),
        "selected_normals": (selected_count, depth, 3),
        "selected_rx_id": (selected_count,),
        "selected_tx": (selected_count, 3),
        "selected_rx": (selected_count, 3),
        "tx_power": (selected_count,),
        "eps_r": (selected_count, depth),
        "sigma_e": (selected_count, depth),
        "mu_r": (selected_count, depth),
        "gain": (selected_count, depth),
        "first_face": (selected_count,),
        "material_id": (selected_count,),
        "material_sequence": (selected_count, depth),
        "first_hit": (selected_count, 3),
        "first_normal": (selected_count, 3),
    }
    for key, shape in expected_shapes.items():
        if tuple(exported[key].shape) != shape:
            raise ValueError(
                f"_channel.deterministic_reflection_sequence_compact returned bad {key} shape"
            )
    return exported


def deterministic_diffraction_order1_compact(
    *,
    valid: torch.Tensor,
    rx_id: torch.Tensor,
    depth: torch.Tensor,
    edge_id: torch.Tensor,
    delay_s: torch.Tensor,
    x_re: torch.Tensor,
    x_im: torch.Tensor,
    y_re: torch.Tensor,
    y_im: torch.Tensor,
    z_re: torch.Tensor,
    z_im: torch.Tensor,
    interaction_position: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=1)
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("depth", depth, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("edge_id", edge_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("delay_s", delay_s, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("x_re", x_re, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("x_im", x_im, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y_re", y_re, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y_im", y_im, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z_re", z_re, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z_im", z_im, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "interaction_position",
        interaction_position,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    count = int(valid.shape[0])
    for name, tensor in {
        "rx_id": rx_id,
        "depth": depth,
        "edge_id": edge_id,
        "delay_s": delay_s,
        "x_re": x_re,
        "x_im": x_im,
        "y_re": y_re,
        "y_im": y_im,
        "z_re": z_re,
        "z_im": z_im,
    }.items():
        if tensor.shape != (count,):
            raise ValueError(f"{name} must match valid")
    if interaction_position.shape != (count, 3):
        raise ValueError("interaction_position must match valid")

    exported = _required_native_op("deterministic_diffraction_order1_compact")(
        valid,
        rx_id,
        depth,
        edge_id,
        delay_s,
        x_re,
        x_im,
        y_re,
        y_im,
        z_re,
        z_im,
        interaction_position,
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_diffraction_order1_compact must return a dict"
        )
    expected_fields = {
        "rx_id",
        "depth",
        "edge_id",
        "delay_s",
        "x_re",
        "x_im",
        "y_re",
        "y_im",
        "z_re",
        "z_im",
        "interaction_position",
    }
    if set(exported) != expected_fields:
        raise ValueError(
            "_channel.deterministic_diffraction_order1_compact returned unexpected fields"
        )
    selected_count = int(exported["rx_id"].shape[0])
    validate_cuda_tensor("rx_id", exported["rx_id"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("depth", exported["depth"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("edge_id", exported["edge_id"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("delay_s", exported["delay_s"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("x_re", exported["x_re"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("x_im", exported["x_im"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y_re", exported["y_re"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y_im", exported["y_im"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z_re", exported["z_re"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z_im", exported["z_im"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "interaction_position",
        exported["interaction_position"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    for key in expected_fields - {"interaction_position"}:
        if exported[key].shape != (selected_count,):
            raise ValueError(
                f"_channel.deterministic_diffraction_order1_compact returned bad {key} shape"
            )
    if exported["interaction_position"].shape != (selected_count, 3):
        raise ValueError(
            "_channel.deterministic_diffraction_order1_compact returned bad interaction_position shape"
        )
    return exported


def deterministic_sort_order(
    valid: torch.Tensor,
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    depth: torch.Tensor,
    component_id: torch.Tensor,
    primitive_id: torch.Tensor,
    edge_id: torch.Tensor,
    primitive_sequence: torch.Tensor,
) -> torch.Tensor:
    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=1)
    for name, tensor in (
        ("tx_id", tx_id),
        ("rx_id", rx_id),
        ("depth", depth),
        ("component_id", component_id),
        ("primitive_id", primitive_id),
        ("edge_id", edge_id),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.int32, ndim=1)
        if tensor.shape != valid.shape:
            raise ValueError(f"{name} must match valid")
    validate_cuda_tensor(
        "primitive_sequence", primitive_sequence, dtype=torch.int32, ndim=2
    )
    if primitive_sequence.shape[0] != valid.shape[0]:
        raise ValueError("primitive_sequence must match valid rows")
    out = _required_native_op("deterministic_sort_order")(
        valid,
        tx_id,
        rx_id,
        depth,
        component_id,
        primitive_id,
        edge_id,
        primitive_sequence,
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel.deterministic_sort_order must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.long, ndim=1)
    if out.shape != valid.shape:
        raise ValueError("_channel.deterministic_sort_order returned bad shape")
    return out
