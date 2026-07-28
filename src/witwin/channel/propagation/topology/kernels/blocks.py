from __future__ import annotations

import torch

from witwin.channel.runtime import (
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)


def path_los_export(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_polarizations: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "tx_polarizations",
        tx_polarizations,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must have one value per transmitter")
    if tx_polarizations.shape != tx_positions.shape:
        raise ValueError("tx_polarizations must match tx_positions shape")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    exported = _required_native_op("path_los_export")(
        tx_positions, tx_power, rx_positions, float(frequency_hz), tx_polarizations
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.path_los_export must return a dict")
    return exported


_PATH_BLOCK_SCHEMA: tuple[tuple[str, torch.dtype], ...] = (
    ("valid", torch.bool),
    ("tx_id", torch.int32),
    ("rx_id", torch.int32),
    ("depth", torch.int32),
    ("component_id", torch.int32),
    ("primitive_id", torch.int32),
    ("edge_id", torch.int32),
    ("path_length_m", torch.float32),
    ("delay_s", torch.float32),
    ("path_gain", torch.float32),
)


_DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA: tuple[tuple[str, torch.dtype], ...] = (
    ("path_field", torch.complex64),
    ("interaction_position", torch.float32),
    ("interaction_normal", torch.float32),
    ("material_id", torch.int32),
    ("primitive_sequence", torch.int32),
    ("material_sequence", torch.int32),
    ("interaction_positions", torch.float32),
    ("interaction_normals", torch.float32),
)


def _validate_path_block(name: str, block: dict[str, torch.Tensor]) -> None:
    if not isinstance(block, dict):
        raise TypeError(f"{name} must be a dict")
    expected_shape: tuple[int, ...] | None = None
    for key, dtype in _PATH_BLOCK_SCHEMA:
        tensor = block.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}.{key} must be a torch.Tensor")
        validate_cuda_tensor(f"{name}.{key}", tensor, dtype=dtype, ndim=1)
        if expected_shape is None:
            expected_shape = tuple(tensor.shape)
        elif tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name}.{key} must share the path count")


def _validate_deterministic_topology_block(
    name: str, block: dict[str, torch.Tensor], sequence_width: int
) -> None:
    _validate_path_block(name, block)
    _validate_topology_extra_fields(
        name,
        block,
        int(sequence_width),
        {key: True for key, _dtype in _DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA},
    )


def _validate_topology_extra_fields(
    name: str,
    block: dict[str, torch.Tensor],
    sequence_width: int,
    expected_presence: dict[str, bool],
) -> None:
    path_count = int(block["valid"].shape[0])
    for key, dtype in _DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA:
        present = key in block
        if present != expected_presence[key]:
            state = "include" if expected_presence[key] else "omit"
            raise TypeError(f"{name}.{key} must {state} the concat schema")
        if not present:
            continue
        tensor = block[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}.{key} must be a torch.Tensor")
        if key in {"interaction_position", "interaction_normal"}:
            validate_cuda_tensor(
                f"{name}.{key}", tensor, dtype=dtype, ndim=2, trailing_shape=(3,)
            )
            expected = (path_count, 3)
        elif key in {"primitive_sequence", "material_sequence"}:
            validate_cuda_tensor(f"{name}.{key}", tensor, dtype=dtype, ndim=2)
            expected = (path_count, sequence_width)
        elif key in {"interaction_positions", "interaction_normals"}:
            validate_cuda_tensor(f"{name}.{key}", tensor, dtype=dtype, ndim=3)
            expected = (path_count, sequence_width, 3)
        else:
            validate_cuda_tensor(f"{name}.{key}", tensor, dtype=dtype, ndim=1)
            expected = (path_count,)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name}.{key} must have shape {expected}")


def deterministic_concat_topology_blocks(
    blocks: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
    *,
    sequence_width: int,
) -> dict[str, torch.Tensor]:
    if not blocks:
        raise ValueError("blocks must not be empty")
    if sequence_width < 0:
        raise ValueError("sequence_width must be non-negative")
    concat_presence = {
        key: key in blocks[0] for key, _dtype in _DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA
    }
    for index, block in enumerate(blocks):
        _validate_path_block(f"blocks[{index}]", block)
        _validate_topology_extra_fields(
            f"blocks[{index}]", block, int(sequence_width), concat_presence
        )
    exported = _required_native_op("deterministic_concat_topology_blocks")(
        tuple(blocks), int(sequence_width)
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_concat_topology_blocks must return a dict"
        )
    _validate_path_block("deterministic_concat_topology_blocks", exported)
    _validate_topology_extra_fields(
        "deterministic_concat_topology_blocks",
        exported,
        int(sequence_width),
        concat_presence,
    )
    expected_count = sum(int(block["valid"].shape[0]) for block in blocks)
    if exported["valid"].shape != (expected_count,):
        raise ValueError(
            "_channel.deterministic_concat_topology_blocks returned bad path count"
        )
    return exported


def deterministic_gather_topology_block(
    block: dict[str, torch.Tensor],
    order: torch.Tensor,
    *,
    max_count: int,
    sequence_width: int,
) -> dict[str, torch.Tensor]:
    if sequence_width < 0:
        raise ValueError("sequence_width must be non-negative")
    if max_count < -1:
        raise ValueError("max_count must be -1 or non-negative")
    _validate_path_block("block", block)
    field_presence = {
        key: key in block for key, _dtype in _DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA
    }
    _validate_topology_extra_fields("block", block, int(sequence_width), field_presence)
    validate_cuda_tensor("order", order, dtype=torch.long, ndim=1)
    if order.get_device() != block["valid"].get_device():
        raise ValueError("order must share block device")

    exported = _required_native_op("deterministic_gather_topology_block")(
        block,
        order,
        int(max_count),
        int(sequence_width),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_gather_topology_block must return a dict"
        )
    _validate_path_block("deterministic_gather_topology_block", exported)
    _validate_topology_extra_fields(
        "deterministic_gather_topology_block",
        exported,
        int(sequence_width),
        field_presence,
    )
    expected_count = (
        int(order.shape[0])
        if max_count < 0
        else min(int(order.shape[0]), int(max_count))
    )
    if exported["valid"].shape != (expected_count,):
        raise ValueError(
            "_channel.deterministic_gather_topology_block returned bad path count"
        )
    return exported


def path_los_visibility_inputs(
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_id", tx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    if rx_id.shape != tx_id.shape:
        raise ValueError("rx_id must match tx_id")
    exported = _required_native_op("path_los_visibility_inputs")(
        tx_positions, rx_positions, tx_id, rx_id
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.path_los_visibility_inputs must return a dict")
    validate_cuda_tensor(
        "start", exported["start"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "end", exported["end"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    if exported["active"].shape != tx_id.shape:
        raise ValueError(
            "_channel.path_los_visibility_inputs returned bad active shape"
        )
    return exported


def path_filter_los(
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    path_length_m: torch.Tensor,
    delay_s: torch.Tensor,
    path_gain: torch.Tensor,
    visible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx_id", tx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("path_length_m", path_length_m, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("delay_s", delay_s, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    for name, tensor in {
        "rx_id": rx_id,
        "path_length_m": path_length_m,
        "delay_s": delay_s,
        "path_gain": path_gain,
        "visible": visible,
    }.items():
        if tensor.shape != tx_id.shape:
            raise ValueError(f"{name} must match tx_id")
    block = _required_native_op("path_filter_los")(
        tx_id, rx_id, path_length_m, delay_s, path_gain, visible
    )
    _validate_path_block("path_filter_los", block)
    return block


def _validate_path_reflection_candidates(
    name: str, candidates: dict[str, torch.Tensor]
) -> None:
    _validate_path_block(name, candidates)
    path_count = candidates["valid"].shape
    for key in ("seg0_start", "seg0_end", "seg1_start", "seg1_end"):
        tensor = candidates.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}.{key} must be a torch.Tensor")
        validate_cuda_tensor(
            f"{name}.{key}", tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        if tensor.shape[0] != path_count[0]:
            raise ValueError(f"{name}.{key} must share the path count")
    active = candidates.get("active")
    if not isinstance(active, torch.Tensor):
        raise TypeError(f"{name}.active must be a torch.Tensor")
    validate_cuda_tensor(f"{name}.active", active, dtype=torch.bool, ndim=1)
    if tuple(active.shape) != path_count:
        raise ValueError(f"{name}.active must share the path count")


def path_filter_block(
    block: dict[str, torch.Tensor],
    visible0: torch.Tensor,
    visible1: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _validate_path_block("block", block)
    validate_cuda_tensor("visible0", visible0, dtype=torch.bool, ndim=1)
    validate_cuda_tensor("visible1", visible1, dtype=torch.bool, ndim=1)
    if visible0.shape != block["valid"].shape or visible1.shape != block["valid"].shape:
        raise ValueError("visible masks must share the path count")
    out = _required_native_op("path_filter_block")(block, visible0, visible1)
    if not isinstance(out, dict):
        raise TypeError("_channel.path_filter_block must return a dict")
    _validate_path_block("path_filter_block", out)
    return out


def path_diffraction_block(
    rayd_output: tuple[torch.Tensor, ...],
    *,
    tx_index: int,
) -> dict[str, torch.Tensor]:
    if not isinstance(rayd_output, tuple) or len(rayd_output) != 18:
        raise TypeError(
            "rayd_output must be the 18-tensor RayD diffraction path tuple"
        )
    for index in (1, 3, 4, 5):
        validate_cuda_tensor(
            f"rayd_output[{index}]",
            rayd_output[index],
            dtype=torch.int32 if index != 1 else torch.bool,
            ndim=1,
        )
    for index in (8, 9, 10, 11, 12, 13, 14):
        validate_cuda_tensor(
            f"rayd_output[{index}]", rayd_output[index], dtype=torch.float32, ndim=1
        )
    capacity = rayd_output[1].shape
    for index in (3, 4, 5, 8, 9, 10, 11, 12, 13, 14):
        if rayd_output[index].shape != capacity:
            raise ValueError("RayD diffraction path tensors must share capacity")
    if tx_index < 0:
        raise ValueError("tx_index must be non-negative")
    out = _required_native_op("path_diffraction_block")(rayd_output, int(tx_index))
    if not isinstance(out, dict):
        raise TypeError("_channel.path_diffraction_block must return a dict")
    _validate_path_block("path_diffraction_block", out)
    return out


def path_merge_blocks(
    blocks: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
    *,
    tx_count: int,
    max_depth: int,
) -> dict[str, torch.Tensor]:
    if not blocks:
        raise ValueError("blocks must not be empty")
    for index, block in enumerate(blocks):
        _validate_path_block(f"blocks[{index}]", block)
    if tx_count < 0:
        raise ValueError("tx_count must be non-negative")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    out = _required_native_op("path_merge_blocks")(
        tuple(blocks), int(tx_count), int(max_depth)
    )
    if not isinstance(out, dict):
        raise TypeError("_channel.path_merge_blocks must return a dict")
    _validate_path_block("path_merge_blocks", out)
    return out


def path_finalize_blocks(
    los: dict[str, torch.Tensor],
    reflection: dict[str, torch.Tensor],
    diffraction: dict[str, torch.Tensor],
    *,
    max_paths: int | None,
    tx_count: int,
    max_depth: int,
) -> dict[str, torch.Tensor]:
    _validate_path_block("los", los)
    _validate_path_block("reflection", reflection)
    _validate_path_block("diffraction", diffraction)
    if tx_count < 0:
        raise ValueError("tx_count must be non-negative")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    max_paths_value = -1 if max_paths is None else int(max_paths)
    if max_paths_value < -1:
        raise ValueError("max_paths must be positive")
    finalized = _required_native_op("path_finalize_blocks")(
        los,
        reflection,
        diffraction,
        max_paths_value,
        int(tx_count),
        int(max_depth),
    )
    _validate_path_block("path_finalize_blocks", finalized)
    return finalized
