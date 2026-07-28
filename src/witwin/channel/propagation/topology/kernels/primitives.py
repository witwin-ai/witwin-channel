from __future__ import annotations

import torch

from witwin.channel.runtime import (
    native_extension,
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)


def deterministic_component_counts(component_id: torch.Tensor) -> dict[str, int]:
    validate_cuda_tensor("component_id", component_id, dtype=torch.int32, ndim=1)
    exported = _required_native_op("deterministic_component_counts")(component_id)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_component_counts must return a dict"
        )
    counts: dict[str, int] = {}
    for name in ("los", "reflection", "diffraction"):
        value = exported[name]
        if not isinstance(value, int):
            raise TypeError(
                f"_channel.deterministic_component_counts returned non-int {name}"
            )
        counts[name] = value
    return counts


def deterministic_selected_edge_count(edge_id: torch.Tensor) -> int:
    validate_cuda_tensor("edge_id", edge_id, dtype=torch.int32, ndim=1)
    value = _required_native_op("deterministic_selected_edge_count")(edge_id)
    if not isinstance(value, int):
        raise TypeError(
            "_channel.deterministic_selected_edge_count must return an int"
        )
    return value


def core_pack_int2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("x", x, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("y", y, dtype=torch.int32, ndim=1)
    if y.shape != x.shape:
        raise ValueError("x and y must have the same shape")
    out = _required_native_op("core_pack_int2")(x, y)
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel.core_pack_int2 must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.int32, ndim=2, trailing_shape=(2,))
    if out.shape != (x.shape[0], 2):
        raise ValueError("_channel.core_pack_int2 returned an unexpected shape")
    return out


def deterministic_diffraction_state_pack(
    edge_indices: torch.Tensor,
    edge_pos: torch.Tensor,
    edge_dir: torch.Tensor,
    line_min: torch.Tensor,
    line_max: torch.Tensor,
    n0: torch.Tensor,
    n1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    exterior_angle: torch.Tensor,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    tx_power_index: int,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor("edge_indices", edge_indices, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "edge_pos", edge_pos, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "edge_dir", edge_dir, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("line_min", line_min, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_max", line_max, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("n0", n0, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("n1", n1, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("exterior_angle", exterior_angle, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    if not 0 <= int(tx_power_index) < int(tx_power.shape[0]):
        raise ValueError("tx_power_index is out of range")
    states = _required_native_op("deterministic_diffraction_state_pack")(
        edge_indices,
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
        int(tx_power_index),
    )
    if not isinstance(states, tuple) or len(states) != 12:
        raise TypeError(
            "_channel.deterministic_diffraction_state_pack must return 12 tensors"
        )
    validate_cuda_tensor("state_edge_index", states[0], dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "state_edge_pos", states[1], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_edge_dir", states[2], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_line_min", states[3], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("state_line_max", states[4], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_n0", states[5], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_n1", states[6], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_face0", states[7], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_face1", states[8], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_exterior_angle", states[9], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_src", states[10], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_tx_power", states[11], dtype=torch.float32, ndim=1)
    return states


def deterministic_diffraction_state_pack_selected(
    selected: torch.Tensor,
    edge_pos: torch.Tensor,
    edge_dir: torch.Tensor,
    line_min: torch.Tensor,
    line_max: torch.Tensor,
    n0: torch.Tensor,
    n1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    exterior_angle: torch.Tensor,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    tx_power_index: int,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor("selected", selected, dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "edge_pos", edge_pos, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "edge_dir", edge_dir, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("line_min", line_min, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_max", line_max, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("n0", n0, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("n1", n1, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("exterior_angle", exterior_angle, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    if not 0 <= int(tx_power_index) < int(tx_power.shape[0]):
        raise ValueError("tx_power_index is out of range")
    states = _required_native_op("deterministic_diffraction_state_pack_selected")(
        selected,
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
        int(tx_power_index),
    )
    if not isinstance(states, tuple) or len(states) != 12:
        raise TypeError(
            "_channel.deterministic_diffraction_state_pack_selected must return 12 tensors"
        )
    validate_cuda_tensor("state_edge_index", states[0], dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "state_edge_pos", states[1], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_edge_dir", states[2], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_line_min", states[3], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("state_line_max", states[4], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_n0", states[5], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_n1", states[6], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_face0", states[7], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_face1", states[8], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_exterior_angle", states[9], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_src", states[10], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_tx_power", states[11], dtype=torch.float32, ndim=1)
    return states


def mc_selected_edge_indices(selected: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("selected", selected, dtype=torch.bool, ndim=1)
    native = native_extension()
    if native is None or not hasattr(native, "mc_selected_edge_indices"):
        raise RuntimeError(
            "_channel.mc_selected_edge_indices CUDA kernel is required"
        )
    indices = native.mc_selected_edge_indices(selected)
    if not isinstance(indices, torch.Tensor):
        raise TypeError("_channel.mc_selected_edge_indices must return a tensor")
    validate_cuda_tensor("indices", indices, dtype=torch.int32, ndim=1)
    return indices


def path_concat_vec3(
    blocks: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> torch.Tensor:
    if not blocks:
        raise ValueError("blocks must not be empty")
    for index, block in enumerate(blocks):
        validate_cuda_tensor(
            f"blocks[{index}]", block, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
    out = _required_native_op("path_concat_vec3")(tuple(blocks))
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    return out
