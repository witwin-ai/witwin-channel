from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel.runtime.capacity import (
    CapacityFailureState,
    require_capacity_failure_state,
)
from witwin.channel.runtime.symbols import (
    native_extension,
    required_symbol as _required_native_op,
)
from witwin.channel.runtime.tensor_contracts import validate_cuda_tensor


@dataclass(frozen=True, slots=True)
class DiffractionStateCapacityBlock:
    failure_state: CapacityFailureState
    edge_index: torch.Tensor
    edge_position: torch.Tensor
    edge_direction: torch.Tensor
    edge_t_min: torch.Tensor
    edge_t_max: torch.Tensor
    n0: torch.Tensor
    n1: torch.Tensor
    prim0: torch.Tensor
    prim1: torch.Tensor
    exterior_angle: torch.Tensor
    source: torch.Tensor
    source_power: torch.Tensor
    valid: torch.Tensor
    actual_count: torch.Tensor
    overflow: torch.Tensor


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


def _validate_diffraction_state_capacity_inputs(
    active: torch.Tensor,
    state_capacity: int,
    state_tensors: tuple[
        tuple[str, torch.Tensor, torch.dtype, int, tuple[int, ...]], ...
    ],
) -> int:
    validate_cuda_tensor("active", active, dtype=torch.bool, ndim=1)
    state_count = int(active.shape[0])
    for name, tensor, dtype, ndim, trailing_shape in state_tensors:
        validate_cuda_tensor(
            name,
            tensor,
            dtype=dtype,
            ndim=ndim,
            trailing_shape=trailing_shape,
            require_contiguous=False,
        )
        if tensor.shape[0] != state_count:
            raise ValueError(f"{name} must share active row capacity")
        if tensor.get_device() != active.get_device():
            raise ValueError(f"{name} must share active device")
    if isinstance(state_capacity, bool) or not isinstance(state_capacity, int):
        raise TypeError("state_capacity must be an integer")
    if state_capacity < 0:
        raise ValueError("state_capacity must be non-negative")
    return state_count


def _name_diffraction_state_capacity_output(
    raw: object,
    *,
    failure_state: CapacityFailureState,
    capacity: int,
    device: torch.device,
) -> DiffractionStateCapacityBlock:
    if not isinstance(raw, tuple) or len(raw) != 15:
        raise TypeError(
            "_channel.deterministic_diffraction_state_capacity_select "
            "must return 15 tensors"
        )
    for name, tensor, dtype, ndim, trailing_shape in (
        ("edge_index", raw[0], torch.int32, 1, ()),
        ("edge_position", raw[1], torch.float32, 2, (3,)),
        ("edge_direction", raw[2], torch.float32, 2, (3,)),
        ("edge_t_min", raw[3], torch.float32, 1, ()),
        ("edge_t_max", raw[4], torch.float32, 1, ()),
        ("n0", raw[5], torch.float32, 2, (3,)),
        ("n1", raw[6], torch.float32, 2, (3,)),
        ("prim0", raw[7], torch.int32, 1, ()),
        ("prim1", raw[8], torch.int32, 1, ()),
        ("exterior_angle", raw[9], torch.float32, 1, ()),
        ("source", raw[10], torch.float32, 2, (3,)),
        ("source_power", raw[11], torch.float32, 1, ()),
        ("valid", raw[12], torch.bool, 1, ()),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"native diffraction capacity output {name} must be a tensor"
            )
        validate_cuda_tensor(
            name, tensor, dtype=dtype, ndim=ndim, trailing_shape=trailing_shape
        )
        if tensor.shape[0] != capacity:
            raise ValueError(
                f"native diffraction capacity output {name} has wrong capacity"
            )
        if tensor.device != device:
            raise ValueError(
                f"native diffraction capacity output {name} has wrong device"
            )
    for name, tensor, dtype in (
        ("actual_count", raw[13], torch.int32),
        ("overflow", raw[14], torch.bool),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"native diffraction capacity output {name} must be a tensor"
            )
        validate_cuda_tensor(name, tensor, dtype=dtype, ndim=1)
        if tensor.shape != (1,):
            raise ValueError(
                f"native diffraction capacity output {name} must have shape (1,)"
            )
        if tensor.device != device:
            raise ValueError(
                f"native diffraction capacity output {name} has wrong device"
            )
    return DiffractionStateCapacityBlock(failure_state, *raw)


def deterministic_diffraction_state_capacity_select(
    *,
    failure_state: CapacityFailureState,
    active: torch.Tensor,
    edge_index: torch.Tensor,
    edge_position: torch.Tensor,
    edge_direction: torch.Tensor,
    edge_t_min: torch.Tensor,
    edge_t_max: torch.Tensor,
    n0: torch.Tensor,
    n1: torch.Tensor,
    prim0: torch.Tensor,
    prim1: torch.Tensor,
    exterior_angle: torch.Tensor,
    source: torch.Tensor,
    source_power: torch.Tensor,
    state_capacity: int,
) -> DiffractionStateCapacityBlock:
    """Stably gather active rows into a host-shaped CUDA capacity block."""

    state_tensors = (
        ("edge_index", edge_index, torch.int32, 1, ()),
        ("edge_position", edge_position, torch.float32, 2, (3,)),
        ("edge_direction", edge_direction, torch.float32, 2, (3,)),
        ("edge_t_min", edge_t_min, torch.float32, 1, ()),
        ("edge_t_max", edge_t_max, torch.float32, 1, ()),
        ("n0", n0, torch.float32, 2, (3,)),
        ("n1", n1, torch.float32, 2, (3,)),
        ("prim0", prim0, torch.int32, 1, ()),
        ("prim1", prim1, torch.int32, 1, ()),
        ("exterior_angle", exterior_angle, torch.float32, 1, ()),
        ("source", source, torch.float32, 2, (3,)),
        ("source_power", source_power, torch.float32, 1, ()),
    )
    state_count = _validate_diffraction_state_capacity_inputs(
        active, state_capacity, state_tensors
    )
    require_capacity_failure_state(failure_state, device=active.device)

    raw = _required_native_op("deterministic_diffraction_state_capacity_select")(
        failure_state.bits,
        active,
        edge_index,
        edge_position,
        edge_direction,
        edge_t_min,
        edge_t_max,
        n0,
        n1,
        prim0,
        prim1,
        exterior_angle,
        source,
        source_power,
        state_capacity,
    )
    return _name_diffraction_state_capacity_output(
        raw,
        failure_state=failure_state,
        capacity=min(state_capacity, state_count),
        device=active.device,
    )


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
