from __future__ import annotations

import torch

from witwin.channel.propagation.topology import (  # noqa: F401
    mc_sample_directions,
)
from witwin.channel.runtime.native_buffers import (  # noqa: F401
    mc_pack_vec3,
    mc_receiver_grid_points,
    mc_transmitter_tensors,
)
from witwin.channel.runtime.symbols import native_extension, required_symbol
from witwin.channel.runtime.tensor_contracts import validate_cuda_tensor


def mc_reflection_launch_inputs(
    tx_positions: torch.Tensor,
    *,
    tx_index: int,
    sample_count: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    native = native_extension()
    if native is None or not hasattr(native, "mc_reflection_launch_inputs"):
        raise RuntimeError(
            "_channel.mc_reflection_launch_inputs CUDA kernel is required"
        )
    exported = native.mc_reflection_launch_inputs(
        tx_positions, int(tx_index), int(sample_count)
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.mc_reflection_launch_inputs must return a dict"
        )
    validate_cuda_tensor(
        "ray_o", exported["ray_o"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("ray_tmax", exported["ray_tmax"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "tx_pol", exported["tx_pol"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return exported


def _validate_mc_diffraction_discovery_args(
    args: tuple[torch.Tensor, ...], *, counted: bool
) -> None:
    expected = 16 if counted else 15
    if len(args) != expected:
        raise TypeError(f"MC diffraction discovery expects {expected} tensors")
    hit_count_index = 6 if counted else None
    offset = 1 if counted else 0
    names = (
        "tx_pos", "ray_dir", "prim_index", "hit_p", "hit_n", "hit_geo_n"
    )
    for index, name in enumerate(names):
        trailing_shape = (3,) if name not in {"prim_index"} else None
        validate_cuda_tensor(
            name,
            args[index],
            dtype=torch.int32 if name == "prim_index" else torch.float32,
            ndim=1 if name in {"tx_pos", "prim_index"} else 2,
            trailing_shape=trailing_shape,
        )
    if hit_count_index is not None:
        validate_cuda_tensor("hit_count", args[hit_count_index], dtype=torch.int32, ndim=1)
    table_start = 6 + offset
    validate_cuda_tensor(
        "triangle_edge_count", args[table_start], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "triangle_edge_indices", args[table_start + 1], dtype=torch.int32, ndim=2
    )
    for index, name in enumerate(("edge_pos", "edge_dir", "edge_n0", "edge_n1"), start=2):
        validate_cuda_tensor(
            name,
            args[table_start + index],
            dtype=torch.float32,
            ndim=2,
            trailing_shape=(3,),
        )
    for index, name, dtype in (
        (6, "edge_line_min", torch.float32),
        (7, "edge_line_max", torch.float32),
        (8, "edge_adjacent_face1", torch.int32),
    ):
        validate_cuda_tensor(name, args[table_start + index], dtype=dtype, ndim=1)


def mc_diffraction_discover_edges(*args: torch.Tensor) -> torch.Tensor:
    _validate_mc_diffraction_discovery_args(args, counted=False)
    out = required_symbol("mc_diffraction_discover_edges")(*args)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.mc_diffraction_discover_edges must return a tensor"
        )
    return out


def mc_diffraction_discover_edges_counted(*args: torch.Tensor) -> torch.Tensor:
    _validate_mc_diffraction_discovery_args(args, counted=True)
    out = required_symbol("mc_diffraction_discover_edges_counted")(*args)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.mc_diffraction_discover_edges_counted must return a tensor"
        )
    return out


def mc_diffraction_state_wi(
    state_edge_pos: torch.Tensor, state_src: torch.Tensor
) -> torch.Tensor:
    validate_cuda_tensor(
        "state_edge_pos",
        state_edge_pos,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "state_src", state_src, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if state_src.shape != state_edge_pos.shape:
        raise ValueError("state_src must match state_edge_pos shape")
    native = native_extension()
    if native is None or not hasattr(native, "mc_diffraction_state_wi"):
        raise RuntimeError(
            "_channel.mc_diffraction_state_wi CUDA kernel is required"
        )
    state_wi = native.mc_diffraction_state_wi(state_edge_pos, state_src)
    if not isinstance(state_wi, torch.Tensor):
        raise TypeError("_channel.mc_diffraction_state_wi must return a tensor")
    validate_cuda_tensor(
        "state_wi", state_wi, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return state_wi


def mc_diffraction_state_pack(
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
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=0)
    if tx.shape[0] != 3:
        raise ValueError("tx must have shape (3,)")
    native = native_extension()
    if native is None or not hasattr(native, "mc_diffraction_state_pack"):
        raise RuntimeError(
            "_channel.mc_diffraction_state_pack CUDA kernel is required"
        )
    states = native.mc_diffraction_state_pack(
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
    )
    if not isinstance(states, tuple) or len(states) != 12:
        raise TypeError(
            "_channel.mc_diffraction_state_pack must return 12 tensors"
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
    validate_cuda_tensor("state_src_power", states[11], dtype=torch.float32, ndim=1)
    return states
