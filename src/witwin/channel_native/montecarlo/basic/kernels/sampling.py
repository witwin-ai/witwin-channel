from __future__ import annotations

import torch

from witwin.channel_native.runtime.symbols import native_extension
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor


def mc_sample_directions(count: int, reference: torch.Tensor) -> torch.Tensor:
    if count < 0:
        raise ValueError("count must be non-negative")
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)

    native = native_extension()
    if native is None or not hasattr(native, "mc_sample_directions"):
        raise RuntimeError(
            "_channel_native.mc_sample_directions CUDA kernel is required"
        )
    directions = native.mc_sample_directions(int(count), reference)
    if not isinstance(directions, torch.Tensor):
        raise TypeError("_channel_native.mc_sample_directions must return a tensor")
    validate_cuda_tensor(
        "directions", directions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return directions


def mc_transmitter_tensors(
    flat_positions: tuple[float, ...],
    powers: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    if len(flat_positions) % 3 != 0:
        raise ValueError("flat_positions must contain xyz triples")
    if len(flat_positions) // 3 != len(powers):
        raise ValueError("powers must match flat_positions")
    native = native_extension()
    if native is None or not hasattr(native, "mc_transmitter_tensors"):
        raise RuntimeError(
            "_channel_native.mc_transmitter_tensors CUDA helper is required"
        )
    exported = native.mc_transmitter_tensors(flat_positions, powers)
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.mc_transmitter_tensors must return a dict")
    validate_cuda_tensor(
        "positions",
        exported["positions"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor("power", exported["power"], dtype=torch.float32, ndim=1)
    return exported


def mc_pack_vec3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("x", x, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y", y, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z", z, dtype=torch.float32, ndim=1)
    if y.shape != x.shape or z.shape != x.shape:
        raise ValueError("x, y, and z must have the same shape")
    native = native_extension()
    if native is None or not hasattr(native, "mc_pack_vec3"):
        raise RuntimeError("_channel_native.mc_pack_vec3 CUDA kernel is required")
    packed = native.mc_pack_vec3(x, y, z)
    if not isinstance(packed, torch.Tensor):
        raise TypeError("_channel_native.mc_pack_vec3 must return a tensor")
    validate_cuda_tensor(
        "packed", packed, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if packed.shape[0] != x.shape[0]:
        raise ValueError("_channel_native.mc_pack_vec3 returned an unexpected shape")
    return packed


def mc_receiver_grid_points(
    reference: torch.Tensor,
    *,
    origin: tuple[float, float, float],
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    shape: tuple[int, int],
    spacing: tuple[float, float],
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    rows, cols = shape
    if rows < 0 or cols < 0:
        raise ValueError("shape entries must be non-negative")
    if spacing[0] <= 0.0 or spacing[1] <= 0.0:
        raise ValueError("spacing entries must be positive")
    native = native_extension()
    if native is None or not hasattr(native, "mc_receiver_grid_points"):
        raise RuntimeError(
            "_channel_native.mc_receiver_grid_points CUDA kernel is required"
        )
    points = native.mc_receiver_grid_points(
        reference,
        int(rows),
        int(cols),
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
        float(x_axis[0]),
        float(x_axis[1]),
        float(x_axis[2]),
        float(y_axis[0]),
        float(y_axis[1]),
        float(y_axis[2]),
        float(spacing[0]),
        float(spacing[1]),
    )
    if not isinstance(points, torch.Tensor):
        raise TypeError("_channel_native.mc_receiver_grid_points must return a tensor")
    validate_cuda_tensor(
        "points", points, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if points.shape[0] != rows * cols:
        raise ValueError(
            "_channel_native.mc_receiver_grid_points returned an unexpected shape"
        )
    return points


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
            "_channel_native.mc_reflection_launch_inputs CUDA kernel is required"
        )
    exported = native.mc_reflection_launch_inputs(
        tx_positions, int(tx_index), int(sample_count)
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.mc_reflection_launch_inputs must return a dict"
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
            "_channel_native.mc_diffraction_state_wi CUDA kernel is required"
        )
    state_wi = native.mc_diffraction_state_wi(state_edge_pos, state_src)
    if not isinstance(state_wi, torch.Tensor):
        raise TypeError("_channel_native.mc_diffraction_state_wi must return a tensor")
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
            "_channel_native.mc_diffraction_state_pack CUDA kernel is required"
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
            "_channel_native.mc_diffraction_state_pack must return 12 tensors"
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
