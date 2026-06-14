from __future__ import annotations

import torch

from .extension import native_extension
from .metadata import make_metadata, validate_metadata


def validate_cuda_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    ndim: int,
    trailing_shape: tuple[int, ...] = (),
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if trailing_shape and tuple(tensor.shape[-len(trailing_shape) :]) != trailing_shape:
        raise ValueError(f"{name} must end with shape {trailing_shape}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def noop_metadata(*, accumulation_strategy: str = "none") -> dict[str, bool | float | int | str]:
    return make_metadata(
        primitive="noop_metadata",
        accumulation_strategy=accumulation_strategy,
        scheduling_strategy="none",
        ad_status="none",
    )


def path_los_export(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must have one value per transmitter")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    native = native_extension()
    if native is not None and hasattr(native, "path_los_export"):
        exported = native.path_los_export(tx_positions, tx_power, rx_positions, float(frequency_hz))
        if not isinstance(exported, dict):
            raise TypeError("_channel_native.path_los_export must return a dict")
        return exported

    raise RuntimeError("_channel_native.path_los_export CUDA kernel is required")


def mc_los_path_gain_backward(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    frequency_hz: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_cuda_tensor("tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if not isinstance(grad_output, torch.Tensor):
        raise TypeError("grad_output must be a torch.Tensor")
    if grad_output.dtype != torch.float32:
        raise TypeError("grad_output must have dtype torch.float32")
    if not grad_output.is_cuda:
        raise ValueError("grad_output must be a CUDA tensor")
    if grad_output.ndim != 2:
        raise ValueError("grad_output must have 2 dimensions")
    if grad_output.shape != (tx_positions.shape[0], rx_positions.shape[0]):
        raise ValueError("grad_output must match the LoS path-gain matrix shape")
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must have one value per transmitter")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "mc_los_path_gain_backward"):
        raise RuntimeError("_channel_native.mc_los_path_gain_backward CUDA kernel is required")
    gradients = native.mc_los_path_gain_backward(
        tx_positions,
        tx_power,
        rx_positions,
        grad_output,
        float(frequency_hz),
    )
    if not isinstance(gradients, tuple) or len(gradients) != 3:
        raise TypeError("_channel_native.mc_los_path_gain_backward must return 3 tensors")
    validate_cuda_tensor("grad_tx", gradients[0], dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("grad_power", gradients[1], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("grad_rx", gradients[2], dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if gradients[0].shape != tx_positions.shape:
        raise ValueError("_channel_native.mc_los_path_gain_backward returned bad grad_tx shape")
    if gradients[1].shape != tx_power.shape:
        raise ValueError("_channel_native.mc_los_path_gain_backward returned bad grad_power shape")
    if gradients[2].shape != rx_positions.shape:
        raise ValueError("_channel_native.mc_los_path_gain_backward returned bad grad_rx shape")
    return gradients


def mc_los_path_gain_jvp(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_tangent: torch.Tensor,
    power_tangent: torch.Tensor,
    rx_tangent: torch.Tensor,
    has_tx_tangent: bool,
    has_power_tangent: bool,
    has_rx_tangent: bool,
    *,
    frequency_hz: float,
) -> torch.Tensor:
    validate_cuda_tensor("tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must have one value per transmitter")
    if has_tx_tangent:
        validate_cuda_tensor("tx_tangent", tx_tangent, dtype=torch.float32, ndim=2, trailing_shape=(3,))
        if tx_tangent.shape != tx_positions.shape:
            raise ValueError("tx_tangent must match tx_positions")
    if has_power_tangent:
        validate_cuda_tensor("power_tangent", power_tangent, dtype=torch.float32, ndim=1)
        if power_tangent.shape != tx_power.shape:
            raise ValueError("power_tangent must match tx_power")
    if has_rx_tangent:
        validate_cuda_tensor("rx_tangent", rx_tangent, dtype=torch.float32, ndim=2, trailing_shape=(3,))
        if rx_tangent.shape != rx_positions.shape:
            raise ValueError("rx_tangent must match rx_positions")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "mc_los_path_gain_jvp"):
        raise RuntimeError("_channel_native.mc_los_path_gain_jvp CUDA kernel is required")
    out = native.mc_los_path_gain_jvp(
        tx_positions,
        tx_power,
        rx_positions,
        tx_tangent,
        power_tangent,
        rx_tangent,
        bool(has_tx_tangent),
        bool(has_power_tangent),
        bool(has_rx_tangent),
        float(frequency_hz),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel_native.mc_los_path_gain_jvp must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2)
    if out.shape != (tx_positions.shape[0], rx_positions.shape[0]):
        raise ValueError("_channel_native.mc_los_path_gain_jvp returned an unexpected shape")
    return out


def mc_finalize_component_maps(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("reflection", reflection, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("diffraction", diffraction, dtype=torch.float32, ndim=3)
    if reflection.shape != los.shape:
        raise ValueError("reflection must match los shape")
    if diffraction.shape != los.shape:
        raise ValueError("diffraction must match los shape")

    native = native_extension()
    if native is None or not hasattr(native, "mc_finalize_component_maps"):
        raise RuntimeError("_channel_native.mc_finalize_component_maps CUDA kernel is required")
    exported = native.mc_finalize_component_maps(los, reflection, diffraction)
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.mc_finalize_component_maps must return a dict")
    return exported


def mc_component_map_buffer(
    reference: torch.Tensor,
    *,
    tx_count: int,
    dim0: int,
    dim1: int,
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if tx_count < 0 or dim0 < 0 or dim1 < 0:
        raise ValueError("tx_count, dim0, and dim1 must be non-negative")
    native = native_extension()
    if native is None or not hasattr(native, "mc_component_map_buffer"):
        raise RuntimeError("_channel_native.mc_component_map_buffer CUDA kernel is required")
    maps = native.mc_component_map_buffer(reference, int(tx_count), int(dim0), int(dim1))
    if not isinstance(maps, torch.Tensor):
        raise TypeError("_channel_native.mc_component_map_buffer must return a tensor")
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (tx_count, dim0, dim1):
        raise ValueError("_channel_native.mc_component_map_buffer returned an unexpected shape")
    return maps


def mc_store_component_map(
    maps: torch.Tensor,
    source: torch.Tensor,
    *,
    tx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2)
    if source.shape != maps.shape[1:]:
        raise ValueError("source shape must match one maps slot")
    native = native_extension()
    if native is None or not hasattr(native, "mc_store_component_map"):
        raise RuntimeError("_channel_native.mc_store_component_map CUDA kernel is required")
    out = native.mc_store_component_map(maps, source, int(tx_index))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel_native.mc_store_component_map must return a tensor")
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def mc_store_scaled_component_map(
    maps: torch.Tensor,
    source: torch.Tensor,
    scale_values: torch.Tensor,
    *,
    tx_index: int,
    scale_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("scale_values", scale_values, dtype=torch.float32, ndim=1)
    if source.shape != maps.shape[1:]:
        raise ValueError("source shape must match one maps slot")
    native = native_extension()
    if native is None or not hasattr(native, "mc_store_scaled_component_map"):
        raise RuntimeError("_channel_native.mc_store_scaled_component_map CUDA kernel is required")
    out = native.mc_store_scaled_component_map(
        maps,
        source,
        scale_values,
        int(tx_index),
        int(scale_index),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel_native.mc_store_scaled_component_map must return a tensor")
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def mc_sample_directions(count: int, reference: torch.Tensor) -> torch.Tensor:
    if count < 0:
        raise ValueError("count must be non-negative")
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)

    native = native_extension()
    if native is None or not hasattr(native, "mc_sample_directions"):
        raise RuntimeError("_channel_native.mc_sample_directions CUDA kernel is required")
    directions = native.mc_sample_directions(int(count), reference)
    if not isinstance(directions, torch.Tensor):
        raise TypeError("_channel_native.mc_sample_directions must return a tensor")
    validate_cuda_tensor("directions", directions, dtype=torch.float32, ndim=2, trailing_shape=(3,))
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
        raise RuntimeError("_channel_native.mc_transmitter_tensors CUDA helper is required")
    exported = native.mc_transmitter_tensors(flat_positions, powers)
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.mc_transmitter_tensors must return a dict")
    validate_cuda_tensor("positions", exported["positions"], dtype=torch.float32, ndim=2, trailing_shape=(3,))
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
    validate_cuda_tensor("packed", packed, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if packed.shape[0] != x.shape[0]:
        raise ValueError("_channel_native.mc_pack_vec3 returned an unexpected shape")
    return packed


def mc_los_component_maps(los: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    native = native_extension()
    if native is None or not hasattr(native, "mc_los_component_maps"):
        raise RuntimeError("_channel_native.mc_los_component_maps CUDA kernel is required")
    maps = native.mc_los_component_maps(los)
    if not isinstance(maps, torch.Tensor):
        raise TypeError("_channel_native.mc_los_component_maps must return a tensor")
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    return maps


def mc_apply_los_visibility(
    maps: torch.Tensor,
    los: torch.Tensor,
    visible: torch.Tensor,
    *,
    tx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if maps.shape != (los.shape[0], los.shape[2], los.shape[1]):
        raise ValueError("maps must have shape (tx, los_cols, los_rows)")
    native = native_extension()
    if native is None or not hasattr(native, "mc_apply_los_visibility"):
        raise RuntimeError("_channel_native.mc_apply_los_visibility CUDA kernel is required")
    out = native.mc_apply_los_visibility(maps, los, visible.contiguous(), int(tx_index))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel_native.mc_apply_los_visibility must return a tensor")
    return out


def mc_los_visibility_inputs(
    tx_positions: torch.Tensor,
    *,
    tx_index: int,
    rx_count: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if rx_count < 0:
        raise ValueError("rx_count must be non-negative")
    native = native_extension()
    if native is None or not hasattr(native, "mc_los_visibility_inputs"):
        raise RuntimeError("_channel_native.mc_los_visibility_inputs CUDA kernel is required")
    exported = native.mc_los_visibility_inputs(tx_positions, int(tx_index), int(rx_count))
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.mc_los_visibility_inputs must return a dict")
    validate_cuda_tensor("start", exported["start"], dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    return exported


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
        raise RuntimeError("_channel_native.mc_receiver_grid_points CUDA kernel is required")
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
    validate_cuda_tensor("points", points, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if points.shape[0] != rows * cols:
        raise ValueError("_channel_native.mc_receiver_grid_points returned an unexpected shape")
    return points


def mc_reflection_launch_inputs(
    tx_positions: torch.Tensor,
    *,
    tx_index: int,
    sample_count: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    native = native_extension()
    if native is None or not hasattr(native, "mc_reflection_launch_inputs"):
        raise RuntimeError("_channel_native.mc_reflection_launch_inputs CUDA kernel is required")
    exported = native.mc_reflection_launch_inputs(tx_positions, int(tx_index), int(sample_count))
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.mc_reflection_launch_inputs must return a dict")
    validate_cuda_tensor("ray_o", exported["ray_o"], dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("ray_tmax", exported["ray_tmax"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    validate_cuda_tensor("tx_pol", exported["tx_pol"], dtype=torch.float32, ndim=2, trailing_shape=(3,))
    return exported


def mc_diffraction_state_wi(state_edge_pos: torch.Tensor, state_src: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("state_edge_pos", state_edge_pos, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("state_src", state_src, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if state_src.shape != state_edge_pos.shape:
        raise ValueError("state_src must match state_edge_pos shape")
    native = native_extension()
    if native is None or not hasattr(native, "mc_diffraction_state_wi"):
        raise RuntimeError("_channel_native.mc_diffraction_state_wi CUDA kernel is required")
    state_wi = native.mc_diffraction_state_wi(state_edge_pos, state_src)
    if not isinstance(state_wi, torch.Tensor):
        raise TypeError("_channel_native.mc_diffraction_state_wi must return a tensor")
    validate_cuda_tensor("state_wi", state_wi, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    return state_wi


def mc_selected_edge_indices(selected: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("selected", selected, dtype=torch.bool, ndim=1)
    native = native_extension()
    if native is None or not hasattr(native, "mc_selected_edge_indices"):
        raise RuntimeError("_channel_native.mc_selected_edge_indices CUDA kernel is required")
    indices = native.mc_selected_edge_indices(selected)
    if not isinstance(indices, torch.Tensor):
        raise TypeError("_channel_native.mc_selected_edge_indices must return a tensor")
    validate_cuda_tensor("indices", indices, dtype=torch.int32, ndim=1)
    return indices


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
    validate_cuda_tensor("edge_pos", edge_pos, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("edge_dir", edge_dir, dtype=torch.float32, ndim=2, trailing_shape=(3,))
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
        raise RuntimeError("_channel_native.mc_diffraction_state_pack CUDA kernel is required")
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
        raise TypeError("_channel_native.mc_diffraction_state_pack must return 12 tensors")
    validate_cuda_tensor("state_edge_index", states[0], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_edge_pos", states[1], dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("state_edge_dir", states[2], dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("state_line_min", states[3], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("state_line_max", states[4], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("state_n0", states[5], dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("state_n1", states[6], dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("state_face0", states[7], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_face1", states[8], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_exterior_angle", states[9], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("state_src", states[10], dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("state_src_power", states[11], dtype=torch.float32, ndim=1)
    return states


def mc_diffraction_edge_geometry(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_normals: torch.Tensor,
    edge_v0: torch.Tensor,
    edge_v1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    *,
    plane_tol: float,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor("vertices", vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("faces", faces, dtype=torch.int32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("face_normals", face_normals, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("edge_v0", edge_v0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("edge_v1", edge_v1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    if edge_v1.shape != edge_v0.shape or face0.shape != edge_v0.shape or face1.shape != edge_v0.shape:
        raise ValueError("edge_v1, face0, and face1 must match edge_v0 shape")
    native = native_extension()
    if native is None or not hasattr(native, "mc_diffraction_edge_geometry"):
        raise RuntimeError("_channel_native.mc_diffraction_edge_geometry CUDA kernel is required")
    geometry = native.mc_diffraction_edge_geometry(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        float(plane_tol),
    )
    if not isinstance(geometry, tuple) or len(geometry) != 11:
        raise TypeError("_channel_native.mc_diffraction_edge_geometry must return 11 tensors")
    validate_cuda_tensor("selected", geometry[0], dtype=torch.bool, ndim=1)
    validate_cuda_tensor("edge_pos", geometry[1], dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("edge_dir", geometry[2], dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("lengths", geometry[3], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_min", geometry[4], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_max", geometry[5], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("n0", geometry[6], dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("n1", geometry[7], dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("face0_out", geometry[8], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1_out", geometry[9], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("exterior_angle", geometry[10], dtype=torch.float32, ndim=1)
    return geometry


def mc_surface_group_edge_candidates(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_normals: torch.Tensor,
    edge_v0: torch.Tensor,
    edge_v1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    selected: torch.Tensor,
    *,
    plane_tol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_cuda_tensor("vertices", vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("faces", faces, dtype=torch.int32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("face_normals", face_normals, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("edge_v0", edge_v0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("edge_v1", edge_v1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("selected", selected, dtype=torch.bool, ndim=1)
    if edge_v1.shape != edge_v0.shape or face0.shape != edge_v0.shape or face1.shape != edge_v0.shape:
        raise ValueError("edge_v1, face0, and face1 must match edge_v0 shape")
    if selected.shape != edge_v0.shape:
        raise ValueError("selected must match edge_v0 shape")
    native = native_extension()
    if native is None or not hasattr(native, "mc_surface_group_edge_candidates"):
        raise RuntimeError("_channel_native.mc_surface_group_edge_candidates CUDA kernel is required")
    candidates = native.mc_surface_group_edge_candidates(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        selected,
        float(plane_tol),
    )
    if not isinstance(candidates, tuple) or len(candidates) != 2:
        raise TypeError("_channel_native.mc_surface_group_edge_candidates must return 2 tensors")
    counts, indices = candidates
    validate_cuda_tensor("counts", counts, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("indices", indices, dtype=torch.int32, ndim=2)
    if counts.shape[0] != faces.shape[0] or indices.shape[0] != faces.shape[0]:
        raise ValueError("_channel_native.mc_surface_group_edge_candidates returned unexpected shapes")
    return counts, indices


def mc_face_material_tensors(
    material_eps_r: torch.Tensor,
    material_sigma_e: torch.Tensor,
    material_mu_r: torch.Tensor,
    face_material_id: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("material_eps_r", material_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_sigma_e", material_sigma_e, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_mu_r", material_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_material_id", face_material_id, dtype=torch.int32, ndim=1)
    if material_sigma_e.shape != material_eps_r.shape:
        raise ValueError("material_sigma_e must match material_eps_r shape")
    if material_mu_r.shape != material_eps_r.shape:
        raise ValueError("material_mu_r must match material_eps_r shape")

    native = native_extension()
    if native is None or not hasattr(native, "mc_face_material_tensors"):
        raise RuntimeError("_channel_native.mc_face_material_tensors CUDA kernel is required")
    exported = native.mc_face_material_tensors(
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        face_material_id,
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.mc_face_material_tensors must return a dict")
    return exported


def deterministic_los_field(
    path_gain: torch.Tensor,
    path_length_m: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("path_length_m", path_length_m, dtype=torch.float32, ndim=1)
    if path_length_m.shape != path_gain.shape:
        raise ValueError("path_length_m must match path_gain")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "deterministic_los_field"):
        raise RuntimeError("_channel_native.deterministic_los_field CUDA kernel is required")
    exported = native.deterministic_los_field(path_gain, path_length_m, float(frequency_hz))
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.deterministic_los_field must return a dict")
    validate_cuda_tensor("path_gain", exported["path_gain"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_real", exported["field_real"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_imag", exported["field_imag"], dtype=torch.float32, ndim=1)
    if exported["path_gain"].shape != path_gain.shape:
        raise ValueError("_channel_native.deterministic_los_field returned bad shape")
    return exported


def deterministic_diffraction_vector_field(
    x_re: torch.Tensor,
    x_im: torch.Tensor,
    y_re: torch.Tensor,
    y_im: torch.Tensor,
    z_re: torch.Tensor,
    z_im: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("x_re", x_re, dtype=torch.float32, ndim=1)
    for name, tensor in {
        "x_im": x_im,
        "y_re": y_re,
        "y_im": y_im,
        "z_re": z_re,
        "z_im": z_im,
    }.items():
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=1)
        if tensor.shape != x_re.shape:
            raise ValueError(f"{name} must match x_re")

    native = native_extension()
    if native is None or not hasattr(native, "deterministic_diffraction_vector_field"):
        raise RuntimeError("_channel_native.deterministic_diffraction_vector_field CUDA kernel is required")
    exported = native.deterministic_diffraction_vector_field(x_re, x_im, y_re, y_im, z_re, z_im)
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.deterministic_diffraction_vector_field must return a dict")
    validate_cuda_tensor("path_gain", exported["path_gain"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_real", exported["field_real"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_imag", exported["field_imag"], dtype=torch.float32, ndim=1)
    if exported["path_gain"].shape != x_re.shape:
        raise ValueError("_channel_native.deterministic_diffraction_vector_field returned bad shape")
    return exported


def deterministic_reflection_field(
    tx_position: torch.Tensor,
    rx_position: torch.Tensor,
    hit_position: torch.Tensor,
    normal: torch.Tensor,
    tx_power: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx_position", tx_position, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("rx_position", rx_position, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("hit_position", hit_position, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("normal", normal, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("eps_r", eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("sigma_e", sigma_e, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("mu_r", mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("gain", gain, dtype=torch.float32, ndim=1)
    count = tx_position.shape[0]
    if rx_position.shape != tx_position.shape or hit_position.shape != tx_position.shape or normal.shape != tx_position.shape:
        raise ValueError("reflection field vec3 tensors must have matching shape")
    for name, tensor in {
        "tx_power": tx_power,
        "eps_r": eps_r,
        "sigma_e": sigma_e,
        "mu_r": mu_r,
        "gain": gain,
    }.items():
        if tensor.shape[0] != count:
            raise ValueError(f"{name} must match path count")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "deterministic_reflection_field"):
        raise RuntimeError("_channel_native.deterministic_reflection_field CUDA kernel is required")
    exported = native.deterministic_reflection_field(
        tx_position,
        rx_position,
        hit_position,
        normal,
        tx_power,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        float(frequency_hz),
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.deterministic_reflection_field must return a dict")
    validate_cuda_tensor("path_gain", exported["path_gain"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_real", exported["field_real"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_imag", exported["field_imag"], dtype=torch.float32, ndim=1)
    return exported


def deterministic_reflection_sequence_field(
    tx_position: torch.Tensor,
    rx_position: torch.Tensor,
    hit_positions: torch.Tensor,
    normals: torch.Tensor,
    tx_power: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx_position", tx_position, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("rx_position", rx_position, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("hit_positions", hit_positions, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("normals", normals, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("eps_r", eps_r, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("sigma_e", sigma_e, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("mu_r", mu_r, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("gain", gain, dtype=torch.float32, ndim=2)
    count = tx_position.shape[0]
    if rx_position.shape != tx_position.shape:
        raise ValueError("rx_position must match tx_position")
    if hit_positions.shape[0] != count or hit_positions.shape[2] != 3:
        raise ValueError("hit_positions must have shape (path_count, depth, 3)")
    if normals.shape != hit_positions.shape:
        raise ValueError("normals must match hit_positions")
    depth_shape = hit_positions.shape[:2]
    for name, tensor in {
        "eps_r": eps_r,
        "sigma_e": sigma_e,
        "mu_r": mu_r,
        "gain": gain,
    }.items():
        if tensor.shape != depth_shape:
            raise ValueError(f"{name} must have shape (path_count, depth)")
    if tx_power.shape[0] != count:
        raise ValueError("tx_power must match path count")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "deterministic_reflection_sequence_field"):
        raise RuntimeError("_channel_native.deterministic_reflection_sequence_field CUDA kernel is required")
    exported = native.deterministic_reflection_sequence_field(
        tx_position,
        rx_position,
        hit_positions,
        normals,
        tx_power,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        float(frequency_hz),
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.deterministic_reflection_sequence_field must return a dict")
    validate_cuda_tensor("path_gain", exported["path_gain"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_real", exported["field_real"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_imag", exported["field_imag"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("path_length_m", exported["path_length_m"], dtype=torch.float32, ndim=1)
    return exported


def deterministic_accumulate_flat(
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    component_id: torch.Tensor,
    path_gain: torch.Tensor,
    field_real: torch.Tensor,
    field_imag: torch.Tensor,
    *,
    num_tx: int,
    num_rx: int,
    coherent: bool,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx_id", tx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("component_id", component_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_real", field_real, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_imag", field_imag, dtype=torch.float32, ndim=1)
    for name, tensor in {
        "rx_id": rx_id,
        "component_id": component_id,
        "path_gain": path_gain,
        "field_real": field_real,
        "field_imag": field_imag,
    }.items():
        if tensor.shape != tx_id.shape:
            raise ValueError(f"{name} must match tx_id shape")
    if num_tx < 0 or num_rx < 0:
        raise ValueError("num_tx and num_rx must be non-negative")

    native = native_extension()
    if native is None or not hasattr(native, "deterministic_accumulate_flat"):
        raise RuntimeError("_channel_native.deterministic_accumulate_flat CUDA kernel is required")
    exported = native.deterministic_accumulate_flat(
        tx_id,
        rx_id,
        component_id,
        path_gain,
        field_real,
        field_imag,
        int(num_tx),
        int(num_rx),
        bool(coherent),
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.deterministic_accumulate_flat must return a dict")
    validate_cuda_tensor("power_total", exported["power_total"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("field_total_real", exported["field_total_real"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("field_total_imag", exported["field_total_imag"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("component_power", exported["component_power"], dtype=torch.float32, ndim=3)
    validate_cuda_tensor("component_field_real", exported["component_field_real"], dtype=torch.float32, ndim=3)
    validate_cuda_tensor("component_field_imag", exported["component_field_imag"], dtype=torch.float32, ndim=3)
    expected_component_shape = (3, int(num_tx), int(num_rx))
    if tuple(exported["power_total"].shape) != (int(num_tx), int(num_rx)):
        raise ValueError("_channel_native.deterministic_accumulate_flat returned bad power_total shape")
    if tuple(exported["component_power"].shape) != expected_component_shape:
        raise ValueError("_channel_native.deterministic_accumulate_flat returned bad component shape")
    return exported
