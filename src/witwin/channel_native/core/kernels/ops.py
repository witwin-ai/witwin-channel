from __future__ import annotations

import torch

from witwin.channel_native.core.rayd_native_handles import (  # noqa: F401
    _raydn_module_handle,
    _raydn_scene_handle_id,
)
from witwin.channel_native.propagation.topology.kernels.blocks import (  # noqa: F401
    _DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA,
    _PATH_BLOCK_SCHEMA,
    _validate_deterministic_topology_block,
    _validate_path_block,
    _validate_path_reflection_candidates,
    _validate_topology_extra_fields,
    deterministic_concat_topology_blocks,
    deterministic_gather_topology_block,
    path_diffraction_block,
    path_filter_block,
    path_filter_los,
    path_finalize_blocks,
    path_los_export,
    path_los_visibility_inputs,
    path_merge_blocks,
)
from witwin.channel_native.propagation.topology.kernels.candidates import (  # noqa: F401
    path_diffraction_paths_order1,
    path_reflection_candidates,
)
from witwin.channel_native.propagation.topology.kernels.compaction import (  # noqa: F401
    deterministic_diffraction_order1_compact,
    deterministic_reflection_order1_compact,
    deterministic_reflection_sequence_compact,
    deterministic_sort_order,
)
from witwin.channel_native.propagation.topology.kernels.construction import (  # noqa: F401
    deterministic_face_anchor_points,
    deterministic_face_sequence_chunk,
    deterministic_los_topology_block,
    deterministic_mapped_face_sequence_chunk,
    deterministic_pad_topology_sequences,
    deterministic_reflection_epc_input_batch,
    deterministic_repeat_range,
    deterministic_topology_base_fields,
    deterministic_topology_default_fields,
)
from witwin.channel_native.propagation.topology.kernels.primitives import (  # noqa: F401
    core_pack_int2,
    deterministic_component_counts,
    deterministic_diffraction_state_pack,
    deterministic_diffraction_state_pack_selected,
    deterministic_selected_edge_count,
    mc_selected_edge_indices,
    path_concat_vec3,
)
from witwin.channel_native.scattering.kernels.functional import (  # noqa: F401
    scattering_event_probabilities,
    scattering_table_eval,
    scattering_table_pdf,
    scattering_table_sample,
)
from witwin.channel_native.materials.kernels.functional import (  # noqa: F401
    _EM_LAYER_STACK_FIELDS,
    bdpt_face_material_tensors,
    bdpt_face_material_tensors_from_host,
    em_layer_stack_backward,
    em_layer_stack_eval,
    em_layer_stack_jvp,
    mc_face_material_tensors,
)
from witwin.channel_native.materials.kernels.autograd import (  # noqa: F401
    _EmLayerStackAdFunction,
    em_layer_stack_ad,
)
from witwin.channel_native.propagation.fields.kernels.autograd import (  # noqa: F401
    _CoupledRdPrepareAdFunction,
    _FieldCoupledRdAdFunction,
    _FieldDiffractionWedgeAdFunction,
    _FieldFreeSpaceAdFunction,
    _FieldProjectComplex3AdFunction,
    _FieldReflectionSequenceAdFunction,
    _FieldTransmissionSequenceAdFunction,
    coupled_rd_prepare_ad,
    field_coupled_rd_ad,
    field_diffraction_wedge_ad,
    field_free_space_ad,
    field_project_complex3_ad,
    field_reflection_sequence_ad,
    field_transmission_sequence_ad,
)
from witwin.channel_native.propagation.fields.kernels.functional import (  # noqa: F401
    _COUPLED_OUTPUT_FIELDS,
    _FIELD_AD_OUTPUT_FIELDS,
    _FIELD_AD_TANGENT_FIELDS,
    _WEDGE_OUTPUT_FIELDS,
    field_coupled_rd,
    field_diffraction_wedge,
    field_free_space,
    field_free_space_backward,
    field_free_space_jvp,
    field_project_complex3,
    field_reflection_sequence,
    field_reflection_sequence_backward,
    field_reflection_sequence_jvp,
    field_transmission_sequence,
    field_transmission_sequence_backward,
    field_transmission_sequence_jvp,
)
from witwin.channel_native.propagation.geometry.kernels.bridge import (  # noqa: F401
    bdpt_diffraction_accumulation_forward,
    bdpt_diffraction_discover_edges,
    bdpt_diffraction_discover_edges_counted,
    bdpt_intersect_forward,
    bdpt_reflection_accumulation_forward,
    bdpt_visibility_forward,
    raydn_coupled_rd_geometry_forward,
    raydn_diffraction_accumulation_forward,
    raydn_diffraction_discover_edges,
    raydn_diffraction_discover_edges_counted,
    raydn_diffraction_paths_order1_forward,
    raydn_reflection_accumulation_forward,
    raydn_reflection_epc_paths_forward,
    raydn_trace_reflections_forward,
    raydn_visibility_forward,
)
from witwin.channel_native.propagation.geometry.kernels.autograd import (  # noqa: F401
    _RaydnFaceNormalsAdFunction,
    _RaydnIntersectAdFunction,
    _RaydnReflectionEpcPathsAdFunction,
    _RaydnTraceReflectionsAdFunction,
    _epc_paths_frozen_winner_checks,
    raydn_face_normals_ad,
    raydn_intersect_ad,
    raydn_intersect_backward,
    raydn_intersect_jvp,
    raydn_reflection_epc_paths_ad,
    raydn_reflection_epc_paths_backward,
    raydn_reflection_epc_paths_jvp,
    raydn_scene_face_normals_backward,
    raydn_scene_face_normals_jvp,
    raydn_trace_reflections_ad,
    raydn_trace_reflections_backward,
    raydn_trace_reflections_forward_tape,
    raydn_trace_reflections_jvp,
)
from witwin.channel_native.propagation.geometry.kernels.primitives import (  # noqa: F401
    bdpt_diffraction_edge_geometry,
    bdpt_surface_group_edge_candidates,
    core_diffraction_edge_count,
    deterministic_face_groups,
    deterministic_normalize_vec3,
    deterministic_reflect_points,
    deterministic_surface_face_groups,
    mc_diffraction_edge_geometry,
    mc_surface_group_edge_candidates,
)
from witwin.channel_native.runtime import symbols as _native_symbols
from witwin.channel_native.runtime.autograd_contracts import (  # noqa: F401
    _ad_active_ctx,
    _ad_check_active,
    _ad_check_optional_grad,
    _ad_check_rows,
    _ad_check_tangent_vec3,
    _ad_checked_tangent,
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_geometry_live,
    _ad_geometry_tangent,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_raise_composed_transforms,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
    _ad_still_wrapped,
)
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor
from witwin.channel_native.scene.kernels.rayd_scene import (  # noqa: F401
    raydn_scene_create,
    raydn_scene_edge_records,
)

from .metadata import make_metadata, noop_metadata, validate_metadata  # noqa: F401


native_extension = _native_symbols.native_extension


def _required_native_op(name: str):
    return _native_symbols._required_symbol(native_extension(), name)


def bdpt_zero_matrix(reference: torch.Tensor, *, rows: int, cols: int) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    native = native_extension()
    if native is None or not hasattr(native, "bdpt_zero_matrix"):
        raise RuntimeError("_channel_native.bdpt_zero_matrix CUDA kernel is required")
    out = native.bdpt_zero_matrix(reference, int(rows), int(cols))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel_native.bdpt_zero_matrix must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2)
    if out.shape != (int(rows), int(cols)):
        raise ValueError(
            "_channel_native.bdpt_zero_matrix returned an unexpected shape"
        )
    return out


def bdpt_store_point_component_column(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    rx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("target", target, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=3)
    if source.shape[0] != target.shape[0] or source.shape[1:] != (1, 1):
        raise ValueError("source must have shape (tx, 1, 1)")
    if rx_index < 0 or rx_index >= target.shape[1]:
        raise ValueError("rx_index is out of range")
    native = _required_native_op("bdpt_store_point_component_column")
    out = native(target, source, int(rx_index))
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_store_point_component_column must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2)
    if out.shape != target.shape:
        raise ValueError(
            "_channel_native.bdpt_store_point_component_column returned an unexpected shape"
        )
    return out


def bdpt_finalize_point_components(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("reflection", reflection, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("diffraction", diffraction, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("transmission", transmission, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("scattering", scattering, dtype=torch.float32, ndim=2)
    if (
        reflection.shape != los.shape
        or diffraction.shape != los.shape
        or transmission.shape != los.shape
        or scattering.shape != los.shape
    ):
        raise ValueError("point component matrices must have matching shapes")
    exported = _required_native_op("bdpt_finalize_point_components")(
        los, reflection, diffraction, transmission, scattering
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_finalize_point_components must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=2
    )
    if exported["path_gain"].shape != los.shape:
        raise ValueError(
            "_channel_native.bdpt_finalize_point_components returned bad path_gain shape"
        )
    for name in (
        "los_power",
        "reflection_power",
        "diffraction_power",
        "transmission_power",
        "scattering_power",
    ):
        validate_cuda_tensor(name, exported[name], dtype=torch.float32, ndim=0)
    return exported


def bdpt_point_component_power(
    path_gain: torch.Tensor, *, include_los: bool
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=2)
    native = native_extension()
    if native is None or not hasattr(native, "bdpt_point_component_power"):
        raise RuntimeError(
            "_channel_native.bdpt_point_component_power CUDA kernel is required"
        )
    exported = native.bdpt_point_component_power(path_gain, bool(include_los))
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.bdpt_point_component_power must return a dict")
    for name in ("los", "reflection", "diffraction"):
        validate_cuda_tensor(name, exported[name], dtype=torch.float32, ndim=0)
    return exported


def bdpt_transmitter_tensors(
    flat_positions: tuple[float, ...],
    powers: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    if len(flat_positions) % 3 != 0:
        raise ValueError("flat_positions must contain xyz triples")
    if len(flat_positions) // 3 != len(powers):
        raise ValueError("powers must match flat_positions")
    exported = _required_native_op("bdpt_transmitter_tensors")(flat_positions, powers)
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.bdpt_transmitter_tensors must return a dict")
    validate_cuda_tensor(
        "positions",
        exported["positions"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor("power", exported["power"], dtype=torch.float32, ndim=1)
    return exported


def bdpt_host_vec3_tensor(flat_positions: tuple[float, ...]) -> torch.Tensor:
    if len(flat_positions) % 3 != 0:
        raise ValueError("flat_positions must contain xyz triples")
    powers = tuple(1.0 for _ in range(len(flat_positions) // 3))
    return bdpt_transmitter_tensors(flat_positions, powers)["positions"]


def bdpt_receiver_grid_points(
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
    points = _required_native_op("bdpt_receiver_grid_points")(
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
        raise TypeError(
            "_channel_native.bdpt_receiver_grid_points must return a tensor"
        )
    validate_cuda_tensor(
        "points", points, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if points.shape[0] != rows * cols:
        raise ValueError(
            "_channel_native.bdpt_receiver_grid_points returned an unexpected shape"
        )
    return points


def bdpt_los_export(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
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
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must have one value per transmitter")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    exported = _required_native_op("bdpt_los_export")(
        tx_positions,
        tx_power,
        rx_positions,
        float(frequency_hz),
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.bdpt_los_export must return a dict")
    validate_cuda_tensor(
        "path_gain_matrix", exported["path_gain_matrix"], dtype=torch.float32, ndim=2
    )
    return exported


def bdpt_los_component_maps(los: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    maps = _required_native_op("bdpt_los_component_maps")(los)
    if not isinstance(maps, torch.Tensor):
        raise TypeError("_channel_native.bdpt_los_component_maps must return a tensor")
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != los.shape:
        raise ValueError(
            "_channel_native.bdpt_los_component_maps returned an unexpected shape"
        )
    return maps


def bdpt_los_component_maps_from_matrix(
    los: torch.Tensor, *, rows: int, cols: int
) -> torch.Tensor:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    if los.shape[1] != int(rows) * int(cols):
        raise ValueError("los columns must match rows * cols")
    maps = _required_native_op("bdpt_los_component_maps_from_matrix")(
        los, int(rows), int(cols)
    )
    if not isinstance(maps, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_los_component_maps_from_matrix must return a tensor"
        )
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (los.shape[0], int(cols), int(rows)):
        raise ValueError(
            "_channel_native.bdpt_los_component_maps_from_matrix returned an unexpected shape"
        )
    return maps


def bdpt_los_visibility_inputs(
    tx_positions: torch.Tensor,
    *,
    tx_index: int,
    rx_count: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if rx_count < 0:
        raise ValueError("rx_count must be non-negative")
    exported = _required_native_op("bdpt_los_visibility_inputs")(
        tx_positions, int(tx_index), int(rx_count)
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.bdpt_los_visibility_inputs must return a dict")
    validate_cuda_tensor(
        "start", exported["start"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    return exported


def bdpt_apply_los_visibility(
    maps: torch.Tensor,
    los: torch.Tensor,
    visible: torch.Tensor,
    *,
    tx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if maps.shape[0] != los.shape[0] or los.shape[1] != maps.shape[1] * maps.shape[2]:
        raise ValueError("los must match maps flattened grid layout")
    out = _required_native_op("bdpt_apply_los_visibility")(
        maps, los, visible, int(tx_index)
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_apply_los_visibility must return a tensor"
        )
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def bdpt_component_map_buffer(
    reference: torch.Tensor,
    *,
    tx_count: int,
    dim0: int,
    dim1: int,
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if tx_count < 0 or dim0 < 0 or dim1 < 0:
        raise ValueError("tx_count, dim0, and dim1 must be non-negative")
    maps = _required_native_op("bdpt_component_map_buffer")(
        reference, int(tx_count), int(dim0), int(dim1)
    )
    if not isinstance(maps, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_component_map_buffer must return a tensor"
        )
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (tx_count, dim0, dim1):
        raise ValueError(
            "_channel_native.bdpt_component_map_buffer returned an unexpected shape"
        )
    return maps


def bdpt_store_component_map(
    maps: torch.Tensor,
    source: torch.Tensor,
    *,
    tx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2)
    out = _required_native_op("bdpt_store_component_map")(maps, source, int(tx_index))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel_native.bdpt_store_component_map must return a tensor")
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def bdpt_store_scaled_component_map(
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
    out = _required_native_op("bdpt_store_scaled_component_map")(
        maps,
        source,
        scale_values,
        int(tx_index),
        int(scale_index),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_store_scaled_component_map must return a tensor"
        )
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def bdpt_finalize_component_maps(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("reflection", reflection, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("diffraction", diffraction, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("transmission", transmission, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("scattering", scattering, dtype=torch.float32, ndim=3)
    if (
        reflection.shape != los.shape
        or diffraction.shape != los.shape
        or transmission.shape != los.shape
        or scattering.shape != los.shape
    ):
        raise ValueError("component maps must share shape")
    exported = _required_native_op("bdpt_finalize_component_maps")(
        los, reflection, diffraction, transmission, scattering
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_finalize_component_maps must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=3
    )
    if exported["path_gain"].shape != los.shape:
        raise ValueError(
            "_channel_native.bdpt_finalize_component_maps returned bad path_gain shape"
        )
    for name in (
        "los_power",
        "reflection_power",
        "diffraction_power",
        "transmission_power",
        "scattering_power",
    ):
        validate_cuda_tensor(name, exported[name], dtype=torch.float32, ndim=0)
    return exported


def bdpt_sample_directions(
    count: int, reference: torch.Tensor, *, seed: int
) -> torch.Tensor:
    if count < 0:
        raise ValueError("count must be non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    directions = _required_native_op("bdpt_sample_directions")(
        int(count), reference, int(seed)
    )
    if not isinstance(directions, torch.Tensor):
        raise TypeError("_channel_native.bdpt_sample_directions must return a tensor")
    validate_cuda_tensor(
        "directions", directions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return directions


def bdpt_reflection_launch_inputs(
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
    exported = _required_native_op("bdpt_reflection_launch_inputs")(
        tx_positions, int(tx_index), int(sample_count)
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_reflection_launch_inputs must return a dict"
        )
    validate_cuda_tensor(
        "ray_o", exported["ray_o"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("ray_tmax", exported["ray_tmax"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "tx_pol", exported["tx_pol"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_id", exported["tx_id"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "light_seed", exported["light_seed"], dtype=torch.int64, ndim=1
    )
    return exported


def bdpt_diffraction_state_wi(
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
    state_wi = _required_native_op("bdpt_diffraction_state_wi")(
        state_edge_pos, state_src
    )
    if not isinstance(state_wi, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_diffraction_state_wi must return a tensor"
        )
    validate_cuda_tensor(
        "state_wi", state_wi, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return state_wi


def bdpt_selected_edge_indices(selected: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("selected", selected, dtype=torch.bool, ndim=1)
    indices = _required_native_op("bdpt_selected_edge_indices")(selected)
    if not isinstance(indices, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_selected_edge_indices must return a tensor"
        )
    validate_cuda_tensor("indices", indices, dtype=torch.int32, ndim=1)
    return indices


def bdpt_diffraction_state_pack(
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
    states = _required_native_op("bdpt_diffraction_state_pack")(
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
            "_channel_native.bdpt_diffraction_state_pack must return 12 tensors"
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


def bdpt_pack_vec3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("x", x, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y", y, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z", z, dtype=torch.float32, ndim=1)
    packed = _required_native_op("bdpt_pack_vec3")(x, y, z)
    if not isinstance(packed, torch.Tensor):
        raise TypeError("_channel_native.bdpt_pack_vec3 must return a tensor")
    validate_cuda_tensor(
        "packed", packed, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return packed






# ---------------------------------------------------------------------------
# RayDN differentiable geometry (AD-A0).
#
# These entry points expose RayD's fixed-winner differentiable geometry
# kernels across the source-linked C bridge (no rayd.torch import, no torch
# dispatcher). Discrete winner records (prim ids, barycentrics, hit points,
# normals) are detached tape constants; gradients flow only through the
# continuous outputs (t, p, n, image sources, EPC field, path length) with
# respect to mesh vertices, ray origin/direction, and EPC source/receiver.
# They are opt-in AD entry points: no public solver calls them and the
# ad_mode="none" forward paths are untouched.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RayDN reflection EPC paths geometry AD (plan 07 AD-2 layer 1).
#
# The reflection EPC path export (direct-plane mode) is the discovery entry
# the deterministic/path solvers use. Its geometry companions differentiate
# the continuous specular chain (hit points, emitted unit normals, path
# length) with respect to the scene vertex table and the per-row endpoints,
# under a frozen winner: the face sequence, the validity mask, the
# containment resolution and the visibility casts are detached discovery
# records. RayD chains each bounce's plane cotangents to the winner
# triangle's vertices itself, so no hit geometry is ever re-derived here.
# ---------------------------------------------------------------------------
































# ---------------------------------------------------------------------------
# Fixed-topology EM-response AD for the field transport kernels (plan 07
# AD-1 materials/frequency, AD-2 geometry). Differentiable inputs:
# per-bounce/per-layer eps_r, sigma_e, gain, thickness, the carrier frequency
# and the continuous hit geometry (source, target, interaction_positions,
# interaction_normals). The discrete winner (polarizations, tx_power, mu_r,
# material ids, valid masks) stays fixed and fails loudly. path_length_m /
# delay_s become differentiable outputs exactly when a geometry input is on
# the graph; they carry zero cotangent into materials and frequency.
# ---------------------------------------------------------------------------



























# ---------------------------------------------------------------------------
# Plan 07 AD-4: differentiable UTD wedge diffraction (component 2 re-evaluated
# from the frozen topology), receiver projection, coupled R-D transport
# (components 3/4) and the coupled stationary-geometry re-solve. Each
# torch.autograd.Function below is dispatch only; the math lives in
# kernels/field_wedge_ad.cu (RayD's templated dual forward).
# ---------------------------------------------------------------------------

























# Import this compatibility facade only after the legacy bodies above are
# defined.  ``core.scene`` reaches this module while the public package is
# still initializing; the late import lets deterministic's eager public API
# reuse the already-defined facade without exposing a partially built module.
from witwin.channel_native.deterministic.kernels.accumulation import (  # noqa: E402,F401
    _DETERMINISTIC_ACCUM_FIELDS,
    _DeterministicAccumulateFlatAdFunction,
    deterministic_accumulate_flat,
    deterministic_accumulate_flat_ad,
    deterministic_accumulate_flat_backward,
    deterministic_accumulate_flat_jvp,
)
from witwin.channel_native.deterministic.kernels.fields import (  # noqa: E402,F401
    deterministic_delay_to_path_length,
    deterministic_diffraction_vector_field,
    deterministic_field_from_power_phase,
    deterministic_los_field,
    deterministic_pack_complex,
    deterministic_phase_from_field,
    deterministic_phase_from_length,
    deterministic_reflection_field,
    deterministic_reflection_sequence_field,
    deterministic_zero_field_phase,
)
from witwin.channel_native.montecarlo.basic.kernels.sampling import (  # noqa: E402,F401
    mc_diffraction_state_pack,
    mc_diffraction_state_wi,
    mc_pack_vec3,
    mc_receiver_grid_points,
    mc_reflection_launch_inputs,
    mc_sample_directions,
    mc_transmitter_tensors,
)
from witwin.channel_native.montecarlo.basic.kernels.maps import (  # noqa: E402,F401
    _LIGHT_SPEED_M_PER_S_AD,
    _MC_FINALIZE_FIELDS,
    _McFinalizeComponentMapsAdFunction,
    _McDiffractionMapAdFunction,
    _McLosGridMapsAdFunction,
    _McLosPathGainAdFunction,
    _McReflectionMapAdFunction,
    mc_apply_los_visibility,
    mc_component_map_buffer,
    mc_finalize_component_maps,
    mc_finalize_component_maps_ad,
    mc_los_component_maps,
    mc_los_component_maps_adjoint,
    mc_los_component_maps_from_matrix,
    mc_los_grid_maps_ad,
    mc_los_path_gain_ad,
    mc_los_path_gain_backward,
    mc_los_path_gain_jvp,
    mc_los_visibility_inputs,
    mc_point_component_power,
    mc_reflection_ad_max_depth,
    mc_sionna_diffraction_tape_accumulate,
    mc_sionna_diffraction_tape_accumulate_ad,
    mc_sionna_diffraction_tape_accumulate_backward,
    mc_sionna_diffraction_tape_accumulate_jvp,
    mc_sionna_reflection_accumulate,
    mc_sionna_reflection_accumulate_ad,
    mc_sionna_reflection_accumulate_backward,
    mc_sionna_reflection_accumulate_jvp,
    mc_store_component_map,
    mc_store_scaled_component_map,
    mc_zero_matrix,
)
from witwin.channel_native.montecarlo.bdpt.kernels.paths import (  # noqa: E402,F401
    _BDPT_CONNECTION_SCHEMA,
    _BDPT_SUBPATH_SCHEMA,
    _bdpt_mis_mode_id,
    _validate_bdpt_connection_samples,
    _validate_bdpt_subpath_state,
    bdpt_accumulate_connection_samples,
    bdpt_compact_connection_samples,
    bdpt_concat_connection_samples,
    bdpt_connection_variance,
    bdpt_count_valid_connection_samples,
    bdpt_diffraction_connection_samples_from_tape,
    bdpt_diffraction_point_connection_samples,
    bdpt_empty_subpath_state,
    bdpt_endpoint_connection_samples,
    bdpt_endpoint_connection_visibility_inputs,
    bdpt_endpoint_subpath_state,
    bdpt_filter_connection_samples,
    bdpt_launch_state,
    bdpt_mis_weights,
    bdpt_reflected_light_subpath_state,
    bdpt_subpath_intersection_inputs,
    bdpt_transmitted_light_subpath_state,
)
