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
from witwin.channel_native.materials.kernels.contracts import (  # noqa: F401
    _validate_layer_csr,
)
from witwin.channel_native.propagation.geometry.kernels.bridge import (  # noqa: F401
    _BDPT_INTERSECTION_FIELDS,
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
from witwin.channel_native.runtime import torch_compat
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


def bdpt_launch_state(
    reference: torch.Tensor,
    *,
    tx_count: int,
    samples: int,
    sample_streams: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if tx_count < 0:
        raise ValueError("tx_count must be non-negative")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if sample_streams <= 0:
        raise ValueError("sample_streams must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    native = native_extension()
    if native is None or not hasattr(native, "bdpt_launch_state"):
        raise RuntimeError("_channel_native.bdpt_launch_state CUDA kernel is required")
    exported = native.bdpt_launch_state(
        reference,
        int(tx_count),
        int(samples),
        int(sample_streams),
        int(seed),
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.bdpt_launch_state must return a dict")
    expected = int(tx_count) * int(samples) * int(sample_streams)
    for name in ("tx_id", "sample_id", "stream_id"):
        validate_cuda_tensor(name, exported[name], dtype=torch.int32, ndim=1)
        if exported[name].shape != (expected,):
            raise ValueError(
                f"_channel_native.bdpt_launch_state returned bad {name} shape"
            )
    for name in ("light_seed",):
        validate_cuda_tensor(name, exported[name], dtype=torch.int64, ndim=1)
        if exported[name].shape != (expected,):
            raise ValueError(
                f"_channel_native.bdpt_launch_state returned bad {name} shape"
            )
    return exported


_BDPT_SUBPATH_SCHEMA: dict[str, tuple[torch.dtype, tuple[int | None, ...]]] = {
    "origin": (torch.float32, (None, 3)),
    "direction": (torch.float32, (None, 3)),
    "throughput_real": (torch.float32, (None,)),
    "throughput_imag": (torch.float32, (None,)),
    "pdf_forward": (torch.float32, (None,)),
    "pdf_reverse": (torch.float32, (None,)),
    "depth": (torch.int32, (None,)),
    "component_mask": (torch.int32, (None,)),
    "primitive_id": (torch.int32, (None,)),
    "edge_id": (torch.int32, (None,)),
    "tx_id": (torch.int32, (None,)),
    "rx_id": (torch.int32, (None,)),
    "grid_linear_id": (torch.int32, (None,)),
    "valid": (torch.bool, (None,)),
    "path_length": (torch.float32, (None,)),
    "field_real": (torch.float32, (None, 3)),
    "field_imag": (torch.float32, (None, 3)),
    "source_power": (torch.float32, (None,)),
    "event_type": (torch.int32, (None,)),
}


def _validate_bdpt_subpath_state(
    name: str, exported: dict[str, torch.Tensor], expected_count: int | None
) -> None:
    if not isinstance(exported, dict):
        raise TypeError(f"{name} must be a dict")
    if set(exported) != set(_BDPT_SUBPATH_SCHEMA):
        raise ValueError(f"{name} returned unexpected fields")
    inferred_count: int | None = expected_count
    for field, (dtype, shape_spec) in _BDPT_SUBPATH_SCHEMA.items():
        tensor = exported[field]
        validate_cuda_tensor(field, tensor, dtype=dtype, ndim=len(shape_spec))
        if inferred_count is None:
            inferred_count = int(tensor.shape[0])
        expected_shape = tuple(
            inferred_count if dim is None else dim for dim in shape_spec
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} returned bad {field} shape")


_BDPT_CONNECTION_SCHEMA: dict[str, tuple[torch.dtype, tuple[int | None, ...]]] = {
    "topology": (torch.int32, (None, 4)),
    "contribution": (torch.float32, (None,)),
    "pdf": (torch.float32, (None,)),
    "mis_weight": (torch.float32, (None,)),
    "component_id": (torch.int32, (None,)),
    "valid": (torch.bool, (None,)),
    "tx_id": (torch.int32, (None,)),
    "rx_id": (torch.int32, (None,)),
    "grid_linear_id": (torch.int32, (None,)),
    "light_depth": (torch.int32, (None,)),
    "sensor_depth": (torch.int32, (None,)),
    "path_length_m": (torch.float32, (None,)),
}


def _validate_bdpt_connection_samples(
    name: str,
    exported: dict[str, torch.Tensor],
    expected_count: int | None,
) -> None:
    if not isinstance(exported, dict):
        raise TypeError(f"{name} must be a dict")
    if set(exported) != set(_BDPT_CONNECTION_SCHEMA):
        raise ValueError(f"{name} returned unexpected fields")
    inferred_count = expected_count
    for field, (dtype, shape_spec) in _BDPT_CONNECTION_SCHEMA.items():
        tensor = exported[field]
        validate_cuda_tensor(
            f"{name}.{field}", tensor, dtype=dtype, ndim=len(shape_spec)
        )
        if inferred_count is None:
            inferred_count = int(tensor.shape[0])
        expected_shape = tuple(
            inferred_count if dim is None else dim for dim in shape_spec
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} returned bad {field} shape")


def _bdpt_mis_mode_id(mis: str) -> int:
    if mis == "none":
        return 0
    if mis == "balance":
        return 1
    if mis == "power_heuristic":
        return 2
    raise ValueError("mis is not supported")


def bdpt_empty_subpath_state(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    exported = _required_native_op("bdpt_empty_subpath_state")(reference)
    _validate_bdpt_subpath_state(
        "_channel_native.bdpt_empty_subpath_state", exported, 0
    )
    return exported


def bdpt_endpoint_subpath_state(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor,
    rx_polarization: torch.Tensor,
    launch_tx_id: torch.Tensor,
    light_seed: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "tx_polarization", tx_polarization, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_polarization", rx_polarization, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("launch_tx_id", launch_tx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("light_seed", light_seed, dtype=torch.int64, ndim=1)
    if tx_power.shape != (tx_positions.shape[0],):
        raise ValueError("tx_power must match tx_positions")
    if tx_polarization.shape != tx_positions.shape:
        raise ValueError("tx_polarization must match tx_positions")
    if rx_polarization.shape != rx_positions.shape:
        raise ValueError("rx_polarization must match rx_positions")
    if light_seed.shape != launch_tx_id.shape:
        raise ValueError("light_seed must match launch_tx_id")
    device = tx_positions.get_device()
    if (
        tx_power.get_device() != device
        or tx_polarization.get_device() != device
        or rx_positions.get_device() != device
        or rx_polarization.get_device() != device
        or launch_tx_id.get_device() != device
        or light_seed.get_device() != device
    ):
        raise ValueError("BDPT endpoint tensors must share one CUDA device")
    exported = _required_native_op("bdpt_endpoint_subpath_state")(
        tx_positions,
        tx_power,
        tx_polarization,
        rx_positions,
        rx_polarization,
        launch_tx_id,
        light_seed,
    )
    if not isinstance(exported, dict) or set(exported) != {"light", "sensor"}:
        raise TypeError(
            "_channel_native.bdpt_endpoint_subpath_state must return light/sensor dicts"
        )
    light = exported["light"]
    sensor = exported["sensor"]
    _validate_bdpt_subpath_state(
        "_channel_native.bdpt_endpoint_subpath_state.light",
        light,
        int(launch_tx_id.shape[0]),
    )
    _validate_bdpt_subpath_state(
        "_channel_native.bdpt_endpoint_subpath_state.sensor",
        sensor,
        int(rx_positions.shape[0]),
    )
    return {"light": light, "sensor": sensor}


def bdpt_subpath_intersection_inputs(
    subpath: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("subpath", subpath, None)
    exported = _required_native_op("bdpt_subpath_intersection_inputs")(subpath)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_subpath_intersection_inputs must return a dict"
        )
    if set(exported) != {"ray_o", "ray_d", "ray_tmax", "active"}:
        raise ValueError(
            "_channel_native.bdpt_subpath_intersection_inputs returned unexpected fields"
        )
    validate_cuda_tensor(
        "ray_o", exported["ray_o"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "ray_d", exported["ray_d"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("ray_tmax", exported["ray_tmax"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    if (
        exported["ray_o"].shape != subpath["origin"].shape
        or exported["ray_d"].shape != subpath["direction"].shape
    ):
        raise ValueError(
            "_channel_native.bdpt_subpath_intersection_inputs returned bad ray shape"
        )
    if exported["active"].shape != subpath["valid"].shape:
        raise ValueError(
            "_channel_native.bdpt_subpath_intersection_inputs returned bad active shape"
        )
    if exported["ray_tmax"].shape != (0,):
        raise ValueError(
            "_channel_native.bdpt_subpath_intersection_inputs returned bad ray_tmax shape"
        )
    return exported


def bdpt_reflected_light_subpath_state(
    light: dict[str, torch.Tensor],
    intersection: dict[str, torch.Tensor],
    *,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_eps_r: torch.Tensor,
    material_sigma_e: torch.Tensor,
    material_mu_r: torch.Tensor,
    material_thickness: torch.Tensor,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    if not isinstance(intersection, dict) or set(intersection) != set(
        _BDPT_INTERSECTION_FIELDS
    ):
        raise ValueError("intersection returned unexpected fields")
    count = int(light["origin"].shape[0])
    validate_cuda_tensor(
        "intersection.t", intersection["t"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "intersection.p",
        intersection["p"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.n",
        intersection["n"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.global_prim_id",
        intersection["global_prim_id"],
        dtype=torch.int32,
        ndim=1,
    )
    validate_cuda_tensor("material_gain", material_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_valid", material_valid, dtype=torch.bool, ndim=1)
    validate_cuda_tensor("material_eps_r", material_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "material_sigma_e", material_sigma_e, dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor("material_mu_r", material_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "material_thickness", material_thickness, dtype=torch.float32, ndim=1
    )
    if int(material_gain.shape[0]) != int(material_valid.shape[0]):
        raise ValueError("material_gain and material_valid must have matching length")
    for name, tensor in (
        ("material_eps_r", material_eps_r),
        ("material_sigma_e", material_sigma_e),
        ("material_mu_r", material_mu_r),
        ("material_thickness", material_thickness),
    ):
        if int(tensor.shape[0]) != int(material_gain.shape[0]):
            raise ValueError(f"{name} must match material_gain length")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if (
        material_gain.get_device() != light["origin"].get_device()
        or material_valid.get_device() != light["origin"].get_device()
        or material_eps_r.get_device() != light["origin"].get_device()
        or material_sigma_e.get_device() != light["origin"].get_device()
        or material_mu_r.get_device() != light["origin"].get_device()
        or material_thickness.get_device() != light["origin"].get_device()
    ):
        raise ValueError("material tensors must share light device")
    for name in ("t", "p", "n", "global_prim_id"):
        if int(intersection[name].shape[0]) != count:
            raise ValueError("intersection must match light subpath count")
        if intersection[name].get_device() != light["origin"].get_device():
            raise ValueError("intersection tensors must share light device")
    exported = _required_native_op("bdpt_reflected_light_subpath_state")(
        light,
        intersection,
        material_gain,
        material_valid,
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        material_thickness,
        float(frequency_hz),
    )
    _validate_bdpt_subpath_state(
        "_channel_native.bdpt_reflected_light_subpath_state", exported, count
    )
    return exported


def bdpt_transmitted_light_subpath_state(
    light: dict[str, torch.Tensor],
    intersection: dict[str, torch.Tensor],
    *,
    face_material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    if not isinstance(intersection, dict) or set(intersection) != set(
        _BDPT_INTERSECTION_FIELDS
    ):
        raise ValueError("intersection returned unexpected fields")
    count = int(light["origin"].shape[0])
    validate_cuda_tensor(
        "intersection.t", intersection["t"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "intersection.p",
        intersection["p"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.n",
        intersection["n"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.global_prim_id",
        intersection["global_prim_id"],
        dtype=torch.int32,
        ndim=1,
    )
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    device = light["origin"].get_device()
    if face_material_id.get_device() != device:
        raise ValueError("face_material_id must share light device")
    _validate_layer_csr(
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        device,
    )
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    for name in ("t", "p", "n", "global_prim_id"):
        if int(intersection[name].shape[0]) != count:
            raise ValueError("intersection must match light subpath count")
        if intersection[name].get_device() != device:
            raise ValueError("intersection tensors must share light device")
    exported = _required_native_op("bdpt_transmitted_light_subpath_state")(
        light,
        intersection,
        face_material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
    )
    _validate_bdpt_subpath_state(
        "_channel_native.bdpt_transmitted_light_subpath_state", exported, count
    )
    return exported


def bdpt_endpoint_connection_samples(
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    *,
    frequency_hz: float,
    samples_per_tx: int,
    max_paths: int | None = None,
    mis: str = "power_heuristic",
    beta: float = 2.0,
    strategy_count: int = 1,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    _validate_bdpt_subpath_state("sensor", sensor, None)
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if samples_per_tx <= 0:
        raise ValueError("samples_per_tx must be positive")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if strategy_count != 1:
        raise ValueError("endpoint connections support exactly one strategy")
    max_paths_value = -1 if max_paths is None else int(max_paths)
    if max_paths_value < -1:
        raise ValueError("max_paths must be non-negative")
    mode_id = _bdpt_mis_mode_id(mis)
    expected_total = int(light["origin"].shape[0]) * int(sensor["origin"].shape[0])
    expected_count = (
        expected_total if max_paths is None else min(int(max_paths), expected_total)
    )
    exported = _required_native_op("bdpt_endpoint_connection_samples")(
        light,
        sensor,
        float(frequency_hz),
        int(samples_per_tx),
        int(mode_id),
        float(beta),
        int(strategy_count),
        int(max_paths_value),
    )
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_endpoint_connection_samples", exported, expected_count
    )
    return exported


def bdpt_endpoint_connection_visibility_inputs(
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    *,
    sample_count: int,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    _validate_bdpt_subpath_state("sensor", sensor, None)
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    expected_total = int(light["origin"].shape[0]) * int(sensor["origin"].shape[0])
    if int(sample_count) > expected_total:
        raise ValueError("sample_count exceeds endpoint pair count")
    exported = _required_native_op("bdpt_endpoint_connection_visibility_inputs")(
        light,
        sensor,
        int(sample_count),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_endpoint_connection_visibility_inputs must return a dict"
        )
    if set(exported) != {"start", "end", "active"}:
        raise ValueError(
            "_channel_native.bdpt_endpoint_connection_visibility_inputs returned unexpected fields"
        )
    validate_cuda_tensor(
        "start", exported["start"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "end", exported["end"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    if tuple(exported["start"].shape) != (int(sample_count), 3):
        raise ValueError(
            "_channel_native.bdpt_endpoint_connection_visibility_inputs returned bad start shape"
        )
    if exported["end"].shape != exported["start"].shape or exported["active"].shape != (
        int(sample_count),
    ):
        raise ValueError(
            "_channel_native.bdpt_endpoint_connection_visibility_inputs returned bad visibility shape"
        )
    return exported


def bdpt_accumulate_connection_samples(
    samples: dict[str, torch.Tensor],
    *,
    tx_count: int,
    rx_count: int,
    accumulation_strategy: str = "atomic",
) -> dict[str, torch.Tensor]:
    _validate_bdpt_connection_samples("samples", samples, None)
    if tx_count < 0 or rx_count < 0:
        raise ValueError("tx_count and rx_count must be non-negative")
    strategy_ids = {"atomic": 0, "staged": 1, "compact": 2}
    if accumulation_strategy not in strategy_ids:
        raise ValueError(
            "accumulation_strategy must be 'atomic', 'staged', or 'compact'"
        )
    exported = _required_native_op("bdpt_accumulate_connection_samples")(
        samples,
        int(tx_count),
        int(rx_count),
        int(strategy_ids[accumulation_strategy]),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_accumulate_connection_samples must return a dict"
        )
    if set(exported) != {
        "path_gain",
        "los",
        "reflection",
        "diffraction",
        "transmission",
        "scattering",
    }:
        raise ValueError(
            "_channel_native.bdpt_accumulate_connection_samples returned unexpected fields"
        )
    for name, tensor in exported.items():
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=2)
        if tuple(tensor.shape) != (int(tx_count), int(rx_count)):
            raise ValueError(
                f"_channel_native.bdpt_accumulate_connection_samples returned bad {name} shape"
            )
    return exported


def bdpt_filter_connection_samples(
    samples: dict[str, torch.Tensor],
    visible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_connection_samples("samples", samples, None)
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if visible.shape != samples["valid"].shape:
        raise ValueError("visible must match samples")
    if visible.get_device() != samples["valid"].get_device():
        raise ValueError("visible must share samples device")
    exported = _required_native_op("bdpt_filter_connection_samples")(samples, visible)
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_filter_connection_samples", exported, None
    )
    return exported


def bdpt_count_valid_connection_samples(samples: dict[str, torch.Tensor]) -> int:
    _validate_bdpt_connection_samples("samples", samples, None)
    count = _required_native_op("bdpt_count_valid_connection_samples")(samples)
    if not isinstance(count, int):
        raise TypeError(
            "_channel_native.bdpt_count_valid_connection_samples must return an int"
        )
    if count < 0 or count > int(samples["valid"].shape[0]):
        raise ValueError(
            "_channel_native.bdpt_count_valid_connection_samples returned bad count"
        )
    return count


def bdpt_compact_connection_samples(
    samples: dict[str, torch.Tensor],
    *,
    max_paths: int | None = None,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_connection_samples("samples", samples, None)
    max_paths_value = -1 if max_paths is None else int(max_paths)
    if max_paths_value < -1:
        raise ValueError("max_paths must be non-negative")
    exported = _required_native_op("bdpt_compact_connection_samples")(
        samples, int(max_paths_value)
    )
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_compact_connection_samples", exported, None
    )
    if max_paths is not None and int(exported["valid"].shape[0]) > int(max_paths):
        raise ValueError(
            "_channel_native.bdpt_compact_connection_samples exceeded max_paths"
        )
    return exported


def bdpt_concat_connection_samples(
    samples: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("samples must not be empty")
    expected_count = 0
    for index, block in enumerate(samples):
        _validate_bdpt_connection_samples(f"samples[{index}]", block, None)
        expected_count += int(block["valid"].shape[0])
    exported = _required_native_op("bdpt_concat_connection_samples")(tuple(samples))
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_concat_connection_samples", exported, expected_count
    )
    return exported


def bdpt_connection_variance(
    samples: dict[str, torch.Tensor],
    *,
    tx_count: int,
    rx_count: int,
    samples_per_tx: int,
) -> torch.Tensor:
    _validate_bdpt_connection_samples("samples", samples, None)
    if tx_count < 0 or rx_count < 0:
        raise ValueError("tx_count and rx_count must be non-negative")
    if samples_per_tx <= 0:
        raise ValueError("samples_per_tx must be positive")
    variance = _required_native_op("bdpt_connection_variance")(
        samples,
        int(tx_count),
        int(rx_count),
        int(samples_per_tx),
    )
    if not isinstance(variance, torch.Tensor):
        raise TypeError("_channel_native.bdpt_connection_variance must return a tensor")
    validate_cuda_tensor("variance", variance, dtype=torch.float32, ndim=2)
    if tuple(variance.shape) != (int(tx_count), int(rx_count)):
        raise ValueError("_channel_native.bdpt_connection_variance returned bad shape")
    return variance


def bdpt_mis_weights(
    pdf: torch.Tensor,
    strategy_pdf_sum: torch.Tensor,
    *,
    mis: str,
    beta: float = 2.0,
) -> torch.Tensor:
    validate_cuda_tensor("pdf", pdf, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "strategy_pdf_sum", strategy_pdf_sum, dtype=torch.float32, ndim=0
    )
    mode_id = _bdpt_mis_mode_id(mis)
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "bdpt_mis_weights"):
        raise RuntimeError("_channel_native.bdpt_mis_weights CUDA kernel is required")
    weights = native.bdpt_mis_weights(pdf, strategy_pdf_sum, int(mode_id), float(beta))
    if not isinstance(weights, torch.Tensor):
        raise TypeError("_channel_native.bdpt_mis_weights must return a tensor")
    validate_cuda_tensor("weights", weights, dtype=torch.float32, ndim=1)
    if weights.shape != pdf.shape:
        raise ValueError(
            "_channel_native.bdpt_mis_weights returned an unexpected shape"
        )
    return weights


def bdpt_diffraction_connection_samples_from_tape(
    tape: dict[str, torch.Tensor],
    states: tuple[torch.Tensor, ...],
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    *,
    tx_index: int,
    state_count: int,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    grid_cell_area: float,
    wavelength: float,
    direct_samples: int,
    keller_samples: int,
    mis: str = "power_heuristic",
    beta: float = 2.0,
    strategy_count: int = 1,
) -> dict[str, torch.Tensor]:
    expected_tape = {
        "active": torch.bool,
        "state_idx": torch.int32,
        "cell": torch.int32,
        "material_idx": torch.int32,
        "edge_u": torch.float32,
    }
    if set(tape) != set(expected_tape):
        raise ValueError("diffraction tape returned unexpected fields")
    inferred_count: int | None = None
    for field, dtype in expected_tape.items():
        tensor = tape[field]
        validate_cuda_tensor(f"tape.{field}", tensor, dtype=dtype, ndim=1)
        if inferred_count is None:
            inferred_count = int(tensor.shape[0])
        if int(tensor.shape[0]) != inferred_count:
            raise ValueError("diffraction tape fields must share shape")
    if len(states) != 12:
        raise ValueError("states must contain 12 tensors")
    validate_cuda_tensor("material_gain", material_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_valid", material_valid, dtype=torch.bool, ndim=1)
    if material_gain.shape != material_valid.shape:
        raise ValueError("material_gain and material_valid must have matching shape")
    if state_count < 0:
        raise ValueError("state_count must be non-negative")
    if direct_samples < 0 or keller_samples < 0:
        raise ValueError("sample counts must be non-negative")
    if strategy_count <= 0:
        raise ValueError("strategy_count must be positive")
    actual_strategy_count = int(direct_samples > 0) + int(keller_samples > 0)
    if strategy_count != actual_strategy_count:
        raise ValueError("strategy_count must match enabled direct/Keller proposals")
    if mis == "none" and actual_strategy_count != 1:
        raise ValueError("mis='none' requires exactly one diffraction proposal")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    mode_id = _bdpt_mis_mode_id(mis)
    exported = _required_native_op("bdpt_diffraction_connection_samples_from_tape")(
        tape,
        tuple(states),
        material_gain,
        material_valid,
        int(tx_index),
        int(state_count),
        int(grid_axis),
        float(grid_position),
        float(grid_coord0_min),
        float(grid_coord0_max),
        float(grid_coord1_min),
        float(grid_coord1_max),
        int(grid_resolution0),
        int(grid_resolution1),
        float(grid_cell_area),
        float(wavelength),
        int(direct_samples),
        int(keller_samples),
        int(mode_id),
        float(beta),
        int(strategy_count),
    )
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_diffraction_connection_samples_from_tape",
        exported,
        inferred_count,
    )
    return exported


def bdpt_diffraction_point_connection_samples(
    rx_positions: torch.Tensor,
    states: tuple[torch.Tensor, ...],
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    *,
    tx_index: int,
    state_count: int,
    direct_samples: int,
    keller_samples: int,
    seed: int,
    wavelength: float,
    mis: str = "power_heuristic",
    beta: float = 2.0,
    strategy_count: int = 1,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if len(states) != 12:
        raise ValueError("states must contain 12 tensors")
    validate_cuda_tensor("material_gain", material_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_valid", material_valid, dtype=torch.bool, ndim=1)
    if material_gain.shape != material_valid.shape:
        raise ValueError("material_gain and material_valid must have matching shape")
    if state_count < 0:
        raise ValueError("state_count must be non-negative")
    if direct_samples < 0 or keller_samples < 0:
        raise ValueError("sample counts must be non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if wavelength <= 0.0:
        raise ValueError("wavelength must be positive")
    if strategy_count <= 0:
        raise ValueError("strategy_count must be positive")
    actual_strategy_count = int(direct_samples > 0) + int(keller_samples > 0)
    if strategy_count != actual_strategy_count:
        raise ValueError("strategy_count must match enabled direct/Keller proposals")
    if mis == "none" and actual_strategy_count != 1:
        raise ValueError("mis='none' requires exactly one diffraction proposal")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    mode_id = _bdpt_mis_mode_id(mis)
    exported = _required_native_op("bdpt_diffraction_point_connection_samples")(
        rx_positions,
        tuple(states),
        material_gain,
        material_valid,
        int(tx_index),
        int(state_count),
        int(direct_samples),
        int(keller_samples),
        int(seed),
        float(wavelength),
        int(mode_id),
        float(beta),
        int(strategy_count),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_diffraction_point_connection_samples must return a dict"
        )
    expected = {
        "samples",
        "source_start",
        "source_end",
        "target_start",
        "target_end",
        "visibility_active",
    }
    if set(exported) != expected:
        raise ValueError(
            "_channel_native.bdpt_diffraction_point_connection_samples returned unexpected fields"
        )
    sample_count = int(rx_positions.shape[0]) * (
        int(direct_samples) + int(keller_samples)
    )
    if state_count == 0:
        sample_count = 0
    samples = exported["samples"]
    if not isinstance(samples, dict):
        raise TypeError(
            "_channel_native.bdpt_diffraction_point_connection_samples samples must be a dict"
        )
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_diffraction_point_connection_samples.samples",
        samples,
        sample_count,
    )
    for name in ("source_start", "source_end", "target_start", "target_end"):
        tensor = exported[name]
        validate_cuda_tensor(
            name, tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        if tuple(tensor.shape) != (sample_count, 3):
            raise ValueError(
                f"_channel_native.bdpt_diffraction_point_connection_samples returned bad {name} shape"
            )
    active = exported["visibility_active"]
    validate_cuda_tensor("visibility_active", active, dtype=torch.bool, ndim=1)
    if tuple(active.shape) != (sample_count,):
        raise ValueError(
            "_channel_native.bdpt_diffraction_point_connection_samples returned bad visibility_active shape"
        )
    return exported


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





















def mc_finalize_component_maps(
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
    if reflection.shape != los.shape:
        raise ValueError("reflection must match los shape")
    if diffraction.shape != los.shape:
        raise ValueError("diffraction must match los shape")
    if transmission.shape != los.shape:
        raise ValueError("transmission must match los shape")
    if scattering.shape != los.shape:
        raise ValueError("scattering must match los shape")

    native = native_extension()
    if native is None or not hasattr(native, "mc_finalize_component_maps"):
        raise RuntimeError(
            "_channel_native.mc_finalize_component_maps CUDA kernel is required"
        )
    exported = native.mc_finalize_component_maps(
        los, reflection, diffraction, transmission, scattering
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.mc_finalize_component_maps must return a dict")
    return exported


_MC_FINALIZE_FIELDS = (
    "path_gain",
    "los_power",
    "reflection_power",
    "diffraction_power",
    "transmission_power",
    "scattering_power",
)


class _McFinalizeComponentMapsAdFunction(torch.autograd.Function):
    """Differentiable component-map finalization (plan 07 AD-3).

    The finalize kernel is a purely linear elementwise sum plus per-component
    power reductions, so the map cotangent is the path_gain cotangent viewed
    back to map layout plus the broadcast power cotangent, and the
    pushforward is the finalize kernel itself applied to the tangents.
    """

    @staticmethod
    def forward(los, reflection, diffraction, transmission, scattering):
        out = mc_finalize_component_maps(
            los, reflection, diffraction, transmission, scattering
        )
        return tuple(out[name] for name in _MC_FINALIZE_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.map_shape = tuple(inputs[0].shape)
        primal = torch.autograd.forward_ad.unpack_dual(inputs[0]).primal
        ctx.save_for_forward(primal)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_path_gain, *grad_powers):
        tx_count, dim0, dim1 = ctx.map_shape
        base = (
            grad_path_gain.unflatten(1, (dim0, dim1))
            if grad_path_gain is not None
            else None
        )
        grads = []
        for index in range(5):
            if not ctx.needs_input_grad[index]:
                grads.append(None)
                continue
            grad_power = grad_powers[index]
            if grad_power is None:
                grads.append(base)
            elif base is None:
                grads.append(grad_power.expand(ctx.map_shape))
            else:
                grads.append(base + grad_power)
        return tuple(grads)

    @staticmethod
    def jvp(ctx, t_los, t_reflection, t_diffraction, t_transmission, t_scattering):
        tangents = [
            _ad_native_tangent_or_none(value) for value in
            (t_los, t_reflection, t_diffraction, t_transmission, t_scattering)
        ]
        if all(value is None for value in tangents):
            return (None,) * len(_MC_FINALIZE_FIELDS)
        (reference,) = ctx.saved_tensors
        zero = None
        filled = []
        for value in tangents:
            if value is None:
                if zero is None:
                    zero = torch.zeros_like(_ad_native_tensor(reference))
                value = zero
            filled.append(value)
        with torch_compat.disable_functorch():
            out = mc_finalize_component_maps(*filled)
        return tuple(out[name] for name in _MC_FINALIZE_FIELDS)


def mc_finalize_component_maps_ad(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`mc_finalize_component_maps`."""

    values = _McFinalizeComponentMapsAdFunction.apply(
        los, reflection, diffraction, transmission, scattering
    )
    return dict(zip(_MC_FINALIZE_FIELDS, values, strict=True))


def mc_los_component_maps_adjoint(
    grad_maps: torch.Tensor,
    visible: torch.Tensor | None,
) -> torch.Tensor:
    """Adjoint of the (visibility-masked) LoS component-map layout."""

    if not isinstance(grad_maps, torch.Tensor):
        raise TypeError("grad_maps must be a torch.Tensor")
    if grad_maps.dtype != torch.float32 or not grad_maps.is_cuda:
        raise TypeError("grad_maps must be a float32 CUDA tensor")
    if grad_maps.ndim != 3:
        raise ValueError("grad_maps must have 3 dimensions")
    if visible is None:
        visible = grad_maps.new_empty((0, 0), dtype=torch.bool)
    out = _required_native_op("mc_los_component_maps_adjoint")(grad_maps, visible)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.mc_los_component_maps_adjoint must return a tensor"
        )
    return out


class _McLosGridMapsAdFunction(torch.autograd.Function):
    """Differentiable LoS/transmission component-map layout (plan 07 AD-3).

    The forward is the primal layout kernel plus the per-tx visibility mask
    application, a permutation times a frozen 0/1 mask of the (tx, cells)
    matrix; its adjoint is one masked gather kernel and its pushforward is
    the forward itself on the tangent matrix.
    """

    @staticmethod
    def forward(matrix, visible, rows, cols):
        maps = mc_los_component_maps_from_matrix(matrix, rows=rows, cols=cols)
        if visible is not None:
            for tx_index in range(int(matrix.shape[0])):
                mc_apply_los_visibility(
                    maps, matrix, visible[tx_index], tx_index=tx_index
                )
        return maps

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        matrix, visible, rows, cols = inputs
        ctx.rows = int(rows)
        ctx.cols = int(cols)
        ctx.has_visible = visible is not None
        if visible is not None:
            ctx.save_for_backward(visible)
            ctx.save_for_forward(visible)
        else:
            ctx.save_for_backward()
            ctx.save_for_forward()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_maps):
        if grad_maps is None or not ctx.needs_input_grad[0]:
            return None, None, None, None
        visible = ctx.saved_tensors[0] if ctx.has_visible else None
        return (
            mc_los_component_maps_adjoint(grad_maps, visible),
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, t_matrix, _t_visible, _t_rows, _t_cols):
        tangent = _ad_native_tangent_or_none(t_matrix)
        if tangent is None:
            return None
        visible = ctx.saved_tensors[0] if ctx.has_visible else None
        with torch_compat.disable_functorch():
            maps = mc_los_component_maps_from_matrix(
                tangent, rows=ctx.rows, cols=ctx.cols
            )
            if visible is not None:
                visible = _ad_native_tensor(visible)
                for tx_index in range(int(tangent.shape[0])):
                    mc_apply_los_visibility(
                        maps, tangent, visible[tx_index], tx_index=tx_index
                    )
        return maps


def mc_los_grid_maps_ad(
    matrix: torch.Tensor,
    visible: torch.Tensor | None,
    *,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Differentiable grid component maps from a (tx, cells) matrix."""

    return _McLosGridMapsAdFunction.apply(matrix, visible, int(rows), int(cols))


def mc_sionna_reflection_accumulate(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    trace_valid: torch.Tensor,
    trace_t: torch.Tensor,
    trace_prim: torch.Tensor,
    face_normals: torch.Tensor,
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_thickness: torch.Tensor,
    *,
    contribution_depth: int,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    solid_angle_per_ray: float,
    grid_cell_area: float,
) -> torch.Tensor:
    native = native_extension()
    if native is None or not hasattr(native, "mc_sionna_reflection_accumulate"):
        raise RuntimeError(
            "_channel_native.mc_sionna_reflection_accumulate CUDA kernel is required"
        )
    return native.mc_sionna_reflection_accumulate(
        ray_o,
        ray_d,
        trace_valid,
        trace_t,
        trace_prim,
        face_normals,
        material_eta_r,
        material_sigma,
        material_gain,
        material_valid,
        material_thickness,
        int(contribution_depth),
        int(grid_axis),
        float(grid_position),
        float(grid_coord0_min),
        float(grid_coord0_max),
        float(grid_coord1_min),
        float(grid_coord1_max),
        int(grid_resolution0),
        int(grid_resolution1),
        float(wavelength),
        float(solid_angle_per_ray),
        float(grid_cell_area),
    )


def mc_sionna_diffraction_tape_accumulate(*args: object) -> torch.Tensor:
    native = native_extension()
    if native is None or not hasattr(native, "mc_sionna_diffraction_tape_accumulate"):
        raise RuntimeError(
            "_channel_native.mc_sionna_diffraction_tape_accumulate CUDA kernel is required"
        )
    output = native.mc_sionna_diffraction_tape_accumulate(*args)
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "_channel_native.mc_sionna_diffraction_tape_accumulate must return a tensor"
        )
    return output


# ---------------------------------------------------------------------------
# MC basic incoherent power-map AD (plan 07 AD-3). The solver emits a REAL
# power map assembled from per-component maps; the differentiable inputs are
# the compiled material store leaves (per-face eta_r / sigma_e / gain /
# thickness, per-layer CSR parameters), the carrier frequency and the LoS
# endpoint positions. The RayD trace and sampling tapes, the sampled ray
# directions, the deposit binning and the visibility masks are all frozen
# winners (plan 07 section 4): gradients cover the continuous part only. The
# reflection deposit weight is analytically independent of the ray origin
# (the weight is |Gamma|^2 * solid_angle * (lambda/4pi)^2 / (A_cell * |cos|),
# with the 1/d^2 spreading carried by the frozen ray density), so the
# transmitter-position gradient of the reflection map is exactly zero and is
# returned without a kernel launch.
# ---------------------------------------------------------------------------

def mc_reflection_ad_max_depth() -> int:
    """Depth cap of the native reflection radiomap AD companions.

    The backward/jvp kernels stage per-bounce state in fixed-size register
    arrays, so ``contribution_depth`` must not exceed this cap. Exposed so the
    solver can reject an over-deep AD configuration before any launch.
    """

    return int(_required_native_op("mc_reflection_ad_max_depth")())


def mc_sionna_reflection_accumulate_backward(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    trace_valid: torch.Tensor,
    trace_t: torch.Tensor,
    trace_prim: torch.Tensor,
    face_normals: torch.Tensor,
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_thickness: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    need_materials: bool,
    need_frequency: bool,
    contribution_depth: int,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    solid_angle_per_ray: float,
    grid_cell_area: float,
    wavelength_dfreq: float,
) -> tuple[torch.Tensor, ...]:
    gradients = _required_native_op("mc_sionna_reflection_accumulate_backward")(
        ray_o,
        ray_d,
        trace_valid,
        trace_t,
        trace_prim,
        face_normals,
        material_eta_r,
        material_sigma,
        material_gain,
        material_valid,
        material_thickness,
        grad_output,
        bool(need_materials),
        bool(need_frequency),
        int(contribution_depth),
        int(grid_axis),
        float(grid_position),
        float(grid_coord0_min),
        float(grid_coord0_max),
        float(grid_coord1_min),
        float(grid_coord1_max),
        int(grid_resolution0),
        int(grid_resolution1),
        float(wavelength),
        float(solid_angle_per_ray),
        float(grid_cell_area),
        float(wavelength_dfreq),
    )
    if not isinstance(gradients, tuple) or len(gradients) != 5:
        raise TypeError(
            "_channel_native.mc_sionna_reflection_accumulate_backward must "
            "return 5 tensors"
        )
    return gradients


def mc_sionna_reflection_accumulate_jvp(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    trace_valid: torch.Tensor,
    trace_t: torch.Tensor,
    trace_prim: torch.Tensor,
    face_normals: torch.Tensor,
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_thickness: torch.Tensor,
    tangent_eta_r: torch.Tensor | None,
    tangent_sigma: torch.Tensor | None,
    tangent_gain: torch.Tensor | None,
    tangent_thickness: torch.Tensor | None,
    *,
    contribution_depth: int,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    solid_angle_per_ray: float,
    grid_cell_area: float,
    wavelength_tangent: float,
) -> torch.Tensor:
    output = _required_native_op("mc_sionna_reflection_accumulate_jvp")(
        ray_o,
        ray_d,
        trace_valid,
        trace_t,
        trace_prim,
        face_normals,
        material_eta_r,
        material_sigma,
        material_gain,
        material_valid,
        material_thickness,
        tangent_eta_r if tangent_eta_r is not None else material_eta_r,
        tangent_sigma if tangent_sigma is not None else material_sigma,
        tangent_gain if tangent_gain is not None else material_gain,
        tangent_thickness if tangent_thickness is not None else material_thickness,
        tangent_eta_r is not None,
        tangent_sigma is not None,
        tangent_gain is not None,
        tangent_thickness is not None,
        int(contribution_depth),
        int(grid_axis),
        float(grid_position),
        float(grid_coord0_min),
        float(grid_coord0_max),
        float(grid_coord1_min),
        float(grid_coord1_max),
        int(grid_resolution0),
        int(grid_resolution1),
        float(wavelength),
        float(solid_angle_per_ray),
        float(grid_cell_area),
        float(wavelength_tangent),
    )
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "_channel_native.mc_sionna_reflection_accumulate_jvp must return a tensor"
        )
    return output


class _McReflectionMapAdFunction(torch.autograd.Function):
    """Differentiable Sionna reflection radiomap for one transmitter.

    Differentiable inputs: per-face eta_r / sigma_e / gain / thickness, the
    carrier frequency and the transmitter anchor. The trace tape, sampled
    directions, face normals and validity masks are frozen winners and fail
    loudly when a gradient is requested. The deposit weight is independent of
    the ray origin, so the transmitter anchor receives an exact zero gradient
    without a kernel launch (its binning influence is discrete and frozen).
    """

    @staticmethod
    def forward(
        tx_anchor,
        eta_r,
        sigma_e,
        gain,
        thickness,
        frequency,
        ray_o,
        ray_d,
        trace_valid,
        trace_t,
        trace_prim,
        face_normals,
        material_valid,
        params,
    ):
        return mc_sionna_reflection_accumulate(
            ray_o,
            ray_d,
            trace_valid,
            trace_t,
            trace_prim,
            face_normals,
            eta_r,
            sigma_e,
            gain,
            material_valid,
            thickness,
            contribution_depth=params["contribution_depth"],
            grid_axis=params["grid_axis"],
            grid_position=params["grid_position"],
            grid_coord0_min=params["grid_coord0_min"],
            grid_coord0_max=params["grid_coord0_max"],
            grid_coord1_min=params["grid_coord1_min"],
            grid_coord1_max=params["grid_coord1_max"],
            grid_resolution0=params["grid_resolution0"],
            grid_resolution1=params["grid_resolution1"],
            wavelength=params["wavelength"],
            solid_angle_per_ray=params["solid_angle_per_ray"],
            grid_cell_area=params["grid_cell_area"],
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:5]
        )
        frequency = inputs[5]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.params = inputs[13]
        ctx.anchor_shape = tuple(inputs[0].shape)
        saved = (*primals[1:], *inputs[6:13])
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        none_grads = (None,) * 14
        _ad_reject_fixed_inputs(
            "mc_sionna_reflection_accumulate_ad",
            ctx.needs_input_grad,
            (
                (6, "ray_o"),
                (7, "ray_d"),
                (8, "trace_valid"),
                (9, "trace_t"),
                (10, "trace_prim"),
                (11, "face_normals"),
                (12, "material_valid"),
            ),
        )
        if grad_output is None:
            return none_grads
        (
            eta_r,
            sigma_e,
            gain,
            thickness,
            ray_o,
            ray_d,
            trace_valid,
            trace_t,
            trace_prim,
            face_normals,
            material_valid,
        ) = ctx.saved_tensors
        need_anchor = bool(ctx.needs_input_grad[0])
        need_eta = bool(ctx.needs_input_grad[1])
        need_sigma = bool(ctx.needs_input_grad[2])
        need_gain = bool(ctx.needs_input_grad[3])
        need_thickness = bool(ctx.needs_input_grad[4])
        need_materials = need_eta or need_sigma or need_gain or need_thickness
        need_frequency = bool(ctx.needs_input_grad[5])
        grad_anchor = None
        if need_anchor:
            # The deposit weight carries no ray-origin dependence; the frozen
            # binning is the only door and it is discrete. Exact zero.
            grad_anchor = mc_zero_matrix(
                ray_o, rows=ctx.anchor_shape[0], cols=ctx.anchor_shape[1]
            )
        if not (need_materials or need_frequency):
            return (grad_anchor,) + (None,) * 13
        params = ctx.params
        wavelength = float(params["wavelength"])
        gradients = mc_sionna_reflection_accumulate_backward(
            ray_o,
            ray_d,
            trace_valid,
            trace_t,
            trace_prim,
            face_normals,
            eta_r,
            sigma_e,
            gain,
            material_valid,
            thickness,
            grad_output,
            need_materials=need_materials,
            need_frequency=need_frequency,
            contribution_depth=params["contribution_depth"],
            grid_axis=params["grid_axis"],
            grid_position=params["grid_position"],
            grid_coord0_min=params["grid_coord0_min"],
            grid_coord0_max=params["grid_coord0_max"],
            grid_coord1_min=params["grid_coord1_min"],
            grid_coord1_max=params["grid_coord1_max"],
            grid_resolution0=params["grid_resolution0"],
            grid_resolution1=params["grid_resolution1"],
            wavelength=wavelength,
            solid_angle_per_ray=params["solid_angle_per_ray"],
            grid_cell_area=params["grid_cell_area"],
            wavelength_dfreq=-wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD,
        )
        grad_frequency = (
            _ad_frequency_grad(gradients[4], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            grad_anchor,
            gradients[0] if need_eta else None,
            gradients[1] if need_sigma else None,
            gradients[2] if need_gain else None,
            gradients[3] if need_thickness else None,
            grad_frequency,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_anchor,
        t_eta,
        t_sigma,
        t_gain,
        t_thickness,
        t_frequency,
        t_ray_o,
        t_ray_d,
        t_trace_valid,
        t_trace_t,
        t_trace_prim,
        t_face_normals,
        t_material_valid,
        _t_params,
    ):
        _ad_reject_fixed_tangents(
            "mc_sionna_reflection_accumulate_ad",
            (
                (t_ray_o, "ray_o"),
                (t_ray_d, "ray_d"),
                (t_face_normals, "face_normals"),
            ),
        )
        saved = ctx.saved_tensors
        face_shape = tuple(saved[0].shape)
        tangent_eta = _ad_checked_tangent(
            "mc_sionna_reflection_accumulate_ad tangent_eta_r",
            _ad_native_tangent_or_none(t_eta),
            face_shape,
        )
        tangent_sigma = _ad_checked_tangent(
            "mc_sionna_reflection_accumulate_ad tangent_sigma_e",
            _ad_native_tangent_or_none(t_sigma),
            face_shape,
        )
        tangent_gain = _ad_checked_tangent(
            "mc_sionna_reflection_accumulate_ad tangent_gain",
            _ad_native_tangent_or_none(t_gain),
            face_shape,
        )
        tangent_thickness = _ad_checked_tangent(
            "mc_sionna_reflection_accumulate_ad tangent_thickness",
            _ad_native_tangent_or_none(t_thickness),
            face_shape,
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        params = ctx.params
        if (
            tangent_eta is None
            and tangent_sigma is None
            and tangent_gain is None
            and tangent_thickness is None
            and tangent_frequency == 0.0
        ):
            if _ad_native_tangent_or_none(
                t_anchor if isinstance(t_anchor, torch.Tensor) else None
            ) is None:
                return None
            # Transmitter-anchor-only tangent: the deposit weight carries no
            # ray-origin dependence (frozen binning), so the map tangent is
            # exactly zero. A concrete zero tensor is required: this torch
            # build rejects a None jvp when an input tangent is live.
            return mc_zero_matrix(
                _ad_native_tensor(saved[4]),
                rows=params["grid_resolution1"],
                cols=params["grid_resolution0"],
            )
        (
            eta_r,
            sigma_e,
            gain,
            thickness,
            ray_o,
            ray_d,
            trace_valid,
            trace_t,
            trace_prim,
            face_normals,
            material_valid,
        ) = saved
        wavelength = float(params["wavelength"])
        with torch_compat.disable_functorch():
            return mc_sionna_reflection_accumulate_jvp(
                _ad_native_tensor(ray_o),
                _ad_native_tensor(ray_d),
                _ad_native_tensor(trace_valid),
                _ad_native_tensor(trace_t),
                _ad_native_tensor(trace_prim),
                _ad_native_tensor(face_normals),
                _ad_native_tensor(eta_r),
                _ad_native_tensor(sigma_e),
                _ad_native_tensor(gain),
                _ad_native_tensor(material_valid),
                _ad_native_tensor(thickness),
                tangent_eta,
                tangent_sigma,
                tangent_gain,
                tangent_thickness,
                contribution_depth=params["contribution_depth"],
                grid_axis=params["grid_axis"],
                grid_position=params["grid_position"],
                grid_coord0_min=params["grid_coord0_min"],
                grid_coord0_max=params["grid_coord0_max"],
                grid_coord1_min=params["grid_coord1_min"],
                grid_coord1_max=params["grid_coord1_max"],
                grid_resolution0=params["grid_resolution0"],
                grid_resolution1=params["grid_resolution1"],
                wavelength=wavelength,
                solid_angle_per_ray=params["solid_angle_per_ray"],
                grid_cell_area=params["grid_cell_area"],
                wavelength_tangent=(
                    -wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD
                )
                * tangent_frequency,
            )


def mc_sionna_reflection_accumulate_ad(
    tx_anchor: torch.Tensor,
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_gain: torch.Tensor,
    material_thickness: torch.Tensor,
    frequency: torch.Tensor | float,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    trace_valid: torch.Tensor,
    trace_t: torch.Tensor,
    trace_prim: torch.Tensor,
    face_normals: torch.Tensor,
    material_valid: torch.Tensor,
    *,
    contribution_depth: int,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    solid_angle_per_ray: float,
    grid_cell_area: float,
) -> torch.Tensor:
    """Differentiable :func:`mc_sionna_reflection_accumulate` (one tx)."""

    params = {
        "contribution_depth": int(contribution_depth),
        "grid_axis": int(grid_axis),
        "grid_position": float(grid_position),
        "grid_coord0_min": float(grid_coord0_min),
        "grid_coord0_max": float(grid_coord0_max),
        "grid_coord1_min": float(grid_coord1_min),
        "grid_coord1_max": float(grid_coord1_max),
        "grid_resolution0": int(grid_resolution0),
        "grid_resolution1": int(grid_resolution1),
        "wavelength": float(wavelength),
        "solid_angle_per_ray": float(solid_angle_per_ray),
        "grid_cell_area": float(grid_cell_area),
    }
    return _McReflectionMapAdFunction.apply(
        tx_anchor,
        material_eta_r,
        material_sigma,
        material_gain,
        material_thickness,
        frequency,
        ray_o,
        ray_d,
        trace_valid,
        trace_t,
        trace_prim,
        face_normals,
        material_valid,
        params,
    )


def mc_sionna_diffraction_tape_accumulate_backward(
    tape_tensors: tuple[torch.Tensor, ...],
    state_tensors: tuple[torch.Tensor, ...],
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_mu_r: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_thickness: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    need_materials: bool,
    need_source: bool,
    need_frequency: bool,
    grid_axis: int,
    grid_position: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    grid_cell_area: float,
    seed: int,
    total_edge_length: float,
    wavelength_dfreq: float,
) -> tuple[torch.Tensor, ...]:
    gradients = _required_native_op("mc_sionna_diffraction_tape_accumulate_backward")(
        *tape_tensors,
        *state_tensors,
        material_eta_r,
        material_sigma,
        material_mu_r,
        material_gain,
        material_valid,
        material_thickness,
        grad_output,
        bool(need_materials),
        bool(need_source),
        bool(need_frequency),
        int(grid_axis),
        float(grid_position),
        int(grid_resolution0),
        int(grid_resolution1),
        float(wavelength),
        float(grid_cell_area),
        int(seed),
        float(total_edge_length),
        float(wavelength_dfreq),
    )
    if not isinstance(gradients, tuple) or len(gradients) != 6:
        raise TypeError(
            "_channel_native.mc_sionna_diffraction_tape_accumulate_backward "
            "must return 6 tensors"
        )
    return gradients


def mc_sionna_diffraction_tape_accumulate_jvp(
    tape_tensors: tuple[torch.Tensor, ...],
    state_tensors: tuple[torch.Tensor, ...],
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_mu_r: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_thickness: torch.Tensor,
    tangent_eta_r: torch.Tensor | None,
    tangent_sigma: torch.Tensor | None,
    tangent_gain: torch.Tensor | None,
    tangent_thickness: torch.Tensor | None,
    tangent_source: torch.Tensor | None,
    *,
    grid_axis: int,
    grid_position: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    grid_cell_area: float,
    seed: int,
    total_edge_length: float,
    wavelength_tangent: float,
) -> torch.Tensor:
    output = _required_native_op("mc_sionna_diffraction_tape_accumulate_jvp")(
        *tape_tensors,
        *state_tensors,
        material_eta_r,
        material_sigma,
        material_mu_r,
        material_gain,
        material_valid,
        material_thickness,
        tangent_eta_r if tangent_eta_r is not None else material_eta_r,
        tangent_sigma if tangent_sigma is not None else material_sigma,
        tangent_gain if tangent_gain is not None else material_gain,
        tangent_thickness if tangent_thickness is not None else material_thickness,
        tangent_source
        if tangent_source is not None
        else material_eta_r.new_empty((3,)),
        tangent_eta_r is not None,
        tangent_sigma is not None,
        tangent_gain is not None,
        tangent_thickness is not None,
        tangent_source is not None,
        int(grid_axis),
        float(grid_position),
        int(grid_resolution0),
        int(grid_resolution1),
        float(wavelength),
        float(grid_cell_area),
        int(seed),
        float(total_edge_length),
        float(wavelength_tangent),
    )
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "_channel_native.mc_sionna_diffraction_tape_accumulate_jvp must "
            "return a tensor"
        )
    return output


class _McDiffractionMapAdFunction(torch.autograd.Function):
    """Differentiable Sionna diffraction radiomap for one transmitter.

    Differentiable inputs: per-face eta_r / sigma_e / gain / thickness, the
    carrier frequency and the transmitter anchor (the state sources are the
    anchor broadcast per winner edge state, so the anchor gradient is the
    per-state source gradient summed natively). The RayD sampling tape
    (active / state / cell / u), the per-lane Keller-cone azimuth, the edge
    state tables and the deposit binning are frozen winners and fail loudly
    when a gradient is requested; the continuous source dependence (incident
    spherical wave, incidence angles, cone orientation and the recomputed
    plane-crossing Jacobian) flows through the templated dual row.
    """

    @staticmethod
    def forward(
        tx_anchor,
        eta_r,
        sigma_e,
        gain,
        thickness,
        frequency,
        tape_active,
        tape_state,
        tape_cell,
        tape_u,
        state_edge_pos,
        state_edge_dir,
        state_t_min,
        state_t_max,
        state_n0,
        state_n1,
        state_prim0,
        state_prim1,
        state_exterior_angle,
        state_src,
        state_src_power,
        material_mu_r,
        material_valid,
        params,
    ):
        return mc_sionna_diffraction_tape_accumulate(
            tape_active,
            tape_state,
            tape_cell,
            tape_u,
            state_edge_pos,
            state_edge_dir,
            state_t_min,
            state_t_max,
            state_n0,
            state_n1,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            eta_r,
            sigma_e,
            material_mu_r,
            gain,
            material_valid,
            thickness,
            params["grid_axis"],
            params["grid_position"],
            params["grid_coord0_min"],
            params["grid_coord0_max"],
            params["grid_coord1_min"],
            params["grid_coord1_max"],
            params["grid_resolution0"],
            params["grid_resolution1"],
            params["wavelength"],
            params["grid_cell_area"],
            params["seed"],
            params["total_edge_length"],
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:5]
        )
        frequency = inputs[5]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.params = inputs[23]
        ctx.anchor_shape = tuple(inputs[0].shape)
        saved = (*primals[1:], *inputs[6:23])
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        none_grads = (None,) * 24
        _ad_reject_fixed_inputs(
            "mc_sionna_diffraction_tape_accumulate_ad",
            ctx.needs_input_grad,
            (
                (6, "tape_active"),
                (7, "tape_state"),
                (8, "tape_cell"),
                (9, "tape_u"),
                (10, "state_edge_pos"),
                (11, "state_edge_dir"),
                (12, "state_t_min"),
                (13, "state_t_max"),
                (14, "state_n0"),
                (15, "state_n1"),
                (18, "state_exterior_angle"),
                (19, "state_src"),
                (20, "state_src_power"),
                (21, "material_mu_r"),
                (22, "material_valid"),
            ),
        )
        if grad_output is None:
            return none_grads
        saved = ctx.saved_tensors
        (eta_r, sigma_e, gain, thickness) = saved[:4]
        tape_tensors = saved[4:8]
        state_tensors = saved[8:19]
        material_mu_r = saved[19]
        material_valid = saved[20]
        need_anchor = bool(ctx.needs_input_grad[0])
        need_eta = bool(ctx.needs_input_grad[1])
        need_sigma = bool(ctx.needs_input_grad[2])
        need_gain = bool(ctx.needs_input_grad[3])
        need_thickness = bool(ctx.needs_input_grad[4])
        need_materials = need_eta or need_sigma or need_gain or need_thickness
        need_frequency = bool(ctx.needs_input_grad[5])
        if not (need_anchor or need_materials or need_frequency):
            return none_grads
        params = ctx.params
        wavelength = float(params["wavelength"])
        gradients = mc_sionna_diffraction_tape_accumulate_backward(
            tuple(tape_tensors),
            tuple(state_tensors),
            eta_r,
            sigma_e,
            material_mu_r,
            gain,
            material_valid,
            thickness,
            grad_output,
            need_materials=need_materials,
            need_source=need_anchor,
            need_frequency=need_frequency,
            grid_axis=params["grid_axis"],
            grid_position=params["grid_position"],
            grid_resolution0=params["grid_resolution0"],
            grid_resolution1=params["grid_resolution1"],
            wavelength=wavelength,
            grid_cell_area=params["grid_cell_area"],
            seed=params["seed"],
            total_edge_length=params["total_edge_length"],
            wavelength_dfreq=-wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD,
        )
        grad_frequency = (
            _ad_frequency_grad(gradients[5], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            gradients[4] if need_anchor else None,
            gradients[0] if need_eta else None,
            gradients[1] if need_sigma else None,
            gradients[2] if need_gain else None,
            gradients[3] if need_thickness else None,
            grad_frequency,
        ) + (None,) * 18

    @staticmethod
    def jvp(ctx, t_anchor, t_eta, t_sigma, t_gain, t_thickness, t_frequency, *t_rest):
        _ad_reject_fixed_tangents(
            "mc_sionna_diffraction_tape_accumulate_ad",
            (
                (t_rest[4], "state_edge_pos"),
                (t_rest[5], "state_edge_dir"),
                (t_rest[8], "state_n0"),
                (t_rest[9], "state_n1"),
                (t_rest[13], "state_src"),
                (t_rest[14], "state_src_power"),
            ),
        )
        saved = ctx.saved_tensors
        face_shape = tuple(saved[0].shape)
        tangent_eta = _ad_checked_tangent(
            "mc_sionna_diffraction_tape_accumulate_ad tangent_eta_r",
            _ad_native_tangent_or_none(t_eta),
            face_shape,
        )
        tangent_sigma = _ad_checked_tangent(
            "mc_sionna_diffraction_tape_accumulate_ad tangent_sigma_e",
            _ad_native_tangent_or_none(t_sigma),
            face_shape,
        )
        tangent_gain = _ad_checked_tangent(
            "mc_sionna_diffraction_tape_accumulate_ad tangent_gain",
            _ad_native_tangent_or_none(t_gain),
            face_shape,
        )
        tangent_thickness = _ad_checked_tangent(
            "mc_sionna_diffraction_tape_accumulate_ad tangent_thickness",
            _ad_native_tangent_or_none(t_thickness),
            face_shape,
        )
        tangent_anchor = _ad_native_tangent_or_none(
            t_anchor if isinstance(t_anchor, torch.Tensor) else None
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        params = ctx.params
        if (
            tangent_eta is None
            and tangent_sigma is None
            and tangent_gain is None
            and tangent_thickness is None
            and tangent_anchor is None
            and tangent_frequency == 0.0
        ):
            return None
        (eta_r, sigma_e, gain, thickness) = saved[:4]
        tape_tensors = tuple(_ad_native_tensor(value) for value in saved[4:8])
        state_tensors = tuple(_ad_native_tensor(value) for value in saved[8:19])
        wavelength = float(params["wavelength"])
        with torch_compat.disable_functorch():
            return mc_sionna_diffraction_tape_accumulate_jvp(
                tape_tensors,
                state_tensors,
                _ad_native_tensor(eta_r),
                _ad_native_tensor(sigma_e),
                _ad_native_tensor(saved[19]),
                _ad_native_tensor(gain),
                _ad_native_tensor(saved[20]),
                _ad_native_tensor(thickness),
                tangent_eta,
                tangent_sigma,
                tangent_gain,
                tangent_thickness,
                tangent_anchor,
                grid_axis=params["grid_axis"],
                grid_position=params["grid_position"],
                grid_resolution0=params["grid_resolution0"],
                grid_resolution1=params["grid_resolution1"],
                wavelength=wavelength,
                grid_cell_area=params["grid_cell_area"],
                seed=params["seed"],
                total_edge_length=params["total_edge_length"],
                wavelength_tangent=(
                    -wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD
                )
                * tangent_frequency,
            )


def mc_sionna_diffraction_tape_accumulate_ad(
    tx_anchor: torch.Tensor,
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_gain: torch.Tensor,
    material_thickness: torch.Tensor,
    frequency: torch.Tensor | float,
    tape_tensors: tuple[torch.Tensor, ...],
    state_tensors: tuple[torch.Tensor, ...],
    material_mu_r: torch.Tensor,
    material_valid: torch.Tensor,
    *,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    grid_cell_area: float,
    seed: int,
    total_edge_length: float,
) -> torch.Tensor:
    """Differentiable :func:`mc_sionna_diffraction_tape_accumulate` (one tx)."""

    if len(tape_tensors) != 4:
        raise ValueError("tape_tensors must hold (active, state, cell, u)")
    if len(state_tensors) != 11:
        raise ValueError("state_tensors must hold the 11 packed state tables")
    params = {
        "grid_axis": int(grid_axis),
        "grid_position": float(grid_position),
        "grid_coord0_min": float(grid_coord0_min),
        "grid_coord0_max": float(grid_coord0_max),
        "grid_coord1_min": float(grid_coord1_min),
        "grid_coord1_max": float(grid_coord1_max),
        "grid_resolution0": int(grid_resolution0),
        "grid_resolution1": int(grid_resolution1),
        "wavelength": float(wavelength),
        "grid_cell_area": float(grid_cell_area),
        "seed": int(seed),
        "total_edge_length": float(total_edge_length),
    }
    return _McDiffractionMapAdFunction.apply(
        tx_anchor,
        material_eta_r,
        material_sigma,
        material_gain,
        material_thickness,
        frequency,
        *tape_tensors,
        *state_tensors,
        material_mu_r,
        material_valid,
        params,
    )




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
    _McLosPathGainAdFunction,
    mc_apply_los_visibility,
    mc_component_map_buffer,
    mc_los_component_maps,
    mc_los_component_maps_from_matrix,
    mc_los_path_gain_ad,
    mc_los_path_gain_backward,
    mc_los_path_gain_jvp,
    mc_los_visibility_inputs,
    mc_point_component_power,
    mc_store_component_map,
    mc_store_scaled_component_map,
    mc_zero_matrix,
)
