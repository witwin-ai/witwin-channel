# Copyright Xingyu Chen.
# Native Monte Carlo kernel facades.

"""Native Monte Carlo kernel facades."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping
from typing import Any, Protocol, cast

import torch

from witwin.channel.kernels.spec import (
    AdInputs,
    NativeOp,
    NativeOps,
    frequency_meta,
    opaque,
    save,
    tensor as tensor_arg,
)
from witwin.channel.kernels.topology import path_los_export
from witwin.channel.runtime import (
    CapacityFailureBit,
    CapacityFailureState,
    _ad_checked_tangent,
    _ad_first_order_only,
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_geometry_tangent,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _rayd_scene_resource,
    disable_functorch,
    require_capacity_failure_state,
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)

_native = NativeOps(lambda symbol: _required_native_op(symbol))
_BDPT_ACCUMULATE_CONNECTION_SAMPLES = _native.op(
    "bdpt_accumulate_connection_samples",
    opaque("samples"), opaque("tx_count"), opaque("rx_count"), opaque("accumulation_strategy"),
    opaque("combine_domain"), tensor_arg("coeff_real"), tensor_arg("coeff_imag"),
)
_BDPT_COMPACT_CONNECTION_SAMPLES = _native.op(
    "bdpt_compact_connection_samples",
    opaque("samples"), opaque("max_paths"),
)
_BDPT_CONCAT_CONNECTION_SAMPLES = _native.op("bdpt_concat_connection_samples", opaque("samples"))
_BDPT_CONNECTION_VARIANCE = _native.op(
    "bdpt_connection_variance",
    opaque("samples"), opaque("tx_count"), opaque("rx_count"), opaque("samples_per_tx"),
)
_BDPT_COUNT_VALID_CONNECTION_SAMPLES = _native.op(
    "bdpt_count_valid_connection_samples",
    opaque("samples"),
)
_BDPT_ENDPOINT_CONNECTION_SAMPLES = _native.op(
    "bdpt_endpoint_connection_samples",
    opaque("light"), opaque("sensor"), opaque("frequency_hz"), opaque("samples_per_tx"),
    opaque("mode_id"), opaque("beta"), opaque("strategy_count"), opaque("max_paths"),
)
_BDPT_ENDPOINT_CONNECTION_VISIBILITY_INPUTS = _native.op(
    "bdpt_endpoint_connection_visibility_inputs",
    opaque("light"), opaque("sensor"), opaque("sample_count"),
)
_BDPT_ENDPOINT_SUBPATH_STATE = _native.op(
    "bdpt_endpoint_subpath_state",
    tensor_arg("tx_positions"), tensor_arg("tx_power"), tensor_arg("tx_polarization"),
    tensor_arg("rx_positions"), tensor_arg("rx_polarization"), tensor_arg("launch_tx_id"),
    tensor_arg("light_seed"),
)
_BDPT_FILTER_CONNECTION_SAMPLES = _native.op(
    "bdpt_filter_connection_samples",
    opaque("samples"), tensor_arg("visible"),
)
_BDPT_FINALIZE_COMPONENT_MAPS = _native.op(
    "bdpt_finalize_component_maps",
    tensor_arg("los"), tensor_arg("reflection"), tensor_arg("diffraction"),
    tensor_arg("transmission"), tensor_arg("scattering"),
)
_BDPT_FINALIZE_COMPONENT_MAPS_BACKWARD = _native.op(
    "bdpt_finalize_component_maps_backward",
    tensor_arg("los"), tensor_arg("reflection"), tensor_arg("diffraction"),
    tensor_arg("transmission"), tensor_arg("scattering"), opaque("grad_path_gain"),
    opaque("grad_los_power"), opaque("grad_reflection_power"), opaque("grad_diffraction_power"),
    opaque("grad_transmission_power"), opaque("grad_scattering_power"),
    opaque("need_grad_components"),
)
_BDPT_FINALIZE_COMPONENT_MAPS_JVP = _native.op(
    "bdpt_finalize_component_maps_jvp",
    tensor_arg("los"), tensor_arg("reflection"), tensor_arg("diffraction"),
    tensor_arg("transmission"), tensor_arg("scattering"), opaque("tangent_los"),
    opaque("tangent_reflection"), opaque("tangent_diffraction"), opaque("tangent_transmission"),
    opaque("tangent_scattering"),
)
_BDPT_FINALIZE_POINT_COMPONENTS = _native.op(
    "bdpt_finalize_point_components",
    tensor_arg("los"), tensor_arg("reflection"), tensor_arg("diffraction"),
    tensor_arg("transmission"), tensor_arg("scattering"),
)
_BDPT_FINALIZE_POINT_COMPONENTS_BACKWARD = _native.op(
    "bdpt_finalize_point_components_backward",
    tensor_arg("los"), tensor_arg("reflection"), tensor_arg("diffraction"),
    tensor_arg("transmission"), tensor_arg("scattering"), opaque("grad_path_gain"),
    opaque("grad_los_power"), opaque("grad_reflection_power"), opaque("grad_diffraction_power"),
    opaque("grad_transmission_power"), opaque("grad_scattering_power"),
    opaque("need_grad_components"),
)
_BDPT_FINALIZE_POINT_COMPONENTS_JVP = _native.op(
    "bdpt_finalize_point_components_jvp",
    tensor_arg("los"), tensor_arg("reflection"), tensor_arg("diffraction"),
    tensor_arg("transmission"), tensor_arg("scattering"), opaque("tangent_los"),
    opaque("tangent_reflection"), opaque("tangent_diffraction"), opaque("tangent_transmission"),
    opaque("tangent_scattering"),
)
_BDPT_LAUNCH_STATE = _native.op(
    "bdpt_launch_state",
    tensor_arg("reference"), opaque("tx_count"), opaque("samples"), opaque("sample_streams"),
    opaque("seed"),
)
_BDPT_LOS_COMPONENT_MAPS_FROM_MATRIX = _native.op(
    "bdpt_los_component_maps_from_matrix",
    tensor_arg("los"), opaque("rows"), opaque("cols"),
)
_BDPT_RECEIVER_GRID_POINTS = _native.op(
    "bdpt_receiver_grid_points",
    tensor_arg("reference"), opaque("rows"), opaque("cols"), opaque("origin_x"), opaque("origin_y"),
    opaque("origin_z"), opaque("x_axis_x"), opaque("x_axis_y"), opaque("x_axis_z"),
    opaque("y_axis_x"), opaque("y_axis_y"), opaque("y_axis_z"), opaque("spacing0"),
    opaque("spacing1"),
)
_BDPT_TRANSMITTER_TENSORS = _native.op(
    "bdpt_transmitter_tensors",
    opaque("flat_positions"), opaque("powers"),
)
_BDPT_WAVEFRONT_BACKWARD = _native.op(
    "bdpt_wavefront_backward",
    opaque("scene"), tensor_arg("tx_positions"), tensor_arg("tx_power"),
    tensor_arg("tx_polarization"), tensor_arg("rx_positions"), tensor_arg("rx_polarization"),
    opaque("materials"), opaque("scatter"), tensor_arg("grad_transmission"),
    tensor_arg("grad_scattering"), opaque("frequency_hz"), opaque("samples"), opaque("max_depth"),
    opaque("seed"), opaque("mode_id"), opaque("beta"), opaque("emit_mixed"),
    opaque("max_scattering_order"), opaque("scene_diagonal"), opaque("chunk_samples"),
    opaque("pair_tile"), opaque("needs"),
)
_BDPT_WAVEFRONT_JVP = _native.op(
    "bdpt_wavefront_jvp",
    opaque("scene"), tensor_arg("tx_positions"), tensor_arg("tx_power"),
    tensor_arg("tx_polarization"), tensor_arg("rx_positions"), tensor_arg("rx_polarization"),
    opaque("materials"), opaque("scatter"), tensor_arg("base_transmission"),
    tensor_arg("base_scattering"), opaque("frequency_hz"), opaque("samples"), opaque("max_depth"),
    opaque("seed"), opaque("mode_id"), opaque("beta"), opaque("emit_mixed"),
    opaque("max_scattering_order"), opaque("scene_diagonal"), opaque("chunk_samples"),
    opaque("pair_tile"), opaque("tangents"),
)
_BDPT_WAVEFRONT_TRANSMISSION = _native.op(
    "bdpt_wavefront_transmission",
    opaque("scene"), tensor_arg("tx_positions"), tensor_arg("tx_power"),
    tensor_arg("tx_polarization"), tensor_arg("rx_positions"), tensor_arg("rx_polarization"),
    opaque("materials"), opaque("scatter"), tensor_arg("base_transmission"),
    tensor_arg("base_scattering"), opaque("frequency_hz"), opaque("samples"), opaque("max_depth"),
    opaque("seed"), opaque("mode_id"), opaque("beta"), opaque("emit_mixed"),
    opaque("max_scattering_order"), opaque("scene_diagonal"), opaque("chunk_samples"),
    opaque("pair_tile"), opaque("rows_capacity"),
)
_MC_APPLY_LOS_VISIBILITY = _native.op(
    "mc_apply_los_visibility",
    tensor_arg("maps"), tensor_arg("los"), tensor_arg("visible"), opaque("tx_index"),
)
_MC_CAPACITY_FAILURE_COMPONENT_MAPS_SANITIZE = _native.op(
    "mc_capacity_failure_component_maps_sanitize",
    tensor_arg("failure_state"), tensor_arg("los"), tensor_arg("reflection"),
    tensor_arg("diffraction"), tensor_arg("transmission"), tensor_arg("scattering"),
)
_MC_CAPACITY_FAILURE_COMPONENT_MAPS_SANITIZE_BACKWARD = _native.op(
    "mc_capacity_failure_component_maps_sanitize_backward",
    tensor_arg("failure_state"), tensor_arg("reference"), opaque("grad_los"),
    opaque("grad_reflection"), opaque("grad_diffraction"), opaque("grad_transmission"),
    opaque("grad_scattering"),
)
_MC_CAPACITY_FAILURE_COMPONENT_MAPS_SANITIZE_JVP = _native.op(
    "mc_capacity_failure_component_maps_sanitize_jvp",
    tensor_arg("failure_state"), tensor_arg("reference"), opaque("tangent_los"),
    opaque("tangent_reflection"), opaque("tangent_diffraction"), opaque("tangent_transmission"),
    opaque("tangent_scattering"),
)
_MC_COMPONENT_MAP_BUFFER = _native.op(
    "mc_component_map_buffer",
    tensor_arg("reference"), opaque("tx_count"), opaque("dim0"), opaque("dim1"),
)
_MC_DIFFRACTION_DISCOVER_EDGES = _native.op(
    "mc_diffraction_discover_edges",
    tensor_arg("tx_pos"), tensor_arg("ray_dir"), tensor_arg("prim_index"), tensor_arg("hit_p"),
    tensor_arg("hit_n"), tensor_arg("hit_geo_n"), tensor_arg("triangle_edge_count"),
    tensor_arg("triangle_edge_indices"), tensor_arg("edge_pos"), tensor_arg("edge_dir"),
    tensor_arg("edge_n0"), tensor_arg("edge_n1"), tensor_arg("edge_line_min"),
    tensor_arg("edge_line_max"), tensor_arg("edge_adjacent_face1"),
)
_MC_DIFFRACTION_DISCOVER_EDGES_COUNTED = _native.op(
    "mc_diffraction_discover_edges_counted",
    tensor_arg("tx_pos"), tensor_arg("ray_dir"), tensor_arg("prim_index"), tensor_arg("hit_p"),
    tensor_arg("hit_n"), tensor_arg("hit_geo_n"), tensor_arg("hit_count"),
    tensor_arg("triangle_edge_count"), tensor_arg("triangle_edge_indices"), tensor_arg("edge_pos"),
    tensor_arg("edge_dir"), tensor_arg("edge_n0"), tensor_arg("edge_n1"),
    tensor_arg("edge_line_min"), tensor_arg("edge_line_max"), tensor_arg("edge_adjacent_face1"),
)
_MC_DIFFRACTION_STATE_WI = _native.op(
    "mc_diffraction_state_wi",
    tensor_arg("state_edge_pos"), tensor_arg("state_src"),
)
_MC_FINALIZE_COMPONENT_MAPS = _native.op(
    "mc_finalize_component_maps",
    tensor_arg("los"), tensor_arg("reflection"), tensor_arg("diffraction"),
    tensor_arg("transmission"), tensor_arg("scattering"),
)
_MC_LOS_COMPONENT_MAPS_ADJOINT = _native.op(
    "mc_los_component_maps_adjoint",
    tensor_arg("grad_maps"), tensor_arg("visible"),
)
_MC_LOS_COMPONENT_MAPS_FROM_MATRIX = _native.op(
    "mc_los_component_maps_from_matrix",
    tensor_arg("los"), opaque("rows"), opaque("cols"),
)
_MC_LOS_PATH_GAIN_BACKWARD = _native.op(
    "mc_los_path_gain_backward",
    tensor_arg("tx_positions"), tensor_arg("tx_power"), tensor_arg("rx_positions"),
    tensor_arg("grad_output"), opaque("frequency_hz"), tensor_arg("tx_pol"),
)
_MC_LOS_PATH_GAIN_JVP = _native.op(
    "mc_los_path_gain_jvp",
    tensor_arg("tx_positions"), tensor_arg("tx_power"), tensor_arg("rx_positions"),
    tensor_arg("tx_tangent"), tensor_arg("power_tangent"), tensor_arg("rx_tangent"),
    opaque("has_tx_tangent"), opaque("has_power_tangent"), opaque("has_rx_tangent"),
    opaque("frequency_hz"), opaque("frequency_tangent"), tensor_arg("tx_pol"),
)
_MC_LOS_VISIBILITY_INPUTS = _native.op(
    "mc_los_visibility_inputs",
    tensor_arg("tx_positions"), opaque("tx_index"), opaque("rx_count"),
)
_MC_POINT_DIFFRACTION_TAPE = _native.op(
    "mc_point_diffraction_tape",
    tensor_arg("sample_state"), tensor_arg("edge_pos"), tensor_arg("edge_dir"), tensor_arg("t_min"),
    tensor_arg("t_max"), tensor_arg("n0"), tensor_arg("nn"), tensor_arg("source"),
    tensor_arg("points"), opaque("point_begin"), opaque("point_count"), opaque("seed"),
    opaque("point_radius"),
)
_MC_REFLECTION_AD_MAX_DEPTH = _native.op(
    "mc_reflection_ad_max_depth",
)
_MC_REFLECTION_APPEND_FINAL = _native.op(
    "mc_reflection_append_final",
    tensor_arg("trace_valid"), tensor_arg("trace_t"), tensor_arg("trace_prim"),
    tensor_arg("final_valid"), tensor_arg("final_t"), tensor_arg("final_prim"),
    tensor_arg("final_offset"),
)
_MC_REFLECTION_FINAL_SEGMENT = _native.op(
    "mc_reflection_final_segment",
    tensor_arg("ray_o"), tensor_arg("ray_d"), tensor_arg("trace_valid"), tensor_arg("trace_t"),
    tensor_arg("trace_prim"), tensor_arg("face_normals"), tensor_arg("material_valid"),
    opaque("axis"), opaque("plane_position"), opaque("coord0_min"), opaque("coord0_max"),
    opaque("coord1_min"), opaque("coord1_max"), opaque("resolution0"), opaque("resolution1"),
    opaque("cell_area"), tensor_arg("points"), opaque("point_radius"), opaque("wavelength"),
    opaque("first_block"), tensor_arg("final_o"), tensor_arg("final_d"), tensor_arg("final_tmax"),
    tensor_arg("final_offset"), tensor_arg("final_active"),
)
_MC_REFLECTION_RAYS = _native.op(
    "mc_reflection_rays",
    tensor_arg("tx_positions"), opaque("bounds"), opaque("sample_count"), opaque("seed"),
)
_MC_SCENE_BOUNDS = _native.op("mc_scene_bounds", tensor_arg("vertices"))
_MC_SLAB_REFLECTION_ACCUMULATE = _native.op(
    "mc_slab_reflection_accumulate",
    tensor_arg("ray_o"), tensor_arg("ray_d"), tensor_arg("trace_valid"), tensor_arg("trace_t"),
    tensor_arg("trace_prim"), tensor_arg("face_normals"), tensor_arg("eta_r"), tensor_arg("sigma"),
    tensor_arg("gain"), tensor_arg("material_valid"), tensor_arg("thickness"), opaque("layers"),
    opaque("contribution_depth"), opaque("axis"), opaque("plane_position"), opaque("coord0_min"),
    opaque("coord0_max"), opaque("coord1_min"), opaque("coord1_max"), opaque("resolution0"),
    opaque("resolution1"), opaque("wavelength"), tensor_arg("solid_angle"), opaque("cell_area"),
    tensor_arg("tx_pol"), tensor_arg("points"), opaque("point_radius"),
)
_MC_SLAB_REFLECTION_ACCUMULATE_BACKWARD = _native.op(
    "mc_slab_reflection_accumulate_backward",
    tensor_arg("ray_o"), tensor_arg("ray_d"), tensor_arg("trace_valid"), tensor_arg("trace_t"),
    tensor_arg("trace_prim"), tensor_arg("face_normals"), tensor_arg("eta_r"), tensor_arg("sigma"),
    tensor_arg("gain"), tensor_arg("material_valid"), tensor_arg("thickness"), opaque("layers"),
    tensor_arg("grad_output"), opaque("need_materials"), opaque("need_frequency"),
    opaque("contribution_depth"), opaque("axis"), opaque("plane_position"), opaque("coord0_min"),
    opaque("coord0_max"), opaque("coord1_min"), opaque("coord1_max"), opaque("resolution0"),
    opaque("resolution1"), opaque("wavelength"), tensor_arg("solid_angle"), opaque("cell_area"),
    opaque("wavelength_dfreq"), tensor_arg("tx_pol"), tensor_arg("points"), opaque("point_radius"),
)
_MC_SLAB_REFLECTION_ACCUMULATE_JVP = _native.op(
    "mc_slab_reflection_accumulate_jvp",
    tensor_arg("ray_o"), tensor_arg("ray_d"), tensor_arg("trace_valid"), tensor_arg("trace_t"),
    tensor_arg("trace_prim"), tensor_arg("face_normals"), tensor_arg("eta_r"), tensor_arg("sigma"),
    tensor_arg("gain"), tensor_arg("material_valid"), tensor_arg("thickness"), opaque("layers"),
    tensor_arg("tangent_eta_r"), tensor_arg("tangent_sigma"), tensor_arg("tangent_gain"),
    tensor_arg("tangent_thickness"), opaque("has_tangent_eta_r"), opaque("has_tangent_sigma"),
    opaque("has_tangent_gain"), opaque("has_tangent_thickness"), opaque("contribution_depth"),
    opaque("axis"), opaque("plane_position"), opaque("coord0_min"), opaque("coord0_max"),
    opaque("coord1_min"), opaque("coord1_max"), opaque("resolution0"), opaque("resolution1"),
    opaque("wavelength"), tensor_arg("solid_angle"), opaque("cell_area"),
    opaque("wavelength_tangent"), tensor_arg("tx_pol"), tensor_arg("points"),
    opaque("point_radius"),
)
_MC_STORE_COMPONENT_MAP = _native.op(
    "mc_store_component_map",
    tensor_arg("maps"), tensor_arg("source"), opaque("tx_index"),
)
_MC_STORE_SCALED_COMPONENT_MAP = _native.op(
    "mc_store_scaled_component_map",
    tensor_arg("maps"), tensor_arg("source"), tensor_arg("scale_values"), opaque("tx_index"),
    opaque("scale_index"),
)
_MC_TRANSMISSION_WALL_PRODUCT = _native.op(
    "mc_transmission_wall_product",
    tensor_arg("valid"), tensor_arg("num_hits"), tensor_arg("reached_target"),
    tensor_arg("direction"), tensor_arg("normal"), tensor_arg("global_primitive_id"),
    tensor_arg("face_material_id"), tensor_arg("geometry_mode_id"), tensor_arg("layer_offset"),
    tensor_arg("layer_count"), tensor_arg("layer_thickness_m"), tensor_arg("layer_eps_r"),
    tensor_arg("layer_sigma_e"), tensor_arg("layer_mu_r"), tensor_arg("pair_polarization"),
    tensor_arg("base_power"), opaque("frequency_hz"), tensor_arg("capacity_failure_state"),
    opaque("failure_bit"),
)
_MC_TRANSMISSION_WALL_PRODUCT_BACKWARD = _native.op(
    "mc_transmission_wall_product_backward",
    tensor_arg("valid"), tensor_arg("num_hits"), tensor_arg("reached_target"),
    tensor_arg("direction"), tensor_arg("normal"), tensor_arg("global_primitive_id"),
    tensor_arg("face_material_id"), tensor_arg("geometry_mode_id"), tensor_arg("layer_offset"),
    tensor_arg("layer_count"), tensor_arg("layer_thickness_m"), tensor_arg("layer_eps_r"),
    tensor_arg("layer_sigma_e"), tensor_arg("layer_mu_r"), tensor_arg("pair_polarization"),
    tensor_arg("base_power"), opaque("frequency_hz"), tensor_arg("capacity_failure_state"),
    opaque("failure_bit"), opaque("grad_scaled_power"), opaque("grad_transmittance"),
)
_MC_TRANSMISSION_WALL_PRODUCT_JVP = _native.op(
    "mc_transmission_wall_product_jvp",
    tensor_arg("valid"), tensor_arg("num_hits"), tensor_arg("reached_target"),
    tensor_arg("direction"), tensor_arg("normal"), tensor_arg("global_primitive_id"),
    tensor_arg("face_material_id"), tensor_arg("geometry_mode_id"), tensor_arg("layer_offset"),
    tensor_arg("layer_count"), tensor_arg("layer_thickness_m"), tensor_arg("layer_eps_r"),
    tensor_arg("layer_sigma_e"), tensor_arg("layer_mu_r"), tensor_arg("pair_polarization"),
    tensor_arg("base_power"), opaque("frequency_hz"), tensor_arg("capacity_failure_state"),
    opaque("failure_bit"), opaque("tangent_direction"), opaque("tangent_normal"),
    opaque("tangent_layer_thickness_m"), opaque("tangent_layer_eps_r"),
    opaque("tangent_layer_sigma_e"), opaque("tangent_base_power"), opaque("tangent_frequency"),
)
_MC_UTD_DIFFRACTION_TAPE_ACCUMULATE = _native.op(
    "mc_utd_diffraction_tape_accumulate",
    tensor_arg("tape_active"), tensor_arg("tape_state"), tensor_arg("tape_cell"),
    tensor_arg("tape_u"), tensor_arg("edge_pos"), tensor_arg("edge_dir"), tensor_arg("t_min"),
    tensor_arg("t_max"), tensor_arg("n0"), tensor_arg("nn"), tensor_arg("prim0"),
    tensor_arg("prim1"), tensor_arg("exterior_angle"), tensor_arg("source"),
    tensor_arg("source_power"), tensor_arg("eta_r"), tensor_arg("sigma"), tensor_arg("mu_r"),
    tensor_arg("gain"), tensor_arg("material_valid"), tensor_arg("thickness"), opaque("axis"),
    opaque("plane"), opaque("c0min"), opaque("c0max"), opaque("c1min"), opaque("c1max"),
    opaque("r0"), opaque("r1"), opaque("wavelength"), opaque("cell_area"), opaque("seed"),
    opaque("total_edge_length"), tensor_arg("tx_pol"), tensor_arg("tape_lane"),
    tensor_arg("points"), opaque("point_radius"), opaque("edge_samples"),
)
_MC_UTD_DIFFRACTION_TAPE_ACCUMULATE_BACKWARD = _native.op(
    "mc_utd_diffraction_tape_accumulate_backward",
    tensor_arg("tape_active"), tensor_arg("tape_state"), tensor_arg("tape_cell"),
    tensor_arg("tape_u"), tensor_arg("edge_pos"), tensor_arg("edge_dir"), tensor_arg("t_min"),
    tensor_arg("t_max"), tensor_arg("n0"), tensor_arg("nn"), tensor_arg("prim0"),
    tensor_arg("prim1"), tensor_arg("exterior_angle"), tensor_arg("source"),
    tensor_arg("source_power"), tensor_arg("eta_r"), tensor_arg("sigma"), tensor_arg("mu_r"),
    tensor_arg("gain"), tensor_arg("material_valid"), tensor_arg("thickness"),
    tensor_arg("grad_output"), opaque("need_materials"), opaque("need_source"),
    opaque("need_frequency"), opaque("axis"), opaque("plane"), opaque("r0"), opaque("r1"),
    opaque("wavelength"), opaque("cell_area"), opaque("seed"), opaque("total_edge_length"),
    opaque("wavelength_dfreq"), tensor_arg("tx_pol"), tensor_arg("tape_lane"), tensor_arg("points"),
    opaque("point_radius"), opaque("edge_samples"),
)
_MC_UTD_DIFFRACTION_TAPE_ACCUMULATE_JVP = _native.op(
    "mc_utd_diffraction_tape_accumulate_jvp",
    tensor_arg("tape_active"), tensor_arg("tape_state"), tensor_arg("tape_cell"),
    tensor_arg("tape_u"), tensor_arg("edge_pos"), tensor_arg("edge_dir"), tensor_arg("t_min"),
    tensor_arg("t_max"), tensor_arg("n0"), tensor_arg("nn"), tensor_arg("prim0"),
    tensor_arg("prim1"), tensor_arg("exterior_angle"), tensor_arg("source"),
    tensor_arg("source_power"), tensor_arg("eta_r"), tensor_arg("sigma"), tensor_arg("mu_r"),
    tensor_arg("gain"), tensor_arg("material_valid"), tensor_arg("thickness"),
    tensor_arg("tangent_eta_r"), tensor_arg("tangent_sigma"), tensor_arg("tangent_gain"),
    tensor_arg("tangent_thickness"), tensor_arg("tangent_source"), opaque("has_tangent_eta_r"),
    opaque("has_tangent_sigma"), opaque("has_tangent_gain"), opaque("has_tangent_thickness"),
    opaque("has_tangent_source"), opaque("axis"), opaque("plane"), opaque("r0"), opaque("r1"),
    opaque("wavelength"), opaque("cell_area"), opaque("seed"), opaque("total_edge_length"),
    opaque("wavelength_tangent"), tensor_arg("tx_pol"), tensor_arg("tape_lane"),
    tensor_arg("points"), opaque("point_radius"), opaque("edge_samples"),
)
_MC_ZERO_MATRIX = _native.op(
    "mc_zero_matrix",
    tensor_arg("reference"), opaque("rows"), opaque("cols"),
)


# -------------------------------------------------------------------------
# basic capacity
# -------------------------------------------------------------------------
_MC_COMPONENT_MAP_FIELDS = (
    "los",
    "reflection",
    "diffraction",
    "transmission",
    "scattering",
)


def _validate_capacity_component_maps(maps: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if len(maps) != len(_MC_COMPONENT_MAP_FIELDS):
        raise ValueError("MC capacity sanitizer requires five component maps")
    reference = maps[0]
    for name, value in zip(_MC_COMPONENT_MAP_FIELDS, maps, strict=True):
        validate_cuda_tensor(name, value, dtype=torch.float32, ndim=3)
        if value.shape != reference.shape:
            raise ValueError(f"{name} must match los shape")
        if value.device != reference.device:
            raise ValueError(f"{name} must share the los device")
    return reference


def _capacity_component_maps_result(
    exported: object, *, reference: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(exported, dict) or set(exported) != set(_MC_COMPONENT_MAP_FIELDS):
        raise TypeError("native MC capacity map sanitizer returned bad fields")
    values = tuple(exported[name] for name in _MC_COMPONENT_MAP_FIELDS)
    for name, value in zip(_MC_COMPONENT_MAP_FIELDS, values, strict=True):
        validate_cuda_tensor(name, value, dtype=torch.float32, ndim=3)
        if value.shape != reference.shape or value.device != reference.device:
            raise ValueError(f"native MC capacity sanitizer returned bad {name}")
    return values


def _mc_capacity_failure_component_maps_sanitize_native(
    failure_state_bits: torch.Tensor, *maps: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    reference = _validate_capacity_component_maps(maps)
    exported = _MC_CAPACITY_FAILURE_COMPONENT_MAPS_SANITIZE(
        failure_state_bits, *maps
    )
    return _capacity_component_maps_result(exported, reference=reference)


def _mc_capacity_failure_component_maps_sanitize_backward_native(
    failure_state_bits: torch.Tensor, reference: torch.Tensor, *gradients: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    exported = _MC_CAPACITY_FAILURE_COMPONENT_MAPS_SANITIZE_BACKWARD(
        failure_state_bits, reference, *gradients
    )
    return _capacity_component_maps_result(exported, reference=reference)


def _mc_capacity_failure_component_maps_sanitize_jvp_native(
    failure_state_bits: torch.Tensor, reference: torch.Tensor, *tangents: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    exported = _MC_CAPACITY_FAILURE_COMPONENT_MAPS_SANITIZE_JVP(
        failure_state_bits, reference, *tangents
    )
    return _capacity_component_maps_result(exported, reference=reference)


_SANITIZE_AD = AdInputs(
    "mc_capacity_failure_component_maps_sanitize", opaque("failure_state_bits"),
    *(opaque(name, grad="diff") for name in _MC_COMPONENT_MAP_FIELDS),
)


class _McCapacityFailureComponentMapsSanitizeFunction(torch.autograd.Function):
    """Native identity/zero transaction boundary for all MC Basic maps."""

    @staticmethod
    def forward(failure_state_bits: Any, *maps: Any) -> Any:
        return _mc_capacity_failure_component_maps_sanitize_native(
            failure_state_bits, *maps
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        (failure_state_bits,) = _SANITIZE_AD.primals(inputs, "failure_state_bits")
        reference = torch.autograd.forward_ad.unpack_dual(output[0]).primal
        ctx.save_for_backward(failure_state_bits, reference)
        ctx.save_for_forward(failure_state_bits, reference)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, *gradients: Any) -> Any:
        if not _SANITIZE_AD.needs(ctx, *_MC_COMPONENT_MAP_FIELDS):
            return _SANITIZE_AD.none
        failure_state_bits, reference = ctx.saved_tensors
        values = _mc_capacity_failure_component_maps_sanitize_backward_native(
            failure_state_bits, reference, *gradients
        )
        return _SANITIZE_AD.grads(ctx, **dict(zip(_MC_COMPONENT_MAP_FIELDS, values, strict=True)))

    @staticmethod
    def jvp(ctx: Any, _failure_state_tangent: Any, *tangents: Any) -> Any:
        tangents = tuple(_ad_native_tangent_or_none(value) for value in tangents)
        if all(value is None for value in tangents):
            return (None,) * len(_MC_COMPONENT_MAP_FIELDS)
        failure_state_bits, reference = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with disable_functorch():
            return _mc_capacity_failure_component_maps_sanitize_jvp_native(
                failure_state_bits, reference, *tangents
            )


def mc_capacity_failure_component_maps_sanitize(
    los: torch.Tensor, reflection: torch.Tensor, diffraction: torch.Tensor,
    transmission: torch.Tensor, scattering: torch.Tensor, *, failure_state: CapacityFailureState,
) -> dict[str, torch.Tensor]:
    """Make every MC Basic component map inert after transaction failure."""

    maps = (los, reflection, diffraction, transmission, scattering)
    reference = _validate_capacity_component_maps(maps)
    require_capacity_failure_state(failure_state, device=reference.device)
    values = _McCapacityFailureComponentMapsSanitizeFunction.apply(
        failure_state.bits, *maps
    )
    return dict(zip(_MC_COMPONENT_MAP_FIELDS, values, strict=True))


# -------------------------------------------------------------------------
# basic maps
# -------------------------------------------------------------------------
_LIGHT_SPEED_M_PER_S_AD = 299_792_458.0

_MC_FINALIZE_FIELDS = (
    "path_gain",
    "los_power",
    "reflection_power",
    "diffraction_power",
    "transmission_power",
    "scattering_power",
)


def mc_finalize_component_maps(
    los: torch.Tensor, reflection: torch.Tensor, diffraction: torch.Tensor,
    transmission: torch.Tensor, scattering: torch.Tensor,
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

    exported = _MC_FINALIZE_COMPONENT_MAPS(
        los, reflection, diffraction, transmission, scattering
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.mc_finalize_component_maps must return a dict")
    return exported


_FINALIZE_AD = AdInputs(
    "mc_finalize_component_maps",
    *(opaque(name, grad="diff") for name in _MC_COMPONENT_MAP_FIELDS),
)


class _McFinalizeComponentMapsAdFunction(torch.autograd.Function):
    """Native linear map finalization and power-reduction AD (solver derivatives)."""

    @staticmethod
    def forward(
        los: Any, reflection: Any, diffraction: Any, transmission: Any, scattering: Any,
    ) -> Any:
        out = mc_finalize_component_maps(
            los, reflection, diffraction, transmission, scattering
        )
        return tuple(out[name] for name in _MC_FINALIZE_FIELDS)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        ctx.map_shape = tuple(_FINALIZE_AD.get(inputs, "los").shape)
        ctx.save_for_forward(*_FINALIZE_AD.primals(inputs, "los"))

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, grad_path_gain: Any, *grad_powers: Any) -> Any:
        tx_count, dim0, dim1 = ctx.map_shape
        base = (
            grad_path_gain.unflatten(1, (dim0, dim1))
            if grad_path_gain is not None
            else None
        )
        grads: dict[str, torch.Tensor | None] = {}
        for name, grad_power in zip(_MC_COMPONENT_MAP_FIELDS, grad_powers, strict=True):
            if not _FINALIZE_AD.need(ctx, name):
                continue
            if grad_power is None:
                grads[name] = base
            elif base is None:
                grads[name] = grad_power.expand(ctx.map_shape)
            else:
                grads[name] = base + grad_power
        return _FINALIZE_AD.place(**grads)

    @staticmethod
    def jvp(
        ctx: Any, t_los: Any, t_reflection: Any, t_diffraction: Any, t_transmission: Any,
        t_scattering: Any,
    ) -> Any:
        tangents = [
            _ad_native_tangent_or_none(value)
            for value in (
                t_los,
                t_reflection,
                t_diffraction,
                t_transmission,
                t_scattering,
            )
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
        with disable_functorch():
            out = mc_finalize_component_maps(*filled)
        return tuple(out[name] for name in _MC_FINALIZE_FIELDS)


def mc_finalize_component_maps_ad(
    los: torch.Tensor, reflection: torch.Tensor, diffraction: torch.Tensor,
    transmission: torch.Tensor, scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`mc_finalize_component_maps`."""

    values = _McFinalizeComponentMapsAdFunction.apply(
        los, reflection, diffraction, transmission, scattering
    )
    return dict(zip(_MC_FINALIZE_FIELDS, values, strict=True))


def mc_los_component_maps_adjoint(
    grad_maps: torch.Tensor, visible: torch.Tensor | None,
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
    out = _MC_LOS_COMPONENT_MAPS_ADJOINT(grad_maps, visible)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.mc_los_component_maps_adjoint must return a tensor"
        )
    return out


_LOS_GRID_MAPS_AD = AdInputs(
    "mc_los_grid_maps", opaque("matrix", grad="diff"), opaque("visible"), opaque("rows"),
    opaque("cols"),
)


class _McLosGridMapsAdFunction(torch.autograd.Function):
    """Native AD for the visibility-masked matrix-to-map layout (solver derivatives)."""

    @staticmethod
    def forward(matrix: Any, visible: Any, rows: Any, cols: Any) -> Any:
        maps = mc_los_component_maps_from_matrix(matrix, rows=rows, cols=cols)
        if visible is not None:
            for tx_index in range(int(matrix.shape[0])):
                mc_apply_los_visibility(
                    maps, matrix, visible[tx_index], tx_index=tx_index
                )
        return maps

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
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
    @_ad_first_order_only
    def backward(ctx: Any, grad_maps: Any) -> Any:
        if grad_maps is None or not _LOS_GRID_MAPS_AD.need(ctx, "matrix"):
            return _LOS_GRID_MAPS_AD.none
        visible = ctx.saved_tensors[0] if ctx.has_visible else None
        return _LOS_GRID_MAPS_AD.place(matrix=mc_los_component_maps_adjoint(grad_maps, visible))

    @staticmethod
    def jvp(ctx: Any, t_matrix: Any, _t_visible: Any, _t_rows: Any, _t_cols: Any) -> Any:
        tangent = _ad_native_tangent_or_none(t_matrix)
        if tangent is None:
            return None
        visible = ctx.saved_tensors[0] if ctx.has_visible else None
        with disable_functorch():
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
    matrix: torch.Tensor, visible: torch.Tensor | None, *, rows: int, cols: int,
) -> torch.Tensor:
    """Differentiable grid component maps from a (tx, cells) matrix."""

    return _McLosGridMapsAdFunction.apply(matrix, visible, int(rows), int(cols))


def mc_zero_matrix(reference: torch.Tensor, *, rows: int, cols: int) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    out = _MC_ZERO_MATRIX(reference, int(rows), int(cols))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel.mc_zero_matrix must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2)
    if out.shape != (int(rows), int(cols)):
        raise ValueError("_channel.mc_zero_matrix returned an unexpected shape")
    return out


def mc_component_map_buffer(
    reference: torch.Tensor, *, tx_count: int, dim0: int, dim1: int,
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if tx_count < 0 or dim0 < 0 or dim1 < 0:
        raise ValueError("tx_count, dim0, and dim1 must be non-negative")
    maps = _MC_COMPONENT_MAP_BUFFER(
        reference, int(tx_count), int(dim0), int(dim1)
    )
    if not isinstance(maps, torch.Tensor):
        raise TypeError("_channel.mc_component_map_buffer must return a tensor")
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (tx_count, dim0, dim1):
        raise ValueError(
            "_channel.mc_component_map_buffer returned an unexpected shape"
        )
    return maps


def mc_store_component_map(
    maps: torch.Tensor, source: torch.Tensor, *, tx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2)
    if source.shape != maps.shape[1:]:
        raise ValueError("source shape must match one maps slot")
    out = _MC_STORE_COMPONENT_MAP(maps, source, int(tx_index))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel.mc_store_component_map must return a tensor")
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def mc_store_scaled_component_map(
    maps: torch.Tensor, source: torch.Tensor, scale_values: torch.Tensor, *, tx_index: int,
    scale_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("scale_values", scale_values, dtype=torch.float32, ndim=1)
    if source.shape != maps.shape[1:]:
        raise ValueError("source shape must match one maps slot")
    out = _MC_STORE_SCALED_COMPONENT_MAP(
        maps,
        source,
        scale_values,
        int(tx_index),
        int(scale_index),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.mc_store_scaled_component_map must return a tensor"
        )
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def mc_los_component_maps_from_matrix(los: torch.Tensor, *, rows: int, cols: int) -> torch.Tensor:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    if los.shape[1] != rows * cols:
        raise ValueError("los columns must match rows * cols")
    maps = _MC_LOS_COMPONENT_MAPS_FROM_MATRIX(
        los, int(rows), int(cols)
    )
    if not isinstance(maps, torch.Tensor):
        raise TypeError(
            "_channel.mc_los_component_maps_from_matrix must return a tensor"
        )
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (los.shape[0], rows, cols):
        raise ValueError(
            "_channel.mc_los_component_maps_from_matrix returned an unexpected shape"
        )
    return maps


def mc_apply_los_visibility(
    maps: torch.Tensor, los: torch.Tensor, visible: torch.Tensor, *, tx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if maps.shape[0] != los.shape[0] or los.shape[1] != maps.shape[1] * maps.shape[2]:
        raise ValueError("los must match maps flattened grid layout")
    out = _MC_APPLY_LOS_VISIBILITY(
        maps, los, visible, int(tx_index)
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel.mc_apply_los_visibility must return a tensor")
    return out


def mc_los_visibility_inputs(
    tx_positions: torch.Tensor, *, tx_index: int, rx_count: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if rx_count < 0:
        raise ValueError("rx_count must be non-negative")
    exported = _MC_LOS_VISIBILITY_INPUTS(
        tx_positions, int(tx_index), int(rx_count)
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.mc_los_visibility_inputs must return a dict")
    validate_cuda_tensor(
        "start", exported["start"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    return exported


def mc_los_path_gain_backward(
    tx_positions: torch.Tensor, tx_power: torch.Tensor, rx_positions: torch.Tensor,
    grad_output: torch.Tensor, tx_polarizations: torch.Tensor, *, frequency_hz: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    gradients = _MC_LOS_PATH_GAIN_BACKWARD(
        tx_positions,
        tx_power,
        rx_positions,
        grad_output,
        float(frequency_hz),
        tx_polarizations,
    )
    if not isinstance(gradients, tuple) or len(gradients) != 4:
        raise TypeError(
            "_channel.mc_los_path_gain_backward must return 4 tensors"
        )
    validate_cuda_tensor(
        "grad_tx", gradients[0], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("grad_power", gradients[1], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "grad_rx", gradients[2], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("grad_frequency", gradients[3], dtype=torch.float32, ndim=1)
    if gradients[0].shape != tx_positions.shape:
        raise ValueError(
            "_channel.mc_los_path_gain_backward returned bad grad_tx shape"
        )
    if gradients[1].shape != tx_power.shape:
        raise ValueError(
            "_channel.mc_los_path_gain_backward returned bad grad_power shape"
        )
    if gradients[2].shape != rx_positions.shape:
        raise ValueError(
            "_channel.mc_los_path_gain_backward returned bad grad_rx shape"
        )
    if gradients[3].shape != (1,):
        raise ValueError(
            "_channel.mc_los_path_gain_backward returned bad grad_frequency shape"
        )
    return gradients


def mc_los_path_gain_jvp(
    tx_positions: torch.Tensor, tx_power: torch.Tensor, rx_positions: torch.Tensor,
    tx_tangent: torch.Tensor, power_tangent: torch.Tensor, rx_tangent: torch.Tensor,
    has_tx_tangent: bool, has_power_tangent: bool, has_rx_tangent: bool,
    tx_polarizations: torch.Tensor, *, frequency_hz: float, frequency_tangent: float = 0.0,
) -> torch.Tensor:
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
    if has_tx_tangent:
        validate_cuda_tensor(
            "tx_tangent", tx_tangent, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        if tx_tangent.shape != tx_positions.shape:
            raise ValueError("tx_tangent must match tx_positions")
    if has_power_tangent:
        validate_cuda_tensor(
            "power_tangent", power_tangent, dtype=torch.float32, ndim=1
        )
        if power_tangent.shape != tx_power.shape:
            raise ValueError("power_tangent must match tx_power")
    if has_rx_tangent:
        validate_cuda_tensor(
            "rx_tangent", rx_tangent, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        if rx_tangent.shape != rx_positions.shape:
            raise ValueError("rx_tangent must match rx_positions")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    out = _MC_LOS_PATH_GAIN_JVP(
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
        float(frequency_tangent),
        tx_polarizations,
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel.mc_los_path_gain_jvp must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2)
    if out.shape != (tx_positions.shape[0], rx_positions.shape[0]):
        raise ValueError(
            "_channel.mc_los_path_gain_jvp returned an unexpected shape"
        )
    return out


_LOS_PATH_GAIN_AD = AdInputs(
    "mc_los_path_gain_ad", opaque("tx_positions", grad="diff"), opaque("tx_power", grad="fixed"),
    opaque("rx_positions", grad="diff"), opaque("frequency", grad="diff"),
    opaque("frequency_value"), opaque("tx_pol"),
)


class _McLosPathGainAdFunction(torch.autograd.Function):
    """Differentiable LoS power-gain matrix (solver derivatives).

    Differentiable inputs: tx_positions, rx_positions and the carrier
    frequency. tx_power stays fixed under the AD contract; requesting
    its gradient fails loudly. The forward is the primal path_los_export
    kernel; only the path_gain_matrix output is exposed.
    """

    @staticmethod
    def forward(
        tx_positions: Any, tx_power: Any, rx_positions: Any, frequency: Any, frequency_value: Any,
        tx_pol: Any,
    ) -> Any:
        exported = path_los_export(
            tx_positions,
            tx_power,
            rx_positions,
            tx_pol,
            frequency_hz=frequency_value,
        )
        return exported["path_gain_matrix"]

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        primals = _LOS_PATH_GAIN_AD.primals(
            inputs, "tx_positions", "tx_power", "rx_positions", "tx_pol"
        )
        ctx.frequency_value = _LOS_PATH_GAIN_AD.get(inputs, "frequency_value")
        ctx.frequency_meta = frequency_meta(_LOS_PATH_GAIN_AD.get(inputs, "frequency"))
        save(ctx, *primals)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, grad_output: Any) -> Any:
        _LOS_PATH_GAIN_AD.reject_fixed_grads(ctx)
        need_tx = _LOS_PATH_GAIN_AD.need(ctx, "tx_positions")
        need_rx = _LOS_PATH_GAIN_AD.need(ctx, "rx_positions")
        need_frequency = _LOS_PATH_GAIN_AD.need(ctx, "frequency")
        if grad_output is None or not (need_tx or need_rx or need_frequency):
            return _LOS_PATH_GAIN_AD.none
        tx_positions, tx_power, rx_positions, tx_pol = ctx.saved_tensors
        grad_tx, _grad_power, grad_rx, grad_frequency = mc_los_path_gain_backward(
            tx_positions,
            tx_power,
            rx_positions,
            grad_output,
            tx_pol,
            frequency_hz=ctx.frequency_value,
        )
        return _LOS_PATH_GAIN_AD.grads(
            ctx, tx_positions=grad_tx, rx_positions=grad_rx,
            frequency=(
                _ad_frequency_grad(grad_frequency, ctx.frequency_meta) if need_frequency else None
            ),
        )

    @staticmethod
    def jvp(
        ctx: Any, t_tx: Any, t_power: Any, t_rx: Any, t_frequency: Any, _t_frequency_value: Any,
        _t_tx_pol: Any,
    ) -> Any:
        tangents = (t_tx, t_power, t_rx, t_frequency, _t_frequency_value, _t_tx_pol)
        _LOS_PATH_GAIN_AD.reject_fixed_tangents(tangents)
        saved = ctx.saved_tensors
        tangent_tx = _LOS_PATH_GAIN_AD.geometry_tangent(tangents, "tx_positions", saved[0], "tx")
        tangent_rx = _LOS_PATH_GAIN_AD.geometry_tangent(tangents, "rx_positions", saved[2], "rx")
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if tangent_tx is None and tangent_rx is None and tangent_frequency == 0.0:
            return None
        tx_positions, tx_power, rx_positions, tx_pol = saved
        with disable_functorch():
            return mc_los_path_gain_jvp(
                _ad_native_tensor(tx_positions),
                _ad_native_tensor(tx_power),
                _ad_native_tensor(rx_positions),
                tangent_tx
                if tangent_tx is not None
                else _ad_native_tensor(tx_positions),
                _ad_native_tensor(tx_power),
                tangent_rx
                if tangent_rx is not None
                else _ad_native_tensor(rx_positions),
                tangent_tx is not None,
                False,
                tangent_rx is not None,
                _ad_native_tensor(tx_pol),
                frequency_hz=ctx.frequency_value,
                frequency_tangent=tangent_frequency,
            )


def mc_los_path_gain_ad(
    tx_positions: torch.Tensor, tx_power: torch.Tensor, rx_positions: torch.Tensor,
    tx_polarizations: torch.Tensor, *, frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> torch.Tensor:
    """Differentiable LoS path-gain matrix (endpoints and frequency).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam); when not
    supplied it is read here, exactly once per apply. ``tx_polarizations`` is
    the fixed per-transmitter polarization (frozen winner) driving the dipole
    sin^2 pattern; its endpoint dependence is differentiated natively.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    return _McLosPathGainAdFunction.apply(
        tx_positions,
        tx_power,
        rx_positions,
        frequency,
        float(frequency_value),
        tx_polarizations,
    )


def _point_rows(points: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    """Point receiver rows for the reflection accumulate ops; grids pass an empty (0, 3) table."""

    if points is None:
        return reference.new_empty((0, 3))
    validate_cuda_tensor("points", points, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    return points


def _mc_reflection_layers(
    layers: tuple[torch.Tensor, ...] | None, material_eta_r: torch.Tensor,
) -> list[torch.Tensor]:
    """Native layer bundle: per-face material ids plus the six CSR layer tables.

    Without a bundle every face uses its scalar single-slab view.
    """

    if layers is not None:
        return list(layers)
    device = material_eta_r.device
    empty_i = torch.empty((0,), device=device, dtype=torch.int32)
    empty_f = torch.empty((0,), device=device, dtype=torch.float32)
    return [
        # Never read: an empty layer store marks every face single-layer.
        torch.empty(material_eta_r.shape, device=device, dtype=torch.int32),
        empty_i, empty_i, empty_f, empty_f, empty_f, empty_f,
    ]


def mc_slab_reflection_accumulate(
    ray_o: torch.Tensor, ray_d: torch.Tensor, trace_valid: torch.Tensor, trace_t: torch.Tensor,
    trace_prim: torch.Tensor, face_normals: torch.Tensor, material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor, material_gain: torch.Tensor, material_valid: torch.Tensor,
    material_thickness: torch.Tensor, *, tx_pol: torch.Tensor, contribution_depth: int,
    grid_axis: int, grid_position: float, grid_coord0_min: float, grid_coord0_max: float,
    grid_coord1_min: float, grid_coord1_max: float, grid_resolution0: int, grid_resolution1: int,
    wavelength: float, solid_angle_per_ray: torch.Tensor, grid_cell_area: float,
    points: torch.Tensor | None = None, point_radius: float = -0.01,
    layers: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """Accumulate finite-thickness slab specular reflections into a radiomap.

    Convention. This family reproduces one law only, and that law is a
    material law rather than a sampling rule: the ITU finite-thickness
    single-slab Fresnel TE/TM coefficient pair over ``eta_r`` / ``sigma_e`` /
    ``gain`` / ``thickness`` / ``wavelength``, applied per bounce in the
    ``(s_hat, p_in) -> (s_hat, p_out)`` frame and deposited incoherently as
    ``|E|^2 * solid_angle * (lambda / 4pi)^2 / (cell_area * |cos|)`` at the
    first plane crossing that the next trace hit does not occlude. It
    deliberately does NOT reproduce the hardcoded-vertical radiomap source
    convention, under which every launched ray leaves the transmitter with
    unit magnitude and a fixed z-hat polarization regardless of launch
    direction. Here the launched field is the unnormalized transverse
    projection of the true transmitter polarization instead, so every deposit
    carries the short-dipole ``sin^2(theta)`` pattern ( polarization
    consistency with the LoS and diffraction maps). Level parity with a
    unit-magnitude-source radiomap is therefore knowingly abandoned; a
    near-axis level offset is the intended behaviour, not a defect.

    With ``points`` of shape ``(P, 3)`` (P > 0) the deposits go to point receivers instead and
    the result has shape ``(1, P)``: a segment passing within the capture radius of a point
    deposits ``|E|^2 * solid_angle * (lambda / 4pi)^2 / (pi * radius^2)``. The radius is
    ``point_radius`` when positive, else ``-point_radius`` times the straight transmitter
    distance (at least 1 mm). ``solid_angle_per_ray`` is a one-element device tensor.

    ``layers`` (per-face material ids and the CSR layer store) lets a multilayer material reflect
    as its whole stack; single-layer materials keep the slab law above.
    """

    return _MC_SLAB_REFLECTION_ACCUMULATE(
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
        _mc_reflection_layers(layers, material_eta_r),
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
        solid_angle_per_ray,
        float(grid_cell_area),
        tx_pol,
        _point_rows(points, ray_o),
        float(point_radius),
    )


def mc_reflection_ad_max_depth() -> int:
    """Depth cap of the native reflection radiomap AD companions.

    The backward/jvp kernels stage per-bounce state in fixed-size register
    arrays, so ``contribution_depth`` must not exceed this cap. Exposed so the
    solver can reject an over-deep AD configuration before any launch.
    """

    return int(_MC_REFLECTION_AD_MAX_DEPTH())


def mc_slab_reflection_accumulate_backward(
    ray_o: torch.Tensor, ray_d: torch.Tensor, trace_valid: torch.Tensor, trace_t: torch.Tensor,
    trace_prim: torch.Tensor, face_normals: torch.Tensor, material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor, material_gain: torch.Tensor, material_valid: torch.Tensor,
    material_thickness: torch.Tensor, grad_output: torch.Tensor, *, tx_pol: torch.Tensor,
    need_materials: bool, need_frequency: bool, contribution_depth: int, grid_axis: int,
    grid_position: float, grid_coord0_min: float, grid_coord0_max: float, grid_coord1_min: float,
    grid_coord1_max: float, grid_resolution0: int, grid_resolution1: int, wavelength: float,
    solid_angle_per_ray: torch.Tensor, grid_cell_area: float, wavelength_dfreq: float,
    points: torch.Tensor | None = None, point_radius: float = -0.01,
    layers: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    gradients = _MC_SLAB_REFLECTION_ACCUMULATE_BACKWARD(
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
        _mc_reflection_layers(layers, material_eta_r),
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
        solid_angle_per_ray,
        float(grid_cell_area),
        float(wavelength_dfreq),
        tx_pol,
        _point_rows(points, ray_o),
        float(point_radius),
    )
    if not isinstance(gradients, tuple) or len(gradients) != 5:
        raise TypeError(
            "_channel.mc_slab_reflection_accumulate_backward must "
            "return 5 tensors"
        )
    return gradients


def mc_slab_reflection_accumulate_jvp(
    ray_o: torch.Tensor, ray_d: torch.Tensor, trace_valid: torch.Tensor, trace_t: torch.Tensor,
    trace_prim: torch.Tensor, face_normals: torch.Tensor, material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor, material_gain: torch.Tensor, material_valid: torch.Tensor,
    material_thickness: torch.Tensor, tangent_eta_r: torch.Tensor | None,
    tangent_sigma: torch.Tensor | None, tangent_gain: torch.Tensor | None,
    tangent_thickness: torch.Tensor | None, *, tx_pol: torch.Tensor, contribution_depth: int,
    grid_axis: int, grid_position: float, grid_coord0_min: float, grid_coord0_max: float,
    grid_coord1_min: float, grid_coord1_max: float, grid_resolution0: int, grid_resolution1: int,
    wavelength: float, solid_angle_per_ray: torch.Tensor, grid_cell_area: float,
    wavelength_tangent: float, points: torch.Tensor | None = None, point_radius: float = -0.01,
    layers: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    output = _MC_SLAB_REFLECTION_ACCUMULATE_JVP(
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
        _mc_reflection_layers(layers, material_eta_r),
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
        solid_angle_per_ray,
        float(grid_cell_area),
        float(wavelength_tangent),
        tx_pol,
        _point_rows(points, ray_o),
        float(point_radius),
    )
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "_channel.mc_slab_reflection_accumulate_jvp must return a tensor"
        )
    return output


def _refuse_multilayer_material_ad(params: dict[str, object], *requested: bool) -> None:
    if params.get("multilayer", False) and any(requested):
        raise NotImplementedError(
            "material derivatives (eta_r, sigma_e, thickness) of reflection off a multilayer "
            "material are not supported; gain and frequency derivatives are"
        )


# Per-face material inputs of the one-transmitter radiomap Functions, in apply order.
_MAP_MATERIALS = ("eta_r", "sigma_e", "gain", "thickness")
_MAP_DIFF_INPUTS = (
    opaque("tx_anchor", grad="diff"),
    *(opaque(name, grad="diff") for name in _MAP_MATERIALS),
    opaque("frequency", grad="diff"),
)
# The frozen trace tape of the reflection map: a gradient fails loudly, and a tangent on the
# continuous ray origin, direction and face normals does too.
_REFLECTION_MAP_FROZEN = (
    "ray_o", "ray_d", "trace_valid", "trace_t", "trace_prim", "face_normals", "material_valid",
)
_REFLECTION_MAP_AD = AdInputs(
    "mc_slab_reflection_accumulate_ad", _MAP_DIFF_INPUTS,
    *(
        opaque(name, grad="fixed", tangent=name in ("ray_o", "ray_d", "face_normals"))
        for name in _REFLECTION_MAP_FROZEN
    ),
    opaque("params"),
)


class _McReflectionMapAdFunction(torch.autograd.Function):
    """Differentiable slab reflection radiomap for one transmitter.

    Differentiable inputs: per-face eta_r / sigma_e / gain / thickness, the
    carrier frequency and the transmitter anchor. The trace tape, sampled
    directions, face normals and validity masks are frozen winners and fail
    loudly when a gradient is requested. The deposit weight is independent of
    the ray origin, so the transmitter anchor receives an exact zero gradient
    without a kernel launch (its binning influence is discrete and frozen).
    """

    @staticmethod
    def forward(
        tx_anchor: Any, eta_r: Any, sigma_e: Any, gain: Any, thickness: Any, frequency: Any,
        ray_o: Any, ray_d: Any, trace_valid: Any, trace_t: Any, trace_prim: Any, face_normals: Any,
        material_valid: Any, params: Any,
    ) -> Any:
        return mc_slab_reflection_accumulate(
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
            tx_pol=params["tx_pol"],
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
            points=params["points"],
            point_radius=params["point_radius"],
            layers=params.get("layers"),
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        ad = _REFLECTION_MAP_AD
        ctx.frequency_meta = frequency_meta(ad.get(inputs, "frequency"))
        ctx.params = ad.get(inputs, "params")
        ctx.anchor_shape = tuple(ad.get(inputs, "tx_anchor").shape)
        save(
            ctx, *ad.primals(inputs, *_MAP_MATERIALS),
            *(ad.get(inputs, name) for name in _REFLECTION_MAP_FROZEN),
        )

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, grad_output: Any) -> Any:
        ad = _REFLECTION_MAP_AD
        none_grads = ad.none
        ad.reject_fixed_grads(ctx)
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
        need_anchor = ad.need(ctx, "tx_anchor")
        need_eta = ad.need(ctx, "eta_r")
        need_sigma = ad.need(ctx, "sigma_e")
        need_gain = ad.need(ctx, "gain")
        need_thickness = ad.need(ctx, "thickness")
        need_materials = need_eta or need_sigma or need_gain or need_thickness
        _refuse_multilayer_material_ad(ctx.params, need_eta, need_sigma, need_thickness)
        need_frequency = ad.need(ctx, "frequency")
        grad_anchor = None
        if need_anchor:
            # The deposit weight carries no ray-origin dependence; the frozen
            # binning is the only door and it is discrete. Exact zero.
            grad_anchor = mc_zero_matrix(
                ray_o, rows=ctx.anchor_shape[0], cols=ctx.anchor_shape[1]
            )
        if not (need_materials or need_frequency):
            return ad.place(tx_anchor=grad_anchor)
        params = ctx.params
        wavelength = float(params["wavelength"])
        gradients = mc_slab_reflection_accumulate_backward(
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
            tx_pol=params["tx_pol"],
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
            points=params["points"],
            point_radius=params["point_radius"],
            layers=params.get("layers"),
            wavelength_dfreq=-wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD,
        )
        grad_frequency = (
            _ad_frequency_grad(gradients[4], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return ad.place(
            tx_anchor=grad_anchor,
            **{
                name: gradients[index]
                for index, name in enumerate(_MAP_MATERIALS)
                if ad.need(ctx, name)
            },
            frequency=grad_frequency,
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        ad = _REFLECTION_MAP_AD
        ad.reject_fixed_tangents(tangents)
        saved = ctx.saved_tensors
        face_shape = tuple(saved[0].shape)
        tangent_eta, tangent_sigma, tangent_gain, tangent_thickness = (
            ad.checked_tangent(tangents, name, face_shape) for name in _MAP_MATERIALS
        )
        tangent_frequency = _ad_frequency_tangent(ad.tangent(tangents, "frequency"))
        t_anchor = ad.tangent(tangents, "tx_anchor")
        params = ctx.params
        _refuse_multilayer_material_ad(
            ctx.params, *(value is not None for value in (tangent_eta, tangent_sigma, tangent_thickness))
        )
        if (
            tangent_eta is None
            and tangent_sigma is None
            and tangent_gain is None
            and tangent_thickness is None
            and tangent_frequency == 0.0
        ):
            if (
                _ad_native_tangent_or_none(
                    t_anchor if isinstance(t_anchor, torch.Tensor) else None
                )
                is None
            ):
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
        with disable_functorch():
            return mc_slab_reflection_accumulate_jvp(
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
                tx_pol=params["tx_pol"],
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
                points=params["points"],
                point_radius=params["point_radius"],
                layers=params.get("layers"),
                wavelength_tangent=(-wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD)
                * tangent_frequency,
            )


def mc_slab_reflection_accumulate_ad(
    tx_anchor: torch.Tensor, material_eta_r: torch.Tensor, material_sigma: torch.Tensor,
    material_gain: torch.Tensor, material_thickness: torch.Tensor, frequency: torch.Tensor | float,
    ray_o: torch.Tensor, ray_d: torch.Tensor, trace_valid: torch.Tensor, trace_t: torch.Tensor,
    trace_prim: torch.Tensor, face_normals: torch.Tensor, material_valid: torch.Tensor, *,
    tx_pol: torch.Tensor, contribution_depth: int, grid_axis: int, grid_position: float,
    grid_coord0_min: float, grid_coord0_max: float, grid_coord1_min: float, grid_coord1_max: float,
    grid_resolution0: int, grid_resolution1: int, wavelength: float,
    solid_angle_per_ray: torch.Tensor, grid_cell_area: float, points: torch.Tensor | None = None,
    point_radius: float = -0.01, layers: tuple[torch.Tensor, ...] | None = None,
    multilayer: bool = False,
) -> torch.Tensor:
    """Differentiable :func:`mc_slab_reflection_accumulate` (one tx)."""

    params = {
        "tx_pol": tx_pol,
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
        "solid_angle_per_ray": solid_angle_per_ray,
        "grid_cell_area": float(grid_cell_area),
        "points": points,
        "point_radius": float(point_radius),
        "layers": layers,
        "multilayer": bool(multilayer),
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


def _point_tape_args(
    point_tape: tuple[torch.Tensor, torch.Tensor, float, int] | None, reference: torch.Tensor,
) -> tuple[object, ...]:
    """Trailing (tape_lane, points, point_radius, edge_samples) of the diffraction tape ops."""

    if point_tape is None:
        return (
            reference.new_empty((0,), dtype=torch.int32),
            reference.new_empty((0, 3)),
            -0.01,
            0,
        )
    lane, points, radius, edge_samples = point_tape
    return (lane, points, float(radius), int(edge_samples))


def mc_utd_diffraction_tape_accumulate(
    *args: object, point_tape: tuple[torch.Tensor, torch.Tensor, float, int] | None = None,
) -> torch.Tensor:
    """Accumulate UTD diffraction power over the sampled Keller-cone tape.

    Convention. The convention this family adopts is a Monte Carlo per-sample
    acceptance interval, not a material law: RayD proposes the complete Keller
    cone, a lane whose edge azimuth exceeds the wedge exterior angle is
    rejected, and an accepted lane keeps the full-cone ``1 / (2pi)`` proposal
    density, so its weight stays ``2pi`` rather than the accepted interval
    width. Everything else is this package's own UTD: the full
    Kouyoumjian-Pathak pair over the stored finite-thickness slab face
    operators (fixed-point plus stored-ops convention, ``selectStationaryPoint
    = 0`` and ``mat.omega = 0`` at the pair call), pseudo-infinite ``+-1e5``
    edge bounds, and the true per-transmitter polarization rather than a
    hardcoded vertical source. It deliberately does NOT reproduce the
    diffracted field level or the cell-by-cell lit set of a radiomap built on
    that hardcoded unit-magnitude vertical source; the short-dipole source
    pattern alone separates the two, so a level delta is expected.

    ``point_tape = (tape_lane, points, point_radius, edge_samples)`` switches to point
    receivers: ``tape_cell`` names the receiver, the result is ``(1, P)`` (pass resolution0 = P,
    resolution1 = 1) and a deposit is the flux through the capture sphere's cross-section.
    """

    reference = args[13]
    assert isinstance(reference, torch.Tensor)
    output = _MC_UTD_DIFFRACTION_TAPE_ACCUMULATE(
        *args, *_point_tape_args(point_tape, reference)
    )
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "_channel.mc_utd_diffraction_tape_accumulate must return a tensor"
        )
    return output


# ---------------------------------------------------------------------------
# MC basic incoherent power-map AD (solver derivatives). The solver emits a REAL
# power map assembled from per-component maps; the differentiable inputs are
# the compiled material store leaves (per-face eta_r / sigma_e / gain /
# thickness, per-layer CSR parameters), the carrier frequency and the LoS
# endpoint positions. The RayD trace and sampling tapes, the sampled ray
# directions, the deposit binning and the visibility masks are all frozen
# winners (the AD contract): gradients cover the continuous part only. The
# reflection deposit weight is analytically independent of the ray origin
# (the weight is |Gamma|^2 * solid_angle * (lambda/4pi)^2 / (A_cell * |cos|),
# with the 1/d^2 spreading carried by the frozen ray density), so the
# transmitter-position gradient of the reflection map is exactly zero and is
# returned without a kernel launch.
# ---------------------------------------------------------------------------


def mc_utd_diffraction_tape_accumulate_backward(
    tape_tensors: tuple[torch.Tensor, ...], state_tensors: tuple[torch.Tensor, ...],
    material_eta_r: torch.Tensor, material_sigma: torch.Tensor, material_mu_r: torch.Tensor,
    material_gain: torch.Tensor, material_valid: torch.Tensor, material_thickness: torch.Tensor,
    grad_output: torch.Tensor, *, tx_pol: torch.Tensor, need_materials: bool, need_source: bool,
    need_frequency: bool, grid_axis: int, grid_position: float, grid_resolution0: int,
    grid_resolution1: int, wavelength: float, grid_cell_area: float, seed: int,
    total_edge_length: float, wavelength_dfreq: float,
    point_tape: tuple[torch.Tensor, torch.Tensor, float, int] | None = None,
) -> tuple[torch.Tensor, ...]:
    gradients = _MC_UTD_DIFFRACTION_TAPE_ACCUMULATE_BACKWARD(
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
        tx_pol,
        *_point_tape_args(point_tape, material_eta_r),
    )
    if not isinstance(gradients, tuple) or len(gradients) != 6:
        raise TypeError(
            "_channel.mc_utd_diffraction_tape_accumulate_backward "
            "must return 6 tensors"
        )
    return gradients


def mc_utd_diffraction_tape_accumulate_jvp(
    tape_tensors: tuple[torch.Tensor, ...], state_tensors: tuple[torch.Tensor, ...],
    material_eta_r: torch.Tensor, material_sigma: torch.Tensor, material_mu_r: torch.Tensor,
    material_gain: torch.Tensor, material_valid: torch.Tensor, material_thickness: torch.Tensor,
    tangent_eta_r: torch.Tensor | None, tangent_sigma: torch.Tensor | None,
    tangent_gain: torch.Tensor | None, tangent_thickness: torch.Tensor | None,
    tangent_source: torch.Tensor | None, *, tx_pol: torch.Tensor, grid_axis: int,
    grid_position: float, grid_resolution0: int, grid_resolution1: int, wavelength: float,
    grid_cell_area: float, seed: int, total_edge_length: float, wavelength_tangent: float,
    point_tape: tuple[torch.Tensor, torch.Tensor, float, int] | None = None,
) -> torch.Tensor:
    output = _MC_UTD_DIFFRACTION_TAPE_ACCUMULATE_JVP(
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
        tx_pol,
        *_point_tape_args(point_tape, material_eta_r),
    )
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "_channel.mc_utd_diffraction_tape_accumulate_jvp must "
            "return a tensor"
        )
    return output


# The frozen sampling tape and edge-state tables of the diffraction map. Requesting their
# gradient fails loudly; a tangent fails only on the continuous edge geometry and source.
_DIFFRACTION_MAP_FROZEN = (
    "tape_active", "tape_state", "tape_cell", "tape_u", "state_edge_pos", "state_edge_dir",
    "state_t_min", "state_t_max", "state_n0", "state_n1", "state_prim0", "state_prim1",
    "state_exterior_angle", "state_src", "state_src_power", "material_mu_r", "material_valid",
)
_DIFFRACTION_MAP_TANGENT_FIXED = (
    "state_edge_pos", "state_edge_dir", "state_n0", "state_n1", "state_src", "state_src_power",
)
_DIFFRACTION_MAP_AD = AdInputs(
    "mc_utd_diffraction_tape_accumulate_ad", _MAP_DIFF_INPUTS,
    *(
        opaque(
            name,
            grad="none" if name in ("state_prim0", "state_prim1") else "fixed",
            tangent=name in _DIFFRACTION_MAP_TANGENT_FIXED,
        )
        for name in _DIFFRACTION_MAP_FROZEN
    ),
    opaque("params"),
)


class _McDiffractionMapAdFunction(torch.autograd.Function):
    """Differentiable UTD diffraction radiomap for one transmitter.

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
        tx_anchor: Any, eta_r: Any, sigma_e: Any, gain: Any, thickness: Any, frequency: Any,
        tape_active: Any, tape_state: Any, tape_cell: Any, tape_u: Any, state_edge_pos: Any,
        state_edge_dir: Any, state_t_min: Any, state_t_max: Any, state_n0: Any, state_n1: Any,
        state_prim0: Any, state_prim1: Any, state_exterior_angle: Any, state_src: Any,
        state_src_power: Any, material_mu_r: Any, material_valid: Any, params: Any,
    ) -> Any:
        return mc_utd_diffraction_tape_accumulate(
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
            params["tx_pol"],
            point_tape=params.get("point_tape"),
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        ad = _DIFFRACTION_MAP_AD
        ctx.frequency_meta = frequency_meta(ad.get(inputs, "frequency"))
        ctx.params = ad.get(inputs, "params")
        ctx.anchor_shape = tuple(ad.get(inputs, "tx_anchor").shape)
        save(
            ctx, *ad.primals(inputs, *_MAP_MATERIALS),
            *(ad.get(inputs, name) for name in _DIFFRACTION_MAP_FROZEN),
        )

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, grad_output: Any) -> Any:
        ad = _DIFFRACTION_MAP_AD
        none_grads = ad.none
        ad.reject_fixed_grads(ctx)
        if grad_output is None:
            return none_grads
        saved = ctx.saved_tensors
        (eta_r, sigma_e, gain, thickness) = saved[:4]
        tape_tensors = saved[4:8]
        state_tensors = saved[8:19]
        material_mu_r = saved[19]
        material_valid = saved[20]
        need_anchor = ad.need(ctx, "tx_anchor")
        need_materials = ad.needs(ctx, *_MAP_MATERIALS)
        need_frequency = ad.need(ctx, "frequency")
        if not (need_anchor or need_materials or need_frequency):
            return none_grads
        params = ctx.params
        wavelength = float(params["wavelength"])
        gradients = mc_utd_diffraction_tape_accumulate_backward(
            tuple(tape_tensors),
            tuple(state_tensors),
            eta_r,
            sigma_e,
            material_mu_r,
            gain,
            material_valid,
            thickness,
            grad_output,
            tx_pol=params["tx_pol"],
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
            point_tape=params.get("point_tape"),
            wavelength_dfreq=-wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD,
        )
        grad_frequency = (
            _ad_frequency_grad(gradients[5], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return ad.grads(
            ctx, tx_anchor=gradients[4], **dict(zip(_MAP_MATERIALS, gradients[:4], strict=True)),
            frequency=grad_frequency,
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        ad = _DIFFRACTION_MAP_AD
        ad.reject_fixed_tangents(tangents)
        saved = ctx.saved_tensors
        face_shape = tuple(saved[0].shape)
        tangent_eta, tangent_sigma, tangent_gain, tangent_thickness = (
            ad.checked_tangent(tangents, name, face_shape) for name in _MAP_MATERIALS
        )
        t_anchor = ad.tangent(tangents, "tx_anchor")
        tangent_anchor = _ad_native_tangent_or_none(
            t_anchor if isinstance(t_anchor, torch.Tensor) else None
        )
        tangent_frequency = _ad_frequency_tangent(ad.tangent(tangents, "frequency"))
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
        with disable_functorch():
            return mc_utd_diffraction_tape_accumulate_jvp(
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
                tx_pol=params["tx_pol"],
                grid_axis=params["grid_axis"],
                grid_position=params["grid_position"],
                grid_resolution0=params["grid_resolution0"],
                grid_resolution1=params["grid_resolution1"],
                wavelength=wavelength,
                grid_cell_area=params["grid_cell_area"],
                seed=params["seed"],
                total_edge_length=params["total_edge_length"],
                point_tape=params.get("point_tape"),
                wavelength_tangent=(-wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD)
                * tangent_frequency,
            )


def mc_utd_diffraction_tape_accumulate_ad(
    tx_anchor: torch.Tensor, material_eta_r: torch.Tensor, material_sigma: torch.Tensor,
    material_gain: torch.Tensor, material_thickness: torch.Tensor, frequency: torch.Tensor | float,
    tape_tensors: tuple[torch.Tensor, ...], state_tensors: tuple[torch.Tensor, ...],
    material_mu_r: torch.Tensor, material_valid: torch.Tensor, *, tx_pol: torch.Tensor,
    grid_axis: int, grid_position: float, grid_coord0_min: float, grid_coord0_max: float,
    grid_coord1_min: float, grid_coord1_max: float, grid_resolution0: int, grid_resolution1: int,
    wavelength: float, grid_cell_area: float, seed: int, total_edge_length: float,
    point_tape: tuple[torch.Tensor, torch.Tensor, float, int] | None = None,
) -> torch.Tensor:
    """Differentiable :func:`mc_utd_diffraction_tape_accumulate` (one tx)."""

    if len(tape_tensors) != 4:
        raise ValueError("tape_tensors must hold (active, state, cell, u)")
    if len(state_tensors) != 11:
        raise ValueError("state_tensors must hold the 11 packed state tables")
    params = {
        "tx_pol": tx_pol,
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
        "point_tape": point_tape,
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


# -------------------------------------------------------------------------
# basic sampling
# -------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PointDiffractionTape:
    """Capture-sphere diffraction tape of a point-receiver range and its visibility legs."""

    active: torch.Tensor
    state: torch.Tensor
    cell: torch.Tensor
    u: torch.Tensor
    lane: torch.Tensor
    leg_source: torch.Tensor
    leg_edge: torch.Tensor
    leg_receiver: torch.Tensor


def mc_point_diffraction_tape(
    sample_state: torch.Tensor, states: tuple[torch.Tensor, ...], points: torch.Tensor, *,
    point_begin: int, point_count: int, seed: int, point_radius: float,
) -> PointDiffractionTape:
    """One tape entry per (lane, receiver) of ``points[point_begin:point_begin + point_count]``.

    ``states`` are the packed edge state tables (``edge_pos, edge_dir, t_min, t_max, n0, nn,
    ..., src, ...``) in the order ``deterministic_diffraction_state_pack`` returns them after
    its edge-index column.
    """

    out = _MC_POINT_DIFFRACTION_TAPE(
        sample_state,
        states[0],
        states[1],
        states[2],
        states[3],
        states[4],
        states[5],
        states[9],
        points,
        int(point_begin),
        int(point_count),
        int(seed),
        float(point_radius),
    )
    if not isinstance(out, tuple) or len(out) != 8:
        raise TypeError("_channel.mc_point_diffraction_tape must return 8 tensors")
    return PointDiffractionTape(*out)


def mc_reflection_rays(
    tx_positions: torch.Tensor, *, bounds: torch.Tensor | None, sample_count: int, seed: int,
) -> dict[str, torch.Tensor]:
    """Seeded lattice launch rays for every transmitter, folded into the scene bounds.

    Returns ``ray_o``/``ray_d`` of shape ``(T * N, 3)`` (transmitter-major), ``active`` and the
    per-transmitter ``solid_angle`` each ray carries, shape ``(T,)``.
    """

    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if bounds is not None:
        validate_cuda_tensor("bounds", bounds, dtype=torch.float32, ndim=1)
    exported = _MC_REFLECTION_RAYS(
        tx_positions, bounds, int(sample_count), int(seed)
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.mc_reflection_rays must return a dict")
    return exported


def mc_scene_bounds(vertices: torch.Tensor) -> torch.Tensor:
    """Axis-aligned bounds ``[min_x, min_y, min_z, max_x, max_y, max_z]`` of scene vertices."""

    validate_cuda_tensor("vertices", vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    bounds = _MC_SCENE_BOUNDS(vertices)
    if not isinstance(bounds, torch.Tensor):
        raise TypeError("_channel.mc_scene_bounds must return a tensor")
    return bounds


@dataclass(frozen=True, slots=True)
class ReflectionFinalSegments:
    """Spawn points, directions, distance bounds and activity of the final reflected segments."""

    origin: torch.Tensor
    direction: torch.Tensor
    tmax: torch.Tensor
    offset: torch.Tensor
    active: torch.Tensor


def mc_reflection_final_segment(
    ray_o: torch.Tensor, ray_d: torch.Tensor, trace: tuple[torch.Tensor, ...],
    face_normals: torch.Tensor, material_valid: torch.Tensor, *, receiver: Mapping[str, Any],
    wavelength: float, segments: ReflectionFinalSegments | None,
) -> ReflectionFinalSegments:
    """Fold one receiver block into the bounded final-segment request (``segments=None`` starts)."""

    rays = int(ray_o.shape[0])
    if segments is None:
        segments = ReflectionFinalSegments(
            origin=torch.empty_like(ray_o),
            direction=torch.empty_like(ray_o),
            tmax=ray_o.new_empty((rays,)),
            offset=ray_o.new_empty((rays,)),
            active=torch.empty((rays,), dtype=torch.bool, device=ray_o.device),
        )
        first_block = True
    else:
        first_block = False
    _MC_REFLECTION_FINAL_SEGMENT(
        ray_o,
        ray_d,
        trace[0],
        trace[1],
        trace[2],
        face_normals,
        material_valid,
        int(receiver["grid_axis"]),
        float(receiver["grid_position"]),
        float(receiver["grid_coord0_min"]),
        float(receiver["grid_coord0_max"]),
        float(receiver["grid_coord1_min"]),
        float(receiver["grid_coord1_max"]),
        int(receiver["grid_resolution0"]),
        int(receiver["grid_resolution1"]),
        float(receiver["grid_cell_area"]),
        _point_rows(receiver.get("points"), ray_o),
        float(receiver.get("point_radius", -0.01)),
        float(wavelength),
        bool(first_block),
        segments.origin,
        segments.direction,
        segments.tmax,
        segments.offset,
        segments.active,
    )
    return segments


def mc_reflection_append_final(
    trace: tuple[torch.Tensor, ...], final_trace: tuple[torch.Tensor, ...],
    segments: ReflectionFinalSegments,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trace table with the bounded final-segment hit appended as its last column."""

    out = _MC_REFLECTION_APPEND_FINAL(
        trace[0],
        trace[1],
        trace[2],
        final_trace[0],
        final_trace[1],
        final_trace[2],
        segments.offset,
    )
    if not isinstance(out, tuple) or len(out) != 3:
        raise TypeError("_channel.mc_reflection_append_final must return 3 tensors")
    return out[0], out[1], out[2]


def _validate_mc_diffraction_discovery_args(
    args: tuple[torch.Tensor, ...], *, counted: bool,
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
        trailing_shape: tuple[int, ...] = (3,) if name not in {"prim_index"} else ()
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
    out = _MC_DIFFRACTION_DISCOVER_EDGES(*args)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.mc_diffraction_discover_edges must return a tensor"
        )
    return out


def mc_diffraction_discover_edges_counted(*args: torch.Tensor) -> torch.Tensor:
    _validate_mc_diffraction_discovery_args(args, counted=True)
    out = _MC_DIFFRACTION_DISCOVER_EDGES_COUNTED(*args)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.mc_diffraction_discover_edges_counted must return a tensor"
        )
    return out


def mc_diffraction_state_wi(state_edge_pos: torch.Tensor, state_src: torch.Tensor) -> torch.Tensor:
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
    state_wi = _MC_DIFFRACTION_STATE_WI(state_edge_pos, state_src)
    if not isinstance(state_wi, torch.Tensor):
        raise TypeError("_channel.mc_diffraction_state_wi must return a tensor")
    validate_cuda_tensor(
        "state_wi", state_wi, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return state_wi


# -------------------------------------------------------------------------
# basic transmission
# -------------------------------------------------------------------------
_CONTRACT_FAILURE_BIT = int(CapacityFailureBit.PAIR_CONTRACT_ERROR)


@dataclass(frozen=True, slots=True)
class McTransmissionWallProduct:
    """Fixed-capacity resident outputs of the MC wall-product estimator."""

    scaled_power: torch.Tensor
    transmittance: torch.Tensor
    wall_count: torch.Tensor
    penetrated: torch.Tensor

# The 16 tensor inputs of the wall-product estimator, in binding order.
_WallProductInputs = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]


def _validate_inputs(
    valid: torch.Tensor, num_hits: torch.Tensor, reached_target: torch.Tensor,
    direction: torch.Tensor, normal: torch.Tensor, global_primitive_id: torch.Tensor,
    face_material_id: torch.Tensor, geometry_mode_id: torch.Tensor, layer_offset: torch.Tensor,
    layer_count: torch.Tensor, layer_thickness_m: torch.Tensor, layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, pair_polarization: torch.Tensor,
    base_power: torch.Tensor, failure_state: CapacityFailureState, *, frequency_hz: float,
) -> tuple[int, int]:
    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=2)
    rows, hit_capacity = valid.shape
    validate_cuda_tensor("num_hits", num_hits, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("reached_target", reached_target, dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "direction", direction, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "normal", normal, dtype=torch.float32, ndim=3, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "global_primitive_id", global_primitive_id, dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "geometry_mode_id", geometry_mode_id, dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor("layer_offset", layer_offset, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("layer_count", layer_count, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "layer_thickness_m", layer_thickness_m, dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor("layer_eps_r", layer_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("layer_sigma_e", layer_sigma_e, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("layer_mu_r", layer_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "pair_polarization",
        pair_polarization,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor("base_power", base_power, dtype=torch.float32, ndim=1)
    if num_hits.shape != (rows,):
        raise ValueError("num_hits must match valid rows")
    if reached_target.shape != (rows,):
        raise ValueError("reached_target must match valid rows")
    if direction.shape != (rows, 3):
        raise ValueError("direction must have shape (N, 3)")
    if normal.shape != (rows, hit_capacity, 3):
        raise ValueError("normal must have shape (N, D, 3)")
    if global_primitive_id.shape != valid.shape:
        raise ValueError("global_primitive_id must match valid")
    if pair_polarization.shape != (rows, 3):
        raise ValueError("pair_polarization must have shape (N, 3)")
    if base_power.shape != (rows,):
        raise ValueError("base_power must match valid rows")
    material_count = layer_offset.shape[0]
    if layer_count.shape != (material_count,):
        raise ValueError("layer_count must match layer_offset")
    if geometry_mode_id.shape != (material_count,):
        raise ValueError("geometry_mode_id must match material rows")
    layer_length = layer_thickness_m.shape[0]
    if any(
        tensor.shape != (layer_length,)
        for tensor in (layer_eps_r, layer_sigma_e, layer_mu_r)
    ):
        raise ValueError("layer property tensors must have one shared length")
    device = valid.device
    for name, tensor in (
        ("num_hits", num_hits),
        ("reached_target", reached_target),
        ("direction", direction),
        ("normal", normal),
        ("global_primitive_id", global_primitive_id),
        ("face_material_id", face_material_id),
        ("geometry_mode_id", geometry_mode_id),
        ("layer_offset", layer_offset),
        ("layer_count", layer_count),
        ("layer_thickness_m", layer_thickness_m),
        ("layer_eps_r", layer_eps_r),
        ("layer_sigma_e", layer_sigma_e),
        ("layer_mu_r", layer_mu_r),
        ("pair_polarization", pair_polarization),
        ("base_power", base_power),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must share the valid device")
    require_capacity_failure_state(failure_state, device=device)
    _validate_frequency_hz(frequency_hz)
    return int(rows), int(hit_capacity)


def _arguments(
    valid: torch.Tensor, num_hits: torch.Tensor, reached_target: torch.Tensor,
    direction: torch.Tensor, normal: torch.Tensor, global_primitive_id: torch.Tensor,
    face_material_id: torch.Tensor, geometry_mode_id: torch.Tensor, layer_offset: torch.Tensor,
    layer_count: torch.Tensor, layer_thickness_m: torch.Tensor, layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, pair_polarization: torch.Tensor,
    base_power: torch.Tensor, frequency_hz: float, failure_state: CapacityFailureState,
) -> tuple[object, ...]:
    return (
        valid,
        num_hits,
        reached_target,
        direction,
        normal,
        global_primitive_id,
        face_material_id,
        geometry_mode_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        pair_polarization,
        base_power,
        float(frequency_hz),
        failure_state.bits,
        _CONTRACT_FAILURE_BIT,
    )


def _result(exported: object, *, rows: int) -> McTransmissionWallProduct:
    if not isinstance(exported, dict):
        raise TypeError("native MC transmission wall product must return a dict")
    scaled_power = exported["scaled_power"]
    transmittance = exported["transmittance"]
    wall_count = exported["wall_count"]
    penetrated = exported["penetrated"]
    validate_cuda_tensor("scaled_power", scaled_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("transmittance", transmittance, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("wall_count", wall_count, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("penetrated", penetrated, dtype=torch.bool, ndim=1)
    if any(
        tensor.shape != (rows,)
        for tensor in (scaled_power, transmittance, wall_count, penetrated)
    ):
        raise ValueError("native MC transmission wall product returned wrong rows")
    return McTransmissionWallProduct(
        scaled_power=scaled_power,
        transmittance=transmittance,
        wall_count=wall_count,
        penetrated=penetrated,
    )


def mc_transmission_wall_product(
    valid: torch.Tensor, num_hits: torch.Tensor, reached_target: torch.Tensor,
    direction: torch.Tensor, normal: torch.Tensor, global_primitive_id: torch.Tensor,
    face_material_id: torch.Tensor, geometry_mode_id: torch.Tensor, layer_offset: torch.Tensor,
    layer_count: torch.Tensor, layer_thickness_m: torch.Tensor, layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, pair_polarization: torch.Tensor,
    base_power: torch.Tensor, failure_state: CapacityFailureState, *, frequency_hz: float,
) -> McTransmissionWallProduct:
    """Evaluate the live segment penetration fixed-capacity MC estimator."""

    rows, _ = _validate_inputs(
        valid,
        num_hits,
        reached_target,
        direction,
        normal,
        global_primitive_id,
        face_material_id,
        geometry_mode_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        pair_polarization,
        base_power,
        failure_state,
        frequency_hz=float(frequency_hz),
    )
    exported = _MC_TRANSMISSION_WALL_PRODUCT(
        *_arguments(
            valid,
            num_hits,
            reached_target,
            direction,
            normal,
            global_primitive_id,
            face_material_id,
            geometry_mode_id,
            layer_offset,
            layer_count,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            pair_polarization,
            base_power,
            float(frequency_hz),
            failure_state,
        )
    )
    return _result(exported, rows=rows)


def mc_transmission_wall_product_backward(
    *inputs: torch.Tensor, frequency_hz: float, failure_state: CapacityFailureState,
    grad_scaled_power: torch.Tensor | None, grad_transmittance: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 16:
        raise ValueError("MC transmission wall-product backward requires 16 inputs")
    fixed = cast(_WallProductInputs, inputs)
    rows, _ = _validate_inputs(*fixed, failure_state, frequency_hz=float(frequency_hz))
    for name, gradient in (
        ("grad_scaled_power", grad_scaled_power),
        ("grad_transmittance", grad_transmittance),
    ):
        if gradient is not None:
            validate_cuda_tensor(
                name,
                gradient,
                dtype=torch.float32,
                ndim=1,
                require_contiguous=False,
            )
            if gradient.shape != (rows,):
                raise ValueError(f"{name} must match the output rows")
    exported = _MC_TRANSMISSION_WALL_PRODUCT_BACKWARD(
        *_arguments(*fixed, float(frequency_hz), failure_state),
        grad_scaled_power,
        grad_transmittance,
    )
    if not isinstance(exported, tuple) or len(exported) != 7:
        raise TypeError(
            "native MC transmission wall-product backward must return 7 tensors"
        )
    named = dict(zip(_WALL_PRODUCT_TENSORS, inputs, strict=True))
    expected = (
        *((torch.float32, named[name].shape) for name in _WALL_PRODUCT_DIFF),
        (torch.float32, (1,)),
    )
    for index, (tensor, (dtype, shape)) in enumerate(
        zip(exported, expected, strict=True)
    ):
        validate_cuda_tensor(f"gradient[{index}]", tensor, dtype=dtype, ndim=len(shape))
        if tensor.shape != shape:
            raise ValueError(f"gradient[{index}] has the wrong shape")
    return exported


def mc_transmission_wall_product_jvp(
    *inputs: torch.Tensor, frequency_hz: float, failure_state: CapacityFailureState,
    tangent_direction: torch.Tensor | None, tangent_normal: torch.Tensor | None,
    tangent_layer_thickness_m: torch.Tensor | None, tangent_layer_eps_r: torch.Tensor | None,
    tangent_layer_sigma_e: torch.Tensor | None, tangent_base_power: torch.Tensor | None,
    tangent_frequency: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 16:
        raise ValueError("MC transmission wall-product JVP requires 16 inputs")
    fixed = cast(_WallProductInputs, inputs)
    rows, _ = _validate_inputs(*fixed, failure_state, frequency_hz=float(frequency_hz))
    exported = _MC_TRANSMISSION_WALL_PRODUCT_JVP(
        *_arguments(*fixed, float(frequency_hz), failure_state),
        tangent_direction,
        tangent_normal,
        tangent_layer_thickness_m,
        tangent_layer_eps_r,
        tangent_layer_sigma_e,
        tangent_base_power,
        float(tangent_frequency),
    )
    if not isinstance(exported, dict):
        raise TypeError("native MC transmission wall-product JVP must return a dict")
    scaled_power = exported["scaled_power"]
    transmittance = exported["transmittance"]
    for name, tensor in (
        ("scaled_power", scaled_power),
        ("transmittance", transmittance),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=1)
        if tensor.shape != (rows,):
            raise ValueError(
                f"native MC transmission wall-product JVP {name} has wrong rows"
            )
    return scaled_power, transmittance


# The apply inputs of the wall product: 16 typed tensors, the frequency (tensor or float), its
# host value and the capacity failure record. Ids, masks, mu_r and the polarizations are fixed.
_WALL_PRODUCT_DIFF = (
    "direction", "normal", "layer_thickness_m", "layer_eps_r", "layer_sigma_e", "base_power",
)
_WALL_PRODUCT_AD = AdInputs(
    "mc_transmission_wall_product_ad",
    *(
        opaque(name, grad="diff" if name in _WALL_PRODUCT_DIFF else "fixed")
        for name in (
            "valid", "num_hits", "reached_target", "direction", "normal", "global_primitive_id",
            "face_material_id", "geometry_mode_id", "layer_offset", "layer_count",
            "layer_thickness_m", "layer_eps_r", "layer_sigma_e", "layer_mu_r",
            "pair_polarization", "base_power",
        )
    ),
    opaque("frequency", grad="diff"),
    opaque("frequency_value"),
    opaque("capacity_failure_state", grad="fixed"),
)
_WALL_PRODUCT_TENSORS = _WALL_PRODUCT_AD.names[:16]


class _McTransmissionWallProductAd(torch.autograd.Function):
    @staticmethod
    def forward(*inputs: Any) -> Any:
        ad = _WALL_PRODUCT_AD
        failure_state = CapacityFailureState(record=ad.get(inputs, "capacity_failure_state"))
        result = mc_transmission_wall_product(
            *cast(
                _WallProductInputs, tuple(ad.get(inputs, name) for name in _WALL_PRODUCT_TENSORS)
            ),
            failure_state, frequency_hz=ad.get(inputs, "frequency_value"),
        )
        return (
            result.scaled_power,
            result.transmittance,
            result.wall_count,
            result.penetrated,
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ad = _WALL_PRODUCT_AD
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(output[2], output[3])
        ctx.frequency_value = float(ad.get(inputs, "frequency_value"))
        ctx.frequency_meta = frequency_meta(ad.get(inputs, "frequency"))
        save(ctx, *ad.primals(inputs, *_WALL_PRODUCT_TENSORS, "capacity_failure_state"))

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx: Any, grad_scaled: Any, grad_transmittance: Any, _grad_count: Any,
        _grad_penetrated: Any,
    ) -> Any:
        ad = _WALL_PRODUCT_AD
        ad.reject_fixed_grads(ctx)
        if (grad_scaled is None and grad_transmittance is None) or not ad.needs(
            ctx, *_WALL_PRODUCT_DIFF, "frequency"
        ):
            return ad.none
        saved = ctx.saved_tensors
        gradients = mc_transmission_wall_product_backward(
            *saved[:16],
            frequency_hz=ctx.frequency_value,
            failure_state=CapacityFailureState(record=saved[16]),
            grad_scaled_power=grad_scaled,
            grad_transmittance=grad_transmittance,
        )
        return ad.grads(
            ctx,
            **dict(zip(_WALL_PRODUCT_DIFF, gradients[:6], strict=True)),
            frequency=(
                _ad_frequency_grad(gradients[6], ctx.frequency_meta)
                if ad.need(ctx, "frequency")
                else None
            ),
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        ad = _WALL_PRODUCT_AD
        ad.reject_fixed_tangents(tangents)
        saved = ctx.saved_tensors
        inputs = tuple(_ad_native_tensor(value) for value in saved[:16])
        primal = dict(zip(_WALL_PRODUCT_TENSORS, inputs, strict=True))
        tangent_direction, tangent_normal = (
            _ad_geometry_tangent(f"tangent_{name}", ad.tangent(tangents, name), primal[name])
            for name in ("direction", "normal")
        )
        continuous_tangents = tuple(
            _ad_checked_tangent(
                f"tangent_{name}",
                _ad_native_tangent_or_none(ad.tangent(tangents, name)),
                tuple(primal[name].shape),
            )
            for name in _WALL_PRODUCT_DIFF[2:]
        )
        tangent_frequency = _ad_frequency_tangent(ad.tangent(tangents, "frequency"))
        if (
            tangent_direction is None
            and tangent_normal is None
            and all(value is None for value in continuous_tangents)
            and tangent_frequency == 0.0
        ):
            return None, None, None, None
        with disable_functorch():
            scaled, transmittance = mc_transmission_wall_product_jvp(
                *inputs,
                frequency_hz=ctx.frequency_value,
                failure_state=CapacityFailureState(record=_ad_native_tensor(saved[16])),
                tangent_direction=tangent_direction,
                tangent_normal=tangent_normal,
                tangent_layer_thickness_m=continuous_tangents[0],
                tangent_layer_eps_r=continuous_tangents[1],
                tangent_layer_sigma_e=continuous_tangents[2],
                tangent_base_power=continuous_tangents[3],
                tangent_frequency=tangent_frequency,
            )
        return scaled, transmittance, None, None


def mc_transmission_wall_product_ad(
    valid: torch.Tensor, num_hits: torch.Tensor, reached_target: torch.Tensor,
    direction: torch.Tensor, normal: torch.Tensor, global_primitive_id: torch.Tensor,
    face_material_id: torch.Tensor, geometry_mode_id: torch.Tensor, layer_offset: torch.Tensor,
    layer_count: torch.Tensor, layer_thickness_m: torch.Tensor, layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, pair_polarization: torch.Tensor,
    base_power: torch.Tensor, frequency: torch.Tensor | float, failure_state: CapacityFailureState,
    *, frequency_value: float | None = None,
) -> McTransmissionWallProduct:
    """Differentiable fixed-topology wall product with native VJP/JVP."""

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _McTransmissionWallProductAd.apply(
        valid,
        num_hits,
        reached_target,
        direction,
        normal,
        global_primitive_id,
        face_material_id,
        geometry_mode_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        pair_polarization,
        base_power,
        frequency,
        float(frequency_value),
        failure_state.record,
    )
    return McTransmissionWallProduct(*values)


def _validate_frequency_hz(frequency_hz: float) -> None:
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")


__all__ = [
    "McTransmissionWallProduct",
    "mc_transmission_wall_product",
    "mc_transmission_wall_product_ad",
    "mc_transmission_wall_product_backward",
    "mc_transmission_wall_product_jvp",
]



def bdpt_finalize_point_components(
    los: torch.Tensor, reflection: torch.Tensor, diffraction: torch.Tensor,
    transmission: torch.Tensor, scattering: torch.Tensor,
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
    exported = _BDPT_FINALIZE_POINT_COMPONENTS(
        los, reflection, diffraction, transmission, scattering
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.bdpt_finalize_point_components must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=2
    )
    if exported["path_gain"].shape != los.shape:
        raise ValueError(
            "_channel.bdpt_finalize_point_components returned bad path_gain shape"
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


def bdpt_transmitter_tensors(
    flat_positions: tuple[float, ...], powers: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    if len(flat_positions) % 3 != 0:
        raise ValueError("flat_positions must contain xyz triples")
    if len(flat_positions) // 3 != len(powers):
        raise ValueError("powers must match flat_positions")
    exported = _BDPT_TRANSMITTER_TENSORS(flat_positions, powers)
    if not isinstance(exported, dict):
        raise TypeError("_channel.bdpt_transmitter_tensors must return a dict")
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
    reference: torch.Tensor, *, origin: tuple[float, float, float],
    x_axis: tuple[float, float, float], y_axis: tuple[float, float, float], shape: tuple[int, int],
    spacing: tuple[float, float],
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    rows, cols = shape
    if rows < 0 or cols < 0:
        raise ValueError("shape entries must be non-negative")
    if spacing[0] <= 0.0 or spacing[1] <= 0.0:
        raise ValueError("spacing entries must be positive")
    points = _BDPT_RECEIVER_GRID_POINTS(
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
            "_channel.bdpt_receiver_grid_points must return a tensor"
        )
    validate_cuda_tensor(
        "points", points, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if points.shape[0] != rows * cols:
        raise ValueError(
            "_channel.bdpt_receiver_grid_points returned an unexpected shape"
        )
    return points


def bdpt_los_component_maps_from_matrix(los: torch.Tensor, *, rows: int, cols: int) -> torch.Tensor:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    if los.shape[1] != int(rows) * int(cols):
        raise ValueError("los columns must match rows * cols")
    maps = _BDPT_LOS_COMPONENT_MAPS_FROM_MATRIX(
        los, int(rows), int(cols)
    )
    if not isinstance(maps, torch.Tensor):
        raise TypeError(
            "_channel.bdpt_los_component_maps_from_matrix must return a tensor"
        )
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (los.shape[0], int(rows), int(cols)):
        raise ValueError(
            "_channel.bdpt_los_component_maps_from_matrix returned an unexpected shape"
        )
    return maps


def bdpt_finalize_component_maps(
    los: torch.Tensor, reflection: torch.Tensor, diffraction: torch.Tensor,
    transmission: torch.Tensor, scattering: torch.Tensor,
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
    exported = _BDPT_FINALIZE_COMPONENT_MAPS(
        los, reflection, diffraction, transmission, scattering
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.bdpt_finalize_component_maps must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=3
    )
    if exported["path_gain"].shape != los.shape:
        raise ValueError(
            "_channel.bdpt_finalize_component_maps returned bad path_gain shape"
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


# ---------------------------------------------------------------------------
# BDPT finalize AD facades. The
# finalize map is linear (elementwise sum into path_gain + per-component 0-dim
# power reductions), so the backward is the transpose scaling and the jvp is the
# forward map on the tangents; both are deterministic with no atomics. Pure
# dispatch, no Torch physics.
# ---------------------------------------------------------------------------

_BDPT_FINALIZE_COMPONENT_GRADS = (
    "grad_los",
    "grad_reflection",
    "grad_diffraction",
    "grad_transmission",
    "grad_scattering",
)
_BDPT_FINALIZE_TANGENTS = (
    "tangent_path_gain",
    "tangent_los_power",
    "tangent_reflection_power",
    "tangent_diffraction_power",
    "tangent_transmission_power",
    "tangent_scattering_power",
)


def _bdpt_finalize_backward(
    op: NativeOp, ndim: int, los: torch.Tensor, reflection: torch.Tensor, diffraction: torch.Tensor,
    transmission: torch.Tensor, scattering: torch.Tensor, *, grad_path_gain: torch.Tensor | None,
    grad_los_power: torch.Tensor | None, grad_reflection_power: torch.Tensor | None,
    grad_diffraction_power: torch.Tensor | None, grad_transmission_power: torch.Tensor | None,
    grad_scattering_power: torch.Tensor | None, need_grad_components: bool,
) -> dict[str, torch.Tensor | None]:
    for name, tensor in (
        ("los", los),
        ("reflection", reflection),
        ("diffraction", diffraction),
        ("transmission", transmission),
        ("scattering", scattering),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=ndim)
        if tensor.shape != los.shape:
            raise ValueError("component tensors must share shape")
    if grad_path_gain is not None:
        validate_cuda_tensor(
            "grad_path_gain",
            grad_path_gain,
            dtype=torch.float32,
            ndim=ndim,
            require_contiguous=False,
        )
    for name, tensor in (
        ("grad_los_power", grad_los_power),
        ("grad_reflection_power", grad_reflection_power),
        ("grad_diffraction_power", grad_diffraction_power),
        ("grad_transmission_power", grad_transmission_power),
        ("grad_scattering_power", grad_scattering_power),
    ):
        if tensor is not None:
            validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=0)
    exported = op(
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        grad_path_gain,
        grad_los_power,
        grad_reflection_power,
        grad_diffraction_power,
        grad_transmission_power,
        grad_scattering_power,
        bool(need_grad_components),
    )
    if not isinstance(exported, dict) or set(exported) != set(
        _BDPT_FINALIZE_COMPONENT_GRADS
    ):
        raise TypeError(f"_channel.{op.symbol} returned unexpected fields")
    return exported


def _bdpt_finalize_jvp(
    op: NativeOp, ndim: int, los: torch.Tensor, reflection: torch.Tensor, diffraction: torch.Tensor,
    transmission: torch.Tensor, scattering: torch.Tensor, *, tangent_los: torch.Tensor | None,
    tangent_reflection: torch.Tensor | None, tangent_diffraction: torch.Tensor | None,
    tangent_transmission: torch.Tensor | None, tangent_scattering: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    for name, tensor in (
        ("los", los),
        ("reflection", reflection),
        ("diffraction", diffraction),
        ("transmission", transmission),
        ("scattering", scattering),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=ndim)
        if tensor.shape != los.shape:
            raise ValueError("component tensors must share shape")
    exported = op(
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        tangent_los,
        tangent_reflection,
        tangent_diffraction,
        tangent_transmission,
        tangent_scattering,
    )
    if not isinstance(exported, dict) or set(exported) != set(_BDPT_FINALIZE_TANGENTS):
        raise TypeError(f"_channel.{op.symbol} returned unexpected fields")
    return exported


def bdpt_finalize_point_components_backward(
    los: torch.Tensor, reflection: torch.Tensor, diffraction: torch.Tensor,
    transmission: torch.Tensor, scattering: torch.Tensor, *,
    grad_path_gain: torch.Tensor | None = None, grad_los_power: torch.Tensor | None = None,
    grad_reflection_power: torch.Tensor | None = None,
    grad_diffraction_power: torch.Tensor | None = None,
    grad_transmission_power: torch.Tensor | None = None,
    grad_scattering_power: torch.Tensor | None = None, need_grad_components: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`bdpt_finalize_point_components` (point-result finalization)."""

    return _bdpt_finalize_backward(
        _BDPT_FINALIZE_POINT_COMPONENTS_BACKWARD,
        2,
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        grad_path_gain=grad_path_gain,
        grad_los_power=grad_los_power,
        grad_reflection_power=grad_reflection_power,
        grad_diffraction_power=grad_diffraction_power,
        grad_transmission_power=grad_transmission_power,
        grad_scattering_power=grad_scattering_power,
        need_grad_components=need_grad_components,
    )


def bdpt_finalize_point_components_jvp(
    los: torch.Tensor, reflection: torch.Tensor, diffraction: torch.Tensor,
    transmission: torch.Tensor, scattering: torch.Tensor, *,
    tangent_los: torch.Tensor | None = None, tangent_reflection: torch.Tensor | None = None,
    tangent_diffraction: torch.Tensor | None = None,
    tangent_transmission: torch.Tensor | None = None,
    tangent_scattering: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`bdpt_finalize_point_components` (point-result finalization)."""

    return _bdpt_finalize_jvp(
        _BDPT_FINALIZE_POINT_COMPONENTS_JVP,
        2,
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        tangent_los=tangent_los,
        tangent_reflection=tangent_reflection,
        tangent_diffraction=tangent_diffraction,
        tangent_transmission=tangent_transmission,
        tangent_scattering=tangent_scattering,
    )


def bdpt_finalize_component_maps_backward(
    los: torch.Tensor, reflection: torch.Tensor, diffraction: torch.Tensor,
    transmission: torch.Tensor, scattering: torch.Tensor, *,
    grad_path_gain: torch.Tensor | None = None, grad_los_power: torch.Tensor | None = None,
    grad_reflection_power: torch.Tensor | None = None,
    grad_diffraction_power: torch.Tensor | None = None,
    grad_transmission_power: torch.Tensor | None = None,
    grad_scattering_power: torch.Tensor | None = None, need_grad_components: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`bdpt_finalize_component_maps` (map finalization, 3-D maps)."""

    return _bdpt_finalize_backward(
        _BDPT_FINALIZE_COMPONENT_MAPS_BACKWARD,
        3,
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        grad_path_gain=grad_path_gain,
        grad_los_power=grad_los_power,
        grad_reflection_power=grad_reflection_power,
        grad_diffraction_power=grad_diffraction_power,
        grad_transmission_power=grad_transmission_power,
        grad_scattering_power=grad_scattering_power,
        need_grad_components=need_grad_components,
    )


def bdpt_finalize_component_maps_jvp(
    los: torch.Tensor, reflection: torch.Tensor, diffraction: torch.Tensor,
    transmission: torch.Tensor, scattering: torch.Tensor, *,
    tangent_los: torch.Tensor | None = None, tangent_reflection: torch.Tensor | None = None,
    tangent_diffraction: torch.Tensor | None = None,
    tangent_transmission: torch.Tensor | None = None,
    tangent_scattering: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`bdpt_finalize_component_maps` (map finalization, 3-D maps)."""

    return _bdpt_finalize_jvp(
        _BDPT_FINALIZE_COMPONENT_MAPS_JVP,
        3,
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        tangent_los=tangent_los,
        tangent_reflection=tangent_reflection,
        tangent_diffraction=tangent_diffraction,
        tangent_transmission=tangent_transmission,
        tangent_scattering=tangent_scattering,
    )


# -------------------------------------------------------------------------
# bdpt paths
# -------------------------------------------------------------------------
def bdpt_launch_state(
    reference: torch.Tensor, *, tx_count: int, samples: int, sample_streams: int, seed: int,
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

    exported = _BDPT_LAUNCH_STATE(
        reference,
        int(tx_count),
        int(samples),
        int(sample_streams),
        int(seed),
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.bdpt_launch_state must return a dict")
    expected = int(tx_count) * int(samples) * int(sample_streams)
    for name in ("tx_id", "sample_id", "stream_id"):
        validate_cuda_tensor(name, exported[name], dtype=torch.int32, ndim=1)
        if exported[name].shape != (expected,):
            raise ValueError(
                f"_channel.bdpt_launch_state returned bad {name} shape"
            )
    for name in ("light_seed",):
        validate_cuda_tensor(name, exported[name], dtype=torch.int64, ndim=1)
        if exported[name].shape != (expected,):
            raise ValueError(
                f"_channel.bdpt_launch_state returned bad {name} shape"
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
    name: str, exported: dict[str, torch.Tensor], expected_count: int | None,
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
    name: str, exported: dict[str, torch.Tensor], expected_count: int | None,
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


# The six accumulate outputs, in the order every caller reads them. The set is
# used for the BDPT AD key-set check (the coherent forward may add
# bin-sum buffers as extra keys).
_BDPT_COMPONENT_MATRIX_ORDER = (
    "path_gain",
    "los",
    "reflection",
    "diffraction",
    "transmission",
    "scattering",
)
_BDPT_COMPONENT_MATRIX_FIELDS = frozenset(_BDPT_COMPONENT_MATRIX_ORDER)

# BDPT AD: the coherent forward returns the per-component phasor bin
# sums S_b as non-differentiable outputs; the coherent backward reads them as
# explicit args in this order (real/imag per accumulating component). Absent for
# the power domain.
_BDPT_ACCUMULATE_BIN_SUM_ORDER = (
    "los_re",
    "los_im",
    "reflection_re",
    "reflection_im",
    "diffraction_re",
    "diffraction_im",
    "transmission_re",
    "transmission_im",
    "scattering_re",
    "scattering_im",
)


def _bdpt_accumulate_bin_sum_args(
    combine_domain: str, bin_sums: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor | None, ...]:
    """Expand the coherent forward's phasor bin sums into the ten positional
    ``los_re..scattering_im`` args the native accumulate VJP/JVP consume.

    BDPT AD (retained forward bin sums): the coherent backward/jvp read the
    per-component bin sums ``S_b`` retained by the forward, so no in-backward
    re-reduction and no sample coefficients are needed. The power domain takes
    no bin sums; every slot is ``None``."""

    bins = tuple(bin_sums)
    if combine_domain == "coherent":
        if len(bins) != len(_BDPT_ACCUMULATE_BIN_SUM_ORDER):
            raise ValueError(
                "coherent accumulate backward/jvp requires the ten forward "
                "phasor bin sums"
            )
        return bins
    if bins:
        raise ValueError("power-domain accumulate takes no bin sums")
    return (None,) * len(_BDPT_ACCUMULATE_BIN_SUM_ORDER)


def _bdpt_mis_mode_id(mis: str) -> int:
    if mis == "none":
        return 0
    if mis == "balance":
        return 1
    if mis == "power_heuristic":
        return 2
    raise ValueError("mis is not supported")


def bdpt_endpoint_subpath_state(
    tx_positions: torch.Tensor, tx_power: torch.Tensor, tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor, rx_polarization: torch.Tensor, launch_tx_id: torch.Tensor,
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
    exported = _BDPT_ENDPOINT_SUBPATH_STATE(
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
            "_channel.bdpt_endpoint_subpath_state must return light/sensor dicts"
        )
    light = exported["light"]
    sensor = exported["sensor"]
    _validate_bdpt_subpath_state(
        "_channel.bdpt_endpoint_subpath_state.light",
        light,
        int(launch_tx_id.shape[0]),
    )
    _validate_bdpt_subpath_state(
        "_channel.bdpt_endpoint_subpath_state.sensor",
        sensor,
        int(rx_positions.shape[0]),
    )
    return {"light": light, "sensor": sensor}


def bdpt_endpoint_connection_samples(
    light: dict[str, torch.Tensor], sensor: dict[str, torch.Tensor], *, frequency_hz: float,
    samples_per_tx: int, max_paths: int | None = None, mis: str = "power_heuristic",
    beta: float = 2.0, strategy_count: int = 1,
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
    exported: dict[str, torch.Tensor] = _BDPT_ENDPOINT_CONNECTION_SAMPLES(
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
        "_channel.bdpt_endpoint_connection_samples", exported, expected_count
    )
    return exported


def bdpt_endpoint_connection_visibility_inputs(
    light: dict[str, torch.Tensor], sensor: dict[str, torch.Tensor], *, sample_count: int,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    _validate_bdpt_subpath_state("sensor", sensor, None)
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    expected_total = int(light["origin"].shape[0]) * int(sensor["origin"].shape[0])
    if int(sample_count) > expected_total:
        raise ValueError("sample_count exceeds endpoint pair count")
    exported = _BDPT_ENDPOINT_CONNECTION_VISIBILITY_INPUTS(
        light,
        sensor,
        int(sample_count),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.bdpt_endpoint_connection_visibility_inputs must return a dict"
        )
    if set(exported) != {"start", "end", "active"}:
        raise ValueError(
            "_channel.bdpt_endpoint_connection_visibility_inputs returned unexpected fields"
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
            "_channel.bdpt_endpoint_connection_visibility_inputs returned bad start shape"
        )
    if exported["end"].shape != exported["start"].shape or exported["active"].shape != (
        int(sample_count),
    ):
        raise ValueError(
            "_channel.bdpt_endpoint_connection_visibility_inputs returned bad visibility shape"
        )
    return exported


def _resolve_accumulate_coeffs(
    samples: dict[str, torch.Tensor], combine_domain: str, coeff_real: torch.Tensor | None,
    coeff_imag: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate coherent phasor planes or create empty placeholders for power-domain accumulation."""

    if combine_domain != "coherent":
        empty = torch.empty(
            (0,), device=samples["contribution"].device, dtype=torch.float32
        )
        return empty, empty
    if coeff_real is None or coeff_imag is None:
        raise ValueError("coherent combine requires coeff_real and coeff_imag")
    for name, tensor in (("coeff_real", coeff_real), ("coeff_imag", coeff_imag)):
        # The coefficients arrive as .real/.imag strided views of the
        # natively-computed complex path field; the one-time layout copy
        # happens at the C++ ABI boundary (mc hot-path layout-copy rule),
        # so contiguity is not required here.
        validate_cuda_tensor(
            name, tensor, dtype=torch.float32, ndim=1, require_contiguous=False
        )
        if tensor.shape != samples["contribution"].shape:
            raise ValueError(f"{name} must match connection-sample rows")
        if tensor.get_device() != samples["contribution"].get_device():
            raise ValueError(f"{name} must share the connection-sample device")
    return coeff_real, coeff_imag


def bdpt_accumulate_connection_samples(
    samples: dict[str, torch.Tensor], *, tx_count: int, rx_count: int,
    accumulation_strategy: str = "atomic", combine_domain: str = "power",
    coeff_real: torch.Tensor | None = None, coeff_imag: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Accumulate connection samples into per-component matrices.

    ``combine_domain='power'`` (default) is the incoherent per-path power
    accumulation, bit-identical to the pre-coherent combination behaviour; the coefficient
    tensors are ignored. ``combine_domain='coherent'`` (coherent combination, opt-in) sums
    the complex projected field coefficient (``coeff_real``/``coeff_imag``,
    row-aligned to ``samples``) into per-(tx, rx, component) phasor bins and
    finalizes ``|sum|^2``; the ``accumulation_strategy`` perf axis stays
    orthogonal (the coherent phasor sum always uses the atomic-double
    reduction).
    """

    _validate_bdpt_connection_samples("samples", samples, None)
    if tx_count < 0 or rx_count < 0:
        raise ValueError("tx_count and rx_count must be non-negative")
    strategy_ids = {"atomic": 0, "staged": 1, "compact": 2}
    if accumulation_strategy not in strategy_ids:
        raise ValueError(
            "accumulation_strategy must be 'atomic', 'staged', or 'compact'"
        )
    combine_ids = {"power": 0, "coherent": 1}
    if combine_domain not in combine_ids:
        raise ValueError("combine_domain must be 'power' or 'coherent'")
    coeff_real, coeff_imag = _resolve_accumulate_coeffs(
        samples, combine_domain, coeff_real, coeff_imag
    )
    exported = _BDPT_ACCUMULATE_CONNECTION_SAMPLES(
        samples,
        int(tx_count),
        int(rx_count),
        int(strategy_ids[accumulation_strategy]),
        int(combine_ids[combine_domain]),
        coeff_real,
        coeff_imag,
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.bdpt_accumulate_connection_samples must return a dict"
        )
    # Coherent BDPT accumulation: under combine_domain='coherent' the
    # forward additionally returns its per-component phasor bin-sum buffers
    # (``S_b``) as non-differentiable outputs so the coherent backward can read
    # them without a second atomic-double reduction. The primal component
    # matrices are unchanged bitwise; a subset check accepts the extra keys and
    # keeps the public return the six component matrices only.
    if not _BDPT_COMPONENT_MATRIX_FIELDS.issubset(exported):
        raise ValueError(
            "_channel.bdpt_accumulate_connection_samples returned unexpected fields"
        )
    for name in _BDPT_COMPONENT_MATRIX_ORDER:
        tensor = exported[name]
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=2)
        if tuple(tensor.shape) != (int(tx_count), int(rx_count)):
            raise ValueError(
                f"_channel.bdpt_accumulate_connection_samples returned bad {name} shape"
            )
    return {name: exported[name] for name in _BDPT_COMPONENT_MATRIX_ORDER}


def bdpt_filter_connection_samples(
    samples: dict[str, torch.Tensor], visible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_connection_samples("samples", samples, None)
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if visible.shape != samples["valid"].shape:
        raise ValueError("visible must match samples")
    if visible.get_device() != samples["valid"].get_device():
        raise ValueError("visible must share samples device")
    exported: dict[str, torch.Tensor] = _BDPT_FILTER_CONNECTION_SAMPLES(samples, visible)
    _validate_bdpt_connection_samples(
        "_channel.bdpt_filter_connection_samples", exported, None
    )
    return exported


def bdpt_count_valid_connection_samples(samples: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the number of valid rows as a 0-dim int64 CUDA tensor (no host copy)."""

    _validate_bdpt_connection_samples("samples", samples, None)
    count = _BDPT_COUNT_VALID_CONNECTION_SAMPLES(samples)
    if not isinstance(count, torch.Tensor):
        raise TypeError(
            "_channel.bdpt_count_valid_connection_samples must return a tensor"
        )
    validate_cuda_tensor("count", count, dtype=torch.int64, ndim=0)
    return count


def bdpt_compact_connection_samples(
    samples: dict[str, torch.Tensor], *, max_paths: int | None = None,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_connection_samples("samples", samples, None)
    max_paths_value = -1 if max_paths is None else int(max_paths)
    if max_paths_value < -1:
        raise ValueError("max_paths must be non-negative")
    exported: dict[str, torch.Tensor] = _BDPT_COMPACT_CONNECTION_SAMPLES(
        samples, int(max_paths_value)
    )
    _validate_bdpt_connection_samples(
        "_channel.bdpt_compact_connection_samples", exported, None
    )
    if max_paths is not None and int(exported["valid"].shape[0]) > int(max_paths):
        raise ValueError(
            "_channel.bdpt_compact_connection_samples exceeded max_paths"
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
    exported: dict[str, torch.Tensor] = _BDPT_CONCAT_CONNECTION_SAMPLES(tuple(samples))
    _validate_bdpt_connection_samples(
        "_channel.bdpt_concat_connection_samples", exported, expected_count
    )
    return exported


def bdpt_connection_variance(
    samples: dict[str, torch.Tensor], *, tx_count: int, rx_count: int, samples_per_tx: int,
) -> torch.Tensor:
    _validate_bdpt_connection_samples("samples", samples, None)
    if tx_count < 0 or rx_count < 0:
        raise ValueError("tx_count and rx_count must be non-negative")
    if samples_per_tx <= 0:
        raise ValueError("samples_per_tx must be positive")
    variance = _BDPT_CONNECTION_VARIANCE(
        samples,
        int(tx_count),
        int(rx_count),
        int(samples_per_tx),
    )
    if not isinstance(variance, torch.Tensor):
        raise TypeError("_channel.bdpt_connection_variance must return a tensor")
    validate_cuda_tensor("variance", variance, dtype=torch.float32, ndim=2)
    if tuple(variance.shape) != (int(tx_count), int(rx_count)):
        raise ValueError("_channel.bdpt_connection_variance returned bad shape")
    return variance


_BDPT_WAVEFRONT_MATERIAL_FIELDS = (
    "material_id",
    "valid",
    "gain",
    "eps_r",
    "sigma_e",
    "mu_r",
    "thickness",
    "layer_offset",
    "layer_count",
    "layer_thickness_m",
    "layer_eps_r",
    "layer_sigma_e",
    "layer_mu_r",
)


class _KirchhoffTableValues(Protocol):
    """The Kirchhoff table values and sampling CDFs the BDPT stack packs."""

    @property
    def f_te(self) -> torch.Tensor: ...
    @property
    def f_tm(self) -> torch.Tensor: ...
    @property
    def conditional_cdf(self) -> torch.Tensor: ...
    @property
    def sample_density(self) -> torch.Tensor: ...
    @property
    def marginal_cdf(self) -> torch.Tensor: ...


class _KirchhoffRuntime(Protocol):
    """A rough-material runtime record carrying its Kirchhoff table."""

    @property
    def table(self) -> _KirchhoffTableValues: ...


def bdpt_scatter_table_stack(
    runtimes: Mapping[int, _KirchhoffRuntime], material_bundle: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Stack the rough-material Kirchhoff tables and their sampling CDFs per slot.

    Structural packing only: ``material_slot[m]`` is the slot of material ``m``
    (``-1`` without a table) and every table block is located by its offsets and
    ``[nti, npi, nto, npo]`` dims. ``f_te`` / ``f_tm`` keep their autograd graph.
    """

    device = material_bundle["scatter_model_id"].device
    material_count = int(material_bundle["scatter_model_id"].shape[0])
    slot = torch.full((material_count,), -1, dtype=torch.int32, device=device)
    parts: dict[str, list[torch.Tensor]] = {
        "f_te": [], "f_tm": [], "conditional_cdf": [], "sample_density": [], "marginal_cdf": [],
    }
    offsets: list[int] = []
    marginal_offsets: list[int] = []
    dims: list[list[int]] = []
    running = 0
    marginal_running = 0
    for index, material in enumerate(sorted(runtimes)):
        table = runtimes[material].table
        slot[material] = index
        for name in ("f_te", "f_tm", "conditional_cdf", "sample_density", "marginal_cdf"):
            value = getattr(table, name)
            # Table values keep their graph (their derivative is a wavefront
            # leaf); the sampling CDFs are frozen.
            if name not in ("f_te", "f_tm"):
                value = value.detach()
            parts[name].append(value.to(device=device, dtype=torch.float32).reshape(-1))
        offsets.append(running)
        marginal_offsets.append(marginal_running)
        dims.append([int(size) for size in table.f_te.shape])
        running += int(table.f_te.numel())
        marginal_running += int(table.marginal_cdf.numel())
    stack = {name: torch.cat(values).contiguous() for name, values in parts.items()}
    stack["material_slot"] = slot
    stack["table_offset"] = torch.tensor(offsets, dtype=torch.int64, device=device)
    stack["marginal_offset"] = torch.tensor(marginal_offsets, dtype=torch.int64, device=device)
    stack["table_dims"] = torch.tensor(dims, dtype=torch.int32, device=device).reshape(-1, 4)
    for name in ("scatter_model_id", "rough_sigma_h_m", "rough_axis_rad"):
        stack[name] = material_bundle[name].contiguous()
    return stack


def _bdpt_wavefront_arguments(
    rayd: object, tx_positions: torch.Tensor, tx_power: torch.Tensor, tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor, rx_polarization: torch.Tensor,
    material_bundle: dict[str, torch.Tensor], first: torch.Tensor, second: torch.Tensor, *,
    scatter: dict[str, torch.Tensor] | None, frequency_hz: float, samples: int, max_depth: int,
    seed: int, mis_mode: int, beta: float, emit_mixed: bool, max_scattering_order: int,
    scene_diagonal: float, chunk_samples: int, pair_tile: int,
) -> list[object]:
    """Validated positional arguments shared by the three wavefront passes."""

    for name, tensor in (
        ("tx_positions", tx_positions),
        ("tx_polarization", tx_polarization),
        ("rx_positions", rx_positions),
        ("rx_polarization", rx_polarization),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("transmission matrix", first, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("scattering matrix", second, dtype=torch.float32, ndim=2)
    if samples <= 0 or max_depth < 1 or seed < 0 or chunk_samples <= 0 or pair_tile <= 0:
        raise ValueError("samples, max_depth, chunk_samples and pair_tile must be positive")
    materials = {
        name: material_bundle[name].detach() for name in _BDPT_WAVEFRONT_MATERIAL_FIELDS
    }
    if scatter is not None:
        scatter = {name: value.detach() for name, value in scatter.items()}
    return [
        _rayd_scene_resource(rayd),
        tx_positions,
        tx_power.detach().contiguous(),
        tx_polarization.contiguous(),
        rx_positions.contiguous(),
        rx_polarization.contiguous(),
        materials,
        scatter,
        first.detach().contiguous(),
        second.detach().contiguous(),
        float(frequency_hz),
        int(samples),
        int(max_depth),
        int(seed),
        int(mis_mode),
        float(beta),
        bool(emit_mixed),
        int(max_scattering_order),
        float(scene_diagonal),
        int(chunk_samples),
        int(pair_tile),
    ]


def bdpt_wavefront_transmission(
    rayd: object, tx_positions: torch.Tensor, tx_power: torch.Tensor, tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor, rx_polarization: torch.Tensor,
    material_bundle: dict[str, torch.Tensor], base_transmission: torch.Tensor,
    base_scattering: torch.Tensor, *, scatter: dict[str, torch.Tensor] | None, frequency_hz: float,
    samples: int, max_depth: int, seed: int, mis_mode: int, beta: float, emit_mixed: bool,
    max_scattering_order: int, scene_diagonal: float, chunk_samples: int = 1 << 18,
    pair_tile: int = 1 << 20, rows_capacity: int | None = None,
) -> dict[str, object]:
    """Run the fused light-subpath wavefront of every transmitter.

    Returns ``transmission`` and ``scattering`` (the base matrices plus the sampled
    mixed-chain and scatter next-event estimates, float ``[T, R]``) and ``counts``
    (int64 ``[6]``: transmit, reflect and scatter events, mixed rows, scatter
    rows, written rows), all still on the device. With ``rows_capacity`` it also
    returns ``rows``: every accumulated connection row in the 12-field
    connection-sample schema, sorted by (sample, bounce, receiver).
    """

    arguments = _bdpt_wavefront_arguments(
        rayd, tx_positions, tx_power, tx_polarization, rx_positions, rx_polarization,
        material_bundle, base_transmission, base_scattering, scatter=scatter,
        frequency_hz=frequency_hz, samples=samples, max_depth=max_depth, seed=seed,
        mis_mode=mis_mode, beta=beta, emit_mixed=emit_mixed,
        max_scattering_order=max_scattering_order, scene_diagonal=scene_diagonal,
        chunk_samples=chunk_samples, pair_tile=pair_tile,
    )
    out = _BDPT_WAVEFRONT_TRANSMISSION(
        *arguments, -1 if rows_capacity is None else int(rows_capacity)
    )
    expected = {"transmission", "scattering", "counts"} | (
        set() if rows_capacity is None else {"rows"}
    )
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError("_channel.bdpt_wavefront_transmission returned unexpected fields")
    if rows_capacity is not None:
        _validate_bdpt_connection_samples(
            "_channel.bdpt_wavefront_transmission rows", out["rows"], None
        )
    validate_cuda_tensor("transmission", out["transmission"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("scattering", out["scattering"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("counts", out["counts"], dtype=torch.int64, ndim=1)
    return out


BDPT_WAVEFRONT_GRADIENTS = (
    "face_eps_r",
    "face_sigma_e",
    "face_gain",
    "face_thickness",
    "layer_thickness_m",
    "layer_eps_r",
    "layer_sigma_e",
    "f_te",
    "f_tm",
    "tx_power",
    "frequency",
)


def bdpt_wavefront_jvp(
    rayd: object, tx_positions: torch.Tensor, tx_power: torch.Tensor, tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor, rx_polarization: torch.Tensor,
    material_bundle: dict[str, torch.Tensor], base_transmission: torch.Tensor,
    base_scattering: torch.Tensor, *, tangents: Mapping[str, Any],
    scatter: dict[str, torch.Tensor] | None, frequency_hz: float, samples: int, max_depth: int,
    seed: int, mis_mode: int, beta: float, emit_mixed: bool, max_scattering_order: int,
    scene_diagonal: float, chunk_samples: int = 1 << 18, pair_tile: int = 1 << 20,
) -> dict[str, torch.Tensor]:
    """Run the wavefront primal and its forward-mode tangent in one native pass.

    ``tangents`` holds float tangents of the leaves in ``BDPT_WAVEFRONT_GRADIENTS``
    (missing or ``None`` is zero; ``frequency`` is a float) and the
    ``base_transmission`` / ``base_scattering`` matrix tangents.
    """

    arguments = _bdpt_wavefront_arguments(
        rayd, tx_positions, tx_power, tx_polarization, rx_positions, rx_polarization,
        material_bundle, base_transmission, base_scattering, scatter=scatter,
        frequency_hz=frequency_hz, samples=samples, max_depth=max_depth, seed=seed,
        mis_mode=mis_mode, beta=beta, emit_mixed=emit_mixed,
        max_scattering_order=max_scattering_order, scene_diagonal=scene_diagonal,
        chunk_samples=chunk_samples, pair_tile=pair_tile,
    )
    native_tangents: dict[str, object] = {"frequency": float(tangents.get("frequency", 0.0))}
    for name, value in tangents.items():
        if name == "frequency" or value is None:
            continue
        if name not in BDPT_WAVEFRONT_GRADIENTS and name not in ("base_transmission", "base_scattering"):
            raise ValueError(f"unknown BDPT wavefront tangent {name!r}")
        tensor: torch.Tensor = value
        validate_cuda_tensor(f"tangent {name}", tensor, dtype=torch.float32, ndim=tensor.dim())
        native_tangents[name] = tensor.detach().contiguous()
    out = _BDPT_WAVEFRONT_JVP(*arguments, native_tangents)
    expected = {"transmission", "scattering", "tangent_transmission", "tangent_scattering", "counts"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError("_channel.bdpt_wavefront_jvp returned unexpected fields")
    return out


def bdpt_wavefront_backward(
    rayd: object, tx_positions: torch.Tensor, tx_power: torch.Tensor, tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor, rx_polarization: torch.Tensor,
    material_bundle: dict[str, torch.Tensor], grad_transmission: torch.Tensor,
    grad_scattering: torch.Tensor, *, needs: tuple[str, ...],
    scatter: dict[str, torch.Tensor] | None, frequency_hz: float, samples: int, max_depth: int,
    seed: int, mis_mode: int, beta: float, emit_mixed: bool, max_scattering_order: int,
    scene_diagonal: float, chunk_samples: int = 1 << 16, pair_tile: int = 1 << 20,
) -> dict[str, torch.Tensor]:
    """Replay the wavefront and return the float gradients of the leaves in ``needs``."""

    unknown = set(needs) - set(BDPT_WAVEFRONT_GRADIENTS)
    if unknown:
        raise ValueError(f"unknown BDPT wavefront gradients {sorted(unknown)}")
    arguments = _bdpt_wavefront_arguments(
        rayd, tx_positions, tx_power, tx_polarization, rx_positions, rx_polarization,
        material_bundle, grad_transmission, grad_scattering, scatter=scatter,
        frequency_hz=frequency_hz, samples=samples, max_depth=max_depth, seed=seed,
        mis_mode=mis_mode, beta=beta, emit_mixed=emit_mixed,
        max_scattering_order=max_scattering_order, scene_diagonal=scene_diagonal,
        chunk_samples=chunk_samples, pair_tile=pair_tile,
    )
    out = _BDPT_WAVEFRONT_BACKWARD(*arguments, list(needs))
    if not isinstance(out, dict) or set(out) != set(needs):
        raise TypeError("_channel.bdpt_wavefront_backward returned unexpected fields")
    return out
