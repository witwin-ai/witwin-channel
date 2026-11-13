# Copyright Xingyu Chen.
# Native geometry kernel facades.

"""Native geometry kernel facades."""

from __future__ import annotations

import math
from typing import Any, TypedDict, cast

import torch

from witwin.channel.propagation.penetration import (
    SegmentPenetrationBackwardResult,
    SegmentPenetrationJvpResult,
    SegmentPenetrationPolicy,
    SegmentPenetrationResult,
    SegmentPenetrationTapeResult,
)
from witwin.channel.kernels.spec import AdInputs, NativeOps, opaque, tensor as tensor_arg
from witwin.channel.runtime import (
    CapacityFailureBit,
    CapacityFailureState,
    _ad_active_ctx,
    _ad_check_optional_grad,
    _ad_check_rows,
    _ad_check_tangent_vec3,
    _ad_checked_tangent,
    _ad_first_order_only,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _rayd_scene_resource,
    disable_functorch,
    require_capacity_failure_state,
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)

__all__ = [
    "_BDPT_INTERSECTION_FIELDS",
    "_RaydFaceNormalsAdFunction",
    "_RaydReflectionEpcPathsAdFunction",
    "_RaydSegmentPenetrationAdFunction",
    "_epc_paths_frozen_winner_checks",
    "coupled_dd_geometry_forward",
    "coupled_chain_geometry_forward",
    "deterministic_face_groups",
    "deterministic_normalize_vec3",
    "diffraction_tx_visible_state_plan",
    "mc_diffraction_edge_geometry",
    "mc_surface_group_edge_candidates",
    "rayd_diffraction_paths_order1_forward",
    "rayd_diffraction_sample_tape_forward",
    "rayd_face_normals_ad",
    "rayd_reflection_epc_paths_ad",
    "rayd_reflection_epc_paths_backward",
    "rayd_reflection_epc_paths_forward",
    "rayd_reflection_epc_paths_jvp",
    "rayd_scene_face_normals_backward",
    "rayd_scene_face_normals_jvp",
    "rayd_segment_penetration_ad",
    "rayd_segment_penetration_backward",
    "rayd_segment_penetration_forward",
    "rayd_segment_penetration_forward_tape",
    "rayd_segment_penetration_jvp",
    "rayd_trace_reflections_forward",
    "rayd_visibility_forward",
]

_native = NativeOps(lambda symbol: _required_native_op(symbol))
_COUPLED_CHAIN_GEOMETRY_FORWARD = _native.op(
    "coupled_chain_geometry_forward",
    opaque("scene"), tensor_arg("source"), tensor_arg("receiver"), tensor_arg("edge_point"),
    tensor_arg("edge_id"), tensor_arg("prefix_faces"), tensor_arg("prefix_plane_points"),
    tensor_arg("prefix_plane_normals"), tensor_arg("suffix_faces"),
    tensor_arg("suffix_plane_points"), tensor_arg("suffix_plane_normals"),
    tensor_arg("surface_group_id"), tensor_arg("surface_group_size"),
    tensor_arg("surface_group_members"), opaque("min_segment_length"),
)
_COUPLED_DD_GEOMETRY_FORWARD = _native.op(
    "coupled_dd_geometry_forward",
    opaque("scene"), tensor_arg("source"), tensor_arg("receiver"), tensor_arg("edge1_id"),
    tensor_arg("edge1_pos"), tensor_arg("edge1_dir"), tensor_arg("edge1_t_min"),
    tensor_arg("edge1_t_max"), tensor_arg("edge2_id"), tensor_arg("edge2_pos"),
    tensor_arg("edge2_dir"), tensor_arg("edge2_t_min"), tensor_arg("edge2_t_max"),
)
_DETERMINISTIC_FACE_GROUPS = _native.op(
    "deterministic_face_groups",
    tensor_arg("tri_a"), tensor_arg("normals"), tensor_arg("surface_ids"), opaque("quantization"),
)
_DETERMINISTIC_NORMALIZE_VEC3 = _native.op(
    "deterministic_normalize_vec3",
    tensor_arg("values"), opaque("eps"),
)
_DIFFRACTION_TX_VISIBLE_STATE_PLAN = _native.op(
    "diffraction_tx_visible_state_plan",
    opaque("scene"), tensor_arg("tx"), tensor_arg("edge_position"), tensor_arg("edge_direction"),
    tensor_arg("edge_t_min"), tensor_arg("edge_t_max"), opaque("state_source"),
)
_MC_DIFFRACTION_EDGE_GEOMETRY = _native.op(
    "mc_diffraction_edge_geometry",
    tensor_arg("vertices"), tensor_arg("faces"), tensor_arg("face_normals"), tensor_arg("edge_v0"),
    tensor_arg("edge_v1"), tensor_arg("face0"), tensor_arg("face1"), opaque("plane_tol"),
)
_MC_SURFACE_GROUP_EDGE_CANDIDATES = _native.op(
    "mc_surface_group_edge_candidates",
    tensor_arg("vertices"), tensor_arg("faces"), tensor_arg("face_normals"), tensor_arg("edge_v0"),
    tensor_arg("edge_v1"), tensor_arg("face0"), tensor_arg("face1"), tensor_arg("selected"),
    opaque("plane_tol"),
)
_RAYD_DIFFRACTION_PATHS_ORDER1_FORWARD = _native.op(
    "rayd_diffraction_paths_order1_forward",
    opaque("scene"), tensor_arg("tx_pos"), tensor_arg("tx_pol"), tensor_arg("rx_pos"),
    tensor_arg("active"), tensor_arg("state_edge_index"), tensor_arg("state_edge_pos"),
    tensor_arg("state_edge_dir"), tensor_arg("state_edge_t_min"), tensor_arg("state_edge_t_max"),
    tensor_arg("state_n0"), tensor_arg("state_n1"), tensor_arg("state_prim0"),
    tensor_arg("state_prim1"), tensor_arg("state_exterior_angle"), tensor_arg("state_src"),
    tensor_arg("state_src_power"), tensor_arg("material_eta_r"), tensor_arg("material_sigma"),
    tensor_arg("material_mu_r"), tensor_arg("material_gain"), tensor_arg("material_valid"),
    opaque("state_limit"), opaque("capacity"), opaque("wavelength"),
    opaque("isb_taper_width_scale"), opaque("state_source_index"), opaque("face_layers"),
)
_RAYD_DIFFRACTION_SAMPLE_TAPE_FORWARD = _native.op(
    "rayd_diffraction_sample_tape_forward",
    opaque("scene"), opaque("active"), tensor_arg("state_edge_index"), tensor_arg("state_edge_pos"),
    tensor_arg("state_edge_dir"), tensor_arg("state_edge_t_min"), tensor_arg("state_edge_t_max"),
    tensor_arg("state_n0"), tensor_arg("state_n1"), tensor_arg("state_prim0"),
    tensor_arg("state_prim1"), tensor_arg("state_exterior_angle"), tensor_arg("state_src"),
    tensor_arg("state_src_power"), opaque("state_wi"), opaque("state_d0"),
    tensor_arg("material_eta_r"), tensor_arg("material_sigma"), tensor_arg("material_mu_r"),
    tensor_arg("material_gain"), tensor_arg("material_valid"), opaque("state_limit"),
    opaque("grid_axis"), opaque("grid_position"), opaque("grid_coord0_min"),
    opaque("grid_coord0_max"), opaque("grid_coord1_min"), opaque("grid_coord1_max"),
    opaque("grid_resolution0"), opaque("grid_resolution1"), opaque("grid_cell_area"),
    opaque("wavelength"), opaque("direct_samples"), opaque("keller_samples"),
    opaque("suffix_samples"), opaque("seed"), opaque("max_order"), opaque("recursive_state_limit"),
    opaque("recursive_active"), opaque("recursive_state_edge_index"),
    opaque("recursive_state_edge_pos"), opaque("recursive_state_edge_dir"),
    opaque("recursive_state_edge_t_min"), opaque("recursive_state_edge_t_max"),
    opaque("recursive_state_n0"), opaque("recursive_state_n1"), opaque("recursive_state_prim0"),
    opaque("recursive_state_prim1"), opaque("recursive_state_exterior_angle"),
    opaque("export_tape"), opaque("sample_state_index"), opaque("sample_edge_weight"),
)
_RAYD_REFLECTION_EPC_PATHS_BACKWARD = _native.op(
    "rayd_reflection_epc_paths_backward",
    opaque("scene"), tensor_arg("source"), tensor_arg("receiver"), tensor_arg("sequence"),
    tensor_arg("plane_points"), tensor_arg("plane_normals"), tensor_arg("valid"),
    tensor_arg("bounce_count"), opaque("grad_points"), opaque("grad_normals"),
    opaque("grad_path_length"), opaque("need_grad_vertices"), opaque("need_grad_source"),
    opaque("need_grad_receiver"),
)
_RAYD_REFLECTION_EPC_PATHS_FORWARD = _native.op(
    "rayd_reflection_epc_paths_forward",
    opaque("scene"), tensor_arg("source"), tensor_arg("receiver"), opaque("active"),
    tensor_arg("expected_prim_ids"), tensor_arg("direct_plane_points"),
    tensor_arg("direct_plane_normals"), tensor_arg("surface_group_id"),
    tensor_arg("surface_group_size"), tensor_arg("surface_group_members"), opaque("max_bounces"),
    opaque("visibility_ignore_mode"), opaque("min_segment_length"), opaque("receiver_is_interaction"),
)
_RAYD_REFLECTION_EPC_PATHS_JVP = _native.op(
    "rayd_reflection_epc_paths_jvp",
    opaque("scene"), tensor_arg("source"), tensor_arg("receiver"), tensor_arg("sequence"),
    tensor_arg("plane_points"), tensor_arg("plane_normals"), tensor_arg("valid"),
    tensor_arg("bounce_count"), opaque("tangent_vertices"), opaque("tangent_source"),
    opaque("tangent_receiver"),
)
_RAYD_SCENE_FACE_NORMALS_BACKWARD = _native.op(
    "rayd_scene_face_normals_backward",
    opaque("scene"), tensor_arg("grad_face_normals"),
)
_RAYD_SCENE_FACE_NORMALS_JVP = _native.op(
    "rayd_scene_face_normals_jvp",
    opaque("scene"), tensor_arg("tangent_vertices"),
)
_RAYD_SEGMENT_PENETRATION_BACKWARD = _native.op(
    "rayd_segment_penetration_backward",
    opaque("scene"), tensor_arg("origins"), tensor_arg("targets"), opaque("input_active"),
    opaque("input_active_any"), opaque("hit_capacity"), opaque("policy"), opaque("scene_diagonal"),
    tensor_arg("capacity_failure_record"), opaque("failure_bit"), tensor_arg("valid"),
    tensor_arg("num_hits"), tensor_arg("reached_target"), tensor_arg("overflow"),
    tensor_arg("distance"), tensor_arg("direction"), tensor_arg("hit_t"), tensor_arg("position"),
    tensor_arg("normal"), tensor_arg("geometric_normal"), tensor_arg("global_primitive_id"),
    tensor_arg("tape_primitive_id"), tensor_arg("tape_barycentric"),
    tensor_arg("tape_restart_epsilon"), tensor_arg("tape_restart_branch"),
    tensor_arg("tape_restart_tie_mask"), tensor_arg("tape_direction_denominator_branch"),
    opaque("grad_distance"), opaque("grad_direction"), opaque("grad_t"), opaque("grad_position"),
    opaque("grad_normal"), opaque("grad_geometric_normal"), opaque("need_grad_vertices"),
    opaque("need_grad_origins"), opaque("need_grad_targets"),
)
_RAYD_SEGMENT_PENETRATION_FORWARD = _native.op(
    "rayd_segment_penetration_forward",
    opaque("scene"), tensor_arg("origins"), tensor_arg("targets"), opaque("input_active"),
    opaque("input_active_any"), opaque("hit_capacity"), opaque("policy"), opaque("scene_diagonal"),
    tensor_arg("capacity_failure_record"), opaque("failure_bit"),
)
_RAYD_SEGMENT_PENETRATION_FORWARD_TAPE = _native.op(
    "rayd_segment_penetration_forward_tape",
    opaque("scene"), tensor_arg("origins"), tensor_arg("targets"), opaque("input_active"),
    opaque("input_active_any"), opaque("hit_capacity"), opaque("policy"), opaque("scene_diagonal"),
    tensor_arg("capacity_failure_record"), opaque("failure_bit"),
)
_RAYD_SEGMENT_PENETRATION_JVP = _native.op(
    "rayd_segment_penetration_jvp",
    opaque("scene"), tensor_arg("origins"), tensor_arg("targets"), opaque("input_active"),
    opaque("input_active_any"), opaque("hit_capacity"), opaque("policy"), opaque("scene_diagonal"),
    tensor_arg("capacity_failure_record"), opaque("failure_bit"), tensor_arg("valid"),
    tensor_arg("num_hits"), tensor_arg("reached_target"), tensor_arg("overflow"),
    tensor_arg("distance"), tensor_arg("direction"), tensor_arg("hit_t"), tensor_arg("position"),
    tensor_arg("normal"), tensor_arg("geometric_normal"), tensor_arg("global_primitive_id"),
    tensor_arg("tape_primitive_id"), tensor_arg("tape_barycentric"),
    tensor_arg("tape_restart_epsilon"), tensor_arg("tape_restart_branch"),
    tensor_arg("tape_restart_tie_mask"), tensor_arg("tape_direction_denominator_branch"),
    opaque("tangent_vertices"), opaque("tangent_origins"), opaque("tangent_targets"),
)
_RAYD_TRACE_REFLECTIONS_FORWARD = _native.op(
    "rayd_trace_reflections_forward",
    opaque("scene"), tensor_arg("ray_o"), tensor_arg("ray_d"), tensor_arg("ray_tmax"),
    opaque("active"), opaque("max_bounces"),
)
_RAYD_VISIBILITY_FORWARD = _native.op(
    "rayd_visibility_forward",
    opaque("scene"), tensor_arg("start"), tensor_arg("end"), opaque("active"),
)


# ---------------------------------------------------------------------------
# bridge
# ---------------------------------------------------------------------------
_SEGMENT_PENETRATION_FAILURE_BIT = int(CapacityFailureBit.SEGMENT_PENETRATION_FAILURE)
_SEGMENT_PENETRATION_RESULT_FIELDS = (
    "valid",
    "num_hits",
    "reached_target",
    "overflow",
    "distance",
    "direction",
    "t",
    "position",
    "normal",
    "geometric_normal",
    "global_primitive_id",
)
_SEGMENT_PENETRATION_TAPE_FIELDS = (
    "tape_primitive_id",
    "tape_barycentric",
    "tape_restart_epsilon",
    "tape_restart_branch",
    "tape_restart_tie_mask",
    "tape_direction_denominator_branch",
)


def _validate_segment_penetration_inputs(
    origins: torch.Tensor, targets: torch.Tensor, input_active: torch.Tensor | None, *,
    input_active_any: bool,
) -> None:
    validate_cuda_tensor(
        "origins", origins, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "targets", targets, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if targets.shape != origins.shape:
        raise ValueError("targets must match origins")
    if targets.device != origins.device:
        raise ValueError("targets must share the origins device")
    if input_active is not None:
        validate_cuda_tensor("input_active", input_active, dtype=torch.bool, ndim=1)
        if input_active.shape != (origins.shape[0],):
            raise ValueError("input_active must match the segment count")
        if input_active.device != origins.device:
            raise ValueError("input_active must share the origins device")
    if type(input_active_any) is not bool:
        raise TypeError("input_active_any must be a bool")
    if origins.shape[0] > 0 and not input_active_any and input_active is None:
        raise ValueError(
            "input_active_any=false requires an explicit device input_active mask"
        )


def _validate_segment_penetration_host_config(
    hit_capacity: int, policy: SegmentPenetrationPolicy, scene_diagonal: float,
) -> float:
    if type(hit_capacity) is not int:
        raise TypeError("hit_capacity must be an int")
    if hit_capacity < 0:
        raise ValueError("hit_capacity must be non-negative")
    if not isinstance(policy, SegmentPenetrationPolicy):
        raise TypeError("policy must be a SegmentPenetrationPolicy")
    if isinstance(scene_diagonal, bool) or not isinstance(scene_diagonal, (int, float)):
        raise TypeError("scene_diagonal must be a host number")
    scene_diagonal_value = float(scene_diagonal)
    if not math.isfinite(scene_diagonal_value) or scene_diagonal_value < 0.0:
        raise ValueError("scene_diagonal must be finite and non-negative")
    return scene_diagonal_value


def _segment_penetration_request_args(
    scene_resource: object, origins: torch.Tensor, targets: torch.Tensor,
    input_active: torch.Tensor | None, *, input_active_any: bool, hit_capacity: int,
    policy: SegmentPenetrationPolicy, scene_diagonal: float, failure_state: CapacityFailureState,
) -> tuple[object, ...]:
    _validate_segment_penetration_inputs(
        origins, targets, input_active, input_active_any=input_active_any
    )
    scene_diagonal_value = _validate_segment_penetration_host_config(
        hit_capacity, policy, scene_diagonal
    )
    require_capacity_failure_state(failure_state, device=origins.device)
    return (
        _rayd_scene_resource(scene_resource),
        origins,
        targets,
        input_active,
        input_active_any,
        hit_capacity,
        int(policy),
        scene_diagonal_value,
        failure_state.record,
        _SEGMENT_PENETRATION_FAILURE_BIT,
    )


def _segment_penetration_result(
    values: object, *, hit_capacity: int, failure_state: CapacityFailureState,
) -> SegmentPenetrationResult:
    if not isinstance(values, (tuple, list)) or len(values) != len(
        _SEGMENT_PENETRATION_RESULT_FIELDS
    ):
        raise TypeError(
            "_channel.rayd_segment_penetration_forward must return 11 tensors"
        )
    return SegmentPenetrationResult(
        hit_capacity=hit_capacity,
        failure_state=failure_state,
        **dict(zip(_SEGMENT_PENETRATION_RESULT_FIELDS, values, strict=True)),
    )


def _segment_penetration_tape_args(tape: SegmentPenetrationTapeResult) -> tuple[torch.Tensor, ...]:
    result = tape.result
    return (
        *(getattr(result, name) for name in _SEGMENT_PENETRATION_RESULT_FIELDS),
        *(getattr(tape, name) for name in _SEGMENT_PENETRATION_TAPE_FIELDS),
    )


def rayd_segment_penetration_forward(
    scene_resource: object, origins: torch.Tensor, targets: torch.Tensor,
    input_active: torch.Tensor | None, *, input_active_any: bool, hit_capacity: int,
    policy: SegmentPenetrationPolicy, scene_diagonal: float, failure_state: CapacityFailureState,
) -> SegmentPenetrationResult:
    """Dispatch the live RayD segment-penetration primal."""

    args = _segment_penetration_request_args(
        scene_resource,
        origins,
        targets,
        input_active,
        input_active_any=input_active_any,
        hit_capacity=hit_capacity,
        policy=policy,
        scene_diagonal=scene_diagonal,
        failure_state=failure_state,
    )
    values = _RAYD_SEGMENT_PENETRATION_FORWARD(*args)
    return _segment_penetration_result(
        values, hit_capacity=hit_capacity, failure_state=failure_state
    )


def rayd_segment_penetration_forward_tape(
    scene_resource: object, origins: torch.Tensor, targets: torch.Tensor,
    input_active: torch.Tensor | None, *, input_active_any: bool, hit_capacity: int,
    policy: SegmentPenetrationPolicy, scene_diagonal: float, failure_state: CapacityFailureState,
) -> SegmentPenetrationTapeResult:
    """Dispatch the live RayD primal and opaque fixed-winner tape."""

    args = _segment_penetration_request_args(
        scene_resource,
        origins,
        targets,
        input_active,
        input_active_any=input_active_any,
        hit_capacity=hit_capacity,
        policy=policy,
        scene_diagonal=scene_diagonal,
        failure_state=failure_state,
    )
    values = _RAYD_SEGMENT_PENETRATION_FORWARD_TAPE(*args)
    expected = len(_SEGMENT_PENETRATION_RESULT_FIELDS) + len(
        _SEGMENT_PENETRATION_TAPE_FIELDS
    )
    if not isinstance(values, (tuple, list)) or len(values) != expected:
        raise TypeError(
            "_channel.rayd_segment_penetration_forward_tape must return 17 tensors"
        )
    result = _segment_penetration_result(
        values[:11], hit_capacity=hit_capacity, failure_state=failure_state
    )
    return SegmentPenetrationTapeResult(
        result=result,
        **dict(zip(_SEGMENT_PENETRATION_TAPE_FIELDS, values[11:], strict=True)),
    )


def rayd_segment_penetration_backward(
    scene_resource: object, origins: torch.Tensor, targets: torch.Tensor,
    input_active: torch.Tensor | None, *, input_active_any: bool, hit_capacity: int,
    policy: SegmentPenetrationPolicy, scene_diagonal: float, failure_state: CapacityFailureState,
    tape: SegmentPenetrationTapeResult, grad_distance: torch.Tensor | None = None,
    grad_direction: torch.Tensor | None = None, grad_t: torch.Tensor | None = None,
    grad_position: torch.Tensor | None = None, grad_normal: torch.Tensor | None = None,
    grad_geometric_normal: torch.Tensor | None = None, need_grad_vertices: bool = False,
    need_grad_origins: bool = False, need_grad_targets: bool = False,
) -> SegmentPenetrationBackwardResult:
    """Dispatch the native fixed-winner VJP companion."""

    args = _segment_penetration_request_args(
        scene_resource,
        origins,
        targets,
        input_active,
        input_active_any=input_active_any,
        hit_capacity=hit_capacity,
        policy=policy,
        scene_diagonal=scene_diagonal,
        failure_state=failure_state,
    )
    if not isinstance(tape, SegmentPenetrationTapeResult):
        raise TypeError("tape must be a SegmentPenetrationTapeResult")
    if tape.failure_state is not failure_state:
        raise ValueError("tape must retain the exact request failure_state")
    if (
        tape.result.segment_count != origins.shape[0]
        or tape.result.hit_capacity != hit_capacity
    ):
        raise ValueError("tape must match the segment request shape")
    rows = int(origins.shape[0])
    _ad_check_optional_grad("grad_distance", grad_distance, ((rows,),))
    _ad_check_optional_grad("grad_direction", grad_direction, ((rows, 3),))
    _ad_check_optional_grad("grad_t", grad_t, ((rows, hit_capacity),))
    for name, value in (
        ("grad_position", grad_position),
        ("grad_normal", grad_normal),
        ("grad_geometric_normal", grad_geometric_normal),
    ):
        _ad_check_optional_grad(name, value, ((rows, hit_capacity, 3),))
    for name, value in (
        ("need_grad_vertices", need_grad_vertices),
        ("need_grad_origins", need_grad_origins),
        ("need_grad_targets", need_grad_targets),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be a bool")
    values = _RAYD_SEGMENT_PENETRATION_BACKWARD(
        *args,
        *_segment_penetration_tape_args(tape),
        grad_distance,
        grad_direction,
        grad_t,
        grad_position,
        grad_normal,
        grad_geometric_normal,
        need_grad_vertices,
        need_grad_origins,
        need_grad_targets,
    )
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        raise TypeError(
            "_channel.rayd_segment_penetration_backward must return 3 gradients"
        )
    return SegmentPenetrationBackwardResult(*values)


def rayd_segment_penetration_jvp(
    scene_resource: object, origins: torch.Tensor, targets: torch.Tensor,
    input_active: torch.Tensor | None, *, input_active_any: bool, hit_capacity: int,
    policy: SegmentPenetrationPolicy, scene_diagonal: float, failure_state: CapacityFailureState,
    tape: SegmentPenetrationTapeResult, tangent_vertices: torch.Tensor | None = None,
    tangent_origins: torch.Tensor | None = None, tangent_targets: torch.Tensor | None = None,
) -> SegmentPenetrationJvpResult:
    """Dispatch the native fixed-winner JVP companion."""

    args = _segment_penetration_request_args(
        scene_resource,
        origins,
        targets,
        input_active,
        input_active_any=input_active_any,
        hit_capacity=hit_capacity,
        policy=policy,
        scene_diagonal=scene_diagonal,
        failure_state=failure_state,
    )
    if not isinstance(tape, SegmentPenetrationTapeResult):
        raise TypeError("tape must be a SegmentPenetrationTapeResult")
    if tape.failure_state is not failure_state:
        raise ValueError("tape must retain the exact request failure_state")
    if (
        tape.result.segment_count != origins.shape[0]
        or tape.result.hit_capacity != hit_capacity
    ):
        raise ValueError("tape must match the segment request shape")
    rows = int(origins.shape[0])
    _ad_check_tangent_vec3("tangent_vertices", tangent_vertices, None)
    _ad_check_tangent_vec3("tangent_origins", tangent_origins, rows)
    _ad_check_tangent_vec3("tangent_targets", tangent_targets, rows)
    values = _RAYD_SEGMENT_PENETRATION_JVP(
        *args,
        *_segment_penetration_tape_args(tape),
        tangent_vertices,
        tangent_origins,
        tangent_targets,
    )
    if not isinstance(values, (tuple, list)) or len(values) != 6:
        raise TypeError(
            "_channel.rayd_segment_penetration_jvp must return 6 tangents"
        )
    return SegmentPenetrationJvpResult(*values)


def rayd_visibility_forward(
    scene_resource: object, start: torch.Tensor, end: torch.Tensor, active: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor(
        "start", start, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("end", end, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if active is not None:
        validate_cuda_tensor("active", active, dtype=torch.bool, ndim=1)
    out = _RAYD_VISIBILITY_FORWARD(
        _rayd_scene_resource(scene_resource),
        start,
        end,
        active,
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel.rayd_visibility_forward must return a tensor sequence"
        )
    return tuple(out)


_BDPT_INTERSECTION_FIELDS = (
    "t",
    "p",
    "n",
    "geo_n",
    "uv",
    "barycentric",
    "shape_id",
    "prim_id",
    "local_prim_id",
    "global_prim_id",
)


def rayd_diffraction_sample_tape_forward(*args: object) -> tuple[torch.Tensor, ...]:
    if not args:
        raise TypeError(
            "rayd_diffraction_sample_tape_forward requires a typed RayD scene resource"
        )
    native_args = (_rayd_scene_resource(args[0]), *args[1:])
    out = _RAYD_DIFFRACTION_SAMPLE_TAPE_FORWARD(
        *native_args,
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel.rayd_diffraction_sample_tape_forward must return a tensor sequence"
        )
    return tuple(out)


def rayd_trace_reflections_forward(*args: object) -> tuple[torch.Tensor, ...]:
    if not args:
        raise TypeError(
            "rayd_trace_reflections_forward requires a typed RayD scene resource"
        )
    native_args = (_rayd_scene_resource(args[0]), *args[1:])
    out = _RAYD_TRACE_REFLECTIONS_FORWARD(
        *native_args,
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel.rayd_trace_reflections_forward must return a tensor sequence"
        )
    return tuple(out)


def rayd_reflection_epc_paths_forward(*args: object) -> tuple[torch.Tensor, ...]:
    if not args:
        raise TypeError(
            "rayd_reflection_epc_paths_forward requires a typed RayD scene resource"
        )
    native_args = (_rayd_scene_resource(args[0]), *args[1:])
    out = _RAYD_REFLECTION_EPC_PATHS_FORWARD(
        *native_args,
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel.rayd_reflection_epc_paths_forward must return a tensor sequence"
        )
    return tuple(out)


def coupled_chain_geometry_forward(*args: object) -> dict[str, torch.Tensor]:
    """Validate interaction-chain candidates without evaluating a coefficient.

    The transmitter half (to the diffraction point) is traced with the RayD reflection EPC or
    one visibility segment, and the receiver half the same way only where the first is valid.
    The rows carry the interaction sequence, the points and the path length; no field.
    """

    if not args:
        raise TypeError("coupled_chain_geometry_forward requires a typed RayD scene resource")
    native_args = (_rayd_scene_resource(args[0]), *args[1:])
    out = _COUPLED_CHAIN_GEOMETRY_FORWARD(*native_args)
    if not isinstance(out, dict):
        raise TypeError("_channel.coupled_chain_geometry_forward must return a dict")
    required = {
        "valid": (torch.bool, 1),
        "primitive_sequence": (torch.int32, 2),
        "interaction_positions": (torch.float32, 3),
        "interaction_normals": (torch.float32, 3),
        "path_length_m": (torch.float32, 1),
        "delay_s": (torch.float32, 1),
    }
    if set(out) != set(required):
        raise ValueError("coupled chain geometry returned unexpected fields")
    for name, (dtype, ndim) in required.items():
        validate_cuda_tensor(name, out[name], dtype=dtype, ndim=ndim)
    count, width = (int(value) for value in out["primitive_sequence"].shape)
    for name in ("interaction_positions", "interaction_normals"):
        if tuple(out[name].shape) != (count, width, 3):
            raise ValueError(f"{name} must have shape (N, depth, 3)")
    return out

class CoupledDdGeometryOutputs(TypedDict):
    """Two-edge diffraction geometry rows; no field or path gain."""

    valid: torch.Tensor
    interaction_type_sequence: torch.Tensor
    primitive_sequence: torch.Tensor
    edge_sequence: torch.Tensor
    edge1_id: torch.Tensor
    edge2_id: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    edge1_position: torch.Tensor
    edge2_position: torch.Tensor
    path_length_m: torch.Tensor
    delay_s: torch.Tensor
    certificate_rejection_count: torch.Tensor


def coupled_dd_geometry_forward(*args: object) -> CoupledDdGeometryOutputs:
    """Construct two-edge (double) diffraction geometry without a coefficient.

    The native operation runs a convex bracketed Fermat solve for the two-edge
    Keller point pair (Q1 on e1, Q2 on e2) and three RayD segment
    visibility queries (tx->Q1, Q1->Q2, Q2->rx). Both edge ids are recoverable
    from ``edge_sequence`` (slot 0 = e1, slot 1 = e2); ``primitive_sequence`` is
    fully ``-1`` because a double-diffraction row touches no face. The returned
    dictionary intentionally carries no ``path_gain``/``field`` entry; complex
    transport belongs to the unified field phase. ``certificate_rejection_count``
    is a device ``int32[1]`` count of candidates dropped because the float32
    bisection could not certify their stationary point.
    """

    if not args:
        raise TypeError(
            "coupled_dd_geometry_forward requires a typed RayD scene resource"
        )
    native_args = (_rayd_scene_resource(args[0]), *args[1:])
    out = _COUPLED_DD_GEOMETRY_FORWARD(
        *native_args,
    )
    if not isinstance(out, dict):
        raise TypeError(
            "_channel.coupled_dd_geometry_forward must return a dict"
        )
    required = {
        "valid": (torch.bool, 1),
        "interaction_type_sequence": (torch.int32, 2),
        "primitive_sequence": (torch.int32, 2),
        "edge_sequence": (torch.int32, 2),
        "edge1_id": (torch.int32, 1),
        "edge2_id": (torch.int32, 1),
        "interaction_positions": (torch.float32, 3),
        "interaction_normals": (torch.float32, 3),
        "edge1_position": (torch.float32, 2),
        "edge2_position": (torch.float32, 2),
        "path_length_m": (torch.float32, 1),
        "delay_s": (torch.float32, 1),
        "certificate_rejection_count": (torch.int32, 1),
    }
    for name, (dtype, ndim) in required.items():
        value = out.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"coupled DD geometry output {name!r} must be a tensor")
        validate_cuda_tensor(name, value, dtype=dtype, ndim=ndim)
    if "path_gain" in out or "field" in out or "path_field" in out:
        raise ValueError(
            "coupled DD geometry must not expose placeholder physical coefficients"
        )
    count = int(out["valid"].shape[0])
    if out["interaction_type_sequence"].shape != (count, 2):
        raise ValueError("interaction_type_sequence must have shape (N, 2)")
    if out["primitive_sequence"].shape != (count, 2) or out["edge_sequence"].shape != (
        count,
        2,
    ):
        raise ValueError("coupled DD primitive/edge sequences must have shape (N, 2)")
    if out["interaction_positions"].shape != (count, 2, 3):
        raise ValueError("interaction_positions must have shape (N, 2, 3)")
    if out["interaction_normals"].shape != (count, 2, 3):
        raise ValueError("interaction_normals must have shape (N, 2, 3)")
    if out["certificate_rejection_count"].shape != (1,):
        raise ValueError("certificate_rejection_count must have shape (1,)")
    return cast(CoupledDdGeometryOutputs, out)


def rayd_diffraction_paths_order1_forward(*args: object) -> tuple[torch.Tensor, ...]:
    if not args:
        raise TypeError(
            "rayd_diffraction_paths_order1_forward requires a typed RayD scene resource"
        )
    native_args = (_rayd_scene_resource(args[0]), *args[1:])
    out = _RAYD_DIFFRACTION_PATHS_ORDER1_FORWARD(
        *native_args,
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel.rayd_diffraction_paths_order1_forward must return a tensor sequence"
        )
    return tuple(out)


_DIFFRACTION_STATE_CAPACITY = 4_194_304


def diffraction_tx_visible_state_plan(
    scene_resource: object, tx: torch.Tensor, edge_index: torch.Tensor, edge_position: torch.Tensor,
    edge_direction: torch.Tensor, edge_t_min: torch.Tensor, edge_t_max: torch.Tensor,
    n0: torch.Tensor, n1: torch.Tensor, prim0: torch.Tensor, prim1: torch.Tensor,
    exterior_angle: torch.Tensor, source: torch.Tensor, source_power: torch.Tensor, *,
    per_state_source: bool = False,
) -> torch.Tensor:
    """Transmitter visibility of every diffraction state.

    With ``per_state_source`` each state is checked against its own ``source`` row, so one
    launch serves the states of several transmitters; ``tx`` is still validated.
    """

    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1, trailing_shape=(3,))
    validate_cuda_tensor(
        "edge_index", edge_index, dtype=torch.int32, ndim=1, require_contiguous=False
    )
    validate_cuda_tensor(
        "edge_position",
        edge_position,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "edge_direction",
        edge_direction,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor("edge_t_min", edge_t_min, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("edge_t_max", edge_t_max, dtype=torch.float32, ndim=1)
    for name, tensor, dtype, ndim, trailing_shape in (
        ("n0", n0, torch.float32, 2, (3,)),
        ("n1", n1, torch.float32, 2, (3,)),
        ("prim0", prim0, torch.int32, 1, ()),
        ("prim1", prim1, torch.int32, 1, ()),
        ("exterior_angle", exterior_angle, torch.float32, 1, ()),
        ("source", source, torch.float32, 2, (3,)),
        ("source_power", source_power, torch.float32, 1, ()),
    ):
        validate_cuda_tensor(
            name,
            tensor,
            dtype=dtype,
            ndim=ndim,
            trailing_shape=trailing_shape,
            require_contiguous=False,
        )

    state_count = int(edge_position.shape[0])
    if state_count > _DIFFRACTION_STATE_CAPACITY:
        raise ValueError(
            "diffraction transmitter-visible state capacity exceeds 4194304"
        )
    row_shapes = {
        "edge_index": edge_index.shape[:1],
        "edge_direction": edge_direction.shape[:1],
        "edge_t_min": edge_t_min.shape[:1],
        "edge_t_max": edge_t_max.shape[:1],
        "n0": n0.shape[:1],
        "n1": n1.shape[:1],
        "prim0": prim0.shape[:1],
        "prim1": prim1.shape[:1],
        "exterior_angle": exterior_angle.shape[:1],
        "source": source.shape[:1],
        "source_power": source_power.shape[:1],
    }
    mismatched = [name for name, shape in row_shapes.items() if shape != (state_count,)]
    if mismatched:
        raise ValueError(
            "diffraction state tensors must share one row capacity: "
            + ", ".join(mismatched)
        )
    state_tensors = (
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
    )
    if any(tensor.get_device() != tx.get_device() for tensor in state_tensors):
        raise ValueError("tx and diffraction state tensors must share one CUDA device")

    active = _DIFFRACTION_TX_VISIBLE_STATE_PLAN(
        _rayd_scene_resource(scene_resource),
        tx,
        edge_position,
        edge_direction,
        edge_t_min,
        edge_t_max,
        source.contiguous() if per_state_source else None,
    )
    if not isinstance(active, torch.Tensor):
        raise TypeError(
            "_channel.diffraction_tx_visible_state_plan must return a tensor"
        )
    validate_cuda_tensor("active", active, dtype=torch.bool, ndim=1)
    if active.shape != (state_count,):
        raise ValueError(
            "_channel.diffraction_tx_visible_state_plan returned bad shape"
        )
    if active.get_device() != tx.get_device():
        raise ValueError(
            "_channel.diffraction_tx_visible_state_plan returned wrong device"
        )
    return active


def deterministic_normalize_vec3(values: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    validate_cuda_tensor(
        "values", values, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    out = _DETERMINISTIC_NORMALIZE_VEC3(values, float(eps))
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.deterministic_normalize_vec3 must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if out.shape != values.shape:
        raise ValueError(
            "_channel.deterministic_normalize_vec3 returned bad shape"
        )
    return out


def deterministic_face_groups(
    tri_a: torch.Tensor, normals: torch.Tensor, surface_ids: torch.Tensor, *, quantization: float,
) -> dict[str, torch.Tensor | int]:
    validate_cuda_tensor(
        "tri_a", tri_a, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "normals", normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("surface_ids", surface_ids, dtype=torch.int64, ndim=1)
    if normals.shape != tri_a.shape:
        raise ValueError("normals must match tri_a")
    if surface_ids.shape != (tri_a.shape[0],):
        raise ValueError("surface_ids must match tri_a")
    if quantization <= 0.0:
        raise ValueError("quantization must be positive")
    exported = _DETERMINISTIC_FACE_GROUPS(
        tri_a,
        normals,
        surface_ids,
        float(quantization),
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.deterministic_face_groups must return a dict")
    expected_fields = {
        "face_group_id",
        "representative_faces",
        "surface_group_id",
        "surface_group_size",
        "surface_group_members",
        "group_count",
    }
    if set(exported) != expected_fields:
        raise ValueError(
            "_channel.deterministic_face_groups returned unexpected fields"
        )
    face_count = int(tri_a.shape[0])
    validate_cuda_tensor(
        "face_group_id", exported["face_group_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "representative_faces",
        exported["representative_faces"],
        dtype=torch.int64,
        ndim=1,
    )
    validate_cuda_tensor(
        "surface_group_id", exported["surface_group_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "surface_group_size", exported["surface_group_size"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "surface_group_members",
        exported["surface_group_members"],
        dtype=torch.int32,
        ndim=1,
    )
    group_count = exported["group_count"]
    if not isinstance(group_count, int):
        raise TypeError(
            "_channel.deterministic_face_groups returned non-int group_count"
        )
    if exported["face_group_id"].shape != (face_count,) or exported[
        "surface_group_id"
    ].shape != (face_count,):
        raise ValueError(
            "_channel.deterministic_face_groups returned bad face group shape"
        )
    if exported["representative_faces"].shape != (group_count,) or exported[
        "surface_group_size"
    ].shape != (group_count,):
        raise ValueError(
            "_channel.deterministic_face_groups returned bad group shape"
        )
    return exported


def mc_diffraction_edge_geometry(
    vertices: torch.Tensor, faces: torch.Tensor, face_normals: torch.Tensor, edge_v0: torch.Tensor,
    edge_v1: torch.Tensor, face0: torch.Tensor, face1: torch.Tensor, *, plane_tol: float,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor(
        "vertices", vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("faces", faces, dtype=torch.int32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor(
        "face_normals", face_normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("edge_v0", edge_v0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("edge_v1", edge_v1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    if (
        edge_v1.shape != edge_v0.shape
        or face0.shape != edge_v0.shape
        or face1.shape != edge_v0.shape
    ):
        raise ValueError("edge_v1, face0, and face1 must match edge_v0 shape")
    geometry = _MC_DIFFRACTION_EDGE_GEOMETRY(
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
        raise TypeError(
            "_channel.mc_diffraction_edge_geometry must return 11 tensors"
        )
    validate_cuda_tensor("selected", geometry[0], dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "edge_pos", geometry[1], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "edge_dir", geometry[2], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("lengths", geometry[3], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_min", geometry[4], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_max", geometry[5], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "n0", geometry[6], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "n1", geometry[7], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("face0_out", geometry[8], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1_out", geometry[9], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("exterior_angle", geometry[10], dtype=torch.float32, ndim=1)
    return geometry


def mc_surface_group_edge_candidates(
    vertices: torch.Tensor, faces: torch.Tensor, face_normals: torch.Tensor, edge_v0: torch.Tensor,
    edge_v1: torch.Tensor, face0: torch.Tensor, face1: torch.Tensor, selected: torch.Tensor, *,
    plane_tol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_cuda_tensor(
        "vertices", vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("faces", faces, dtype=torch.int32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor(
        "face_normals", face_normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("edge_v0", edge_v0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("edge_v1", edge_v1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("selected", selected, dtype=torch.bool, ndim=1)
    if (
        edge_v1.shape != edge_v0.shape
        or face0.shape != edge_v0.shape
        or face1.shape != edge_v0.shape
    ):
        raise ValueError("edge_v1, face0, and face1 must match edge_v0 shape")
    if selected.shape != edge_v0.shape:
        raise ValueError("selected must match edge_v0 shape")
    candidates = _MC_SURFACE_GROUP_EDGE_CANDIDATES(
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
        raise TypeError(
            "_channel.mc_surface_group_edge_candidates must return 2 tensors"
        )
    counts, indices = candidates
    validate_cuda_tensor("counts", counts, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("indices", indices, dtype=torch.int32, ndim=2)
    if counts.shape[0] != faces.shape[0] or indices.shape[0] != faces.shape[0]:
        raise ValueError(
            "_channel.mc_surface_group_edge_candidates returned unexpected shapes"
        )
    return counts, indices


# ---------------------------------------------------------------------------
# autograd
# ---------------------------------------------------------------------------
_RAYD_RAY_FLAGS_ALL = 0x01 | 0x02 | 0x04


_RAYD_TRACE_REFLECTIONS_AD_FIELDS = (
    "valid",
    "t",
    "image_sources",
    "prim_ids",
    "tape_barycentric",
    "tape_hit_points",
    "tape_normals",
)


def _epc_paths_frozen_winner_checks(
    source: torch.Tensor, receiver: torch.Tensor, sequence: torch.Tensor,
    plane_points: torch.Tensor, plane_normals: torch.Tensor, valid: torch.Tensor,
    bounce_count: torch.Tensor,
) -> tuple[int, int]:
    validate_cuda_tensor(
        "source", source, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "receiver", receiver, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("sequence", sequence, dtype=torch.int32, ndim=2)
    validate_cuda_tensor(
        "plane_points", plane_points, dtype=torch.float32, ndim=3, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "plane_normals",
        plane_normals,
        dtype=torch.float32,
        ndim=3,
        trailing_shape=(3,),
    )
    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=1)
    validate_cuda_tensor("bounce_count", bounce_count, dtype=torch.int32, ndim=1)
    rows = int(source.shape[0])
    bounces = int(sequence.shape[1])
    _ad_check_rows("receiver", receiver, rows)
    _ad_check_rows("sequence", sequence, rows)
    _ad_check_rows("valid", valid, rows)
    _ad_check_rows("bounce_count", bounce_count, rows)
    if bounces < 1:
        raise ValueError("sequence must cover at least one bounce")
    if tuple(plane_points.shape) != (rows, bounces, 3):
        raise ValueError("plane_points must have shape (rows, bounces, 3)")
    if tuple(plane_normals.shape) != (rows, bounces, 3):
        raise ValueError("plane_normals must have shape (rows, bounces, 3)")
    return rows, bounces


def rayd_reflection_epc_paths_backward(
    scene_resource: object, source: torch.Tensor, receiver: torch.Tensor, sequence: torch.Tensor,
    plane_points: torch.Tensor, plane_normals: torch.Tensor, valid: torch.Tensor,
    bounce_count: torch.Tensor, *, grad_points: torch.Tensor | None = None,
    grad_normals: torch.Tensor | None = None, grad_path_length: torch.Tensor | None = None,
    need_grad_vertices: bool = False, need_grad_source: bool = False,
    need_grad_receiver: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    rows, bounces = _epc_paths_frozen_winner_checks(
        source, receiver, sequence, plane_points, plane_normals, valid, bounce_count
    )
    _ad_check_optional_grad("grad_points", grad_points, ((rows, bounces, 3),))
    _ad_check_optional_grad("grad_normals", grad_normals, ((rows, bounces, 3),))
    _ad_check_optional_grad("grad_path_length", grad_path_length, ((rows,),))
    out = _RAYD_REFLECTION_EPC_PATHS_BACKWARD(
        _rayd_scene_resource(scene_resource),
        source,
        receiver,
        sequence,
        plane_points,
        plane_normals,
        valid,
        bounce_count,
        grad_points,
        grad_normals,
        grad_path_length,
        bool(need_grad_vertices),
        bool(need_grad_source),
        bool(need_grad_receiver),
    )
    if not isinstance(out, (tuple, list)) or len(out) != 3:
        raise TypeError(
            "_channel.rayd_reflection_epc_paths_backward must return"
            " 3 gradients"
        )
    return tuple(out)


def rayd_reflection_epc_paths_jvp(
    scene_resource: object, source: torch.Tensor, receiver: torch.Tensor, sequence: torch.Tensor,
    plane_points: torch.Tensor, plane_normals: torch.Tensor, valid: torch.Tensor,
    bounce_count: torch.Tensor, *, tangent_vertices: torch.Tensor | None = None,
    tangent_source: torch.Tensor | None = None, tangent_receiver: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    rows, _bounces = _epc_paths_frozen_winner_checks(
        source, receiver, sequence, plane_points, plane_normals, valid, bounce_count
    )
    _ad_check_tangent_vec3("tangent_vertices", tangent_vertices, None)
    _ad_check_tangent_vec3("tangent_source", tangent_source, rows)
    _ad_check_tangent_vec3("tangent_receiver", tangent_receiver, rows)
    out = _RAYD_REFLECTION_EPC_PATHS_JVP(
        _rayd_scene_resource(scene_resource),
        source,
        receiver,
        sequence,
        plane_points,
        plane_normals,
        valid,
        bounce_count,
        tangent_vertices,
        tangent_source,
        tangent_receiver,
    )
    if not isinstance(out, (tuple, list)) or len(out) != 3:
        raise TypeError(
            "_channel.rayd_reflection_epc_paths_jvp must return 3 tangents"
        )
    return tuple(out)


_EPC_PATHS_AD = AdInputs(
    "rayd_reflection_epc_paths_ad", opaque("scene_resource"),
    *(opaque(name, grad="diff") for name in ("vertices", "source", "receiver")),
    *(
        opaque(name)
        for name in (
            "sequence", "plane_points", "plane_normals", "surface_group_id",
            "surface_group_size", "surface_group_members", "max_bounces",
            "visibility_ignore_mode", "min_segment_length",
        )
    ),
)


class _RaydReflectionEpcPathsAdFunction(torch.autograd.Function):
    """Fixed-winner differentiable RayD reflection EPC paths through the direct typed C++ boundary.

    Forward IS the discovery entry (direct-plane mode) re-launched on the
    frozen winner sequence, so the primal hit points, normals and path length
    are the native discovery values, not a reconstruction. Backward/jvp call
    RayD's chain geometry companions; ``vertices`` must be the scene's global
    vertex table and only routes vertex gradients/tangents.
    """

    @staticmethod
    def forward(
        scene_resource: Any, vertices: Any, source: Any, receiver: Any, sequence: Any,
        plane_points: Any, plane_normals: Any, surface_group_id: Any, surface_group_size: Any,
        surface_group_members: Any, max_bounces: Any, visibility_ignore_mode: Any,
        min_segment_length: Any,
    ) -> Any:
        out = _RAYD_REFLECTION_EPC_PATHS_FORWARD(
            scene_resource,
            source,
            receiver,
            None,
            sequence,
            plane_points,
            plane_normals,
            surface_group_id,
            surface_group_size,
            surface_group_members,
            int(max_bounces),
            int(visibility_ignore_mode),
            float(min_segment_length),
            False,
        )
        return tuple(out)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        (
            scene_resource,
            vertices,
            source,
            receiver,
            sequence,
            plane_points,
            plane_normals,
            _surface_group_id,
            _surface_group_size,
            _surface_group_members,
            max_bounces,
            _visibility_ignore_mode,
            _min_segment_length,
        ) = inputs
        valid, _path_length, resolved_prim_ids, surface_group_ids = output[:4]
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        source = torch.autograd.forward_ad.unpack_dual(source).primal
        receiver = torch.autograd.forward_ad.unpack_dual(receiver).primal
        # Direct-plane mode fills every bounce slot or invalidates the row, so
        # the frozen per-row bounce count is the launch width; invalid rows
        # are skipped by the companions regardless of this value.
        bounce_count = torch.full(
            (int(source.shape[0]),),
            int(max_bounces),
            device=source.device,
            dtype=torch.int32,
        )
        ctx.scene_resource = scene_resource
        ctx.vertices_shape = tuple(vertices.shape)
        ctx.save_for_backward(
            source,
            receiver,
            sequence,
            plane_points,
            plane_normals,
            valid,
            bounce_count,
        )
        ctx.save_for_forward(
            source,
            receiver,
            sequence,
            plane_points,
            plane_normals,
            valid,
            bounce_count,
        )
        ctx.mark_non_differentiable(valid, resolved_prim_ids, surface_group_ids)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        none_grads = _EPC_PATHS_AD.none
        grad_path_length = grad_outputs[1]
        grad_hits = grad_outputs[4]
        grad_unit_normals = grad_outputs[5]
        if grad_path_length is None and grad_hits is None and grad_unit_normals is None:
            return none_grads
        (
            source,
            receiver,
            sequence,
            plane_points,
            plane_normals,
            valid,
            bounce_count,
        ) = ctx.saved_tensors
        need_grad_vertices = _EPC_PATHS_AD.need(ctx, "vertices")
        need_grad_source = _EPC_PATHS_AD.need(ctx, "source")
        need_grad_receiver = _EPC_PATHS_AD.need(ctx, "receiver")
        if not (need_grad_vertices or need_grad_source or need_grad_receiver):
            return none_grads
        grad_vertices, grad_source, grad_receiver = rayd_reflection_epc_paths_backward(
            ctx.scene_resource,
            source,
            receiver,
            sequence,
            plane_points,
            plane_normals,
            valid,
            bounce_count,
            grad_points=grad_hits,
            grad_normals=grad_unit_normals,
            grad_path_length=grad_path_length,
            need_grad_vertices=need_grad_vertices,
            need_grad_source=need_grad_source,
            need_grad_receiver=need_grad_receiver,
        )
        if need_grad_vertices and (
            tuple(cast(torch.Tensor, grad_vertices).shape) != ctx.vertices_shape
        ):
            raise RuntimeError(
                "rayd_reflection_epc_paths_ad vertices must be the scene"
                " global vertex table"
            )
        return _EPC_PATHS_AD.grads(
            ctx, vertices=grad_vertices, source=grad_source, receiver=grad_receiver
        )

    @staticmethod
    def jvp(
        ctx: Any, _grad_handle: Any, grad_vertices: Any, grad_source: Any, grad_receiver: Any,
        *_frozen_tangents: Any,
    ) -> Any:
        (
            source,
            receiver,
            sequence,
            plane_points,
            plane_normals,
            valid,
            bounce_count,
        ) = ctx.saved_tensors
        with disable_functorch():
            tangents = rayd_reflection_epc_paths_jvp(
                ctx.scene_resource,
                _ad_native_tensor(source),
                _ad_native_tensor(receiver),
                _ad_native_tensor(sequence),
                _ad_native_tensor(plane_points),
                _ad_native_tensor(plane_normals),
                _ad_native_tensor(valid),
                _ad_native_tensor(bounce_count),
                tangent_vertices=_ad_checked_tangent(
                    "rayd_reflection_epc_paths_ad tangent_vertices",
                    _ad_native_tangent_or_none(grad_vertices),
                    ctx.vertices_shape,
                ),
                tangent_source=_ad_checked_tangent(
                    "rayd_reflection_epc_paths_ad tangent_source",
                    _ad_native_tangent_or_none(grad_source),
                    tuple(source.shape),
                ),
                tangent_receiver=_ad_checked_tangent(
                    "rayd_reflection_epc_paths_ad tangent_receiver",
                    _ad_native_tangent_or_none(grad_receiver),
                    tuple(receiver.shape),
                ),
            )
        tangent_points, tangent_normals, tangent_path_length = tangents
        return (
            None,
            tangent_path_length,
            None,
            None,
            tangent_points,
            tangent_normals,
        )


_RAYD_REFLECTION_EPC_PATHS_AD_FIELDS = (
    "valid",
    "path_length",
    "resolved_prim_ids",
    "surface_group_ids",
    "hit_positions",
    "normals",
)


def rayd_reflection_epc_paths_ad(
    scene_resource: object, vertices: torch.Tensor, source: torch.Tensor, receiver: torch.Tensor,
    sequence: torch.Tensor, plane_points: torch.Tensor, plane_normals: torch.Tensor,
    surface_group_id: torch.Tensor, surface_group_size: torch.Tensor,
    surface_group_members: torch.Tensor, max_bounces: int, visibility_ignore_mode: int,
    min_segment_length: float,
) -> dict[str, torch.Tensor]:
    """Differentiable RayD reflection EPC paths under the fixed-winner contract.

    ``hit_positions``, ``normals`` and ``path_length`` participate in
    reverse- and forward-mode torch AD with respect to ``vertices``,
    ``source`` and ``receiver``. ``sequence`` is the frozen winner face
    sequence and ``plane_points`` / ``plane_normals`` are its detached plane
    arrays (anchor + unit face normal, gathered per bounce); ``valid`` and
    the resolved ids stay non-differentiable. ``valid`` applies the same
    minimum segment between interactions as discovery.
    """

    validate_cuda_tensor(
        "source", source, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "receiver", receiver, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("sequence", sequence, dtype=torch.int32, ndim=2)
    validate_cuda_tensor(
        "plane_points", plane_points, dtype=torch.float32, ndim=3, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "plane_normals",
        plane_normals,
        dtype=torch.float32,
        ndim=3,
        trailing_shape=(3,),
    )
    if int(sequence.shape[1]) != int(max_bounces):
        raise ValueError("sequence width must equal max_bounces")
    values = _RaydReflectionEpcPathsAdFunction.apply(
        _rayd_scene_resource(scene_resource),
        vertices,
        source,
        receiver,
        sequence,
        plane_points,
        plane_normals,
        surface_group_id,
        surface_group_size,
        surface_group_members,
        int(max_bounces),
        int(visibility_ignore_mode),
        float(min_segment_length),
    )
    return dict(zip(_RAYD_REFLECTION_EPC_PATHS_AD_FIELDS, values, strict=True))


def rayd_scene_face_normals_backward(
    scene_resource: object, grad_face_normals: torch.Tensor,
) -> torch.Tensor:
    # Cotangents from autograd may be strided views; the native kernel
    # consumes explicit strides, so contiguity is deliberately not required.
    if not isinstance(grad_face_normals, torch.Tensor):
        raise TypeError("grad_face_normals must be a torch.Tensor")
    if grad_face_normals.dtype != torch.float32:
        raise TypeError("grad_face_normals must have dtype torch.float32")
    if not grad_face_normals.is_cuda:
        raise ValueError("grad_face_normals must be a CUDA tensor")
    if grad_face_normals.ndim != 2 or grad_face_normals.shape[1] != 3:
        raise ValueError("grad_face_normals must have shape (F, 3)")
    out = _RAYD_SCENE_FACE_NORMALS_BACKWARD(
        _rayd_scene_resource(scene_resource),
        grad_face_normals,
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.rayd_scene_face_normals_backward must return a tensor"
        )
    return out


def rayd_scene_face_normals_jvp(
    scene_resource: object, tangent_vertices: torch.Tensor,
) -> torch.Tensor:
    _ad_check_tangent_vec3("tangent_vertices", tangent_vertices, None)
    if tangent_vertices is None:
        raise ValueError("tangent_vertices is required")
    out = _RAYD_SCENE_FACE_NORMALS_JVP(
        _rayd_scene_resource(scene_resource),
        tangent_vertices,
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.rayd_scene_face_normals_jvp must return a tensor"
        )
    return out


_FACE_NORMALS_AD = AdInputs(
    "rayd_face_normals_ad", opaque("scene_resource"), opaque("vertices", grad="diff"),
    opaque("raw_face_normals"),
)


class _RaydFaceNormalsAdFunction(torch.autograd.Function):
    """Scene unit face-normal table with a graph to the vertex table.

    Forward normalizes the native face-normal export with the same kernel the
    reflection discovery uses; backward/jvp call RayD's face-normal table
    companions (the adjoint/tangent of normalize(cross(v1 - v0, v2 - v0))
    over the scene's global vertex and face tables).
    """

    @staticmethod
    def forward(scene_resource: Any, vertices: Any, raw_face_normals: Any) -> Any:
        return deterministic_normalize_vec3(raw_face_normals, eps=1.0e-6)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        scene_resource, vertices, _raw_face_normals = inputs
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        ctx.scene_resource = scene_resource
        ctx.vertices_shape = tuple(vertices.shape)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, grad_face_normals: Any) -> Any:
        if _FACE_NORMALS_AD.need(ctx, "raw_face_normals"):
            raise RuntimeError(
                "rayd_face_normals_ad differentiates the vertex table only;"
                " the raw face-normal export is a detached scene record"
            )
        if grad_face_normals is None or not _FACE_NORMALS_AD.need(ctx, "vertices"):
            return _FACE_NORMALS_AD.none
        grad_vertices = rayd_scene_face_normals_backward(ctx.scene_resource, grad_face_normals)
        if tuple(grad_vertices.shape) != ctx.vertices_shape:
            raise RuntimeError(
                "rayd_face_normals_ad vertices must be the scene global vertex table"
            )
        return _FACE_NORMALS_AD.place(vertices=grad_vertices)

    @staticmethod
    def jvp(ctx: Any, _grad_handle: Any, grad_vertices: Any, _grad_raw_face_normals: Any) -> Any:
        tangent_vertices = _ad_native_tangent_or_none(grad_vertices)
        if tangent_vertices is None:
            return None
        with disable_functorch():
            return rayd_scene_face_normals_jvp(
                ctx.scene_resource,
                _ad_checked_tangent(
                    "rayd_face_normals_ad tangent_vertices",
                    tangent_vertices,
                    ctx.vertices_shape,
                ),
            )


def rayd_face_normals_ad(
    scene_resource: object, vertices: torch.Tensor, raw_face_normals: torch.Tensor,
) -> torch.Tensor:
    """Scene unit face-normal table, differentiable in the vertex table.

    ``raw_face_normals`` is the detached native export (scene edge records);
    the returned table is its normalization with vertex gradients/tangents
    routed through RayD's face-normal companions under the fixed-winner
    contract (which face a row consumes stays a frozen integer gather).
    """

    validate_cuda_tensor(
        "raw_face_normals",
        raw_face_normals,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    return _RaydFaceNormalsAdFunction.apply(
        _rayd_scene_resource(scene_resource), vertices, raw_face_normals
    )


# ---------------------------------------------------------------------------
# penetration_autograd
# ---------------------------------------------------------------------------
def _segment_penetration_tape_from_values(
    values: tuple[torch.Tensor, ...], *, hit_capacity: int, failure_state: CapacityFailureState,
) -> SegmentPenetrationTapeResult:
    result = SegmentPenetrationResult(
        hit_capacity=hit_capacity,
        failure_state=failure_state,
        valid=values[0],
        num_hits=values[1],
        reached_target=values[2],
        overflow=values[3],
        distance=values[4],
        direction=values[5],
        t=values[6],
        position=values[7],
        normal=values[8],
        geometric_normal=values[9],
        global_primitive_id=values[10],
    )
    return SegmentPenetrationTapeResult(
        result=result,
        tape_primitive_id=values[11],
        tape_barycentric=values[12],
        tape_restart_epsilon=values[13],
        tape_restart_branch=values[14],
        tape_restart_tie_mask=values[15],
        tape_direction_denominator_branch=values[16],
    )


def _segment_penetration_tape_values(
    tape: SegmentPenetrationTapeResult,
) -> tuple[torch.Tensor, ...]:
    result = tape.result
    return (
        result.valid,
        result.num_hits,
        result.reached_target,
        result.overflow,
        result.distance,
        result.direction,
        result.t,
        result.position,
        result.normal,
        result.geometric_normal,
        result.global_primitive_id,
        tape.tape_primitive_id,
        tape.tape_barycentric,
        tape.tape_restart_epsilon,
        tape.tape_restart_branch,
        tape.tape_restart_tie_mask,
        tape.tape_direction_denominator_branch,
    )


_SEGMENT_PENETRATION_AD = AdInputs(
    "rayd_segment_penetration_ad", opaque("scene_resource"),
    *(opaque(name, grad="diff") for name in ("vertices", "origins", "targets")),
    *(
        opaque(name)
        for name in (
            "input_active", "input_active_any", "hit_capacity", "policy", "scene_diagonal",
            "failure_state",
        )
    ),
)


class _RaydSegmentPenetrationAdFunction(torch.autograd.Function):
    """Fixed-winner segment geometry with native RayD VJP and JVP companions."""

    @staticmethod
    def forward(
        scene_resource: Any, vertices: Any, origins: Any, targets: Any, input_active: Any,
        input_active_any: Any, hit_capacity: Any, policy: Any, scene_diagonal: Any,
        failure_state: Any,
    ) -> Any:
        tape = rayd_segment_penetration_forward_tape(
            scene_resource,
            origins,
            targets,
            input_active,
            input_active_any=input_active_any,
            hit_capacity=hit_capacity,
            policy=policy,
            scene_diagonal=scene_diagonal,
            failure_state=failure_state,
        )
        return _segment_penetration_tape_values(tape)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        (
            scene_resource,
            vertices,
            origins,
            targets,
            input_active,
            input_active_any,
            hit_capacity,
            policy,
            scene_diagonal,
            failure_state,
        ) = inputs
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        origins = torch.autograd.forward_ad.unpack_dual(origins).primal
        targets = torch.autograd.forward_ad.unpack_dual(targets).primal
        active_ctx = _ad_active_ctx(input_active, origins)
        saved = (origins, targets, active_ctx, *output)
        ctx.scene_resource = scene_resource
        ctx.input_active_absent = input_active is None
        ctx.input_active_any = input_active_any
        ctx.hit_capacity = hit_capacity
        ctx.policy = policy
        ctx.scene_diagonal = scene_diagonal
        ctx.failure_state = failure_state
        ctx.vertices_shape = tuple(vertices.shape)
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.mark_non_differentiable(
            output[0],
            output[1],
            output[2],
            output[3],
            output[10],
            *output[11:],
        )

    @staticmethod
    def _saved_request(
        ctx: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, SegmentPenetrationTapeResult]:
        origins, targets, active_ctx, *values = ctx.saved_tensors
        input_active = None if ctx.input_active_absent else active_ctx
        tape = _segment_penetration_tape_from_values(
            tuple(values),
            hit_capacity=ctx.hit_capacity,
            failure_state=ctx.failure_state,
        )
        return origins, targets, input_active, tape

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        none_grads = _SEGMENT_PENETRATION_AD.none
        continuous_grads = grad_outputs[4:10]
        if not any(value is not None for value in continuous_grads):
            return none_grads
        need_grad_vertices = _SEGMENT_PENETRATION_AD.need(ctx, "vertices")
        need_grad_origins = _SEGMENT_PENETRATION_AD.need(ctx, "origins")
        need_grad_targets = _SEGMENT_PENETRATION_AD.need(ctx, "targets")
        if not (need_grad_vertices or need_grad_origins or need_grad_targets):
            return none_grads
        origins, targets, input_active, tape = (
            _RaydSegmentPenetrationAdFunction._saved_request(ctx)
        )
        gradients = rayd_segment_penetration_backward(
            ctx.scene_resource,
            origins,
            targets,
            input_active,
            input_active_any=ctx.input_active_any,
            hit_capacity=ctx.hit_capacity,
            policy=ctx.policy,
            scene_diagonal=ctx.scene_diagonal,
            failure_state=ctx.failure_state,
            tape=tape,
            grad_distance=continuous_grads[0],
            grad_direction=continuous_grads[1],
            grad_t=continuous_grads[2],
            grad_position=continuous_grads[3],
            grad_normal=continuous_grads[4],
            grad_geometric_normal=continuous_grads[5],
            need_grad_vertices=need_grad_vertices,
            need_grad_origins=need_grad_origins,
            need_grad_targets=need_grad_targets,
        )
        if (
            need_grad_vertices
            and gradients.grad_vertices is not None
            and tuple(gradients.grad_vertices.shape) != ctx.vertices_shape
        ):
            raise RuntimeError(
                "rayd_segment_penetration_ad vertices must be the scene global"
                " vertex table"
            )
        return _SEGMENT_PENETRATION_AD.grads(
            ctx, vertices=gradients.grad_vertices, origins=gradients.grad_origins,
            targets=gradients.grad_targets,
        )

    @staticmethod
    def jvp(
        ctx: Any, _grad_handle: Any, grad_vertices: Any, grad_origins: Any, grad_targets: Any,
        *_frozen_tangents: Any,
    ) -> Any:
        origins, targets, input_active, tape = (
            _RaydSegmentPenetrationAdFunction._saved_request(ctx)
        )
        native_values = tuple(_ad_native_tensor(value) for value in ctx.saved_tensors)
        native_origins, native_targets, native_active, *native_tape_values = (
            native_values
        )
        native_input_active = None if ctx.input_active_absent else native_active
        native_tape = _segment_penetration_tape_from_values(
            tuple(native_tape_values),
            hit_capacity=ctx.hit_capacity,
            failure_state=ctx.failure_state,
        )
        with disable_functorch():
            tangents = rayd_segment_penetration_jvp(
                ctx.scene_resource,
                native_origins,
                native_targets,
                native_input_active,
                input_active_any=ctx.input_active_any,
                hit_capacity=ctx.hit_capacity,
                policy=ctx.policy,
                scene_diagonal=ctx.scene_diagonal,
                failure_state=ctx.failure_state,
                tape=native_tape,
                tangent_vertices=_ad_checked_tangent(
                    "rayd_segment_penetration_ad tangent_vertices",
                    _ad_native_tangent_or_none(grad_vertices),
                    ctx.vertices_shape,
                ),
                tangent_origins=_ad_checked_tangent(
                    "rayd_segment_penetration_ad tangent_origins",
                    _ad_native_tangent_or_none(grad_origins),
                    tuple(origins.shape),
                ),
                tangent_targets=_ad_checked_tangent(
                    "rayd_segment_penetration_ad tangent_targets",
                    _ad_native_tangent_or_none(grad_targets),
                    tuple(targets.shape),
                ),
            )
        return (
            None,
            None,
            None,
            None,
            tangents.tangent_distance,
            tangents.tangent_direction,
            tangents.tangent_t,
            tangents.tangent_position,
            tangents.tangent_normal,
            tangents.tangent_geometric_normal,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rayd_segment_penetration_ad(
    scene_resource: object, vertices: torch.Tensor, origins: torch.Tensor, targets: torch.Tensor,
    input_active: torch.Tensor | None, *, input_active_any: bool, hit_capacity: int,
    policy: SegmentPenetrationPolicy, scene_diagonal: float, failure_state: CapacityFailureState,
) -> SegmentPenetrationResult:
    """Differentiable live RayD segment-penetration entry."""

    validate_cuda_tensor(
        "vertices", vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    values = _RaydSegmentPenetrationAdFunction.apply(
        _rayd_scene_resource(scene_resource),
        vertices,
        origins,
        targets,
        input_active,
        input_active_any,
        hit_capacity,
        policy,
        scene_diagonal,
        failure_state,
    )
    return SegmentPenetrationResult(
        hit_capacity=hit_capacity,
        failure_state=failure_state,
        valid=values[0],
        num_hits=values[1],
        reached_target=values[2],
        overflow=values[3],
        distance=values[4],
        direction=values[5],
        t=values[6],
        position=values[7],
        normal=values[8],
        geometric_normal=values[9],
        global_primitive_id=values[10],
    )
