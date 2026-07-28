"""Native geometry kernel facades.

Thin facades over the ``_channel`` geometry ABI: the RayD typed forward
bridges (intersection, visibility, reflection tracing, EPC paths, diffraction
sampling, coupled RD/DD geometry, ADR-027 segment penetration, and the
ADR-028 device-resident diffraction state plan), the scene-static geometry
primitives, and the :class:`torch.autograd.Function` companions that dispatch
their registered native backward/JVP entries.

bridge
------
Typed forward entries into the RayD-owned geometry operations. Every entry
validates its contract, requests the required native symbol through
:mod:`witwin.channel.runtime`, dispatches, and converts the native result into
a named typed contract. The ADR-027 segment-penetration family shares the
request/result validators ``_validate_segment_penetration_inputs``,
``_segment_penetration_request_args``, and ``_segment_penetration_result``.

primitives
----------
Scene-static geometry primitives: diffraction edge counting/selection, vector
normalization, point reflection, and the deterministic/Monte Carlo face-group
and edge-candidate exports.

autograd
--------
Differentiable geometry ops. Native ``torch.autograd.Function`` companions for
ray intersection, reflection tracing, EPC reflection paths, and scene face
normals: a plain forward, a ``once_differentiable`` VJP, and a forward-mode
JVP that all dispatch registered native kernels. Torch autograd may dispatch
these companions but never reconstructs the numerical operation. The adjoint
and tangent of the triangle face-normal are owned by the native face-normal
companions, not rebuilt here.

penetration_autograd
--------------------
The ADR-027 segment-penetration ``torch.autograd.Function``. It wraps the
forward-tape bridge above and dispatches the registered native
backward/JVP companions, reassembling the typed
:class:`SegmentPenetrationResult` / :class:`SegmentPenetrationTapeResult`
contracts from the flat native value tuple.
"""

from __future__ import annotations

import math

import torch

from witwin.channel.propagation.penetration import (
    SegmentPenetrationBackwardResult,
    SegmentPenetrationJvpResult,
    SegmentPenetrationPolicy,
    SegmentPenetrationResult,
    SegmentPenetrationTapeResult,
)
from witwin.channel.runtime import (
    CapacityFailureBit,
    CapacityFailureState,
    _ad_active_ctx,
    _ad_check_active,
    _ad_check_optional_grad,
    _ad_check_rows,
    _ad_check_tangent_vec3,
    _ad_checked_tangent,
    _ad_first_order_only,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _rayd_scene_resource,
    disable_functorch,
    native_extension,
    require_capacity_failure_state,
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)

__all__ = [
    "_BDPT_INTERSECTION_FIELDS",
    "_RaydFaceNormalsAdFunction",
    "_RaydIntersectAdFunction",
    "_RaydReflectionEpcPathsAdFunction",
    "_RaydSegmentPenetrationAdFunction",
    "_RaydTraceReflectionsAdFunction",
    "_epc_paths_frozen_winner_checks",
    "core_diffraction_edge_count",
    "coupled_dd_geometry_forward",
    "coupled_rd_geometry_forward",
    "deterministic_face_groups",
    "deterministic_normalize_vec3",
    "deterministic_reflect_points",
    "deterministic_surface_face_groups",
    "diffraction_tx_visible_state_plan",
    "mc_diffraction_edge_geometry",
    "mc_surface_group_edge_candidates",
    "rayd_diffraction_paths_order1_forward",
    "rayd_diffraction_sample_tape_forward",
    "rayd_face_normals_ad",
    "rayd_intersect_ad",
    "rayd_intersect_backward",
    "rayd_intersect_forward",
    "rayd_intersect_jvp",
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
    "rayd_trace_reflections_ad",
    "rayd_trace_reflections_backward",
    "rayd_trace_reflections_forward",
    "rayd_trace_reflections_forward_tape",
    "rayd_trace_reflections_jvp",
    "rayd_visibility_forward",
]


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
    origins: torch.Tensor,
    targets: torch.Tensor,
    input_active: torch.Tensor | None,
    *,
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
    hit_capacity: int,
    policy: SegmentPenetrationPolicy,
    scene_diagonal: float,
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
    scene_resource: object,
    origins: torch.Tensor,
    targets: torch.Tensor,
    input_active: torch.Tensor | None,
    *,
    input_active_any: bool,
    hit_capacity: int,
    policy: SegmentPenetrationPolicy,
    scene_diagonal: float,
    failure_state: CapacityFailureState,
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
        failure_state.bits,
        _SEGMENT_PENETRATION_FAILURE_BIT,
    )


def _segment_penetration_result(
    values: object,
    *,
    hit_capacity: int,
    failure_state: CapacityFailureState,
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


def _segment_penetration_tape_args(
    tape: SegmentPenetrationTapeResult,
) -> tuple[torch.Tensor, ...]:
    result = tape.result
    return (
        *(getattr(result, name) for name in _SEGMENT_PENETRATION_RESULT_FIELDS),
        *(getattr(tape, name) for name in _SEGMENT_PENETRATION_TAPE_FIELDS),
    )


def rayd_segment_penetration_forward(
    scene_resource: object,
    origins: torch.Tensor,
    targets: torch.Tensor,
    input_active: torch.Tensor | None,
    *,
    input_active_any: bool,
    hit_capacity: int,
    policy: SegmentPenetrationPolicy,
    scene_diagonal: float,
    failure_state: CapacityFailureState,
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
    values = _required_native_op("rayd_segment_penetration_forward")(*args)
    return _segment_penetration_result(
        values, hit_capacity=hit_capacity, failure_state=failure_state
    )


def rayd_segment_penetration_forward_tape(
    scene_resource: object,
    origins: torch.Tensor,
    targets: torch.Tensor,
    input_active: torch.Tensor | None,
    *,
    input_active_any: bool,
    hit_capacity: int,
    policy: SegmentPenetrationPolicy,
    scene_diagonal: float,
    failure_state: CapacityFailureState,
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
    values = _required_native_op("rayd_segment_penetration_forward_tape")(*args)
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
    scene_resource: object,
    origins: torch.Tensor,
    targets: torch.Tensor,
    input_active: torch.Tensor | None,
    *,
    input_active_any: bool,
    hit_capacity: int,
    policy: SegmentPenetrationPolicy,
    scene_diagonal: float,
    failure_state: CapacityFailureState,
    tape: SegmentPenetrationTapeResult,
    grad_distance: torch.Tensor | None = None,
    grad_direction: torch.Tensor | None = None,
    grad_t: torch.Tensor | None = None,
    grad_position: torch.Tensor | None = None,
    grad_normal: torch.Tensor | None = None,
    grad_geometric_normal: torch.Tensor | None = None,
    need_grad_vertices: bool = False,
    need_grad_origins: bool = False,
    need_grad_targets: bool = False,
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
    values = _required_native_op("rayd_segment_penetration_backward")(
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
    scene_resource: object,
    origins: torch.Tensor,
    targets: torch.Tensor,
    input_active: torch.Tensor | None,
    *,
    input_active_any: bool,
    hit_capacity: int,
    policy: SegmentPenetrationPolicy,
    scene_diagonal: float,
    failure_state: CapacityFailureState,
    tape: SegmentPenetrationTapeResult,
    tangent_vertices: torch.Tensor | None = None,
    tangent_origins: torch.Tensor | None = None,
    tangent_targets: torch.Tensor | None = None,
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
    values = _required_native_op("rayd_segment_penetration_jvp")(
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
    scene_resource: object,
    start: torch.Tensor,
    end: torch.Tensor,
    active: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor(
        "start", start, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("end", end, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if active is not None:
        validate_cuda_tensor("active", active, dtype=torch.bool, ndim=1)
    out = _required_native_op("rayd_visibility_forward")(
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


def rayd_intersect_forward(
    scene_resource: object,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor | None,
    *,
    flags: int = 7,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "ray_o", ray_o, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "ray_d", ray_d, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("ray_tmax", ray_tmax, dtype=torch.float32, ndim=1)
    if ray_d.shape != ray_o.shape:
        raise ValueError("ray_d must match ray_o")
    if ray_tmax.shape not in ((0,), (ray_o.shape[0],)):
        raise ValueError("ray_tmax must be empty or match ray_o")
    if active is not None:
        validate_cuda_tensor("active", active, dtype=torch.bool, ndim=1)
        if active.shape not in ((0,), (ray_o.shape[0],)):
            raise ValueError("active must be empty or match ray_o")
        if active.get_device() != ray_o.get_device():
            raise ValueError("active must share ray_o device")
    if (
        ray_d.get_device() != ray_o.get_device()
        or ray_tmax.get_device() != ray_o.get_device()
    ):
        raise ValueError("intersection tensors must share one CUDA device")
    if flags < 0:
        raise ValueError("flags must be non-negative")
    out = _required_native_op("rayd_intersect_forward")(
        _rayd_scene_resource(scene_resource),
        ray_o,
        ray_d,
        ray_tmax,
        active,
        int(flags),
    )
    if not isinstance(out, (tuple, list)) or len(out) != len(_BDPT_INTERSECTION_FIELDS):
        raise TypeError("_channel.rayd_intersect_forward must return 10 tensors")
    exported = dict(zip(_BDPT_INTERSECTION_FIELDS, out, strict=True))
    validate_cuda_tensor("t", exported["t"], dtype=torch.float32, ndim=1)
    if exported["t"].shape != (ray_o.shape[0],):
        raise ValueError("_channel.rayd_intersect_forward returned bad t shape")
    for name in ("p", "n", "geo_n", "barycentric"):
        validate_cuda_tensor(
            name, exported[name], dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
    validate_cuda_tensor(
        "uv", exported["uv"], dtype=torch.float32, ndim=2, trailing_shape=(2,)
    )
    for name in ("shape_id", "prim_id", "local_prim_id", "global_prim_id"):
        validate_cuda_tensor(name, exported[name], dtype=torch.int32, ndim=1)
    return exported


def rayd_diffraction_sample_tape_forward(*args: object) -> tuple[torch.Tensor, ...]:
    if not args:
        raise TypeError(
            "rayd_diffraction_sample_tape_forward requires a typed RayD scene resource"
        )
    native_args = (_rayd_scene_resource(args[0]), *args[1:])
    out = _required_native_op("rayd_diffraction_sample_tape_forward")(
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
    out = _required_native_op("rayd_trace_reflections_forward")(
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
    out = _required_native_op("rayd_reflection_epc_paths_forward")(
        *native_args,
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel.rayd_reflection_epc_paths_forward must return a tensor sequence"
        )
    return tuple(out)


def coupled_rd_geometry_forward(*args: object) -> dict[str, torch.Tensor]:
    """Construct reciprocal 1R+1D geometry without evaluating a coefficient.

    The native operation uses image-source edge stationarity, RayD reflection
    EPC, and RayD segment visibility. ``reverse=True`` constructs D->R by
    exchanging endpoints and reversing the interaction sequence. The returned
    dictionary intentionally has no ``path_gain`` or ``field`` entry; coupled
    complex/Jones transport belongs to the unified field phase.
    """

    if not args:
        raise TypeError(
            "coupled_rd_geometry_forward requires a typed RayD scene resource"
        )
    native_args = (_rayd_scene_resource(args[0]), *args[1:])
    out = _required_native_op("coupled_rd_geometry_forward")(
        *native_args,
    )
    if not isinstance(out, dict):
        raise TypeError(
            "_channel.coupled_rd_geometry_forward must return a dict"
        )
    required = {
        "valid": (torch.bool, 1),
        "interaction_type_sequence": (torch.int32, 2),
        "primitive_sequence": (torch.int32, 2),
        "edge_sequence": (torch.int32, 2),
        "face_id": (torch.int32, 1),
        "edge_id": (torch.int32, 1),
        "interaction_positions": (torch.float32, 3),
        "interaction_normals": (torch.float32, 3),
        "reflection_position": (torch.float32, 2),
        "reflection_normal": (torch.float32, 2),
        "edge_position": (torch.float32, 2),
        "edge_direction": (torch.float32, 2),
        "path_length_m": (torch.float32, 1),
        "delay_s": (torch.float32, 1),
    }
    for name, (dtype, ndim) in required.items():
        value = out.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"coupled geometry output {name!r} must be a tensor")
        validate_cuda_tensor(name, value, dtype=dtype, ndim=ndim)
    if "path_gain" in out or "field" in out or "path_field" in out:
        raise ValueError(
            "coupled geometry must not expose placeholder physical coefficients"
        )
    count = int(out["valid"].shape[0])
    if out["interaction_type_sequence"].shape != (count, 2):
        raise ValueError("interaction_type_sequence must have shape (N, 2)")
    if out["primitive_sequence"].shape != (count, 2) or out["edge_sequence"].shape != (
        count,
        2,
    ):
        raise ValueError("coupled primitive/edge sequences must have shape (N, 2)")
    if out["interaction_positions"].shape != (count, 2, 3):
        raise ValueError("interaction_positions must have shape (N, 2, 3)")
    if out["interaction_normals"].shape != (count, 2, 3):
        raise ValueError("interaction_normals must have shape (N, 2, 3)")
    return out


def coupled_dd_geometry_forward(*args: object) -> dict[str, torch.Tensor]:
    """Construct two-edge (double) diffraction geometry without a coefficient.

    The native operation runs an alternating-projection Fermat solve for the
    two-edge Keller point pair (Q1 on e1, Q2 on e2) and three RayD segment
    visibility queries (tx->Q1, Q1->Q2, Q2->rx). Both edge ids are recoverable
    from ``edge_sequence`` (slot 0 = e1, slot 1 = e2); ``primitive_sequence`` is
    fully ``-1`` because a double-diffraction row touches no face. The returned
    dictionary intentionally carries no ``path_gain``/``field`` entry; complex
    transport belongs to the unified field phase.
    """

    if not args:
        raise TypeError(
            "coupled_dd_geometry_forward requires a typed RayD scene resource"
        )
    native_args = (_rayd_scene_resource(args[0]), *args[1:])
    out = _required_native_op("coupled_dd_geometry_forward")(
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
    return out


def rayd_diffraction_paths_order1_forward(*args: object) -> tuple[torch.Tensor, ...]:
    if not args:
        raise TypeError(
            "rayd_diffraction_paths_order1_forward requires a typed RayD scene resource"
        )
    native_args = (_rayd_scene_resource(args[0]), *args[1:])
    out = _required_native_op("rayd_diffraction_paths_order1_forward")(
        *native_args,
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel.rayd_diffraction_paths_order1_forward must return a tensor sequence"
        )
    return tuple(out)


_DIFFRACTION_STATE_CAPACITY = 4_194_304


def diffraction_tx_visible_state_plan(
    scene_resource: object,
    tx: torch.Tensor,
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
) -> torch.Tensor:
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

    active = _required_native_op("diffraction_tx_visible_state_plan")(
        _rayd_scene_resource(scene_resource),
        tx,
        edge_position,
        edge_direction,
        edge_t_min,
        edge_t_max,
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


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def core_diffraction_edge_count(
    *,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_normals: torch.Tensor,
    edge_v0: torch.Tensor,
    edge_v1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    vertical_only: bool,
    vertical_ratio: float,
    boundary_half_plane: bool,
    plane_tol: float,
) -> int:
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
    if faces.shape[0] != face_normals.shape[0]:
        raise ValueError("face_normals must match faces")
    for name, tensor in {"edge_v1": edge_v1, "face0": face0, "face1": face1}.items():
        if tensor.shape != edge_v0.shape:
            raise ValueError(f"{name} must match edge_v0")
    value = _required_native_op("core_diffraction_edge_count")(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        bool(vertical_only),
        float(vertical_ratio),
        bool(boundary_half_plane),
        float(plane_tol),
    )
    if not isinstance(value, int):
        raise TypeError(
            "_channel.core_diffraction_edge_count must return an int"
        )
    return value


def deterministic_normalize_vec3(
    values: torch.Tensor, *, eps: float = 1.0e-6
) -> torch.Tensor:
    validate_cuda_tensor(
        "values", values, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    out = _required_native_op("deterministic_normalize_vec3")(values, float(eps))
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


def deterministic_reflect_points(
    points: torch.Tensor,
    plane_points: torch.Tensor,
    normals: torch.Tensor,
) -> torch.Tensor:
    validate_cuda_tensor(
        "points", points, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "plane_points", plane_points, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "normals", normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if plane_points.shape != points.shape or normals.shape != points.shape:
        raise ValueError("points, plane_points, and normals must have matching shapes")
    out = _required_native_op("deterministic_reflect_points")(
        points, plane_points, normals
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.deterministic_reflect_points must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if out.shape != points.shape:
        raise ValueError(
            "_channel.deterministic_reflect_points returned bad shape"
        )
    return out


def deterministic_face_groups(
    tri_a: torch.Tensor,
    normals: torch.Tensor,
    surface_ids: torch.Tensor,
    *,
    quantization: float,
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
    exported = _required_native_op("deterministic_face_groups")(
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


def deterministic_surface_face_groups(
    surface_ids: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    validate_cuda_tensor("surface_ids", surface_ids, dtype=torch.int64, ndim=1)
    exported = _required_native_op("deterministic_surface_face_groups")(surface_ids)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_surface_face_groups must return a dict"
        )
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
            "_channel.deterministic_surface_face_groups returned unexpected fields"
        )
    face_count = int(surface_ids.shape[0])
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
            "_channel.deterministic_surface_face_groups returned non-int group_count"
        )
    if exported["face_group_id"].shape != (face_count,) or exported[
        "surface_group_id"
    ].shape != (face_count,):
        raise ValueError(
            "_channel.deterministic_surface_face_groups returned bad face group shape"
        )
    if exported["representative_faces"].shape != (group_count,) or exported[
        "surface_group_size"
    ].shape != (group_count,):
        raise ValueError(
            "_channel.deterministic_surface_face_groups returned bad group shape"
        )
    return exported


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
    native = native_extension()
    if native is None or not hasattr(native, "mc_diffraction_edge_geometry"):
        raise RuntimeError(
            "_channel.mc_diffraction_edge_geometry CUDA kernel is required"
        )
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
    native = native_extension()
    if native is None or not hasattr(native, "mc_surface_group_edge_candidates"):
        raise RuntimeError(
            "_channel.mc_surface_group_edge_candidates CUDA kernel is required"
        )
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


def rayd_intersect_backward(
    scene_resource: object,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor | None,
    tape_prim_id: torch.Tensor,
    tape_barycentric: torch.Tensor,
    *,
    grad_t: torch.Tensor | None = None,
    grad_p: torch.Tensor | None = None,
    grad_n: torch.Tensor | None = None,
    grad_geo_n: torch.Tensor | None = None,
    grad_uv: torch.Tensor | None = None,
    grad_barycentric: torch.Tensor | None = None,
    need_grad_vertices: bool = False,
    need_grad_ray_o: bool = False,
    need_grad_ray_d: bool = False,
    need_grad_ray_tmax: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    validate_cuda_tensor(
        "ray_o", ray_o, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "ray_d", ray_d, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("ray_tmax", ray_tmax, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tape_prim_id", tape_prim_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "tape_barycentric", tape_barycentric, dtype=torch.float32, ndim=2
    )
    rows = int(ray_o.shape[0])
    _ad_check_rows("ray_d", ray_d, rows)
    if ray_tmax.shape[0] not in (0, rows):
        raise ValueError("ray_tmax must be empty or match the ray batch size")
    _ad_check_active(active, rows)
    _ad_check_rows("tape_prim_id", tape_prim_id, rows)
    # An empty barycentric tape selects the native width-0 recompute path.
    if tape_barycentric.shape[0] not in (0, rows):
        raise ValueError("tape_barycentric must be empty or match the ray batch size")
    if tape_barycentric.shape[0] and tape_barycentric.shape[1] not in (2, 3):
        raise ValueError("tape_barycentric last dimension must be 2 or 3")
    _ad_check_optional_grad("grad_t", grad_t, ((rows,),))
    _ad_check_optional_grad("grad_p", grad_p, ((rows, 3),))
    _ad_check_optional_grad("grad_n", grad_n, ((rows, 3),))
    _ad_check_optional_grad("grad_geo_n", grad_geo_n, ((rows, 3),))
    _ad_check_optional_grad("grad_uv", grad_uv, ((rows, 2),))
    _ad_check_optional_grad("grad_barycentric", grad_barycentric, ((rows, 3),))
    out = _required_native_op("rayd_intersect_backward")(
        _rayd_scene_resource(scene_resource),
        ray_o,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        grad_t,
        grad_p,
        grad_n,
        grad_geo_n,
        grad_uv,
        grad_barycentric,
        bool(need_grad_vertices),
        bool(need_grad_ray_o),
        bool(need_grad_ray_d),
        bool(need_grad_ray_tmax),
    )
    if not isinstance(out, (tuple, list)) or len(out) != 4:
        raise TypeError(
            "_channel.rayd_intersect_backward must return 4 gradients"
        )
    return tuple(out)


def rayd_intersect_jvp(
    scene_resource: object,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    active: torch.Tensor | None,
    tape_prim_id: torch.Tensor,
    tape_barycentric: torch.Tensor,
    *,
    tangent_vertices: torch.Tensor | None = None,
    tangent_ray_o: torch.Tensor | None = None,
    tangent_ray_d: torch.Tensor | None = None,
    flags: int = _RAYD_RAY_FLAGS_ALL,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor(
        "ray_o", ray_o, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "ray_d", ray_d, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tape_prim_id", tape_prim_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "tape_barycentric", tape_barycentric, dtype=torch.float32, ndim=2
    )
    rows = int(ray_o.shape[0])
    _ad_check_rows("ray_d", ray_d, rows)
    _ad_check_active(active, rows)
    _ad_check_rows("tape_prim_id", tape_prim_id, rows)
    # The native jvp kernel has no width-0 recompute path: the barycentric
    # tape must cover the full ray batch (unlike backward, which accepts an
    # empty tape).
    _ad_check_rows("tape_barycentric", tape_barycentric, rows)
    if rows and tape_barycentric.shape[1] not in (2, 3):
        raise ValueError("tape_barycentric last dimension must be 2 or 3")
    _ad_check_tangent_vec3("tangent_vertices", tangent_vertices, None)
    _ad_check_tangent_vec3("tangent_ray_o", tangent_ray_o, rows)
    _ad_check_tangent_vec3("tangent_ray_d", tangent_ray_d, rows)
    out = _required_native_op("rayd_intersect_jvp")(
        _rayd_scene_resource(scene_resource),
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tangent_vertices,
        tangent_ray_o,
        tangent_ray_d,
        int(flags),
    )
    if not isinstance(out, (tuple, list)) or len(out) != 6:
        raise TypeError("_channel.rayd_intersect_jvp must return 6 tangents")
    return tuple(out)


def rayd_trace_reflections_forward_tape(*args: object) -> tuple[torch.Tensor, ...]:
    if not args:
        raise TypeError(
            "rayd_trace_reflections_forward_tape requires a typed RayD scene resource"
        )
    native_args = (_rayd_scene_resource(args[0]), *args[1:])
    out = _required_native_op("rayd_trace_reflections_forward_tape")(
        *native_args,
    )
    if not isinstance(out, (tuple, list)) or len(out) != 9:
        raise TypeError(
            "_channel.rayd_trace_reflections_forward_tape must return 9 tensors"
        )
    return tuple(out)


def rayd_trace_reflections_backward(
    scene_resource: object,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor | None,
    tape_prim_id: torch.Tensor,
    tape_barycentric: torch.Tensor,
    tape_hit_points: torch.Tensor,
    tape_normals: torch.Tensor,
    image_sources: torch.Tensor,
    *,
    grad_t: torch.Tensor | None = None,
    grad_image_sources: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, ...]:
    validate_cuda_tensor(
        "ray_o", ray_o, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "ray_d", ray_d, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tape_prim_id", tape_prim_id, dtype=torch.int32, ndim=2)
    validate_cuda_tensor(
        "tape_barycentric", tape_barycentric, dtype=torch.float32, ndim=3
    )
    validate_cuda_tensor(
        "tape_hit_points",
        tape_hit_points,
        dtype=torch.float32,
        ndim=3,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "tape_normals", tape_normals, dtype=torch.float32, ndim=3, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "image_sources",
        image_sources,
        dtype=torch.float32,
        ndim=3,
        trailing_shape=(3,),
    )
    validate_cuda_tensor("ray_tmax", ray_tmax, dtype=torch.float32, ndim=1)
    rows = int(ray_o.shape[0])
    _ad_check_rows("ray_d", ray_d, rows)
    if ray_tmax.shape[0] not in (0, rows):
        raise ValueError("ray_tmax must be empty or match the ray batch size")
    _ad_check_active(active, rows)
    _ad_check_rows("tape_prim_id", tape_prim_id, rows)
    bounces = int(tape_prim_id.shape[1])
    if tuple(tape_barycentric.shape[:2]) != (rows, bounces) or tape_barycentric.shape[
        2
    ] not in (2, 3):
        raise ValueError(f"tape_barycentric must have shape ({rows}, {bounces}, 2|3)")
    for name, value in (
        ("tape_hit_points", tape_hit_points),
        ("tape_normals", tape_normals),
        ("image_sources", image_sources),
    ):
        if tuple(value.shape) != (rows, bounces, 3):
            raise ValueError(f"{name} must have shape ({rows}, {bounces}, 3)")
    _ad_check_optional_grad("grad_t", grad_t, ((rows,), (rows, bounces)))
    _ad_check_optional_grad(
        "grad_image_sources", grad_image_sources, ((rows, bounces, 3),)
    )
    out = _required_native_op("rayd_trace_reflections_backward")(
        _rayd_scene_resource(scene_resource),
        ray_o,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        tape_hit_points,
        tape_normals,
        image_sources,
        grad_t,
        grad_image_sources,
    )
    if not isinstance(out, (tuple, list)) or len(out) != 4:
        raise TypeError(
            "_channel.rayd_trace_reflections_backward must return 4 gradients"
        )
    return tuple(out)


def rayd_trace_reflections_jvp(
    scene_resource: object,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    active: torch.Tensor | None,
    tape_prim_id: torch.Tensor,
    tape_barycentric: torch.Tensor,
    tape_hit_points: torch.Tensor,
    tape_normals: torch.Tensor,
    image_sources: torch.Tensor,
    *,
    tangent_vertices: torch.Tensor | None = None,
    tangent_ray_o: torch.Tensor | None = None,
    tangent_ray_d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor(
        "ray_o", ray_o, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "ray_d", ray_d, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tape_prim_id", tape_prim_id, dtype=torch.int32, ndim=2)
    validate_cuda_tensor(
        "tape_barycentric", tape_barycentric, dtype=torch.float32, ndim=3
    )
    validate_cuda_tensor(
        "tape_hit_points",
        tape_hit_points,
        dtype=torch.float32,
        ndim=3,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "tape_normals", tape_normals, dtype=torch.float32, ndim=3, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "image_sources",
        image_sources,
        dtype=torch.float32,
        ndim=3,
        trailing_shape=(3,),
    )
    rows = int(ray_o.shape[0])
    _ad_check_rows("ray_d", ray_d, rows)
    _ad_check_active(active, rows)
    _ad_check_rows("tape_prim_id", tape_prim_id, rows)
    bounces = int(tape_prim_id.shape[1])
    if tuple(tape_barycentric.shape[:2]) != (rows, bounces) or tape_barycentric.shape[
        2
    ] not in (2, 3):
        raise ValueError(f"tape_barycentric must have shape ({rows}, {bounces}, 2|3)")
    for name, value in (
        ("tape_hit_points", tape_hit_points),
        ("tape_normals", tape_normals),
        ("image_sources", image_sources),
    ):
        if tuple(value.shape) != (rows, bounces, 3):
            raise ValueError(f"{name} must have shape ({rows}, {bounces}, 3)")
    _ad_check_tangent_vec3("tangent_vertices", tangent_vertices, None)
    _ad_check_tangent_vec3("tangent_ray_o", tangent_ray_o, rows)
    _ad_check_tangent_vec3("tangent_ray_d", tangent_ray_d, rows)
    out = _required_native_op("rayd_trace_reflections_jvp")(
        _rayd_scene_resource(scene_resource),
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tape_hit_points,
        tape_normals,
        tangent_vertices,
        tangent_ray_o,
        tangent_ray_d,
        image_sources,
    )
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise TypeError(
            "_channel.rayd_trace_reflections_jvp must return 2 tangents"
        )
    return tuple(out)


class _RaydIntersectAdFunction(torch.autograd.Function):
    """Fixed-winner differentiable RayD intersect through the direct typed C++ boundary.

    Inputs: (scene_resource, vertices, ray_o, ray_d, ray_tmax, active).
    ``vertices`` must be the scene's global vertex table (single-structure
    scenes in AD-A0); the forward reads geometry from the native scene and
    the tensor only routes vertex gradients/tangents.
    """

    @staticmethod
    def forward(scene_resource, vertices, ray_o, ray_d, ray_tmax, active):
        out = _required_native_op("rayd_intersect_forward")(
            scene_resource,
            ray_o,
            ray_d,
            ray_tmax,
            active,
            _RAYD_RAY_FLAGS_ALL,
        )
        return tuple(out)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        scene_resource, vertices, ray_o, ray_d, ray_tmax, active = inputs
        barycentric = output[5]
        shape_id, prim_id, local_prim_id, global_prim_id = output[6:10]
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        ray_o = torch.autograd.forward_ad.unpack_dual(ray_o).primal
        ray_d = torch.autograd.forward_ad.unpack_dual(ray_d).primal
        ray_tmax = torch.autograd.forward_ad.unpack_dual(ray_tmax).primal
        active_ctx = _ad_active_ctx(active, ray_o)
        ctx.scene_resource = scene_resource
        ctx.vertices_shape = tuple(vertices.shape)
        ctx.save_for_backward(
            ray_o, ray_d, ray_tmax, active_ctx, global_prim_id, barycentric
        )
        ctx.save_for_forward(ray_o, ray_d, active_ctx, global_prim_id, barycentric)
        ctx.mark_non_differentiable(shape_id, prim_id, local_prim_id, global_prim_id)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *grad_outputs):
        none_grads = (None, None, None, None, None, None)
        if all(value is None for value in grad_outputs[:6]):
            return none_grads
        (
            ray_o,
            ray_d,
            ray_tmax,
            active_ctx,
            tape_prim_id,
            tape_barycentric,
        ) = ctx.saved_tensors
        need_grad_vertices = bool(ctx.needs_input_grad[1])
        need_grad_ray_o = bool(ctx.needs_input_grad[2])
        need_grad_ray_d = bool(ctx.needs_input_grad[3])
        need_grad_ray_tmax = bool(ctx.needs_input_grad[4])
        if not (
            need_grad_vertices
            or need_grad_ray_o
            or need_grad_ray_d
            or need_grad_ray_tmax
        ):
            return none_grads
        grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax = rayd_intersect_backward(
            ctx.scene_resource,
            ray_o,
            ray_d,
            ray_tmax,
            active_ctx,
            tape_prim_id,
            tape_barycentric,
            grad_t=grad_outputs[0],
            grad_p=grad_outputs[1],
            grad_n=grad_outputs[2],
            grad_geo_n=grad_outputs[3],
            grad_uv=grad_outputs[4],
            grad_barycentric=grad_outputs[5],
            need_grad_vertices=need_grad_vertices,
            need_grad_ray_o=need_grad_ray_o,
            need_grad_ray_d=need_grad_ray_d,
            need_grad_ray_tmax=need_grad_ray_tmax,
        )
        if need_grad_vertices and tuple(grad_vertices.shape) != ctx.vertices_shape:
            raise RuntimeError(
                "rayd_intersect_ad vertices must be the scene global vertex table"
            )
        return (
            None,
            grad_vertices if need_grad_vertices else None,
            grad_ray_o if need_grad_ray_o else None,
            grad_ray_d if need_grad_ray_d else None,
            grad_ray_tmax if need_grad_ray_tmax else None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        _grad_handle,
        grad_vertices,
        grad_ray_o,
        grad_ray_d,
        _grad_tmax,
        _grad_active,
    ):
        ray_o, ray_d, active_ctx, tape_prim_id, tape_barycentric = ctx.saved_tensors
        with disable_functorch():
            values = rayd_intersect_jvp(
                ctx.scene_resource,
                _ad_native_tensor(ray_o),
                _ad_native_tensor(ray_d),
                _ad_native_tensor(active_ctx),
                _ad_native_tensor(tape_prim_id),
                _ad_native_tensor(tape_barycentric),
                tangent_vertices=_ad_checked_tangent(
                    "rayd_intersect_ad tangent_vertices",
                    _ad_native_tangent_or_none(grad_vertices),
                    ctx.vertices_shape,
                ),
                tangent_ray_o=_ad_checked_tangent(
                    "rayd_intersect_ad tangent_ray_o",
                    _ad_native_tangent_or_none(grad_ray_o),
                    tuple(ray_o.shape),
                ),
                tangent_ray_d=_ad_checked_tangent(
                    "rayd_intersect_ad tangent_ray_d",
                    _ad_native_tangent_or_none(grad_ray_d),
                    tuple(ray_d.shape),
                ),
                flags=_RAYD_RAY_FLAGS_ALL,
            )
        return (*values, None, None, None, None)


def rayd_intersect_ad(
    scene_resource: object,
    vertices: torch.Tensor,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable RayD intersect under the fixed-winner contract.

    Returns the same fields as :func:`rayd_intersect_forward`; ``t``/``p``/
    ``n``/``geo_n``/``uv``/``barycentric`` participate in reverse- and
    forward-mode torch AD with respect to ``vertices``, ``ray_o`` and
    ``ray_d``. Winner ids stay detached.
    """

    values = _RaydIntersectAdFunction.apply(
        _rayd_scene_resource(scene_resource), vertices, ray_o, ray_d, ray_tmax, active
    )
    return dict(zip(_BDPT_INTERSECTION_FIELDS, values, strict=True))


class _RaydTraceReflectionsAdFunction(torch.autograd.Function):
    """Fixed-winner differentiable RayD reflection chain through the direct typed C++ boundary."""

    @staticmethod
    def forward(scene_resource, vertices, ray_o, ray_d, ray_tmax, active, max_bounces):
        out = _required_native_op("rayd_trace_reflections_forward_tape")(
            scene_resource,
            ray_o,
            ray_d,
            ray_tmax,
            active,
            int(max_bounces),
        )
        (
            valid,
            t,
            image_sources,
            prim_ids,
            _tape_prim_id,
            tape_barycentric,
            tape_hit_points,
            tape_normals,
            _active_ctx,
        ) = out
        return (
            valid,
            t,
            image_sources,
            prim_ids,
            tape_barycentric,
            tape_hit_points,
            tape_normals,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        scene_resource, vertices, ray_o, ray_d, ray_tmax, active, _max_bounces = inputs
        (
            valid,
            _t,
            image_sources,
            prim_ids,
            tape_barycentric,
            tape_hit_points,
            tape_normals,
        ) = output
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        ray_o = torch.autograd.forward_ad.unpack_dual(ray_o).primal
        ray_d = torch.autograd.forward_ad.unpack_dual(ray_d).primal
        ray_tmax = torch.autograd.forward_ad.unpack_dual(ray_tmax).primal
        active_ctx = _ad_active_ctx(active, ray_o)
        ctx.scene_resource = scene_resource
        ctx.vertices_shape = tuple(vertices.shape)
        ctx.save_for_backward(
            ray_o,
            ray_d,
            ray_tmax,
            active_ctx,
            prim_ids,
            tape_barycentric,
            tape_hit_points,
            tape_normals,
            image_sources,
        )
        ctx.save_for_forward(
            ray_o,
            ray_d,
            active_ctx,
            prim_ids,
            tape_barycentric,
            tape_hit_points,
            tape_normals,
            image_sources,
        )
        ctx.mark_non_differentiable(
            valid, prim_ids, tape_barycentric, tape_hit_points, tape_normals
        )

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *grad_outputs):
        none_grads = (None, None, None, None, None, None, None)
        grad_t = grad_outputs[1]
        grad_image_sources = grad_outputs[2]
        if grad_t is None and grad_image_sources is None:
            return none_grads
        (
            ray_o,
            ray_d,
            ray_tmax,
            active_ctx,
            tape_prim_id,
            tape_barycentric,
            tape_hit_points,
            tape_normals,
            image_sources,
        ) = ctx.saved_tensors
        need_grad_vertices = bool(ctx.needs_input_grad[1])
        need_grad_ray_o = bool(ctx.needs_input_grad[2])
        need_grad_ray_d = bool(ctx.needs_input_grad[3])
        need_grad_ray_tmax = bool(ctx.needs_input_grad[4])
        if not (
            need_grad_vertices
            or need_grad_ray_o
            or need_grad_ray_d
            or need_grad_ray_tmax
        ):
            return none_grads
        grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax = (
            rayd_trace_reflections_backward(
                ctx.scene_resource,
                ray_o,
                ray_d,
                ray_tmax,
                active_ctx,
                tape_prim_id,
                tape_barycentric,
                tape_hit_points,
                tape_normals,
                image_sources,
                grad_t=grad_t,
                grad_image_sources=grad_image_sources,
            )
        )
        if need_grad_vertices and tuple(grad_vertices.shape) != ctx.vertices_shape:
            raise RuntimeError(
                "rayd_trace_reflections_ad vertices must be the scene global"
                " vertex table"
            )
        return (
            None,
            grad_vertices if need_grad_vertices else None,
            grad_ray_o if need_grad_ray_o else None,
            grad_ray_d if need_grad_ray_d else None,
            grad_ray_tmax if need_grad_ray_tmax else None,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        _grad_handle,
        grad_vertices,
        grad_ray_o,
        grad_ray_d,
        _grad_tmax,
        _grad_active,
        _grad_max_bounces,
    ):
        (
            ray_o,
            ray_d,
            active_ctx,
            tape_prim_id,
            tape_barycentric,
            tape_hit_points,
            tape_normals,
            image_sources,
        ) = ctx.saved_tensors
        with disable_functorch():
            tangent_t, tangent_image_sources = rayd_trace_reflections_jvp(
                ctx.scene_resource,
                _ad_native_tensor(ray_o),
                _ad_native_tensor(ray_d),
                _ad_native_tensor(active_ctx),
                _ad_native_tensor(tape_prim_id),
                _ad_native_tensor(tape_barycentric),
                _ad_native_tensor(tape_hit_points),
                _ad_native_tensor(tape_normals),
                _ad_native_tensor(image_sources),
                tangent_vertices=_ad_checked_tangent(
                    "rayd_trace_reflections_ad tangent_vertices",
                    _ad_native_tangent_or_none(grad_vertices),
                    ctx.vertices_shape,
                ),
                tangent_ray_o=_ad_checked_tangent(
                    "rayd_trace_reflections_ad tangent_ray_o",
                    _ad_native_tangent_or_none(grad_ray_o),
                    tuple(ray_o.shape),
                ),
                tangent_ray_d=_ad_checked_tangent(
                    "rayd_trace_reflections_ad tangent_ray_d",
                    _ad_native_tangent_or_none(grad_ray_d),
                    tuple(ray_d.shape),
                ),
            )
        return (None, tangent_t, tangent_image_sources, None, None, None, None)


_RAYD_TRACE_REFLECTIONS_AD_FIELDS = (
    "valid",
    "t",
    "image_sources",
    "prim_ids",
    "tape_barycentric",
    "tape_hit_points",
    "tape_normals",
)


def rayd_trace_reflections_ad(
    scene_resource: object,
    vertices: torch.Tensor,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor | None,
    max_bounces: int,
) -> dict[str, torch.Tensor]:
    """Differentiable RayD reflection chain under the fixed-winner contract.

    ``t`` and ``image_sources`` participate in reverse- and forward-mode
    torch AD with respect to ``vertices``, ``ray_o`` and ``ray_d``; the
    reflection chain (prim ids and tape tensors) stays detached.
    """

    values = _RaydTraceReflectionsAdFunction.apply(
        _rayd_scene_resource(scene_resource),
        vertices,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        int(max_bounces),
    )
    return dict(zip(_RAYD_TRACE_REFLECTIONS_AD_FIELDS, values, strict=True))


def _epc_paths_frozen_winner_checks(
    source: torch.Tensor,
    receiver: torch.Tensor,
    sequence: torch.Tensor,
    plane_points: torch.Tensor,
    plane_normals: torch.Tensor,
    valid: torch.Tensor,
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
    scene_resource: object,
    source: torch.Tensor,
    receiver: torch.Tensor,
    sequence: torch.Tensor,
    plane_points: torch.Tensor,
    plane_normals: torch.Tensor,
    valid: torch.Tensor,
    bounce_count: torch.Tensor,
    *,
    grad_points: torch.Tensor | None = None,
    grad_normals: torch.Tensor | None = None,
    grad_path_length: torch.Tensor | None = None,
    need_grad_vertices: bool = False,
    need_grad_source: bool = False,
    need_grad_receiver: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    rows, bounces = _epc_paths_frozen_winner_checks(
        source, receiver, sequence, plane_points, plane_normals, valid, bounce_count
    )
    _ad_check_optional_grad("grad_points", grad_points, ((rows, bounces, 3),))
    _ad_check_optional_grad("grad_normals", grad_normals, ((rows, bounces, 3),))
    _ad_check_optional_grad("grad_path_length", grad_path_length, ((rows,),))
    out = _required_native_op("rayd_reflection_epc_paths_backward")(
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
    scene_resource: object,
    source: torch.Tensor,
    receiver: torch.Tensor,
    sequence: torch.Tensor,
    plane_points: torch.Tensor,
    plane_normals: torch.Tensor,
    valid: torch.Tensor,
    bounce_count: torch.Tensor,
    *,
    tangent_vertices: torch.Tensor | None = None,
    tangent_source: torch.Tensor | None = None,
    tangent_receiver: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    rows, _bounces = _epc_paths_frozen_winner_checks(
        source, receiver, sequence, plane_points, plane_normals, valid, bounce_count
    )
    _ad_check_tangent_vec3("tangent_vertices", tangent_vertices, None)
    _ad_check_tangent_vec3("tangent_source", tangent_source, rows)
    _ad_check_tangent_vec3("tangent_receiver", tangent_receiver, rows)
    out = _required_native_op("rayd_reflection_epc_paths_jvp")(
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
        scene_resource,
        vertices,
        source,
        receiver,
        sequence,
        plane_points,
        plane_normals,
        surface_group_id,
        surface_group_size,
        surface_group_members,
        max_bounces,
        visibility_ignore_mode,
    ):
        out = _required_native_op("rayd_reflection_epc_paths_forward")(
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
        )
        return tuple(out)

    @staticmethod
    def setup_context(ctx, inputs, output):
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
    def backward(ctx, *grad_outputs):
        none_grads = (None,) * 12
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
        need_grad_vertices = bool(ctx.needs_input_grad[1])
        need_grad_source = bool(ctx.needs_input_grad[2])
        need_grad_receiver = bool(ctx.needs_input_grad[3])
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
        if need_grad_vertices and tuple(grad_vertices.shape) != ctx.vertices_shape:
            raise RuntimeError(
                "rayd_reflection_epc_paths_ad vertices must be the scene"
                " global vertex table"
            )
        return (
            None,
            grad_vertices if need_grad_vertices else None,
            grad_source if need_grad_source else None,
            grad_receiver if need_grad_receiver else None,
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
        _grad_handle,
        grad_vertices,
        grad_source,
        grad_receiver,
        *_frozen_tangents,
    ):
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
    scene_resource: object,
    vertices: torch.Tensor,
    source: torch.Tensor,
    receiver: torch.Tensor,
    sequence: torch.Tensor,
    plane_points: torch.Tensor,
    plane_normals: torch.Tensor,
    surface_group_id: torch.Tensor,
    surface_group_size: torch.Tensor,
    surface_group_members: torch.Tensor,
    max_bounces: int,
    visibility_ignore_mode: int,
) -> dict[str, torch.Tensor]:
    """Differentiable RayD reflection EPC paths under the fixed-winner contract.

    ``hit_positions``, ``normals`` and ``path_length`` participate in
    reverse- and forward-mode torch AD with respect to ``vertices``,
    ``source`` and ``receiver``. ``sequence`` is the frozen winner face
    sequence and ``plane_points`` / ``plane_normals`` are its detached plane
    arrays (anchor + unit face normal, gathered per bounce); ``valid`` and
    the resolved ids stay non-differentiable.
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
    )
    return dict(zip(_RAYD_REFLECTION_EPC_PATHS_AD_FIELDS, values, strict=True))


def rayd_scene_face_normals_backward(
    scene_resource: object, grad_face_normals: torch.Tensor
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
    out = _required_native_op("rayd_scene_face_normals_backward")(
        _rayd_scene_resource(scene_resource),
        grad_face_normals,
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.rayd_scene_face_normals_backward must return a tensor"
        )
    return out


def rayd_scene_face_normals_jvp(
    scene_resource: object, tangent_vertices: torch.Tensor
) -> torch.Tensor:
    _ad_check_tangent_vec3("tangent_vertices", tangent_vertices, None)
    if tangent_vertices is None:
        raise ValueError("tangent_vertices is required")
    out = _required_native_op("rayd_scene_face_normals_jvp")(
        _rayd_scene_resource(scene_resource),
        tangent_vertices,
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.rayd_scene_face_normals_jvp must return a tensor"
        )
    return out


class _RaydFaceNormalsAdFunction(torch.autograd.Function):
    """Scene unit face-normal table with a graph to the vertex table.

    Forward normalizes the native face-normal export with the same kernel the
    reflection discovery uses; backward/jvp call RayD's face-normal table
    companions (the adjoint/tangent of normalize(cross(v1 - v0, v2 - v0))
    over the scene's global vertex and face tables).
    """

    @staticmethod
    def forward(scene_resource, vertices, raw_face_normals):
        return deterministic_normalize_vec3(raw_face_normals, eps=1.0e-6)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        scene_resource, vertices, _raw_face_normals = inputs
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        ctx.scene_resource = scene_resource
        ctx.vertices_shape = tuple(vertices.shape)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_face_normals):
        if ctx.needs_input_grad[2]:
            raise RuntimeError(
                "rayd_face_normals_ad differentiates the vertex table only;"
                " the raw face-normal export is a detached scene record"
            )
        if grad_face_normals is None or not ctx.needs_input_grad[1]:
            return (None, None, None)
        grad_vertices = rayd_scene_face_normals_backward(ctx.scene_resource, grad_face_normals)
        if tuple(grad_vertices.shape) != ctx.vertices_shape:
            raise RuntimeError(
                "rayd_face_normals_ad vertices must be the scene global vertex table"
            )
        return (None, grad_vertices, None)

    @staticmethod
    def jvp(ctx, _grad_handle, grad_vertices, _grad_raw_face_normals):
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
    scene_resource: object, vertices: torch.Tensor, raw_face_normals: torch.Tensor
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
    values: tuple[torch.Tensor, ...],
    *,
    hit_capacity: int,
    failure_state: object,
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


class _RaydSegmentPenetrationAdFunction(torch.autograd.Function):
    """Fixed-winner segment geometry with native RayD VJP and JVP companions."""

    @staticmethod
    def forward(
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
    ):
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
    def setup_context(ctx, inputs, output):
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
    def _saved_request(ctx):
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
    def backward(ctx, *grad_outputs):
        none_grads = (None,) * 10
        continuous_grads = grad_outputs[4:10]
        if not any(value is not None for value in continuous_grads):
            return none_grads
        need_grad_vertices = bool(ctx.needs_input_grad[1])
        need_grad_origins = bool(ctx.needs_input_grad[2])
        need_grad_targets = bool(ctx.needs_input_grad[3])
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
        return (
            None,
            gradients.grad_vertices if need_grad_vertices else None,
            gradients.grad_origins if need_grad_origins else None,
            gradients.grad_targets if need_grad_targets else None,
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
        _grad_handle,
        grad_vertices,
        grad_origins,
        grad_targets,
        *_frozen_tangents,
    ):
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
    scene_resource: object,
    vertices: torch.Tensor,
    origins: torch.Tensor,
    targets: torch.Tensor,
    input_active: torch.Tensor | None,
    *,
    input_active_any: bool,
    hit_capacity: int,
    policy: SegmentPenetrationPolicy,
    scene_diagonal: float,
    failure_state: object,
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
