from __future__ import annotations

import math

import torch

from witwin.channel.propagation.models.penetration import (
    SegmentPenetrationBackwardResult,
    SegmentPenetrationJvpResult,
    SegmentPenetrationPolicy,
    SegmentPenetrationResult,
    SegmentPenetrationTapeResult,
)
from witwin.channel.runtime.autograd_contracts import (
    _ad_check_optional_grad,
    _ad_check_tangent_vec3,
)
from witwin.channel.runtime.capacity import (
    CapacityFailureBit,
    CapacityFailureState,
    require_capacity_failure_state,
)
from witwin.channel.runtime.symbols import required_symbol as _required_native_op
from witwin.channel.runtime.tensor_contracts import validate_cuda_tensor
from witwin.channel.runtime.native_resources import _rayd_scene_resource


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
            "_channel_native.rayd_segment_penetration_forward must return 11 tensors"
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
            "_channel_native.rayd_segment_penetration_forward_tape must return 17 tensors"
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
            "_channel_native.rayd_segment_penetration_backward must return 3 gradients"
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
            "_channel_native.rayd_segment_penetration_jvp must return 6 tangents"
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
            "_channel_native.rayd_visibility_forward must return a tensor sequence"
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
        raise TypeError("_channel_native.rayd_intersect_forward must return 10 tensors")
    exported = dict(zip(_BDPT_INTERSECTION_FIELDS, out, strict=True))
    validate_cuda_tensor("t", exported["t"], dtype=torch.float32, ndim=1)
    if exported["t"].shape != (ray_o.shape[0],):
        raise ValueError("_channel_native.rayd_intersect_forward returned bad t shape")
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
            "_channel_native.rayd_diffraction_sample_tape_forward must return a tensor sequence"
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
            "_channel_native.rayd_trace_reflections_forward must return a tensor sequence"
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
            "_channel_native.rayd_reflection_epc_paths_forward must return a tensor sequence"
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
            "_channel_native.coupled_rd_geometry_forward must return a dict"
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
            "_channel_native.coupled_dd_geometry_forward must return a dict"
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
            "_channel_native.rayd_diffraction_paths_order1_forward must return a tensor sequence"
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
            "_channel_native.diffraction_tx_visible_state_plan must return a tensor"
        )
    validate_cuda_tensor("active", active, dtype=torch.bool, ndim=1)
    if active.shape != (state_count,):
        raise ValueError(
            "_channel_native.diffraction_tx_visible_state_plan returned bad shape"
        )
    if active.get_device() != tx.get_device():
        raise ValueError(
            "_channel_native.diffraction_tx_visible_state_plan returned wrong device"
        )
    return active


__all__ = [
    "diffraction_tx_visible_state_plan",
    "rayd_diffraction_sample_tape_forward",
    "coupled_dd_geometry_forward",
    "coupled_rd_geometry_forward",
    "rayd_diffraction_paths_order1_forward",
    "rayd_intersect_forward",
    "rayd_reflection_epc_paths_forward",
    "rayd_trace_reflections_forward",
    "rayd_visibility_forward",
]
