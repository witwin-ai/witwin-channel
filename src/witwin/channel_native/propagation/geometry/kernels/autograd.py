from __future__ import annotations

import torch

from witwin.channel_native.runtime import torch_compat
from witwin.channel_native.runtime.autograd_contracts import (
    _ad_active_ctx,
    _ad_check_active,
    _ad_check_optional_grad,
    _ad_check_rows,
    _ad_check_tangent_vec3,
    _ad_checked_tangent,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
)
from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor
from witwin.channel_native.runtime.native_handles import _raydn_scene_handle_id

from .bridge import _BDPT_INTERSECTION_FIELDS
from .primitives import deterministic_normalize_vec3


_RAYDN_RAY_FLAGS_ALL = 0x01 | 0x02 | 0x04


def raydn_intersect_backward(
    handle: object,
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
    out = _required_native_op("raydn_intersect_backward")(
        _raydn_scene_handle_id(handle),
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
            "_channel_native.raydn_intersect_backward must return 4 gradients"
        )
    return tuple(out)


def raydn_intersect_jvp(
    handle: object,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    active: torch.Tensor | None,
    tape_prim_id: torch.Tensor,
    tape_barycentric: torch.Tensor,
    *,
    tangent_vertices: torch.Tensor | None = None,
    tangent_ray_o: torch.Tensor | None = None,
    tangent_ray_d: torch.Tensor | None = None,
    flags: int = _RAYDN_RAY_FLAGS_ALL,
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
    out = _required_native_op("raydn_intersect_jvp")(
        _raydn_scene_handle_id(handle),
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
        raise TypeError("_channel_native.raydn_intersect_jvp must return 6 tangents")
    return tuple(out)


def raydn_trace_reflections_forward_tape(*args: object) -> tuple[torch.Tensor, ...]:
    if not args:
        raise TypeError(
            "raydn_trace_reflections_forward_tape requires a RayDN scene handle"
        )
    native_args = (_raydn_scene_handle_id(args[0]), *args[1:])
    out = _required_native_op("raydn_trace_reflections_forward_tape")(
        *native_args,
    )
    if not isinstance(out, (tuple, list)) or len(out) != 9:
        raise TypeError(
            "_channel_native.raydn_trace_reflections_forward_tape must return 9 tensors"
        )
    return tuple(out)


def raydn_trace_reflections_backward(
    handle: object,
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
    out = _required_native_op("raydn_trace_reflections_backward")(
        _raydn_scene_handle_id(handle),
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
            "_channel_native.raydn_trace_reflections_backward must return 4 gradients"
        )
    return tuple(out)


def raydn_trace_reflections_jvp(
    handle: object,
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
    out = _required_native_op("raydn_trace_reflections_jvp")(
        _raydn_scene_handle_id(handle),
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
            "_channel_native.raydn_trace_reflections_jvp must return 2 tangents"
        )
    return tuple(out)


class _RaydnIntersectAdFunction(torch.autograd.Function):
    """Fixed-winner differentiable RayDN intersect over the C bridge.

    Inputs: (scene_handle, vertices, ray_o, ray_d, ray_tmax, active).
    ``vertices`` must be the scene's global vertex table (single-structure
    scenes in AD-A0); the forward reads geometry from the native scene and
    the tensor only routes vertex gradients/tangents.
    """

    @staticmethod
    def forward(scene_handle, vertices, ray_o, ray_d, ray_tmax, active):
        out = _required_native_op("bdpt_intersect_forward")(
            int(scene_handle),
            ray_o,
            ray_d,
            ray_tmax,
            active,
            _RAYDN_RAY_FLAGS_ALL,
        )
        return tuple(out)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        scene_handle, vertices, ray_o, ray_d, ray_tmax, active = inputs
        barycentric = output[5]
        shape_id, prim_id, local_prim_id, global_prim_id = output[6:10]
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        ray_o = torch.autograd.forward_ad.unpack_dual(ray_o).primal
        ray_d = torch.autograd.forward_ad.unpack_dual(ray_d).primal
        ray_tmax = torch.autograd.forward_ad.unpack_dual(ray_tmax).primal
        active_ctx = _ad_active_ctx(active, ray_o)
        ctx.scene = int(scene_handle)
        ctx.vertices_shape = tuple(vertices.shape)
        ctx.save_for_backward(
            ray_o, ray_d, ray_tmax, active_ctx, global_prim_id, barycentric
        )
        ctx.save_for_forward(ray_o, ray_d, active_ctx, global_prim_id, barycentric)
        ctx.mark_non_differentiable(shape_id, prim_id, local_prim_id, global_prim_id)

    @staticmethod
    @torch.autograd.function.once_differentiable
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
        grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax = raydn_intersect_backward(
            ctx.scene,
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
                "raydn_intersect_ad vertices must be the scene global vertex table"
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
        with torch_compat.disable_functorch():
            values = raydn_intersect_jvp(
                ctx.scene,
                _ad_native_tensor(ray_o),
                _ad_native_tensor(ray_d),
                _ad_native_tensor(active_ctx),
                _ad_native_tensor(tape_prim_id),
                _ad_native_tensor(tape_barycentric),
                tangent_vertices=_ad_checked_tangent(
                    "raydn_intersect_ad tangent_vertices",
                    _ad_native_tangent_or_none(grad_vertices),
                    ctx.vertices_shape,
                ),
                tangent_ray_o=_ad_checked_tangent(
                    "raydn_intersect_ad tangent_ray_o",
                    _ad_native_tangent_or_none(grad_ray_o),
                    tuple(ray_o.shape),
                ),
                tangent_ray_d=_ad_checked_tangent(
                    "raydn_intersect_ad tangent_ray_d",
                    _ad_native_tangent_or_none(grad_ray_d),
                    tuple(ray_d.shape),
                ),
                flags=_RAYDN_RAY_FLAGS_ALL,
            )
        return (*values, None, None, None, None)


def raydn_intersect_ad(
    handle: object,
    vertices: torch.Tensor,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable RayDN intersect under the fixed-winner contract.

    Returns the same fields as :func:`bdpt_intersect_forward`; ``t``/``p``/
    ``n``/``geo_n``/``uv``/``barycentric`` participate in reverse- and
    forward-mode torch AD with respect to ``vertices``, ``ray_o`` and
    ``ray_d``. Winner ids stay detached.
    """

    values = _RaydnIntersectAdFunction.apply(
        _raydn_scene_handle_id(handle), vertices, ray_o, ray_d, ray_tmax, active
    )
    return dict(zip(_BDPT_INTERSECTION_FIELDS, values, strict=True))


class _RaydnTraceReflectionsAdFunction(torch.autograd.Function):
    """Fixed-winner differentiable RayDN reflection chain over the C bridge."""

    @staticmethod
    def forward(scene_handle, vertices, ray_o, ray_d, ray_tmax, active, max_bounces):
        out = _required_native_op("raydn_trace_reflections_forward_tape")(
            int(scene_handle),
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
        scene_handle, vertices, ray_o, ray_d, ray_tmax, active, _max_bounces = inputs
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
        ctx.scene = int(scene_handle)
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
    @torch.autograd.function.once_differentiable
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
            raydn_trace_reflections_backward(
                ctx.scene,
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
                "raydn_trace_reflections_ad vertices must be the scene global"
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
        with torch_compat.disable_functorch():
            tangent_t, tangent_image_sources = raydn_trace_reflections_jvp(
                ctx.scene,
                _ad_native_tensor(ray_o),
                _ad_native_tensor(ray_d),
                _ad_native_tensor(active_ctx),
                _ad_native_tensor(tape_prim_id),
                _ad_native_tensor(tape_barycentric),
                _ad_native_tensor(tape_hit_points),
                _ad_native_tensor(tape_normals),
                _ad_native_tensor(image_sources),
                tangent_vertices=_ad_checked_tangent(
                    "raydn_trace_reflections_ad tangent_vertices",
                    _ad_native_tangent_or_none(grad_vertices),
                    ctx.vertices_shape,
                ),
                tangent_ray_o=_ad_checked_tangent(
                    "raydn_trace_reflections_ad tangent_ray_o",
                    _ad_native_tangent_or_none(grad_ray_o),
                    tuple(ray_o.shape),
                ),
                tangent_ray_d=_ad_checked_tangent(
                    "raydn_trace_reflections_ad tangent_ray_d",
                    _ad_native_tangent_or_none(grad_ray_d),
                    tuple(ray_d.shape),
                ),
            )
        return (None, tangent_t, tangent_image_sources, None, None, None, None)


_RAYDN_TRACE_REFLECTIONS_AD_FIELDS = (
    "valid",
    "t",
    "image_sources",
    "prim_ids",
    "tape_barycentric",
    "tape_hit_points",
    "tape_normals",
)


def raydn_trace_reflections_ad(
    handle: object,
    vertices: torch.Tensor,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor | None,
    max_bounces: int,
) -> dict[str, torch.Tensor]:
    """Differentiable RayDN reflection chain under the fixed-winner contract.

    ``t`` and ``image_sources`` participate in reverse- and forward-mode
    torch AD with respect to ``vertices``, ``ray_o`` and ``ray_d``; the
    reflection chain (prim ids and tape tensors) stays detached.
    """

    values = _RaydnTraceReflectionsAdFunction.apply(
        _raydn_scene_handle_id(handle),
        vertices,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        int(max_bounces),
    )
    return dict(zip(_RAYDN_TRACE_REFLECTIONS_AD_FIELDS, values, strict=True))


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


def raydn_reflection_epc_paths_backward(
    handle: object,
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
    out = _required_native_op("raydn_reflection_epc_paths_backward")(
        _raydn_scene_handle_id(handle),
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
            "_channel_native.raydn_reflection_epc_paths_backward must return"
            " 3 gradients"
        )
    return tuple(out)


def raydn_reflection_epc_paths_jvp(
    handle: object,
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
    out = _required_native_op("raydn_reflection_epc_paths_jvp")(
        _raydn_scene_handle_id(handle),
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
            "_channel_native.raydn_reflection_epc_paths_jvp must return 3 tangents"
        )
    return tuple(out)


class _RaydnReflectionEpcPathsAdFunction(torch.autograd.Function):
    """Fixed-winner differentiable RayDN reflection EPC paths over the C bridge.

    Forward IS the discovery entry (direct-plane mode) re-launched on the
    frozen winner sequence, so the primal hit points, normals and path length
    are the native discovery values, not a reconstruction. Backward/jvp call
    RayD's chain geometry companions; ``vertices`` must be the scene's global
    vertex table and only routes vertex gradients/tangents.
    """

    @staticmethod
    def forward(
        scene_handle,
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
        out = _required_native_op("raydn_reflection_epc_paths_forward")(
            int(scene_handle),
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
            scene_handle,
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
        ctx.scene = int(scene_handle)
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
    @torch.autograd.function.once_differentiable
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
        grad_vertices, grad_source, grad_receiver = raydn_reflection_epc_paths_backward(
            ctx.scene,
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
                "raydn_reflection_epc_paths_ad vertices must be the scene"
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
        with torch_compat.disable_functorch():
            tangents = raydn_reflection_epc_paths_jvp(
                ctx.scene,
                _ad_native_tensor(source),
                _ad_native_tensor(receiver),
                _ad_native_tensor(sequence),
                _ad_native_tensor(plane_points),
                _ad_native_tensor(plane_normals),
                _ad_native_tensor(valid),
                _ad_native_tensor(bounce_count),
                tangent_vertices=_ad_checked_tangent(
                    "raydn_reflection_epc_paths_ad tangent_vertices",
                    _ad_native_tangent_or_none(grad_vertices),
                    ctx.vertices_shape,
                ),
                tangent_source=_ad_checked_tangent(
                    "raydn_reflection_epc_paths_ad tangent_source",
                    _ad_native_tangent_or_none(grad_source),
                    tuple(source.shape),
                ),
                tangent_receiver=_ad_checked_tangent(
                    "raydn_reflection_epc_paths_ad tangent_receiver",
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


_RAYDN_REFLECTION_EPC_PATHS_AD_FIELDS = (
    "valid",
    "path_length",
    "resolved_prim_ids",
    "surface_group_ids",
    "hit_positions",
    "normals",
)


def raydn_reflection_epc_paths_ad(
    handle: object,
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
    """Differentiable RayDN reflection EPC paths under the fixed-winner contract.

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
    values = _RaydnReflectionEpcPathsAdFunction.apply(
        _raydn_scene_handle_id(handle),
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
    return dict(zip(_RAYDN_REFLECTION_EPC_PATHS_AD_FIELDS, values, strict=True))


def raydn_scene_face_normals_backward(
    handle: object, grad_face_normals: torch.Tensor
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
    out = _required_native_op("raydn_scene_face_normals_backward")(
        _raydn_scene_handle_id(handle),
        grad_face_normals,
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.raydn_scene_face_normals_backward must return a tensor"
        )
    return out


def raydn_scene_face_normals_jvp(
    handle: object, tangent_vertices: torch.Tensor
) -> torch.Tensor:
    _ad_check_tangent_vec3("tangent_vertices", tangent_vertices, None)
    if tangent_vertices is None:
        raise ValueError("tangent_vertices is required")
    out = _required_native_op("raydn_scene_face_normals_jvp")(
        _raydn_scene_handle_id(handle),
        tangent_vertices,
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.raydn_scene_face_normals_jvp must return a tensor"
        )
    return out


class _RaydnFaceNormalsAdFunction(torch.autograd.Function):
    """Scene unit face-normal table with a graph to the vertex table.

    Forward normalizes the native face-normal export with the same kernel the
    reflection discovery uses; backward/jvp call RayD's face-normal table
    companions (the adjoint/tangent of normalize(cross(v1 - v0, v2 - v0))
    over the scene's global vertex and face tables).
    """

    @staticmethod
    def forward(scene_handle, vertices, raw_face_normals):
        return deterministic_normalize_vec3(raw_face_normals, eps=1.0e-6)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        scene_handle, vertices, _raw_face_normals = inputs
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        ctx.scene = int(scene_handle)
        ctx.vertices_shape = tuple(vertices.shape)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_face_normals):
        if ctx.needs_input_grad[2]:
            raise RuntimeError(
                "raydn_face_normals_ad differentiates the vertex table only;"
                " the raw face-normal export is a detached scene record"
            )
        if grad_face_normals is None or not ctx.needs_input_grad[1]:
            return (None, None, None)
        grad_vertices = raydn_scene_face_normals_backward(ctx.scene, grad_face_normals)
        if tuple(grad_vertices.shape) != ctx.vertices_shape:
            raise RuntimeError(
                "raydn_face_normals_ad vertices must be the scene global vertex table"
            )
        return (None, grad_vertices, None)

    @staticmethod
    def jvp(ctx, _grad_handle, grad_vertices, _grad_raw_face_normals):
        tangent_vertices = _ad_native_tangent_or_none(grad_vertices)
        if tangent_vertices is None:
            return None
        with torch_compat.disable_functorch():
            return raydn_scene_face_normals_jvp(
                ctx.scene,
                _ad_checked_tangent(
                    "raydn_face_normals_ad tangent_vertices",
                    tangent_vertices,
                    ctx.vertices_shape,
                ),
            )


def raydn_face_normals_ad(
    handle: object, vertices: torch.Tensor, raw_face_normals: torch.Tensor
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
    return _RaydnFaceNormalsAdFunction.apply(
        _raydn_scene_handle_id(handle), vertices, raw_face_normals
    )
