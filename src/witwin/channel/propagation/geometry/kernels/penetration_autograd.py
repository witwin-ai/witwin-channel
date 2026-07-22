from __future__ import annotations

import torch

from witwin.channel.propagation.models.penetration import (
    SegmentPenetrationPolicy,
    SegmentPenetrationResult,
    SegmentPenetrationTapeResult,
)
from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.autograd_contracts import (
    _ad_active_ctx,
    _ad_checked_tangent,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
)
from witwin.channel.runtime.native_resources import _rayd_scene_resource
from witwin.channel.runtime.tensor_contracts import validate_cuda_tensor

from .bridge import (
    rayd_segment_penetration_backward,
    rayd_segment_penetration_forward_tape,
    rayd_segment_penetration_jvp,
)


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
    @torch.autograd.function.once_differentiable
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
        with torch_compat.disable_functorch():
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


__all__ = ["_RaydSegmentPenetrationAdFunction", "rayd_segment_penetration_ad"]
