"""Native ADR-030 deterministic diffraction pair reduction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.autograd_contracts import (
    _ad_first_order_only,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
)
from witwin.channel.runtime.capacity import (
    CapacityFailureState,
    require_capacity_failure_state,
)
from witwin.channel.runtime.symbols import required_symbol as _required_native_op
from witwin.channel.runtime.tensor_contracts import validate_cuda_tensor


@dataclass(frozen=True, slots=True, eq=False)
class DeterministicDiffractionPairReduction:
    """Pair-major field and power produced by the ADR-030 native reducer."""

    field_xyz: torch.Tensor
    power: torch.Tensor

    def __post_init__(self) -> None:
        validate_cuda_tensor(
            "field_xyz", self.field_xyz, dtype=torch.complex64, ndim=2
        )
        validate_cuda_tensor("power", self.power, dtype=torch.float32, ndim=1)
        if self.field_xyz.shape != (self.power.shape[0], 3):
            raise ValueError("field_xyz must have shape (pair_count, 3)")
        if self.field_xyz.device != self.power.device:
            raise ValueError("field_xyz and power must share a CUDA device")


@dataclass(frozen=True, slots=True, eq=False)
class DeterministicDiffractionPairGradients:
    """Six row-aligned real cotangents for the reducer's field inputs."""

    grad_x_re: torch.Tensor
    grad_x_im: torch.Tensor
    grad_y_re: torch.Tensor
    grad_y_im: torch.Tensor
    grad_z_re: torch.Tensor
    grad_z_im: torch.Tensor

    def __post_init__(self) -> None:
        fields = self.as_tuple()
        for name, value in zip(_GRADIENT_FIELDS, fields, strict=True):
            validate_cuda_tensor(name, value, dtype=torch.float32, ndim=1)
        reference = fields[0]
        for name, value in zip(_GRADIENT_FIELDS[1:], fields[1:], strict=True):
            if value.shape != reference.shape:
                raise ValueError(f"{name} must match grad_x_re shape")
            if value.device != reference.device:
                raise ValueError(f"{name} must share grad_x_re device")

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return (
            self.grad_x_re,
            self.grad_x_im,
            self.grad_y_re,
            self.grad_y_im,
            self.grad_z_re,
            self.grad_z_im,
        )


_INPUT_FIELDS = ("x_re", "x_im", "y_re", "y_im", "z_re", "z_im")
_GRADIENT_FIELDS = tuple(f"grad_{name}" for name in _INPUT_FIELDS)
_RESULT_FIELDS = ("field_xyz", "power")


def _validate_capacity_inputs(
    failure_state: CapacityFailureState,
    reported_count: torch.Tensor,
    valid: torch.Tensor,
    fields: tuple[torch.Tensor, ...],
    *,
    pair_count: int,
    state_capacity: int,
) -> None:
    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=1)
    require_capacity_failure_state(failure_state, device=valid.device)
    validate_cuda_tensor(
        "reported_count", reported_count, dtype=torch.int32, ndim=1
    )
    if reported_count.shape != (1,):
        raise ValueError("reported_count must have shape (1,)")
    if reported_count.device != valid.device:
        raise ValueError("reported_count must share valid device")
    if pair_count < 0:
        raise ValueError("pair_count must be non-negative")
    if state_capacity < 0:
        raise ValueError("state_capacity must be non-negative")
    expected_rows = pair_count * state_capacity
    if valid.shape != (expected_rows,):
        raise ValueError("valid must have shape (pair_count * state_capacity,)")
    for name, value in zip(_INPUT_FIELDS, fields, strict=True):
        validate_cuda_tensor(name, value, dtype=torch.float32, ndim=1)
        if value.shape != valid.shape:
            raise ValueError(f"{name} must match valid shape")
        if value.device != valid.device:
            raise ValueError(f"{name} must share valid device")


def _reduction_from_native(
    out: object, *, pair_count: int
) -> DeterministicDiffractionPairReduction:
    if not isinstance(out, dict) or set(out) != set(_RESULT_FIELDS):
        raise TypeError("native diffraction pair reducer returned invalid fields")
    result = DeterministicDiffractionPairReduction(
        field_xyz=out["field_xyz"], power=out["power"]
    )
    if result.field_xyz.shape != (pair_count, 3):
        raise ValueError("native diffraction pair reducer returned invalid capacity")
    return result


def deterministic_diffraction_pair_reduce(
    failure_state: CapacityFailureState,
    reported_count: torch.Tensor,
    valid: torch.Tensor,
    x_re: torch.Tensor,
    x_im: torch.Tensor,
    y_re: torch.Tensor,
    y_im: torch.Tensor,
    z_re: torch.Tensor,
    z_im: torch.Tensor,
    *,
    pair_count: int,
    state_capacity: int,
) -> DeterministicDiffractionPairReduction:
    """Reduce source-lane fields in exact ascending state order on CUDA."""

    pair_count = int(pair_count)
    state_capacity = int(state_capacity)
    fields = (x_re, x_im, y_re, y_im, z_re, z_im)
    _validate_capacity_inputs(
        failure_state,
        reported_count,
        valid,
        fields,
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    out = _required_native_op("deterministic_diffraction_pair_reduce")(
        failure_state.bits,
        reported_count,
        valid,
        *fields,
        pair_count,
        state_capacity,
    )
    return _reduction_from_native(out, pair_count=pair_count)


def deterministic_diffraction_pair_reduce_backward(
    failure_state: CapacityFailureState,
    valid: torch.Tensor,
    field_xyz: torch.Tensor,
    *,
    grad_field_xyz: torch.Tensor | None = None,
    grad_power: torch.Tensor | None = None,
    pair_count: int,
    state_capacity: int,
) -> DeterministicDiffractionPairGradients:
    """Dispatch the native fixed-valid VJP companion."""

    pair_count = int(pair_count)
    state_capacity = int(state_capacity)
    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=1)
    require_capacity_failure_state(failure_state, device=valid.device)
    out = _required_native_op("deterministic_diffraction_pair_reduce_backward")(
        failure_state.bits,
        valid,
        field_xyz,
        grad_field_xyz,
        grad_power,
        pair_count,
        state_capacity,
    )
    if not isinstance(out, dict) or set(out) != set(_GRADIENT_FIELDS):
        raise TypeError("native diffraction pair reducer VJP returned invalid fields")
    return DeterministicDiffractionPairGradients(
        **{name: out[name] for name in _GRADIENT_FIELDS}
    )


def deterministic_diffraction_pair_reduce_jvp(
    failure_state: CapacityFailureState,
    valid: torch.Tensor,
    field_xyz: torch.Tensor,
    *,
    tangent_x_re: torch.Tensor | None = None,
    tangent_x_im: torch.Tensor | None = None,
    tangent_y_re: torch.Tensor | None = None,
    tangent_y_im: torch.Tensor | None = None,
    tangent_z_re: torch.Tensor | None = None,
    tangent_z_im: torch.Tensor | None = None,
    pair_count: int,
    state_capacity: int,
) -> DeterministicDiffractionPairReduction:
    """Dispatch the native fixed-valid JVP companion."""

    pair_count = int(pair_count)
    state_capacity = int(state_capacity)
    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=1)
    require_capacity_failure_state(failure_state, device=valid.device)
    out = _required_native_op("deterministic_diffraction_pair_reduce_jvp")(
        failure_state.bits,
        valid,
        field_xyz,
        tangent_x_re,
        tangent_x_im,
        tangent_y_re,
        tangent_y_im,
        tangent_z_re,
        tangent_z_im,
        pair_count,
        state_capacity,
    )
    return _reduction_from_native(out, pair_count=pair_count)


class _DeterministicDiffractionPairReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        failure_bits,
        reported_count,
        valid,
        x_re,
        x_im,
        y_re,
        y_im,
        z_re,
        z_im,
        pair_count,
        state_capacity,
    ):
        result = deterministic_diffraction_pair_reduce(
            CapacityFailureState(bits=failure_bits),
            reported_count,
            valid,
            x_re,
            x_im,
            y_re,
            y_im,
            z_re,
            z_im,
            pair_count=int(pair_count),
            state_capacity=int(state_capacity),
        )
        return result.field_xyz, result.power

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        failure_bits, valid = (
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (inputs[0], inputs[2])
        )
        field_xyz, _power = output
        ctx.pair_count = int(inputs[9])
        ctx.state_capacity = int(inputs[10])
        saved = (failure_bits, valid, field_xyz)
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @_ad_first_order_only
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_field_xyz, grad_power):
        if (
            not any(ctx.needs_input_grad[3:9])
            or (grad_field_xyz is None and grad_power is None)
        ):
            return (None,) * 11
        failure_bits, valid, field_xyz = ctx.saved_tensors
        gradients = deterministic_diffraction_pair_reduce_backward(
            CapacityFailureState(bits=failure_bits),
            valid,
            field_xyz,
            grad_field_xyz=grad_field_xyz,
            grad_power=grad_power,
            pair_count=ctx.pair_count,
            state_capacity=ctx.state_capacity,
        ).as_tuple()
        input_gradients = tuple(
            value if ctx.needs_input_grad[index] else None
            for index, value in zip(range(3, 9), gradients, strict=True)
        )
        return (
            None,
            None,
            None,
            *input_gradients,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        _t_failure_bits,
        _t_reported_count,
        _t_valid,
        tangent_x_re,
        tangent_x_im,
        tangent_y_re,
        tangent_y_im,
        tangent_z_re,
        tangent_z_im,
        _t_pair_count,
        _t_state_capacity,
    ):
        tangents = tuple(
            _ad_native_tangent_or_none(value)
            for value in (
                tangent_x_re,
                tangent_x_im,
                tangent_y_re,
                tangent_y_im,
                tangent_z_re,
                tangent_z_im,
            )
        )
        if all(value is None for value in tangents):
            return None, None
        failure_bits, valid, field_xyz = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with torch_compat.disable_functorch():
            result = deterministic_diffraction_pair_reduce_jvp(
                CapacityFailureState(bits=failure_bits),
                valid,
                field_xyz,
                tangent_x_re=tangents[0],
                tangent_x_im=tangents[1],
                tangent_y_re=tangents[2],
                tangent_y_im=tangents[3],
                tangent_z_re=tangents[4],
                tangent_z_im=tangents[5],
                pair_count=ctx.pair_count,
                state_capacity=ctx.state_capacity,
            )
        return result.field_xyz, result.power


def deterministic_diffraction_pair_reduce_ad(
    failure_state: CapacityFailureState,
    reported_count: torch.Tensor,
    valid: torch.Tensor,
    x_re: torch.Tensor,
    x_im: torch.Tensor,
    y_re: torch.Tensor,
    y_im: torch.Tensor,
    z_re: torch.Tensor,
    z_im: torch.Tensor,
    *,
    pair_count: int,
    state_capacity: int,
) -> DeterministicDiffractionPairReduction:
    """Differentiable native ADR-030 pair reduction."""

    pair_count = int(pair_count)
    state_capacity = int(state_capacity)
    _validate_capacity_inputs(
        failure_state,
        reported_count,
        valid,
        (x_re, x_im, y_re, y_im, z_re, z_im),
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    field_xyz, power = _DeterministicDiffractionPairReduceFunction.apply(
        failure_state.bits,
        reported_count,
        valid,
        x_re,
        x_im,
        y_re,
        y_im,
        z_re,
        z_im,
        pair_count,
        state_capacity,
    )
    return DeterministicDiffractionPairReduction(
        field_xyz=field_xyz, power=power
    )


__all__ = [
    "DeterministicDiffractionPairGradients",
    "DeterministicDiffractionPairReduction",
    "deterministic_diffraction_pair_reduce",
    "deterministic_diffraction_pair_reduce_ad",
    "deterministic_diffraction_pair_reduce_backward",
    "deterministic_diffraction_pair_reduce_jvp",
]
