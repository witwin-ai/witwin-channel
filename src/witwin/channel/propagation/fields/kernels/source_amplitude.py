"""Source-amplitude application onto a transported complex3 field (ADR-039).

The field transport kernels publish an unit-excitation complex3 vector and an
excited scalar ``path_field = coefficient * sqrt(tx_power)``, but no excited
vector. This module owns the native facade for that missing output and its
differentiable wrapper. The map is linear in the field vector and the
amplitude is a real per-row constant, so the VJP and the JVP are the same
native scale, and ``tx_power`` stays a frozen primal exactly as it is in every
field transport companion.
"""

from __future__ import annotations

import torch

from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.autograd_contracts import (
    _ad_first_order_only,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
)
from witwin.channel.runtime.symbols import required_symbol as _required_native_op
from witwin.channel.runtime.tensor_contracts import validate_cuda_tensor


def _validate_source_amplitude_inputs(
    field_vector: torch.Tensor, tx_power: torch.Tensor, *, field_name: str
) -> int:
    # Cotangents and tangents arrive as strided views; the native owner
    # canonicalizes them, so layout is not part of this contract.
    validate_cuda_tensor(
        field_name,
        field_vector,
        dtype=torch.complex64,
        ndim=2,
        trailing_shape=(3,),
        require_contiguous=False,
    )
    validate_cuda_tensor(
        "tx_power",
        tx_power,
        dtype=torch.float32,
        ndim=1,
        require_contiguous=False,
    )
    count = int(field_vector.shape[0])
    if tx_power.shape != (count,):
        raise ValueError("tx_power must have one row per complex3 field row")
    return count


def _validate_source_amplitude_result(
    out: object, name: str, count: int, *, operation: str
) -> dict[str, torch.Tensor]:
    if not isinstance(out, dict) or set(out) != {name}:
        raise TypeError(f"_channel.{operation} returned invalid fields")
    validate_cuda_tensor(name, out[name], dtype=torch.complex64, ndim=2)
    if tuple(out[name].shape) != (count, 3):
        raise ValueError(f"{operation} returned an invalid shape")
    return out


def field_source_amplitude_scale(
    field_vector: torch.Tensor, tx_power: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Apply ``sqrt(max(tx_power, 0))`` to a transported complex3 field.

    The transport kernels publish ``field_vector`` for unit source amplitude
    and ``path_field = coefficient * sqrt(tx_power)`` for the excited scalar,
    but no excited vector. This is that missing output, evaluated natively with
    the same amplitude expression, so its receiver projection reproduces
    ``path_field``.
    """

    count = _validate_source_amplitude_inputs(
        field_vector, tx_power, field_name="field_vector"
    )
    out = _required_native_op("field_source_amplitude_scale")(field_vector, tx_power)
    return _validate_source_amplitude_result(
        out, "path_field_vector", count, operation="field_source_amplitude_scale"
    )


def field_source_amplitude_scale_backward(
    tx_power: torch.Tensor, grad_path_field_vector: torch.Tensor
) -> dict[str, torch.Tensor]:
    """VJP of :func:`field_source_amplitude_scale`; ``tx_power`` is frozen."""

    count = _validate_source_amplitude_inputs(
        grad_path_field_vector, tx_power, field_name="grad_path_field_vector"
    )
    out = _required_native_op("field_source_amplitude_scale_backward")(
        tx_power, grad_path_field_vector
    )
    return _validate_source_amplitude_result(
        out,
        "grad_field_vector",
        count,
        operation="field_source_amplitude_scale_backward",
    )


def field_source_amplitude_scale_jvp(
    tx_power: torch.Tensor, tangent_field_vector: torch.Tensor
) -> dict[str, torch.Tensor]:
    """JVP of :func:`field_source_amplitude_scale`; ``tx_power`` is frozen."""

    count = _validate_source_amplitude_inputs(
        tangent_field_vector, tx_power, field_name="tangent_field_vector"
    )
    out = _required_native_op("field_source_amplitude_scale_jvp")(
        tx_power, tangent_field_vector
    )
    return _validate_source_amplitude_result(
        out,
        "tangent_path_field_vector",
        count,
        operation="field_source_amplitude_scale_jvp",
    )


class _FieldSourceAmplitudeScaleAdFunction(torch.autograd.Function):
    """Differentiable ``field_vector * sqrt(max(tx_power, 0))``."""

    @staticmethod
    def forward(field_vector, tx_power):
        out = _required_native_op("field_source_amplitude_scale")(
            field_vector, tx_power
        )
        return out["path_field_vector"]

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        tx_power = torch.autograd.forward_ad.unpack_dual(inputs[1]).primal
        ctx.save_for_backward(tx_power)
        ctx.save_for_forward(tx_power)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_path_field_vector):
        _ad_reject_fixed_inputs(
            "field_source_amplitude_scale_ad",
            ctx.needs_input_grad,
            ((1, "tx_power"),),
        )
        if not ctx.needs_input_grad[0] or grad_path_field_vector is None:
            return (None, None)
        (tx_power,) = ctx.saved_tensors
        out = _required_native_op("field_source_amplitude_scale_backward")(
            tx_power, grad_path_field_vector
        )
        return (out["grad_field_vector"], None)

    @staticmethod
    def jvp(ctx, t_field_vector, t_tx_power):
        _ad_reject_fixed_tangents(
            "field_source_amplitude_scale_ad", ((t_tx_power, "tx_power"),)
        )
        tangent = _ad_native_tangent_or_none(t_field_vector)
        if tangent is None:
            return None
        (tx_power,) = ctx.saved_tensors
        with torch_compat.disable_functorch():
            out = _required_native_op("field_source_amplitude_scale_jvp")(
                _ad_native_tensor(tx_power), tangent
            )
        return out["tangent_path_field_vector"]


def field_source_amplitude_scale_ad(
    field_vector: torch.Tensor, tx_power: torch.Tensor
) -> torch.Tensor:
    """Differentiable :func:`field_source_amplitude_scale` (field vector only)."""

    return _FieldSourceAmplitudeScaleAdFunction.apply(field_vector, tx_power)


__all__ = [
    "_FieldSourceAmplitudeScaleAdFunction",
    "field_source_amplitude_scale",
    "field_source_amplitude_scale_ad",
    "field_source_amplitude_scale_backward",
    "field_source_amplitude_scale_jvp",
]
