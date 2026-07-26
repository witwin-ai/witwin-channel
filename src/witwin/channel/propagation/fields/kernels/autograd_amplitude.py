"""Differentiable source-amplitude application onto a complex3 field.

ADR-039. The field transport kernels publish an unit-excitation complex3
vector and an excited scalar, but no excited vector. This wrapper dispatches
the native owner of that missing output. The map is linear in the field vector
and the amplitude is a real per-row constant, so the VJP and the JVP are the
same native scale, and ``tx_power`` stays a frozen primal exactly as it is in
every field transport companion.
"""

from __future__ import annotations

import torch

from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.autograd_contracts import (
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
)
from witwin.channel.runtime.symbols import required_symbol as _required_native_op


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
    @torch.autograd.function.once_differentiable
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
    "field_source_amplitude_scale_ad",
]
