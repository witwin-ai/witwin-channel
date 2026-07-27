from __future__ import annotations

import torch

from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.autograd_contracts import (
    _ad_first_order_only,
    _ad_geometry_tangent,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
)
from witwin.channel.runtime.symbols import required_symbol as _required_native_op


class _FieldProjectComplex3AdFunction(torch.autograd.Function):
    """Differentiable receiver projection on a frozen polarization basis."""

    @staticmethod
    def forward(field_vector, direction, rx_polarization):
        out = _required_native_op("field_project_complex3")(
            field_vector, direction, rx_polarization
        )
        return (out["coefficient"], out["path_gain"])

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal for value in inputs
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_coefficient, grad_path_gain):
        _ad_reject_fixed_inputs(
            "field_project_complex3_ad",
            ctx.needs_input_grad,
            ((2, "rx_polarization"),),
        )
        need_field = bool(ctx.needs_input_grad[0])
        need_direction = bool(ctx.needs_input_grad[1])
        if not (need_field or need_direction) or (
            grad_coefficient is None and grad_path_gain is None
        ):
            return (None, None, None)
        field_vector, direction, rx_polarization = ctx.saved_tensors
        out = _required_native_op("field_project_complex3_backward")(
            field_vector,
            direction,
            rx_polarization,
            grad_coefficient,
            grad_path_gain,
            need_field,
            need_direction,
        )
        return (
            out["grad_field_vector"] if need_field else None,
            out["grad_direction"] if need_direction else None,
            None,
        )

    @staticmethod
    def jvp(ctx, t_field_vector, t_direction, t_rx_polarization):
        _ad_reject_fixed_tangents(
            "field_project_complex3_ad",
            ((t_rx_polarization, "rx_polarization"),),
        )
        saved = ctx.saved_tensors
        tangent_field = _ad_native_tangent_or_none(t_field_vector)
        tangent_direction = _ad_geometry_tangent(
            "field_project_complex3_ad tangent_direction", t_direction, saved[1]
        )
        if tangent_field is None and tangent_direction is None:
            return (None, None)
        with torch_compat.disable_functorch():
            out = _required_native_op("field_project_complex3_jvp")(
                *(_ad_native_tensor(value) for value in saved),
                tangent_field,
                tangent_direction,
            )
        return (out["tangent_coefficient"], out["tangent_path_gain"])


def field_project_complex3_ad(
    field_vector: torch.Tensor,
    direction: torch.Tensor,
    rx_polarization: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_project_complex3` (field vector + direction)."""

    coefficient, path_gain = _FieldProjectComplex3AdFunction.apply(
        field_vector, direction, rx_polarization
    )
    return {"coefficient": coefficient, "path_gain": path_gain}


__all__ = [
    "_FieldProjectComplex3AdFunction",
    "field_project_complex3_ad",
]
