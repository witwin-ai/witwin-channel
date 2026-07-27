from __future__ import annotations

import torch

from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.autograd_contracts import (
    _ad_checked_tangent,
    _ad_first_order_only,
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
)

from .functional import (
    _EM_LAYER_STACK_FIELDS,
    em_layer_stack_backward,
    em_layer_stack_eval,
    em_layer_stack_jvp,
)


class _EmLayerStackAdFunction(torch.autograd.Function):
    """Differentiable layer-stack r/t coefficients and power budgets.

    Differentiable inputs: cos_theta (per row), the CSR layer thickness /
    eps_r / sigma_e and the carrier frequency. layer_mu_r and the CSR
    topology stay fixed under the plan 07 contract; requesting the mu_r
    gradient fails loudly. Layer gradients accumulate atomically because the
    CSR store is shared by every row.
    """

    @staticmethod
    def forward(
        cos_theta,
        material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        frequency_value,
    ):
        out = em_layer_stack_eval(
            cos_theta,
            material_id,
            layer_offset,
            layer_count,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            frequency_hz=frequency_value,
        )
        return tuple(out[name] for name in _EM_LAYER_STACK_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[8]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:8]
        )
        ctx.frequency_value = inputs[9]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *grad_outputs):
        none_grads = (None,) * 10
        _ad_reject_fixed_inputs(
            "em_layer_stack_ad",
            ctx.needs_input_grad,
            ((7, "layer_mu_r"),),
        )
        need_cos = bool(ctx.needs_input_grad[0])
        need_layers = any(bool(ctx.needs_input_grad[i]) for i in (4, 5, 6))
        need_frequency = bool(ctx.needs_input_grad[8])
        if not (need_cos or need_layers or need_frequency) or all(
            value is None for value in grad_outputs
        ):
            return none_grads
        saved = ctx.saved_tensors
        out = em_layer_stack_backward(
            *saved,
            grad_outputs,
            frequency_hz=ctx.frequency_value,
            need_cos_theta=need_cos,
            need_layers=need_layers,
            need_frequency=need_frequency,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_cos_theta"] if need_cos else None,
            None,
            None,
            None,
            out["grad_layer_thickness_m"] if ctx.needs_input_grad[4] else None,
            out["grad_layer_eps_r"] if ctx.needs_input_grad[5] else None,
            out["grad_layer_sigma_e"] if ctx.needs_input_grad[6] else None,
            None,
            grad_frequency,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_cos_theta,
        _t_material_id,
        _t_layer_offset,
        _t_layer_count,
        t_layer_thickness,
        t_layer_eps_r,
        t_layer_sigma_e,
        t_layer_mu_r,
        t_frequency,
        _t_frequency_value,
    ):
        _ad_reject_fixed_tangents(
            "em_layer_stack_ad", ((t_layer_mu_r, "layer_mu_r"),)
        )
        saved = ctx.saved_tensors
        tangent_cos = _ad_checked_tangent(
            "em_layer_stack_ad tangent_cos_theta",
            _ad_native_tangent_or_none(t_cos_theta),
            tuple(saved[0].shape),
        )
        layer_shape = tuple(saved[4].shape)
        tangent_thickness = _ad_checked_tangent(
            "em_layer_stack_ad tangent_layer_thickness_m",
            _ad_native_tangent_or_none(t_layer_thickness),
            layer_shape,
        )
        tangent_eps = _ad_checked_tangent(
            "em_layer_stack_ad tangent_layer_eps_r",
            _ad_native_tangent_or_none(t_layer_eps_r),
            layer_shape,
        )
        tangent_sigma = _ad_checked_tangent(
            "em_layer_stack_ad tangent_layer_sigma_e",
            _ad_native_tangent_or_none(t_layer_sigma_e),
            layer_shape,
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if (
            tangent_cos is None
            and tangent_thickness is None
            and tangent_eps is None
            and tangent_sigma is None
            and tangent_frequency == 0.0
        ):
            return (None,) * len(_EM_LAYER_STACK_FIELDS)
        with torch_compat.disable_functorch():
            out = em_layer_stack_jvp(
                *(_ad_native_tensor(value) for value in saved),
                frequency_hz=ctx.frequency_value,
                tangent_cos_theta=tangent_cos,
                tangent_layer_thickness=tangent_thickness,
                tangent_layer_eps_r=tangent_eps,
                tangent_layer_sigma_e=tangent_sigma,
                tangent_frequency=tangent_frequency,
            )
        return tuple(out[name] for name in _EM_LAYER_STACK_FIELDS)


def em_layer_stack_ad(
    cos_theta: torch.Tensor,
    material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`em_layer_stack_eval` (plan 07 AD-3).

    ``frequency_value`` is the precomputed host scalar of ``frequency``; a
    seam that applies several Functions per solve reads the 0-d tensor once
    and threads the float here so no Function re-reads it (audit M3). When
    not supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _EmLayerStackAdFunction.apply(
        cos_theta,
        material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        float(frequency_value),
    )
    return dict(zip(_EM_LAYER_STACK_FIELDS, values, strict=True))


__all__ = ["_EmLayerStackAdFunction", "em_layer_stack_ad"]
