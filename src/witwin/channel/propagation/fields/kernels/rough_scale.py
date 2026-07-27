"""Differentiable rough-surface C_r reflection scale (ADR-010 op 3)."""

from __future__ import annotations

import torch

from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.autograd_contracts import (
    _ad_first_order_only,
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_geometry_live,
    _ad_geometry_tangent,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
)

from .functional import (
    _ROUGH_SCALE_OUTPUT_FIELDS,
    _ROUGH_SCALE_TANGENT_FIELDS,
    field_rough_reflection_scale,
    field_rough_reflection_scale_backward,
    field_rough_reflection_scale_jvp,
)


def _grad_or_none(out: dict, key: str, needed: bool) -> torch.Tensor | None:
    return out[key] if needed else None


class _FieldRoughReflectionScaleAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable rough-surface C_r scale (ADR-010 op 3).

    Differentiable inputs: the four reflection field outputs (field_vector,
    coefficient, path_field, path_gain), frequency, and the hit geometry
    (positions, normals, source). sigma_b, rough_b and the realization
    ``replaced`` mask are fixed; requesting a sigma_b gradient fails loudly.
    Positions/normals/source only carry a gradient when the fixed-winner
    geometry AD path is live (matching the previous Torch factor's reach).
    """

    @staticmethod
    def forward(
        field_vector,
        coefficient,
        path_field,
        path_gain,
        positions,
        normals,
        source,
        sigma_b,
        rough_b,
        replaced,
        frequency,
        frequency_value,
        geometry_live,
    ):
        out = field_rough_reflection_scale(
            field_vector,
            coefficient,
            path_field,
            path_gain,
            positions,
            normals,
            source,
            sigma_b,
            rough_b,
            replaced,
            frequency_hz=frequency_value,
        )
        return tuple(out[name] for name in _ROUGH_SCALE_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[10]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:10]
        )
        ctx.frequency_value = inputs[11]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        # Computed by the wrapper, where forward duals are still visible.
        ctx.geometry_live = inputs[12]
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)

    @staticmethod
    @_ad_first_order_only
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
    ):
        none_grads = (None,) * 13
        _ad_reject_fixed_inputs(
            "field_rough_reflection_scale_ad",
            ctx.needs_input_grad,
            ((7, "sigma_b"),),
        )
        needed = tuple(bool(ctx.needs_input_grad[i]) for i in range(11))
        need_field = any(needed[:4])
        need_geometry = any(needed[4:7])
        need_frequency = needed[10]
        grads = (
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
        )
        if not (need_field or need_geometry or need_frequency) or all(
            value is None for value in grads
        ):
            return none_grads
        out = field_rough_reflection_scale_backward(
            *ctx.saved_tensors,
            frequency_hz=ctx.frequency_value,
            grad_field_vector=grad_field_vector,
            grad_coefficient=grad_coefficient,
            grad_path_field=grad_path_field,
            grad_path_gain=grad_path_gain,
            need_field=need_field,
            need_geometry=need_geometry,
            need_frequency=need_frequency,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            _grad_or_none(out, "grad_field_vector", needed[0]),
            _grad_or_none(out, "grad_coefficient", needed[1]),
            _grad_or_none(out, "grad_path_field", needed[2]),
            _grad_or_none(out, "grad_path_gain", needed[3]),
            _grad_or_none(out, "grad_positions", needed[4]),
            _grad_or_none(out, "grad_normals", needed[5]),
            _grad_or_none(out, "grad_source", needed[6]),
            None,
            None,
            None,
            grad_frequency,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_field_vector,
        t_coefficient,
        t_path_field,
        t_path_gain,
        t_positions,
        t_normals,
        t_source,
        t_sigma_b,
        _t_rough_b,
        _t_replaced,
        t_frequency,
        _t_frequency_value,
        _t_geometry_live,
    ):
        _ad_reject_fixed_tangents(
            "field_rough_reflection_scale_ad",
            ((t_sigma_b, "sigma_b"),),
        )
        saved = ctx.saved_tensors
        tangent_field = _ad_native_tangent_or_none(t_field_vector)
        tangent_coef = _ad_native_tangent_or_none(t_coefficient)
        tangent_pf = _ad_native_tangent_or_none(t_path_field)
        tangent_pg = _ad_native_tangent_or_none(t_path_gain)
        tangent_positions = _ad_geometry_tangent(
            "field_rough_reflection_scale_ad tangent_positions",
            t_positions,
            saved[4],
        )
        tangent_normals = _ad_geometry_tangent(
            "field_rough_reflection_scale_ad tangent_normals", t_normals, saved[5]
        )
        tangent_source = _ad_geometry_tangent(
            "field_rough_reflection_scale_ad tangent_source", t_source, saved[6]
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if (
            tangent_field is None
            and tangent_coef is None
            and tangent_pf is None
            and tangent_pg is None
            and tangent_positions is None
            and tangent_normals is None
            and tangent_source is None
            and tangent_frequency == 0.0
        ):
            return (None,) * len(_ROUGH_SCALE_OUTPUT_FIELDS)
        with torch_compat.disable_functorch():
            out = field_rough_reflection_scale_jvp(
                *(_ad_native_tensor(value) for value in saved),
                frequency_hz=ctx.frequency_value,
                tangent_field_vector=tangent_field,
                tangent_coefficient=tangent_coef,
                tangent_path_field=tangent_pf,
                tangent_path_gain=tangent_pg,
                tangent_positions=tangent_positions,
                tangent_normals=tangent_normals,
                tangent_source=tangent_source,
                tangent_frequency=tangent_frequency,
            )
        return tuple(out[name] for name in _ROUGH_SCALE_TANGENT_FIELDS)


def field_rough_reflection_scale_ad(
    field_vector: torch.Tensor,
    coefficient: torch.Tensor,
    path_field: torch.Tensor,
    path_gain: torch.Tensor,
    positions: torch.Tensor,
    normals: torch.Tensor,
    source: torch.Tensor,
    sigma_b: torch.Tensor,
    rough_b: torch.Tensor,
    replaced: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_rough_reflection_scale` (frequency + geometry).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not supplied
    it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldRoughReflectionScaleAdFunction.apply(
        field_vector,
        coefficient,
        path_field,
        path_gain,
        positions,
        normals,
        source,
        sigma_b,
        rough_b,
        replaced,
        frequency,
        float(frequency_value),
        _ad_geometry_live(positions, normals, source),
    )
    return dict(zip(_ROUGH_SCALE_OUTPUT_FIELDS, values, strict=True))


__all__ = [
    "_FieldRoughReflectionScaleAdFunction",
    "field_rough_reflection_scale_ad",
]
