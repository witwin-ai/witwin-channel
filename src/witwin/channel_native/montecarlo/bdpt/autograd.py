"""ADR-022 BDPT fixed-topology AD ``torch.autograd.Function`` wrappers.

Plan-07 companion pattern (``setup_context``, ``once_differentiable``,
``set_materialize_grads(False)``, dual unpacking, ``_ad_reject_fixed_*``) applied
to the six BDPT-owned forward ops (subpath advance x2, endpoint connection,
accumulate, finalize x2). Every wrapper:

- dispatches the SAME registered native forward symbol as ``ad_mode='none'`` so
  the primal values are bitwise identical (no Torch reconstruction of physics);
- carries gradients only for the differentiable set (material eps_r / sigma_e /
  thickness, CSR layers, resident table values ride the scattering companions,
  carrier frequency, tx_power, and the subpath complex field it advances);
- rejects every frozen input loudly (sampled quantities, pdfs, MIS weights,
  visibility masks, the connection/subpath schema layouts, hit geometry — the
  stochastic sampler keeps ``ad_geometry='enumerated_blocks_only'``);
- routes cotangents to the registered ``_backward`` companion and tangents to
  the registered ``_jvp`` companion (never finite differences).

AD-live scalars (``frequency``) cross as a 0-dim tensor on the tape plus a plain
``double`` value at the ABI (ADR-014 / ``field_free_space`` precedent).
"""

from __future__ import annotations

import torch

from witwin.channel_native.runtime import torch_compat
from witwin.channel_native.runtime.autograd_contracts import (
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
)

from .kernels.maps import (
    bdpt_finalize_component_maps,
    bdpt_finalize_component_maps_backward,
    bdpt_finalize_component_maps_jvp,
    bdpt_finalize_point_components,
    bdpt_finalize_point_components_backward,
    bdpt_finalize_point_components_jvp,
)
from .kernels.paths import (
    _BDPT_SUBPATH_SCHEMA,
    bdpt_accumulate_connection_samples_backward,
    bdpt_accumulate_connection_samples_forward_ad,
    bdpt_accumulate_connection_samples_jvp,
    bdpt_endpoint_connection_samples,
    bdpt_endpoint_connection_samples_backward,
    bdpt_endpoint_connection_samples_jvp,
    bdpt_reflected_light_subpath_state,
    bdpt_reflected_light_subpath_state_backward,
    bdpt_reflected_light_subpath_state_jvp,
    bdpt_transmitted_light_subpath_state,
    bdpt_transmitted_light_subpath_state_backward,
    bdpt_transmitted_light_subpath_state_jvp,
)


# Subpath field order and the four differentiable slots (spec 6.1/6.2).
_SUBPATH_FIELDS = tuple(_BDPT_SUBPATH_SCHEMA)
_SUBPATH_DIFF_FIELDS = (
    "field_real",
    "field_imag",
    "throughput_real",
    "throughput_imag",
)
_SUBPATH_DIFF_INDEX = {
    name: _SUBPATH_FIELDS.index(name) for name in _SUBPATH_DIFF_FIELDS
}
_FINALIZE_FIELDS = (
    "path_gain",
    "los_power",
    "reflection_power",
    "diffraction_power",
    "transmission_power",
    "scattering_power",
)


def _subpath_with_fields(
    base: dict[str, torch.Tensor],
    field_real: torch.Tensor,
    field_imag: torch.Tensor,
    throughput_real: torch.Tensor,
    throughput_imag: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Rebuild a subpath dict overriding only the four differentiable slots."""

    out = dict(base)
    out["field_real"] = field_real
    out["field_imag"] = field_imag
    out["throughput_real"] = throughput_real
    out["throughput_imag"] = throughput_imag
    return out


def _subpath_output_tuple(out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(out[name] for name in _SUBPATH_FIELDS)


def _mark_subpath_structural(ctx, output) -> None:
    """Mark every subpath output field non-differentiable except the four
    field/throughput slots that carry the advance's cotangents."""

    structural = [
        output[index]
        for index, name in enumerate(_SUBPATH_FIELDS)
        if name not in _SUBPATH_DIFF_FIELDS
    ]
    ctx.mark_non_differentiable(*structural)


# ---------------------------------------------------------------------------
# 6.1 reflected light subpath advance
# ---------------------------------------------------------------------------


class _BdptReflectedSubpathAdFunction(torch.autograd.Function):
    """Differentiable specular-reflection subpath advance (spec 6.1).

    Differentiable inputs: the upstream light field/throughput (4 tensors) and
    the per-face material eps_r / sigma_e / thickness plus the carrier
    frequency. Frozen (reject loudly): hit-point geometry (the ``intersection``
    dict), material_valid, material_gain, mu_r, and every structural subpath
    field. Hit geometry stays frozen in v1 (stochastic-sampler stance,
    ``ad_geometry='enumerated_blocks_only'``)."""

    @staticmethod
    def forward(
        field_real,
        field_imag,
        throughput_real,
        throughput_imag,
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        material_thickness,
        frequency,
        frequency_value,
        base_light,
        intersection,
        material_gain,
        material_valid,
    ):
        light = _subpath_with_fields(
            base_light, field_real, field_imag, throughput_real, throughput_imag
        )
        out = bdpt_reflected_light_subpath_state(
            light,
            intersection,
            material_gain=material_gain,
            material_valid=material_valid,
            material_eps_r=material_eps_r,
            material_sigma_e=material_sigma_e,
            material_mu_r=material_mu_r,
            material_thickness=material_thickness,
            frequency_hz=frequency_value,
        )
        return _subpath_output_tuple(out)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        (
            field_real,
            field_imag,
            throughput_real,
            throughput_imag,
            material_eps_r,
            material_sigma_e,
            material_mu_r,
            material_thickness,
            frequency,
            frequency_value,
            base_light,
            intersection,
            material_gain,
            material_valid,
        ) = inputs
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (
                field_real,
                field_imag,
                throughput_real,
                throughput_imag,
                material_eps_r,
                material_sigma_e,
                material_mu_r,
                material_thickness,
            )
        )
        ctx.frequency_value = frequency_value
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.base_light = {
            name: torch.autograd.forward_ad.unpack_dual(value).primal
            if isinstance(value, torch.Tensor)
            else value
            for name, value in base_light.items()
        }
        ctx.intersection = intersection
        ctx.material_gain = material_gain
        ctx.material_valid = material_valid
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        _mark_subpath_structural(ctx, output)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        none_grads = (None,) * 14
        _ad_reject_fixed_inputs(
            "bdpt_reflected_light_subpath_state_ad",
            ctx.needs_input_grad,
            ((6, "material_mu_r"),),
        )
        need_field_in = any(bool(ctx.needs_input_grad[i]) for i in range(4))
        need_material = any(bool(ctx.needs_input_grad[i]) for i in (4, 5, 7))
        need_frequency = bool(ctx.needs_input_grad[8])
        grad_field_real = grad_outputs[_SUBPATH_DIFF_INDEX["field_real"]]
        grad_field_imag = grad_outputs[_SUBPATH_DIFF_INDEX["field_imag"]]
        grad_throughput_real = grad_outputs[_SUBPATH_DIFF_INDEX["throughput_real"]]
        grad_throughput_imag = grad_outputs[_SUBPATH_DIFF_INDEX["throughput_imag"]]
        grads = (
            grad_field_real,
            grad_field_imag,
            grad_throughput_real,
            grad_throughput_imag,
        )
        if not (need_field_in or need_material or need_frequency) or all(
            value is None for value in grads
        ):
            return none_grads
        (
            field_real,
            field_imag,
            throughput_real,
            throughput_imag,
            material_eps_r,
            material_sigma_e,
            material_mu_r,
            material_thickness,
        ) = ctx.saved_tensors
        light = _subpath_with_fields(
            ctx.base_light, field_real, field_imag, throughput_real, throughput_imag
        )
        out = bdpt_reflected_light_subpath_state_backward(
            light,
            ctx.intersection,
            ctx.material_gain,
            ctx.material_valid,
            material_eps_r,
            material_sigma_e,
            material_mu_r,
            material_thickness,
            frequency_hz=ctx.frequency_value,
            grad_field_real=grad_field_real,
            grad_field_imag=grad_field_imag,
            grad_throughput_real=grad_throughput_real,
            grad_throughput_imag=grad_throughput_imag,
            need_grad_material=need_material,
            need_grad_field_in=need_field_in,
            need_grad_frequency=need_frequency,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_light_field_real"] if ctx.needs_input_grad[0] else None,
            out["grad_light_field_imag"] if ctx.needs_input_grad[1] else None,
            out["grad_light_throughput_real"] if ctx.needs_input_grad[2] else None,
            out["grad_light_throughput_imag"] if ctx.needs_input_grad[3] else None,
            out["grad_eps_r"] if ctx.needs_input_grad[4] else None,
            out["grad_sigma_e"] if ctx.needs_input_grad[5] else None,
            None,
            out["grad_thickness"] if ctx.needs_input_grad[7] else None,
            grad_frequency,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_field_real,
        t_field_imag,
        t_throughput_real,
        t_throughput_imag,
        t_eps_r,
        t_sigma_e,
        t_mu_r,
        t_thickness,
        t_frequency,
        _t_frequency_value,
        _t_base_light,
        _t_intersection,
        _t_material_gain,
        _t_material_valid,
    ):
        _ad_reject_fixed_tangents(
            "bdpt_reflected_light_subpath_state_ad", ((t_mu_r, "material_mu_r"),)
        )
        saved = ctx.saved_tensors
        light = _subpath_with_fields(ctx.base_light, saved[0], saved[1], saved[2], saved[3])
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        tangents = {
            "tangent_light_field_real": _ad_native_tangent_or_none(t_field_real),
            "tangent_light_field_imag": _ad_native_tangent_or_none(t_field_imag),
            "tangent_light_throughput_real": _ad_native_tangent_or_none(
                t_throughput_real
            ),
            "tangent_light_throughput_imag": _ad_native_tangent_or_none(
                t_throughput_imag
            ),
            "tangent_eps_r": _ad_native_tangent_or_none(t_eps_r),
            "tangent_sigma_e": _ad_native_tangent_or_none(t_sigma_e),
            "tangent_thickness": _ad_native_tangent_or_none(t_thickness),
        }
        if tangent_frequency == 0.0 and all(v is None for v in tangents.values()):
            return (None,) * len(_SUBPATH_FIELDS)
        with torch_compat.disable_functorch():
            out = bdpt_reflected_light_subpath_state_jvp(
                light,
                ctx.intersection,
                ctx.material_gain,
                ctx.material_valid,
                _ad_native_tensor(saved[4]),
                _ad_native_tensor(saved[5]),
                _ad_native_tensor(saved[6]),
                _ad_native_tensor(saved[7]),
                frequency_hz=ctx.frequency_value,
                tangent_frequency=tangent_frequency,
                **tangents,
            )
        return _subpath_tangent_outputs(out)


def _subpath_tangent_outputs(out: dict[str, torch.Tensor]) -> tuple:
    """Map the four native tangent fields onto the full subpath output slots."""

    result: list[torch.Tensor | None] = [None] * len(_SUBPATH_FIELDS)
    result[_SUBPATH_DIFF_INDEX["field_real"]] = out["tangent_field_real"]
    result[_SUBPATH_DIFF_INDEX["field_imag"]] = out["tangent_field_imag"]
    result[_SUBPATH_DIFF_INDEX["throughput_real"]] = out["tangent_throughput_real"]
    result[_SUBPATH_DIFF_INDEX["throughput_imag"]] = out["tangent_throughput_imag"]
    return tuple(result)


def bdpt_reflected_light_subpath_state_ad(
    light: dict[str, torch.Tensor],
    intersection: dict[str, torch.Tensor],
    *,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_eps_r: torch.Tensor,
    material_sigma_e: torch.Tensor,
    material_mu_r: torch.Tensor,
    material_thickness: torch.Tensor,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_reflected_light_subpath_state` (spec 6.1)."""

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    base_light = {
        name: value
        for name, value in light.items()
        if name not in _SUBPATH_DIFF_FIELDS
    }
    values = _BdptReflectedSubpathAdFunction.apply(
        light["field_real"],
        light["field_imag"],
        light["throughput_real"],
        light["throughput_imag"],
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        material_thickness,
        frequency,
        float(frequency_value),
        base_light,
        intersection,
        material_gain,
        material_valid,
    )
    return dict(zip(_SUBPATH_FIELDS, values, strict=True))


# ---------------------------------------------------------------------------
# 6.2 transmitted light subpath advance
# ---------------------------------------------------------------------------


class _BdptTransmittedSubpathAdFunction(torch.autograd.Function):
    """Differentiable slab-transmission subpath advance (spec 6.2).

    Differentiable inputs: the upstream light field/throughput (4 tensors) and
    the CSR layer thickness / eps_r / sigma_e plus the carrier frequency. Frozen
    (reject loudly): hit-point geometry, face_material_id, the CSR index arrays
    (layer_offset / layer_count), layer_mu_r, and every structural subpath
    field."""

    @staticmethod
    def forward(
        field_real,
        field_imag,
        throughput_real,
        throughput_imag,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        frequency_value,
        base_light,
        intersection,
        face_material_id,
        layer_offset,
        layer_count,
    ):
        light = _subpath_with_fields(
            base_light, field_real, field_imag, throughput_real, throughput_imag
        )
        out = bdpt_transmitted_light_subpath_state(
            light,
            intersection,
            face_material_id=face_material_id,
            layer_offset=layer_offset,
            layer_count=layer_count,
            layer_thickness_m=layer_thickness_m,
            layer_eps_r=layer_eps_r,
            layer_sigma_e=layer_sigma_e,
            layer_mu_r=layer_mu_r,
            frequency_hz=frequency_value,
        )
        return _subpath_output_tuple(out)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        (
            field_real,
            field_imag,
            throughput_real,
            throughput_imag,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            frequency,
            frequency_value,
            base_light,
            intersection,
            face_material_id,
            layer_offset,
            layer_count,
        ) = inputs
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (
                field_real,
                field_imag,
                throughput_real,
                throughput_imag,
                layer_thickness_m,
                layer_eps_r,
                layer_sigma_e,
                layer_mu_r,
            )
        )
        ctx.frequency_value = frequency_value
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.base_light = {
            name: torch.autograd.forward_ad.unpack_dual(value).primal
            if isinstance(value, torch.Tensor)
            else value
            for name, value in base_light.items()
        }
        ctx.intersection = intersection
        ctx.face_material_id = face_material_id
        ctx.layer_offset = layer_offset
        ctx.layer_count = layer_count
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        _mark_subpath_structural(ctx, output)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        none_grads = (None,) * 15
        _ad_reject_fixed_inputs(
            "bdpt_transmitted_light_subpath_state_ad",
            ctx.needs_input_grad,
            ((7, "layer_mu_r"),),
        )
        need_field_in = any(bool(ctx.needs_input_grad[i]) for i in range(4))
        need_layers = any(bool(ctx.needs_input_grad[i]) for i in (4, 5, 6))
        need_frequency = bool(ctx.needs_input_grad[8])
        grad_field_real = grad_outputs[_SUBPATH_DIFF_INDEX["field_real"]]
        grad_field_imag = grad_outputs[_SUBPATH_DIFF_INDEX["field_imag"]]
        grad_throughput_real = grad_outputs[_SUBPATH_DIFF_INDEX["throughput_real"]]
        grad_throughput_imag = grad_outputs[_SUBPATH_DIFF_INDEX["throughput_imag"]]
        grads = (
            grad_field_real,
            grad_field_imag,
            grad_throughput_real,
            grad_throughput_imag,
        )
        if not (need_field_in or need_layers or need_frequency) or all(
            value is None for value in grads
        ):
            return none_grads
        (
            field_real,
            field_imag,
            throughput_real,
            throughput_imag,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
        ) = ctx.saved_tensors
        light = _subpath_with_fields(
            ctx.base_light, field_real, field_imag, throughput_real, throughput_imag
        )
        out = bdpt_transmitted_light_subpath_state_backward(
            light,
            ctx.intersection,
            ctx.face_material_id,
            ctx.layer_offset,
            ctx.layer_count,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            frequency_hz=ctx.frequency_value,
            grad_field_real=grad_field_real,
            grad_field_imag=grad_field_imag,
            grad_throughput_real=grad_throughput_real,
            grad_throughput_imag=grad_throughput_imag,
            need_grad_layers=need_layers,
            need_grad_field_in=need_field_in,
            need_grad_frequency=need_frequency,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_light_field_real"] if ctx.needs_input_grad[0] else None,
            out["grad_light_field_imag"] if ctx.needs_input_grad[1] else None,
            out["grad_light_throughput_real"] if ctx.needs_input_grad[2] else None,
            out["grad_light_throughput_imag"] if ctx.needs_input_grad[3] else None,
            out["grad_layer_thickness"] if ctx.needs_input_grad[4] else None,
            out["grad_layer_eps_r"] if ctx.needs_input_grad[5] else None,
            out["grad_layer_sigma_e"] if ctx.needs_input_grad[6] else None,
            None,
            grad_frequency,
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
        t_field_real,
        t_field_imag,
        t_throughput_real,
        t_throughput_imag,
        t_thickness,
        t_eps_r,
        t_sigma_e,
        t_mu_r,
        t_frequency,
        _t_frequency_value,
        _t_base_light,
        _t_intersection,
        _t_face_material_id,
        _t_layer_offset,
        _t_layer_count,
    ):
        _ad_reject_fixed_tangents(
            "bdpt_transmitted_light_subpath_state_ad", ((t_mu_r, "layer_mu_r"),)
        )
        saved = ctx.saved_tensors
        light = _subpath_with_fields(ctx.base_light, saved[0], saved[1], saved[2], saved[3])
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        tangents = {
            "tangent_light_field_real": _ad_native_tangent_or_none(t_field_real),
            "tangent_light_field_imag": _ad_native_tangent_or_none(t_field_imag),
            "tangent_light_throughput_real": _ad_native_tangent_or_none(
                t_throughput_real
            ),
            "tangent_light_throughput_imag": _ad_native_tangent_or_none(
                t_throughput_imag
            ),
            "tangent_layer_thickness": _ad_native_tangent_or_none(t_thickness),
            "tangent_layer_eps_r": _ad_native_tangent_or_none(t_eps_r),
            "tangent_layer_sigma_e": _ad_native_tangent_or_none(t_sigma_e),
        }
        if tangent_frequency == 0.0 and all(v is None for v in tangents.values()):
            return (None,) * len(_SUBPATH_FIELDS)
        with torch_compat.disable_functorch():
            out = bdpt_transmitted_light_subpath_state_jvp(
                light,
                ctx.intersection,
                ctx.face_material_id,
                ctx.layer_offset,
                ctx.layer_count,
                _ad_native_tensor(saved[4]),
                _ad_native_tensor(saved[5]),
                _ad_native_tensor(saved[6]),
                _ad_native_tensor(saved[7]),
                frequency_hz=ctx.frequency_value,
                tangent_frequency=tangent_frequency,
                **tangents,
            )
        return _subpath_tangent_outputs(out)


def bdpt_transmitted_light_subpath_state_ad(
    light: dict[str, torch.Tensor],
    intersection: dict[str, torch.Tensor],
    *,
    face_material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_transmitted_light_subpath_state` (spec 6.2)."""

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    base_light = {
        name: value
        for name, value in light.items()
        if name not in _SUBPATH_DIFF_FIELDS
    }
    values = _BdptTransmittedSubpathAdFunction.apply(
        light["field_real"],
        light["field_imag"],
        light["throughput_real"],
        light["throughput_imag"],
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        float(frequency_value),
        base_light,
        intersection,
        face_material_id,
        layer_offset,
        layer_count,
    )
    return dict(zip(_SUBPATH_FIELDS, values, strict=True))


# ---------------------------------------------------------------------------
# 6.5 / 6.6 finalize (point components and component maps)
# ---------------------------------------------------------------------------


class _BdptFinalizeAdFunction(torch.autograd.Function):
    """Differentiable BDPT finalize (linear map; spec 6.5/6.6).

    The five component matrices/maps are all differentiable; the forward sums
    them into ``path_gain`` and reduces each into a 0-dim power. Backward is the
    native transpose companion, jvp the native forward map on the tangents;
    both deterministic, no atomics."""

    @staticmethod
    def forward(los, reflection, diffraction, transmission, scattering, kind):
        forward = (
            bdpt_finalize_point_components
            if kind == "point"
            else bdpt_finalize_component_maps
        )
        out = forward(los, reflection, diffraction, transmission, scattering)
        return tuple(out[name] for name in _FINALIZE_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        los, reflection, diffraction, transmission, scattering, kind = inputs
        ctx.kind = kind
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (los, reflection, diffraction, transmission, scattering)
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_path_gain, *grad_powers):
        if not any(bool(flag) for flag in ctx.needs_input_grad[:5]):
            return (None,) * 6
        backward = (
            bdpt_finalize_point_components_backward
            if ctx.kind == "point"
            else bdpt_finalize_component_maps_backward
        )
        out = backward(
            *ctx.saved_tensors,
            grad_path_gain=grad_path_gain,
            grad_los_power=grad_powers[0],
            grad_reflection_power=grad_powers[1],
            grad_diffraction_power=grad_powers[2],
            grad_transmission_power=grad_powers[3],
            grad_scattering_power=grad_powers[4],
            need_grad_components=True,
        )
        return (
            out["grad_los"],
            out["grad_reflection"],
            out["grad_diffraction"],
            out["grad_transmission"],
            out["grad_scattering"],
            None,
        )

    @staticmethod
    def jvp(ctx, t_los, t_reflection, t_diffraction, t_transmission, t_scattering, _t_kind):
        tangents = {
            "tangent_los": _ad_native_tangent_or_none(t_los),
            "tangent_reflection": _ad_native_tangent_or_none(t_reflection),
            "tangent_diffraction": _ad_native_tangent_or_none(t_diffraction),
            "tangent_transmission": _ad_native_tangent_or_none(t_transmission),
            "tangent_scattering": _ad_native_tangent_or_none(t_scattering),
        }
        if all(value is None for value in tangents.values()):
            return (None,) * len(_FINALIZE_FIELDS)
        jvp = (
            bdpt_finalize_point_components_jvp
            if ctx.kind == "point"
            else bdpt_finalize_component_maps_jvp
        )
        with torch_compat.disable_functorch():
            out = jvp(*(_ad_native_tensor(value) for value in ctx.saved_tensors), **tangents)
        return tuple(out[name] for name in _FINALIZE_TANGENT_FIELDS)


_FINALIZE_TANGENT_FIELDS = (
    "tangent_path_gain",
    "tangent_los_power",
    "tangent_reflection_power",
    "tangent_diffraction_power",
    "tangent_transmission_power",
    "tangent_scattering_power",
)


def bdpt_finalize_point_components_ad(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_finalize_point_components` (spec 6.5)."""

    values = _BdptFinalizeAdFunction.apply(
        los, reflection, diffraction, transmission, scattering, "point"
    )
    return dict(zip(_FINALIZE_FIELDS, values, strict=True))


def bdpt_finalize_component_maps_ad(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_finalize_component_maps` (spec 6.6)."""

    values = _BdptFinalizeAdFunction.apply(
        los, reflection, diffraction, transmission, scattering, "maps"
    )
    return dict(zip(_FINALIZE_FIELDS, values, strict=True))


# ---------------------------------------------------------------------------
# 6.4 accumulate (power AND coherent)
# ---------------------------------------------------------------------------


_ACCUMULATE_MATRIX_FIELDS = (
    "path_gain",
    "los",
    "reflection",
    "diffraction",
    "transmission",
    "scattering",
)


class _BdptAccumulateAdFunction(torch.autograd.Function):
    """Differentiable connection-sample accumulate, both domains (spec 6.4).

    Differentiable inputs: the row ``contribution`` (power domain) OR the
    complex ``coeff_real``/``coeff_imag`` (coherent domain). Frozen (reject
    loudly): mis_weight, tx_id/rx_id/component_id/valid, the whole index
    structure. The coherent forward retains its per-component phasor bin sums
    ``S_b`` so the coherent backward needs no re-reduction (supervisor ruling)."""

    @staticmethod
    def forward(
        contribution,
        coeff_real,
        coeff_imag,
        base_samples,
        tx_count,
        rx_count,
        accumulation_strategy,
        combine_domain,
    ):
        samples = dict(base_samples)
        samples["contribution"] = contribution
        matrices, bin_sums = bdpt_accumulate_connection_samples_forward_ad(
            samples,
            tx_count=tx_count,
            rx_count=rx_count,
            accumulation_strategy=accumulation_strategy,
            combine_domain=combine_domain,
            coeff_real=coeff_real,
            coeff_imag=coeff_imag,
        )
        # Flat tensor output tuple: the six differentiable component matrices
        # followed by the coherent bin-sum buffers (empty for the power domain),
        # which are marked non-differentiable in setup_context (spec 6.4).
        return tuple(matrices[name] for name in _ACCUMULATE_MATRIX_FIELDS) + tuple(
            bin_sums
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        (
            contribution,
            coeff_real,
            coeff_imag,
            base_samples,
            tx_count,
            rx_count,
            accumulation_strategy,
            combine_domain,
        ) = inputs
        ctx.tx_count = int(tx_count)
        ctx.rx_count = int(rx_count)
        ctx.accumulation_strategy = accumulation_strategy
        ctx.combine_domain = combine_domain
        ctx.base_samples = {
            name: torch.autograd.forward_ad.unpack_dual(value).primal
            if isinstance(value, torch.Tensor)
            else value
            for name, value in base_samples.items()
        }
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (contribution, coeff_real, coeff_imag)
        )
        # The trailing outputs past the six component matrices are the coherent
        # bin sums S_b, retained for the backward but carrying no gradient.
        ctx.bin_sums = tuple(output[len(_ACCUMULATE_MATRIX_FIELDS):])
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        if ctx.bin_sums:
            ctx.mark_non_differentiable(*ctx.bin_sums)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        grad_matrices = grad_outputs[: len(_ACCUMULATE_MATRIX_FIELDS)]
        none_grads = (None,) * 8
        # The coherent VJP/JVP read the forward-retained bin sums (ctx.bin_sums),
        # not the sample coefficients, so the saved coeff tensors are unused here.
        contribution, _coeff_real, _coeff_imag = ctx.saved_tensors
        samples = dict(ctx.base_samples)
        samples["contribution"] = contribution
        need_contribution = bool(ctx.needs_input_grad[0])
        need_coeff = bool(ctx.needs_input_grad[1]) or bool(ctx.needs_input_grad[2])
        if not (need_contribution or need_coeff) or all(
            value is None for value in grad_matrices
        ):
            return none_grads
        out = bdpt_accumulate_connection_samples_backward(
            samples,
            tx_count=ctx.tx_count,
            rx_count=ctx.rx_count,
            combine_domain=ctx.combine_domain,
            bin_sums=ctx.bin_sums,
            grad_path_gain=grad_matrices[0],
            grad_los=grad_matrices[1],
            grad_reflection=grad_matrices[2],
            grad_diffraction=grad_matrices[3],
            grad_transmission=grad_matrices[4],
            grad_scattering=grad_matrices[5],
            need_grad_contribution=need_contribution,
            need_grad_coeff=need_coeff,
        )
        return (
            out["grad_contribution"] if need_contribution else None,
            out["grad_coeff_real"] if bool(ctx.needs_input_grad[1]) else None,
            out["grad_coeff_imag"] if bool(ctx.needs_input_grad[2]) else None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_contribution,
        t_coeff_real,
        t_coeff_imag,
        _t_base_samples,
        _t_tx_count,
        _t_rx_count,
        _t_strategy,
        _t_combine,
    ):
        # The coherent VJP/JVP read the forward-retained bin sums (ctx.bin_sums),
        # not the sample coefficients, so the saved coeff tensors are unused here.
        contribution, _coeff_real, _coeff_imag = ctx.saved_tensors
        samples = dict(ctx.base_samples)
        samples["contribution"] = contribution
        tangent_contribution = _ad_native_tangent_or_none(t_contribution)
        tangent_coeff_real = _ad_native_tangent_or_none(t_coeff_real)
        tangent_coeff_imag = _ad_native_tangent_or_none(t_coeff_imag)
        n_out = len(_ACCUMULATE_MATRIX_FIELDS) + len(ctx.bin_sums)
        if (
            tangent_contribution is None
            and tangent_coeff_real is None
            and tangent_coeff_imag is None
        ):
            return (None,) * n_out
        with torch_compat.disable_functorch():
            out = bdpt_accumulate_connection_samples_jvp(
                samples,
                tx_count=ctx.tx_count,
                rx_count=ctx.rx_count,
                combine_domain=ctx.combine_domain,
                bin_sums=ctx.bin_sums,
                tangent_contribution=tangent_contribution,
                tangent_coeff_real=tangent_coeff_real,
                tangent_coeff_imag=tangent_coeff_imag,
            )
        tangent_matrices = (
            out["tangent_path_gain"],
            out["tangent_los"],
            out["tangent_reflection"],
            out["tangent_diffraction"],
            out["tangent_transmission"],
            out["tangent_scattering"],
        )
        # Bin-sum outputs are non-differentiable: their forward-mode tangent is
        # None.
        return tangent_matrices + (None,) * len(ctx.bin_sums)


def bdpt_accumulate_connection_samples_ad(
    samples: dict[str, torch.Tensor],
    *,
    tx_count: int,
    rx_count: int,
    accumulation_strategy: str = "atomic",
    combine_domain: str = "power",
    coeff_real: torch.Tensor | None = None,
    coeff_imag: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_accumulate_connection_samples` (spec 6.4)."""

    device = samples["contribution"].device
    if combine_domain == "coherent":
        if coeff_real is None or coeff_imag is None:
            raise ValueError("coherent combine requires coeff_real and coeff_imag")
    else:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        coeff_real = empty if coeff_real is None else coeff_real
        coeff_imag = empty if coeff_imag is None else coeff_imag
    base_samples = {
        name: value for name, value in samples.items() if name != "contribution"
    }
    outputs = _BdptAccumulateAdFunction.apply(
        samples["contribution"],
        coeff_real,
        coeff_imag,
        base_samples,
        int(tx_count),
        int(rx_count),
        accumulation_strategy,
        combine_domain,
    )
    matrices = outputs[: len(_ACCUMULATE_MATRIX_FIELDS)]
    return dict(zip(_ACCUMULATE_MATRIX_FIELDS, matrices, strict=True))


# ---------------------------------------------------------------------------
# 6.3 endpoint connection samples
# ---------------------------------------------------------------------------


_CONNECTION_FIELDS = (
    "topology",
    "contribution",
    "pdf",
    "mis_weight",
    "component_id",
    "valid",
    "tx_id",
    "rx_id",
    "grid_linear_id",
    "light_depth",
    "sensor_depth",
    "path_length_m",
)
_CONNECTION_CONTRIBUTION_INDEX = _CONNECTION_FIELDS.index("contribution")


class _BdptEndpointConnectionAdFunction(torch.autograd.Function):
    """Differentiable endpoint (LoS/NEE) connection contribution (spec 6.3).

    Differentiable inputs: the light and sensor subpath fields (8 tensors), the
    carrier frequency and ``tx_power`` (P_src). Frozen (reject loudly): the
    connection length L, samples_per_tx N, visibility, MIS mode, component_id,
    and every structural connection field. Only ``contribution`` is a
    differentiable output; the other 11 schema fields are frozen structure."""

    @staticmethod
    def forward(
        light_field_real,
        light_field_imag,
        light_throughput_real,
        light_throughput_imag,
        sensor_field_real,
        sensor_field_imag,
        sensor_throughput_real,
        sensor_throughput_imag,
        frequency,
        frequency_value,
        tx_power,
        base_light,
        base_sensor,
        params,
    ):
        light = _subpath_with_fields(
            base_light,
            light_field_real,
            light_field_imag,
            light_throughput_real,
            light_throughput_imag,
        )
        sensor = _subpath_with_fields(
            base_sensor,
            sensor_field_real,
            sensor_field_imag,
            sensor_throughput_real,
            sensor_throughput_imag,
        )
        out = bdpt_endpoint_connection_samples(
            light,
            sensor,
            frequency_hz=frequency_value,
            samples_per_tx=params["samples_per_tx"],
            max_paths=params["max_paths"],
            mis=params["mis"],
            beta=params["beta"],
            strategy_count=params["strategy_count"],
        )
        return tuple(out[name] for name in _CONNECTION_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        (
            light_field_real,
            light_field_imag,
            light_throughput_real,
            light_throughput_imag,
            sensor_field_real,
            sensor_field_imag,
            sensor_throughput_real,
            sensor_throughput_imag,
            frequency,
            frequency_value,
            tx_power,
            base_light,
            base_sensor,
            params,
        ) = inputs
        ctx.frequency_value = frequency_value
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.params = params

        def _detach_dict(source):
            return {
                name: torch.autograd.forward_ad.unpack_dual(value).primal
                if isinstance(value, torch.Tensor)
                else value
                for name, value in source.items()
            }

        ctx.base_light = _detach_dict(base_light)
        ctx.base_sensor = _detach_dict(base_sensor)
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (
                light_field_real,
                light_field_imag,
                light_throughput_real,
                light_throughput_imag,
                sensor_field_real,
                sensor_field_imag,
                sensor_throughput_real,
                sensor_throughput_imag,
                tx_power,
            )
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        structural = [
            output[index]
            for index, name in enumerate(_CONNECTION_FIELDS)
            if name != "contribution"
        ]
        ctx.mark_non_differentiable(*structural)

    @staticmethod
    def _light_sensor(ctx):
        saved = ctx.saved_tensors
        light = _subpath_with_fields(
            ctx.base_light, saved[0], saved[1], saved[2], saved[3]
        )
        sensor = _subpath_with_fields(
            ctx.base_sensor, saved[4], saved[5], saved[6], saved[7]
        )
        return light, sensor

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        none_grads = (None,) * 14
        need_field = any(bool(ctx.needs_input_grad[i]) for i in range(8))
        need_frequency = bool(ctx.needs_input_grad[8])
        need_tx_power = bool(ctx.needs_input_grad[10])
        grad_contribution = grad_outputs[_CONNECTION_CONTRIBUTION_INDEX]
        if not (need_field or need_frequency or need_tx_power) or (
            grad_contribution is None
        ):
            return none_grads
        light, sensor = _BdptEndpointConnectionAdFunction._light_sensor(ctx)
        params = ctx.params
        out = bdpt_endpoint_connection_samples_backward(
            light,
            sensor,
            frequency_hz=ctx.frequency_value,
            samples_per_tx=params["samples_per_tx"],
            mis=params["mis"],
            beta=params["beta"],
            strategy_count=params["strategy_count"],
            max_paths=params["max_paths"],
            grad_contribution=grad_contribution,
            need_grad_field=need_field,
            need_grad_frequency=need_frequency,
            need_grad_tx_power=need_tx_power,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_light_field_real"] if ctx.needs_input_grad[0] else None,
            out["grad_light_field_imag"] if ctx.needs_input_grad[1] else None,
            None,
            None,
            out["grad_sensor_field_real"] if ctx.needs_input_grad[4] else None,
            out["grad_sensor_field_imag"] if ctx.needs_input_grad[5] else None,
            None,
            None,
            grad_frequency,
            None,
            out["grad_tx_power"] if need_tx_power else None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        light, sensor = _BdptEndpointConnectionAdFunction._light_sensor(ctx)
        params = ctx.params
        tangent_frequency = _ad_frequency_tangent(tangents[8])
        payload = {
            "tangent_light_field_real": _ad_native_tangent_or_none(tangents[0]),
            "tangent_light_field_imag": _ad_native_tangent_or_none(tangents[1]),
            "tangent_sensor_field_real": _ad_native_tangent_or_none(tangents[4]),
            "tangent_sensor_field_imag": _ad_native_tangent_or_none(tangents[5]),
            "tangent_tx_power": _ad_native_tangent_or_none(tangents[10]),
        }
        if tangent_frequency == 0.0 and all(v is None for v in payload.values()):
            return (None,) * len(_CONNECTION_FIELDS)
        with torch_compat.disable_functorch():
            out = bdpt_endpoint_connection_samples_jvp(
                light,
                sensor,
                frequency_hz=ctx.frequency_value,
                samples_per_tx=params["samples_per_tx"],
                mis=params["mis"],
                beta=params["beta"],
                strategy_count=params["strategy_count"],
                max_paths=params["max_paths"],
                tangent_frequency=tangent_frequency,
                **payload,
            )
        result: list[torch.Tensor | None] = [None] * len(_CONNECTION_FIELDS)
        result[_CONNECTION_CONTRIBUTION_INDEX] = out["tangent_contribution"]
        return tuple(result)


def bdpt_endpoint_connection_samples_ad(
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    tx_power: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
    samples_per_tx: int,
    max_paths: int | None = None,
    mis: str = "power_heuristic",
    beta: float = 2.0,
    strategy_count: int = 1,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_endpoint_connection_samples` (spec 6.3).

    ``tx_power`` (P_src) is carried so its gradient (``grad_tx_power``) is
    accumulated by the native backward; it is not consumed by the primal
    forward (source power rides the light subpath) so the primal is bitwise
    unchanged."""

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    base_light = {
        name: value for name, value in light.items() if name not in _SUBPATH_DIFF_FIELDS
    }
    base_sensor = {
        name: value for name, value in sensor.items() if name not in _SUBPATH_DIFF_FIELDS
    }
    params = {
        "samples_per_tx": int(samples_per_tx),
        "max_paths": max_paths,
        "mis": mis,
        "beta": float(beta),
        "strategy_count": int(strategy_count),
    }
    values = _BdptEndpointConnectionAdFunction.apply(
        light["field_real"],
        light["field_imag"],
        light["throughput_real"],
        light["throughput_imag"],
        sensor["field_real"],
        sensor["field_imag"],
        sensor["throughput_real"],
        sensor["throughput_imag"],
        frequency,
        float(frequency_value),
        tx_power,
        base_light,
        base_sensor,
        params,
    )
    return dict(zip(_CONNECTION_FIELDS, values, strict=True))


__all__ = [
    "bdpt_accumulate_connection_samples_ad",
    "bdpt_endpoint_connection_samples_ad",
    "bdpt_finalize_component_maps_ad",
    "bdpt_finalize_point_components_ad",
    "bdpt_reflected_light_subpath_state_ad",
    "bdpt_transmitted_light_subpath_state_ad",
]
