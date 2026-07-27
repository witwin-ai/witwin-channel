from __future__ import annotations

import torch

from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.autograd_contracts import (
    _ad_checked_tangent,
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
from witwin.channel.runtime.symbols import required_symbol as _required_native_op

from . import _liveness
from .functional import (
    _COUPLED_OUTPUT_FIELDS,
    _FIELD_AD_OUTPUT_FIELDS,
    _FIELD_AD_TANGENT_FIELDS,
    _WEDGE_OUTPUT_FIELDS,
    _validate_wedge_valid,
    field_free_space_backward,
    field_free_space_jvp,
    field_reflection_sequence_backward,
    field_reflection_sequence_jvp,
    field_transmission_sequence_backward,
    field_transmission_sequence_jvp,
)


class _FieldFreeSpaceAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable free-space transport.

    Frequency and endpoints are differentiable; power and polarizations are
    fixed. Float64 inputs use the strict-double companion for gradcheck.
    """

    @staticmethod
    def forward(
        source,
        target,
        tx_power,
        tx_polarization,
        rx_polarization,
        frequency,
        frequency_value,
        liveness,
    ):
        op_name = (
            "field_free_space_fwd64"
            if source.dtype == torch.float64
            else "field_free_space"
        )
        out = _required_native_op(op_name)(
            source,
            target,
            tx_power,
            tx_polarization,
            rx_polarization,
            frequency_value,
        )
        return tuple(out[name] for name in _FIELD_AD_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        (
            source,
            target,
            tx_power,
            tx_polarization,
            rx_polarization,
            frequency,
            frequency_value,
            liveness,
        ) = inputs
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (source, target, tx_power, tx_polarization, rx_polarization)
        )
        ctx.frequency_value = frequency_value
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        # Both decided by the wrapper, where forward duals are still visible;
        # Function.apply unpacks them before this hook runs.
        ctx.geometry_live, ctx.direction_live = liveness
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        _liveness.mark_dead_outputs(ctx, output)

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx,
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        grad_path_length,
        grad_delay,
        grad_direction,
    ):
        none_grads = (None,) * 8
        _ad_reject_fixed_inputs(
            "field_free_space_ad",
            ctx.needs_input_grad,
            (
                (2, "tx_power"),
                (3, "tx_polarization"),
                (4, "rx_polarization"),
            ),
        )
        need_geometry = bool(ctx.needs_input_grad[0]) or bool(
            ctx.needs_input_grad[1]
        )
        need_frequency = bool(ctx.needs_input_grad[5])
        grad_direction = _liveness.direction_cotangent(ctx, grad_direction)
        grads = (
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
            grad_path_length,
            grad_delay,
            grad_direction,
        )
        if not (need_geometry or need_frequency) or all(
            value is None for value in grads
        ):
            return none_grads
        source, target, tx_power, tx_polarization, rx_polarization = ctx.saved_tensors
        out = field_free_space_backward(
            source,
            target,
            tx_power,
            tx_polarization,
            rx_polarization,
            frequency_hz=ctx.frequency_value,
            grad_field_vector=grad_field_vector,
            grad_coefficient=grad_coefficient,
            grad_path_field=grad_path_field,
            grad_path_gain=grad_path_gain,
            grad_path_length=grad_path_length,
            grad_delay=grad_delay,
            grad_direction=grad_direction,
            need_grad_frequency=need_frequency,
            need_grad_geometry=need_geometry,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_source"] if ctx.needs_input_grad[0] else None,
            out["grad_target"] if ctx.needs_input_grad[1] else None,
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
        t_source,
        t_target,
        t_tx_power,
        t_tx_pol,
        t_rx_pol,
        t_frequency,
        _t_frequency_value,
        _t_liveness,
    ):
        _ad_reject_fixed_tangents(
            "field_free_space_ad",
            (
                (t_tx_power, "tx_power"),
                (t_tx_pol, "tx_polarization"),
                (t_rx_pol, "rx_polarization"),
            ),
        )
        saved = ctx.saved_tensors
        tangent_source = _ad_geometry_tangent(
            "field_free_space_ad tangent_source", t_source, saved[0]
        )
        tangent_target = _ad_geometry_tangent(
            "field_free_space_ad tangent_target", t_target, saved[1]
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if (
            tangent_frequency == 0.0
            and tangent_source is None
            and tangent_target is None
        ):
            return (None,) * 7
        source, target, tx_power, tx_polarization, rx_polarization = saved
        with torch_compat.disable_functorch():
            out = field_free_space_jvp(
                _ad_native_tensor(source),
                _ad_native_tensor(target),
                _ad_native_tensor(tx_power),
                _ad_native_tensor(tx_polarization),
                _ad_native_tensor(rx_polarization),
                frequency_hz=ctx.frequency_value,
                tangent_frequency=tangent_frequency,
                tangent_source=tangent_source,
                tangent_target=tangent_target,
            )
        return _liveness.direction_tangents(ctx, out)


def field_free_space_ad(
    source: torch.Tensor,
    target: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
    direction_live: bool = False,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_free_space` (frequency only in AD-1).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not
    supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldFreeSpaceAdFunction.apply(
        source,
        target,
        tx_power,
        tx_polarization,
        rx_polarization,
        frequency,
        float(frequency_value),
        _liveness.ad_liveness(direction_live, source, target),
    )
    return dict(zip(_FIELD_AD_OUTPUT_FIELDS, values, strict=True))


class _FieldReflectionSequenceAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable reflection transport.

    Frequency, hit geometry, and per-bounce material scalars except ``mu_r``
    are differentiable; power, polarizations, and ``mu_r`` stay fixed.
    """

    @staticmethod
    def forward(
        source,
        target,
        interaction_positions,
        interaction_normals,
        tx_power,
        tx_polarization,
        rx_polarization,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        thickness,
        frequency,
        frequency_value,
        liveness,
    ):
        out = _required_native_op("field_reflection_sequence")(
            source,
            target,
            interaction_positions,
            interaction_normals,
            tx_power,
            tx_polarization,
            rx_polarization,
            eps_r,
            sigma_e,
            mu_r,
            gain,
            thickness,
            frequency_value,
        )
        return tuple(out[name] for name in _FIELD_AD_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[12]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:12]
        )
        ctx.frequency_value = inputs[13]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        # Both decided by the wrapper, where forward duals are still visible.
        ctx.geometry_live, ctx.direction_live = inputs[14]
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        _liveness.mark_dead_outputs(ctx, output)

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx,
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        grad_path_length,
        grad_delay,
        grad_direction,
    ):
        none_grads = (None,) * 15
        _ad_reject_fixed_inputs(
            "field_reflection_sequence_ad",
            ctx.needs_input_grad,
            (
                (4, "tx_power"),
                (5, "tx_polarization"),
                (6, "rx_polarization"),
                (9, "mu_r"),
            ),
        )
        need_geometry = any(bool(ctx.needs_input_grad[i]) for i in range(4))
        need_eps = bool(ctx.needs_input_grad[7])
        need_sigma = bool(ctx.needs_input_grad[8])
        need_gain = bool(ctx.needs_input_grad[10])
        need_thickness = bool(ctx.needs_input_grad[11])
        need_frequency = bool(ctx.needs_input_grad[12])
        grad_direction = _liveness.direction_cotangent(ctx, grad_direction)
        grads = (
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
            grad_path_length,
            grad_delay,
            grad_direction,
        )
        if not (
            need_geometry
            or need_eps
            or need_sigma
            or need_gain
            or need_thickness
            or need_frequency
        ) or all(value is None for value in grads):
            return none_grads
        (
            source,
            target,
            interaction_positions,
            interaction_normals,
            tx_power,
            tx_polarization,
            rx_polarization,
            eps_r,
            sigma_e,
            mu_r,
            gain,
            thickness,
        ) = ctx.saved_tensors
        out = field_reflection_sequence_backward(
            source,
            target,
            interaction_positions,
            interaction_normals,
            tx_power,
            tx_polarization,
            rx_polarization,
            eps_r,
            sigma_e,
            mu_r,
            gain,
            thickness,
            frequency_hz=ctx.frequency_value,
            grad_field_vector=grad_field_vector,
            grad_coefficient=grad_coefficient,
            grad_path_field=grad_path_field,
            grad_path_gain=grad_path_gain,
            grad_path_length=grad_path_length,
            grad_delay=grad_delay,
            grad_direction=grad_direction,
            need_grad_eps_r=need_eps,
            need_grad_sigma_e=need_sigma,
            need_grad_gain=need_gain,
            need_grad_thickness=need_thickness,
            need_grad_frequency=need_frequency,
            need_grad_geometry=need_geometry,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_source"] if ctx.needs_input_grad[0] else None,
            out["grad_target"] if ctx.needs_input_grad[1] else None,
            out["grad_interaction_positions"] if ctx.needs_input_grad[2] else None,
            out["grad_interaction_normals"] if ctx.needs_input_grad[3] else None,
            None,
            None,
            None,
            out["grad_eps_r"] if need_eps else None,
            out["grad_sigma_e"] if need_sigma else None,
            None,
            out["grad_gain"] if need_gain else None,
            out["grad_thickness"] if need_thickness else None,
            grad_frequency,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_source,
        t_target,
        t_positions,
        t_normals,
        t_tx_power,
        t_tx_pol,
        t_rx_pol,
        t_eps_r,
        t_sigma_e,
        t_mu_r,
        t_gain,
        t_thickness,
        t_frequency,
        _t_frequency_value,
        _t_liveness,
    ):
        _ad_reject_fixed_tangents(
            "field_reflection_sequence_ad",
            (
                (t_tx_power, "tx_power"),
                (t_tx_pol, "tx_polarization"),
                (t_rx_pol, "rx_polarization"),
                (t_mu_r, "mu_r"),
            ),
        )
        saved = ctx.saved_tensors
        eps_shape = tuple(saved[7].shape)
        tangent_source = _ad_geometry_tangent(
            "field_reflection_sequence_ad tangent_source", t_source, saved[0]
        )
        tangent_target = _ad_geometry_tangent(
            "field_reflection_sequence_ad tangent_target", t_target, saved[1]
        )
        tangent_positions = _ad_geometry_tangent(
            "field_reflection_sequence_ad tangent_interaction_positions",
            t_positions,
            saved[2],
        )
        tangent_normals = _ad_geometry_tangent(
            "field_reflection_sequence_ad tangent_interaction_normals",
            t_normals,
            saved[3],
        )
        tangent_eps = _ad_checked_tangent(
            "field_reflection_sequence_ad tangent_eps_r",
            _ad_native_tangent_or_none(t_eps_r),
            eps_shape,
        )
        tangent_sigma = _ad_checked_tangent(
            "field_reflection_sequence_ad tangent_sigma_e",
            _ad_native_tangent_or_none(t_sigma_e),
            eps_shape,
        )
        tangent_gain = _ad_checked_tangent(
            "field_reflection_sequence_ad tangent_gain",
            _ad_native_tangent_or_none(t_gain),
            eps_shape,
        )
        tangent_thickness = _ad_checked_tangent(
            "field_reflection_sequence_ad tangent_thickness",
            _ad_native_tangent_or_none(t_thickness),
            eps_shape,
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if (
            tangent_eps is None
            and tangent_sigma is None
            and tangent_gain is None
            and tangent_thickness is None
            and tangent_source is None
            and tangent_target is None
            and tangent_positions is None
            and tangent_normals is None
            and tangent_frequency == 0.0
        ):
            return (None,) * 7
        with torch_compat.disable_functorch():
            out = field_reflection_sequence_jvp(
                *(_ad_native_tensor(value) for value in saved),
                frequency_hz=ctx.frequency_value,
                tangent_eps_r=tangent_eps,
                tangent_sigma_e=tangent_sigma,
                tangent_gain=tangent_gain,
                tangent_thickness=tangent_thickness,
                tangent_frequency=tangent_frequency,
                tangent_source=tangent_source,
                tangent_target=tangent_target,
                tangent_interaction_positions=tangent_positions,
                tangent_interaction_normals=tangent_normals,
            )
        return _liveness.direction_tangents(ctx, out)


def field_reflection_sequence_ad(
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    thickness: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
    direction_live: bool = False,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_reflection_sequence` (materials + frequency).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not
    supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldReflectionSequenceAdFunction.apply(
        source,
        target,
        interaction_positions,
        interaction_normals,
        tx_power,
        tx_polarization,
        rx_polarization,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        thickness,
        frequency,
        float(frequency_value),
        _liveness.ad_liveness(
            direction_live, source, target, interaction_positions, interaction_normals
        ),
    )
    return dict(zip(_FIELD_AD_OUTPUT_FIELDS, values, strict=True))


class _FieldTransmissionSequenceAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable transmission transport (plan 07 AD-1/AD-2).

    Differentiable inputs: CSR layer thickness / eps_r / sigma_e, frequency,
    and the hit geometry (source, target, interaction_normals). The straight
    transmission field is independent of the crossing points themselves, so
    interaction_positions receives an exact zero gradient (None). tx_power,
    the polarizations, layer_mu_r, material ids and valid masks are fixed;
    requesting their gradient fails loudly. Layer gradients accumulate
    atomically across paths because the CSR store is shared by every wall
    crossing.
    """

    @staticmethod
    def forward(
        path_valid,
        source,
        target,
        interaction_positions,
        interaction_normals,
        interaction_material_id,
        interaction_valid,
        tx_power,
        tx_polarization,
        rx_polarization,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        frequency_value,
    ):
        out = _required_native_op("field_transmission_sequence")(
            path_valid,
            source,
            target,
            interaction_positions,
            interaction_normals,
            interaction_material_id,
            interaction_valid,
            tx_power,
            tx_polarization,
            rx_polarization,
            layer_offset,
            layer_count,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            frequency_value,
        )
        return tuple(out[name] for name in _FIELD_AD_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[16]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:16]
        )
        ctx.frequency_value = inputs[17]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.geometry_live = _ad_geometry_live(*inputs[1:5])
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        # ADR-043: RayD owns this family's direction seam, so the arrival
        # direction is a declared non-differentiable output on every route.
        ctx.direction_live = False
        _liveness.mark_dead_outputs(ctx, output)

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx,
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        grad_path_length,
        grad_delay,
        _grad_direction,
    ):
        none_grads = (None,) * 18
        _ad_reject_fixed_inputs(
            "field_transmission_sequence_ad",
            ctx.needs_input_grad,
            (
                (7, "tx_power"),
                (8, "tx_polarization"),
                (9, "rx_polarization"),
                (15, "layer_mu_r"),
            ),
        )
        # interaction_positions (index 3) never enters the straight-path
        # field: its gradient is exactly zero, so it does not drive a launch.
        need_geometry = (
            bool(ctx.needs_input_grad[1])
            or bool(ctx.needs_input_grad[2])
            or bool(ctx.needs_input_grad[4])
        )
        need_thickness = bool(ctx.needs_input_grad[12])
        need_eps = bool(ctx.needs_input_grad[13])
        need_sigma = bool(ctx.needs_input_grad[14])
        need_frequency = bool(ctx.needs_input_grad[16])
        grads = (
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
            grad_path_length,
            grad_delay,
        )
        if not (
            need_geometry or need_thickness or need_eps or need_sigma
            or need_frequency
        ) or all(value is None for value in grads):
            return none_grads
        saved = ctx.saved_tensors
        out = field_transmission_sequence_backward(
            *saved,
            frequency_hz=ctx.frequency_value,
            grad_field_vector=grad_field_vector,
            grad_coefficient=grad_coefficient,
            grad_path_field=grad_path_field,
            grad_path_gain=grad_path_gain,
            grad_path_length=grad_path_length,
            grad_delay=grad_delay,
            need_grad_layer_thickness=need_thickness,
            need_grad_layer_eps_r=need_eps,
            need_grad_layer_sigma_e=need_sigma,
            need_grad_frequency=need_frequency,
            need_grad_geometry=need_geometry,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            None,
            out["grad_source"] if ctx.needs_input_grad[1] else None,
            out["grad_target"] if ctx.needs_input_grad[2] else None,
            None,
            out["grad_interaction_normals"] if ctx.needs_input_grad[4] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            out["grad_layer_thickness_m"] if need_thickness else None,
            out["grad_layer_eps_r"] if need_eps else None,
            out["grad_layer_sigma_e"] if need_sigma else None,
            None,
            grad_frequency,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        _t_path_valid,
        t_source,
        t_target,
        t_positions,
        t_normals,
        _t_material_id,
        _t_interaction_valid,
        t_tx_power,
        t_tx_pol,
        t_rx_pol,
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
            "field_transmission_sequence_ad",
            (
                (t_tx_power, "tx_power"),
                (t_tx_pol, "tx_polarization"),
                (t_rx_pol, "rx_polarization"),
                (t_layer_mu_r, "layer_mu_r"),
            ),
        )
        saved = ctx.saved_tensors
        layer_shape = tuple(saved[12].shape)
        tangent_source = _ad_geometry_tangent(
            "field_transmission_sequence_ad tangent_source", t_source, saved[1]
        )
        tangent_target = _ad_geometry_tangent(
            "field_transmission_sequence_ad tangent_target", t_target, saved[2]
        )
        tangent_positions = _ad_geometry_tangent(
            "field_transmission_sequence_ad tangent_interaction_positions",
            t_positions,
            saved[3],
        )
        tangent_normals = _ad_geometry_tangent(
            "field_transmission_sequence_ad tangent_interaction_normals",
            t_normals,
            saved[4],
        )
        tangent_thickness = _ad_checked_tangent(
            "field_transmission_sequence_ad tangent_layer_thickness_m",
            _ad_native_tangent_or_none(t_layer_thickness),
            layer_shape,
        )
        tangent_eps = _ad_checked_tangent(
            "field_transmission_sequence_ad tangent_layer_eps_r",
            _ad_native_tangent_or_none(t_layer_eps_r),
            layer_shape,
        )
        tangent_sigma = _ad_checked_tangent(
            "field_transmission_sequence_ad tangent_layer_sigma_e",
            _ad_native_tangent_or_none(t_layer_sigma_e),
            layer_shape,
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if (
            tangent_thickness is None
            and tangent_eps is None
            and tangent_sigma is None
            and tangent_source is None
            and tangent_target is None
            and tangent_positions is None
            and tangent_normals is None
            and tangent_frequency == 0.0
        ):
            return (None,) * 7
        with torch_compat.disable_functorch():
            out = field_transmission_sequence_jvp(
                *(_ad_native_tensor(value) for value in saved),
                frequency_hz=ctx.frequency_value,
                tangent_layer_thickness_m=tangent_thickness,
                tangent_layer_eps_r=tangent_eps,
                tangent_layer_sigma_e=tangent_sigma,
                tangent_frequency=tangent_frequency,
                tangent_source=tangent_source,
                tangent_target=tangent_target,
                tangent_interaction_positions=tangent_positions,
                tangent_interaction_normals=tangent_normals,
            )
        tangents = tuple(out[name] for name in _FIELD_AD_TANGENT_FIELDS)
        if not ctx.geometry_live:
            return (*tangents[:4], None, None, None)
        return (*tangents, None)


def field_transmission_sequence_ad(
    path_valid: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    interaction_material_id: torch.Tensor,
    interaction_valid: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
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
    """Differentiable :func:`field_transmission_sequence` (layers + frequency).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not
    supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldTransmissionSequenceAdFunction.apply(
        path_valid,
        source,
        target,
        interaction_positions,
        interaction_normals,
        interaction_material_id,
        interaction_valid,
        tx_power,
        tx_polarization,
        rx_polarization,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        float(frequency_value),
    )
    return dict(zip(_FIELD_AD_OUTPUT_FIELDS, values, strict=True))


class _FieldDiffractionWedgeAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable UTD wedge field (plan 07 AD-4).

    Differentiable inputs: both faces' eps_r / sigma_e / gain, frequency,
    endpoints, and optional per-row winner vertices v0/v1 plus each face's
    opposite vertex. The kernel rebuilds edge tables so mesh-vertex gradients
    reach edge geometry. Frozen edge tables, valid masks, mu_r and tx_power
    stay fixed and reject gradients. The stationary edge point is re-solved
    inside the kernel, preserving its endpoint and vertex gradient motion.
    """

    @staticmethod
    def forward(
        valid,
        source,
        target,
        edge_position,
        edge_direction,
        edge_t_min,
        edge_t_max,
        edge_n0,
        edge_n1,
        exterior_angle,
        face0_valid,
        face0_eps_r,
        face0_sigma_e,
        face0_mu_r,
        face0_gain,
        face1_valid,
        face1_eps_r,
        face1_sigma_e,
        face1_mu_r,
        face1_gain,
        tx_power,
        frequency,
        vertex_v0,
        vertex_v1,
        vertex_opp0,
        vertex_opp1,
        edge_boundary,
        frequency_value,
    ):
        out = _required_native_op("field_diffraction_wedge")(
            valid,
            source,
            target,
            edge_position,
            edge_direction,
            edge_t_min,
            edge_t_max,
            edge_n0,
            edge_n1,
            exterior_angle,
            face0_valid,
            face0_eps_r,
            face0_sigma_e,
            face0_mu_r,
            face0_gain,
            face1_valid,
            face1_eps_r,
            face1_sigma_e,
            face1_mu_r,
            face1_gain,
            tx_power,
            frequency_value,
            vertex_v0,
            vertex_v1,
            vertex_opp0,
            vertex_opp1,
            edge_boundary,
            # ISB boundary taper (ADR-017), D member. Always 0.0: taper + AD is
            # refused by the deterministic/path pipelines (gate 3, C1 clearance
            # companion pending), so the differentiable twin never tapers. The
            # argument is threaded for lockstep completeness of the guarded path.
            0.0,
        )
        return tuple(out[name] for name in _WEDGE_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[21]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:21]
        )
        vertex_primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            if isinstance(value, torch.Tensor)
            else value
            for value in inputs[22:27]
        )
        ctx.has_vertices = vertex_primals[0] is not None
        ctx.frequency_value = inputs[27]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.geometry_live = _ad_geometry_live(inputs[1], inputs[2])
        saved = primals + tuple(
            value for value in vertex_primals if value is not None
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    def _unpack_saved(ctx):
        saved = ctx.saved_tensors
        primals = saved[:21]
        vertices = saved[21:26] if ctx.has_vertices else (None,) * 5
        return primals, vertices

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_field_vector, grad_direction):
        none_grads = (None,) * 28
        _ad_reject_fixed_inputs(
            "field_diffraction_wedge_ad",
            ctx.needs_input_grad,
            (
                (0, "valid"),
                (3, "edge_position"),
                (4, "edge_direction"),
                (5, "edge_t_min"),
                (6, "edge_t_max"),
                (7, "edge_n0"),
                (8, "edge_n1"),
                (9, "exterior_angle"),
                (10, "face0_valid"),
                (13, "face0_mu_r"),
                (15, "face1_valid"),
                (18, "face1_mu_r"),
                (20, "tx_power"),
                (26, "edge_boundary"),
            ),
        )
        need_geometry = bool(ctx.needs_input_grad[1]) or bool(ctx.needs_input_grad[2])
        need_material = any(
            bool(ctx.needs_input_grad[index]) for index in (11, 12, 14, 16, 17, 19)
        )
        need_frequency = bool(ctx.needs_input_grad[21])
        need_vertices = any(bool(ctx.needs_input_grad[i]) for i in (22, 23, 24, 25))
        if not (need_geometry or need_material or need_frequency or need_vertices) or (
            grad_field_vector is None and grad_direction is None
        ):
            return none_grads
        primals, vertices = _FieldDiffractionWedgeAdFunction._unpack_saved(ctx)
        out = _required_native_op("field_diffraction_wedge_backward")(
            *primals,
            ctx.frequency_value,
            *vertices,
            grad_field_vector,
            grad_direction,
            need_material,
            need_frequency,
            need_geometry,
            need_vertices,
            # ADR-017 D-member width; always 0.0 (taper + AD is guarded off).
            0.0,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            None,
            out["grad_source"] if ctx.needs_input_grad[1] else None,
            out["grad_target"] if ctx.needs_input_grad[2] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            out["grad_face0_eps_r"] if ctx.needs_input_grad[11] else None,
            out["grad_face0_sigma_e"] if ctx.needs_input_grad[12] else None,
            None,
            out["grad_face0_gain"] if ctx.needs_input_grad[14] else None,
            None,
            out["grad_face1_eps_r"] if ctx.needs_input_grad[16] else None,
            out["grad_face1_sigma_e"] if ctx.needs_input_grad[17] else None,
            None,
            out["grad_face1_gain"] if ctx.needs_input_grad[19] else None,
            None,
            grad_frequency,
            out["grad_vertex_v0"] if ctx.needs_input_grad[22] else None,
            out["grad_vertex_v1"] if ctx.needs_input_grad[23] else None,
            out["grad_vertex_opp0"] if ctx.needs_input_grad[24] else None,
            out["grad_vertex_opp1"] if ctx.needs_input_grad[25] else None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        _ad_reject_fixed_tangents(
            "field_diffraction_wedge_ad",
            (
                (tangents[0], "valid"),
                (tangents[3], "edge_position"),
                (tangents[4], "edge_direction"),
                (tangents[5], "edge_t_min"),
                (tangents[6], "edge_t_max"),
                (tangents[7], "edge_n0"),
                (tangents[8], "edge_n1"),
                (tangents[9], "exterior_angle"),
                (tangents[13], "face0_mu_r"),
                (tangents[18], "face1_mu_r"),
                (tangents[20], "tx_power"),
            ),
        )
        primals, vertices = _FieldDiffractionWedgeAdFunction._unpack_saved(ctx)
        scalar_shape = tuple(primals[11].shape)
        tangent_source = _ad_geometry_tangent(
            "field_diffraction_wedge_ad tangent_source", tangents[1], primals[1])
        tangent_target = _ad_geometry_tangent(
            "field_diffraction_wedge_ad tangent_target", tangents[2], primals[2])
        material_tangents = {}
        for index, name in (
            (11, "face0_eps_r"),
            (12, "face0_sigma_e"),
            (14, "face0_gain"),
            (16, "face1_eps_r"),
            (17, "face1_sigma_e"),
            (19, "face1_gain"),
        ):
            material_tangents[name] = _ad_checked_tangent(
                f"field_diffraction_wedge_ad tangent_{name}",
                _ad_native_tangent_or_none(tangents[index]),
                scalar_shape,
            )
        vertex_tangents = []
        for index, name in (
            (22, "vertex_v0"),
            (23, "vertex_v1"),
            (24, "vertex_opp0"),
            (25, "vertex_opp1"),
        ):
            tangent = tangents[index] if index < len(tangents) else None
            vertex_tangents.append(
                _ad_native_tangent_or_none(
                    tangent if isinstance(tangent, torch.Tensor) else None
                )
            )
        tangent_frequency = _ad_frequency_tangent(tangents[21])
        if (
            tangent_source is None
            and tangent_target is None
            and tangent_frequency == 0.0
            and all(value is None for value in material_tangents.values())
            and all(value is None for value in vertex_tangents)
        ):
            return (None, None)
        with torch_compat.disable_functorch():
            out = _required_native_op("field_diffraction_wedge_jvp")(
                *(_ad_native_tensor(value) for value in primals),
                ctx.frequency_value,
                *(
                    _ad_native_tensor(value) if isinstance(value, torch.Tensor)
                    else value
                    for value in vertices
                ),
                tangent_source,
                tangent_target,
                material_tangents["face0_eps_r"],
                material_tangents["face0_sigma_e"],
                material_tangents["face0_gain"],
                material_tangents["face1_eps_r"],
                material_tangents["face1_sigma_e"],
                material_tangents["face1_gain"],
                float(tangent_frequency),
                *vertex_tangents,
                # ADR-017 D-member width; always 0.0 (taper + AD is guarded off).
                0.0,
            )
        return (out["tangent_field_vector"], out["tangent_direction"])


def field_diffraction_wedge_ad(
    valid: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    edge_position: torch.Tensor,
    edge_direction: torch.Tensor,
    edge_t_min: torch.Tensor,
    edge_t_max: torch.Tensor,
    edge_n0: torch.Tensor,
    edge_n1: torch.Tensor,
    exterior_angle: torch.Tensor,
    face0_valid: torch.Tensor,
    face0_eps_r: torch.Tensor,
    face0_sigma_e: torch.Tensor,
    face0_mu_r: torch.Tensor,
    face0_gain: torch.Tensor,
    face1_valid: torch.Tensor,
    face1_eps_r: torch.Tensor,
    face1_sigma_e: torch.Tensor,
    face1_mu_r: torch.Tensor,
    face1_gain: torch.Tensor,
    tx_power: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
    vertices: tuple[torch.Tensor, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_diffraction_wedge`.

    ``vertices`` optionally supplies the winner edge vertices as
    ``(v0, v1, opp0, opp1, edge_boundary)`` per row; the kernel then rebuilds
    the edge tables from them so mesh-vertex gradients exist (plan 07 section
    9.3 mesh-vertex x diffraction). ``frequency_value`` optionally carries
    the precomputed host scalar of ``frequency`` (one read per solve at the
    seam, audit M3); when not supplied it is read here, exactly once per
    apply.
    """

    _validate_wedge_valid(valid, source)
    if vertices is not None and len(vertices) != 5:
        raise ValueError(
            "vertices must hold (v0, v1, opp0, opp1, edge_boundary) per row"
        )
    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    vertex_args = vertices if vertices is not None else (None,) * 5
    values = _FieldDiffractionWedgeAdFunction.apply(
        valid,
        source,
        target,
        edge_position,
        edge_direction,
        edge_t_min,
        edge_t_max,
        edge_n0,
        edge_n1,
        exterior_angle,
        face0_valid,
        face0_eps_r,
        face0_sigma_e,
        face0_mu_r,
        face0_gain,
        face1_valid,
        face1_eps_r,
        face1_sigma_e,
        face1_mu_r,
        face1_gain,
        tx_power,
        frequency,
        *vertex_args,
        float(frequency_value),
    )
    return dict(zip(_WEDGE_OUTPUT_FIELDS, values, strict=True))


class _FieldCoupledRdAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable coupled R-D transport (plan 07 AD-4).

    Differentiable inputs: eps_r / sigma_e / gain / thickness for the
    reflection wall and both wedge faces (12 scalars per path), frequency,
    and the continuous geometry (source, target, reflection_position,
    edge_position). The wedge axis/normals/exterior angle, the reflection
    normal (a frozen wall plane), mu_r, tx_power and the polarizations stay
    fixed. The pseudo-infinite edge truncation factor is a frozen regularizer
    of the differentiation (see kernels/field_wedge_ad.cu).
    """

    @staticmethod
    def forward(
        source,
        target,
        reflection_position,
        reflection_normal,
        edge_position,
        edge_direction,
        edge_n0,
        edge_n1,
        exterior_angle,
        tx_power,
        tx_polarization,
        rx_polarization,
        refl_eps_r,
        refl_sigma_e,
        refl_mu_r,
        refl_gain,
        refl_thickness,
        w0_eps_r,
        w0_sigma_e,
        w0_mu_r,
        w0_gain,
        w0_thickness,
        w1_eps_r,
        w1_sigma_e,
        w1_mu_r,
        w1_gain,
        w1_thickness,
        frequency,
        reverse,
        frequency_value,
        edge_line_min,
        edge_line_max,
    ):
        out = _required_native_op("field_coupled_rd")(
            source,
            target,
            reflection_position,
            reflection_normal,
            edge_position,
            edge_direction,
            edge_n0,
            edge_n1,
            exterior_angle,
            tx_power,
            tx_polarization,
            rx_polarization,
            refl_eps_r,
            refl_sigma_e,
            refl_mu_r,
            refl_gain,
            refl_thickness,
            w0_eps_r,
            w0_sigma_e,
            w0_mu_r,
            w0_gain,
            w0_thickness,
            w1_eps_r,
            w1_sigma_e,
            w1_mu_r,
            w1_gain,
            w1_thickness,
            edge_line_min,
            edge_line_max,
            frequency_value,
            bool(reverse),
        )
        return tuple(out[name] for name in _COUPLED_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[27]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:27]
        )
        # edge_line_min / edge_line_max (inputs[30], inputs[31]) are frozen edge
        # bounds (G4): non-differentiable, but saved so the backward/jvp
        # companions can forward them to the native coupled kernels in the same
        # position as the primal.
        bounds = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (inputs[30], inputs[31])
        )
        ctx.frequency_value = inputs[29]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.reverse = bool(inputs[28])
        ctx.save_for_backward(*primals, *bounds)
        ctx.save_for_forward(*primals, *bounds)
        ctx.mark_non_differentiable(output[4])

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx,
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        _grad_direction,
    ):
        none_grads = (None,) * 32
        _ad_reject_fixed_inputs(
            "field_coupled_rd_ad",
            ctx.needs_input_grad,
            (
                (3, "reflection_normal"),
                (5, "edge_direction"),
                (6, "edge_n0"),
                (7, "edge_n1"),
                (8, "exterior_angle"),
                (9, "tx_power"),
                (10, "tx_polarization"),
                (11, "rx_polarization"),
                (14, "reflection_mu_r"),
                (19, "wedge_mu_r0"),
                (24, "wedge_mu_r1"),
                (30, "edge_line_min"),
                (31, "edge_line_max"),
            ),
        )
        need_geometry = any(
            bool(ctx.needs_input_grad[index]) for index in (0, 1, 2, 4)
        )
        need_eps = any(bool(ctx.needs_input_grad[index]) for index in (12, 17, 22))
        need_sigma = any(bool(ctx.needs_input_grad[index]) for index in (13, 18, 23))
        need_gain = any(bool(ctx.needs_input_grad[index]) for index in (15, 20, 25))
        need_thickness = any(
            bool(ctx.needs_input_grad[index]) for index in (16, 21, 26)
        )
        need_frequency = bool(ctx.needs_input_grad[27])
        grads = (grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain)
        if not (
            need_geometry
            or need_eps
            or need_sigma
            or need_gain
            or need_thickness
            or need_frequency
        ) or all(value is None for value in grads):
            return none_grads
        saved = ctx.saved_tensors
        out = _required_native_op("field_coupled_rd_backward")(
            *saved,
            ctx.frequency_value,
            ctx.reverse,
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
            need_eps,
            need_sigma,
            need_gain,
            need_thickness,
            need_frequency,
            need_geometry,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )

        def material_column(name: str, column: int, index: int):
            if not ctx.needs_input_grad[index]:
                return None
            return out[name][:, column]

        return (
            out["grad_source"] if ctx.needs_input_grad[0] else None,
            out["grad_target"] if ctx.needs_input_grad[1] else None,
            out["grad_reflection_position"] if ctx.needs_input_grad[2] else None,
            None,
            out["grad_edge_position"] if ctx.needs_input_grad[4] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            material_column("grad_eps_r", 0, 12),
            material_column("grad_sigma_e", 0, 13),
            None,
            material_column("grad_gain", 0, 15),
            material_column("grad_thickness", 0, 16),
            material_column("grad_eps_r", 1, 17),
            material_column("grad_sigma_e", 1, 18),
            None,
            material_column("grad_gain", 1, 20),
            material_column("grad_thickness", 1, 21),
            material_column("grad_eps_r", 2, 22),
            material_column("grad_sigma_e", 2, 23),
            None,
            material_column("grad_gain", 2, 25),
            material_column("grad_thickness", 2, 26),
            grad_frequency,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        _ad_reject_fixed_tangents(
            "field_coupled_rd_ad",
            (
                (tangents[3], "reflection_normal"),
                (tangents[5], "edge_direction"),
                (tangents[6], "edge_n0"),
                (tangents[7], "edge_n1"),
                (tangents[8], "exterior_angle"),
                (tangents[9], "tx_power"),
                (tangents[10], "tx_polarization"),
                (tangents[11], "rx_polarization"),
                (tangents[14], "reflection_mu_r"),
                (tangents[19], "wedge_mu_r0"),
                (tangents[24], "wedge_mu_r1"),
                (tangents[30], "edge_line_min"),
                (tangents[31], "edge_line_max"),
            ),
        )
        saved = ctx.saved_tensors
        scalar_shape = tuple(saved[12].shape)
        tangent_source = _ad_geometry_tangent(
            "field_coupled_rd_ad tangent_source", tangents[0], saved[0]
        )
        tangent_target = _ad_geometry_tangent(
            "field_coupled_rd_ad tangent_target", tangents[1], saved[1]
        )
        tangent_hit = _ad_geometry_tangent(
            "field_coupled_rd_ad tangent_reflection_position",
            tangents[2],
            saved[2],
        )
        tangent_edge = _ad_geometry_tangent(
            "field_coupled_rd_ad tangent_edge_position", tangents[4], saved[4]
        )

        def material_pack(indices: tuple[int, int, int], name: str):
            columns = tuple(
                _ad_checked_tangent(
                    f"field_coupled_rd_ad tangent_{name}",
                    _ad_native_tangent_or_none(tangents[index]),
                    scalar_shape,
                )
                for index in indices
            )
            if all(column is None for column in columns):
                return None
            zero = torch.zeros(
                scalar_shape, device=saved[12].device, dtype=torch.float32
            )
            return torch.stack(
                [zero if column is None else column for column in columns], dim=1
            )

        tangent_eps = material_pack((12, 17, 22), "eps_r")
        tangent_sigma = material_pack((13, 18, 23), "sigma_e")
        tangent_gain = material_pack((15, 20, 25), "gain")
        tangent_thickness = material_pack((16, 21, 26), "thickness")
        tangent_frequency = _ad_frequency_tangent(tangents[27])
        if (
            tangent_source is None
            and tangent_target is None
            and tangent_hit is None
            and tangent_edge is None
            and tangent_eps is None
            and tangent_sigma is None
            and tangent_gain is None
            and tangent_thickness is None
            and tangent_frequency == 0.0
        ):
            return (None,) * 5
        with torch_compat.disable_functorch():
            out = _required_native_op("field_coupled_rd_jvp")(
                *(_ad_native_tensor(value) for value in saved),
                ctx.frequency_value,
                ctx.reverse,
                tangent_source,
                tangent_target,
                tangent_hit,
                tangent_edge,
                tangent_eps,
                tangent_sigma,
                tangent_gain,
                tangent_thickness,
                float(tangent_frequency),
            )
        return (
            out["tangent_field_vector"],
            out["tangent_coefficient"],
            out["tangent_path_field"],
            out["tangent_path_gain"],
            None,
        )


def field_coupled_rd_ad(
    source: torch.Tensor,
    target: torch.Tensor,
    reflection_position: torch.Tensor,
    reflection_normal: torch.Tensor,
    edge_position: torch.Tensor,
    edge_direction: torch.Tensor,
    edge_n0: torch.Tensor,
    edge_n1: torch.Tensor,
    exterior_angle: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    reflection_material: tuple[torch.Tensor, ...],
    wedge_material0: tuple[torch.Tensor, ...],
    wedge_material1: tuple[torch.Tensor, ...],
    edge_line_min: torch.Tensor,
    edge_line_max: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
    reverse: bool,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_coupled_rd` (12 material scalars + frequency + geometry).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not
    supplied it is read here, exactly once per apply.
    """

    if any(
        len(bundle) != 5
        for bundle in (reflection_material, wedge_material0, wedge_material1)
    ):
        raise ValueError(
            "coupled material bundles must contain eps/sigma/mu/gain/thickness"
        )
    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldCoupledRdAdFunction.apply(
        source,
        target,
        reflection_position,
        reflection_normal,
        edge_position,
        edge_direction,
        edge_n0,
        edge_n1,
        exterior_angle,
        tx_power,
        tx_polarization,
        rx_polarization,
        *reflection_material,
        *wedge_material0,
        *wedge_material1,
        frequency,
        bool(reverse),
        float(frequency_value),
        edge_line_min,
        edge_line_max,
    )
    return dict(zip(_COUPLED_OUTPUT_FIELDS, values, strict=True))


class _CoupledRdPrepareAdFunction(torch.autograd.Function):
    """Fixed-winner coupled stationary geometry (plan 07 AD-4).

    Re-solves the image source, the stationary diffraction point on the edge
    and the predicted wall crossing for the frozen winner (wall plane + edge
    line), so the coupled interaction points move with the endpoints on the
    autograd graph. Differentiable inputs: source and receiver.
    """

    @staticmethod
    def forward(
        source,
        receiver,
        plane_point,
        plane_normal,
        edge_pos,
        edge_dir,
        edge_t_min,
        edge_t_max,
    ):
        out = _required_native_op("coupled_rd_prepare")(
            source,
            receiver,
            plane_point,
            plane_normal,
            edge_pos,
            edge_dir,
            edge_t_min,
            edge_t_max,
        )
        active, edge_point, _virtual_source, reflection_point = out
        return (edge_point, reflection_point, active)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:7]
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        ctx.mark_non_differentiable(output[2])

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_edge_point, grad_reflection_point, _grad_active):
        none_grads = (None,) * 8
        _ad_reject_fixed_inputs(
            "coupled_rd_prepare_ad",
            ctx.needs_input_grad,
            (
                (2, "plane_point"),
                (3, "plane_normal"),
                (4, "edge_pos"),
                (5, "edge_dir"),
                (6, "edge_t_min"),
                (7, "edge_t_max"),
            ),
        )
        need_source = bool(ctx.needs_input_grad[0])
        need_receiver = bool(ctx.needs_input_grad[1])
        if not (need_source or need_receiver) or (
            grad_edge_point is None and grad_reflection_point is None
        ):
            return none_grads
        out = _required_native_op("coupled_rd_prepare_backward")(
            *ctx.saved_tensors,
            grad_edge_point,
            grad_reflection_point,
            need_source,
            need_receiver,
        )
        return (
            out["grad_source"] if need_source else None,
            out["grad_receiver"] if need_receiver else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, t_source, t_receiver, *t_rest):
        _ad_reject_fixed_tangents(
            "coupled_rd_prepare_ad",
            (
                (t_rest[0], "plane_point"),
                (t_rest[1], "plane_normal"),
                (t_rest[2], "edge_pos"),
                (t_rest[3], "edge_dir"),
                (t_rest[4], "edge_t_min"),
                (t_rest[5], "edge_t_max"),
            ),
        )
        saved = ctx.saved_tensors
        tangent_source = _ad_geometry_tangent(
            "coupled_rd_prepare_ad tangent_source", t_source, saved[0]
        )
        tangent_receiver = _ad_geometry_tangent(
            "coupled_rd_prepare_ad tangent_receiver", t_receiver, saved[1]
        )
        if tangent_source is None and tangent_receiver is None:
            return (None, None, None)
        with torch_compat.disable_functorch():
            out = _required_native_op("coupled_rd_prepare_jvp")(
                *(_ad_native_tensor(value) for value in saved),
                tangent_source,
                tangent_receiver,
            )
        return (
            out["tangent_edge_point"],
            out["tangent_reflection_point"],
            None,
        )


def coupled_rd_prepare_ad(
    source: torch.Tensor,
    receiver: torch.Tensor,
    plane_point: torch.Tensor,
    plane_normal: torch.Tensor,
    edge_pos: torch.Tensor,
    edge_dir: torch.Tensor,
    edge_t_min: torch.Tensor,
    edge_t_max: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable fixed-winner coupled stationary geometry re-solve."""

    edge_point, reflection_point, active = _CoupledRdPrepareAdFunction.apply(
        source,
        receiver,
        plane_point,
        plane_normal,
        edge_pos,
        edge_dir,
        edge_t_min,
        edge_t_max,
    )
    return {
        "edge_point": edge_point,
        "reflection_point": reflection_point,
        "active": active,
    }


__all__ = [
    "_CoupledRdPrepareAdFunction",
    "_FieldCoupledRdAdFunction",
    "_FieldDiffractionWedgeAdFunction",
    "_FieldFreeSpaceAdFunction",
    "_FieldReflectionSequenceAdFunction",
    "_FieldTransmissionSequenceAdFunction",
    "coupled_rd_prepare_ad",
    "field_coupled_rd_ad",
    "field_diffraction_wedge_ad",
    "field_free_space_ad",
    "field_reflection_sequence_ad",
    "field_transmission_sequence_ad",
]
