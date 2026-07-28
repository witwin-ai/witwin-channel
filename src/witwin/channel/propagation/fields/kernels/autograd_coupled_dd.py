from __future__ import annotations

import torch

from witwin.channel.runtime import (
    _ad_checked_tangent,
    _ad_first_order_only,
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_geometry_tangent,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
    disable_functorch,
    required_symbol as _required_native_op,
)

from .functional import _COUPLED_OUTPUT_FIELDS


# Material grad-request index groups for the four coupled double-diffraction
# wedge faces (wedge1 face0, wedge1 face1, wedge2 face0, wedge2 face1), mirroring
# the native (N, 4) column layout.
_DD_EPS_INDICES = (15, 20, 25, 30)
_DD_SIGMA_INDICES = (16, 21, 26, 31)
_DD_GAIN_INDICES = (18, 23, 28, 33)
_DD_THICKNESS_INDICES = (19, 24, 29, 34)


def _dd_backward_needs(needs_input_grad) -> dict[str, bool]:
    """Per-family gradient-request flags for the coupled-DD backward launch."""
    return {
        "geometry": any(bool(needs_input_grad[index]) for index in (0, 1)),
        "eps": any(bool(needs_input_grad[index]) for index in _DD_EPS_INDICES),
        "sigma": any(bool(needs_input_grad[index]) for index in _DD_SIGMA_INDICES),
        "gain": any(bool(needs_input_grad[index]) for index in _DD_GAIN_INDICES),
        "thickness": any(
            bool(needs_input_grad[index]) for index in _DD_THICKNESS_INDICES
        ),
        "frequency": bool(needs_input_grad[35]),
    }


def _dd_backward_is_noop(needs: dict[str, bool], grads: tuple) -> bool:
    """True when no requested input needs a gradient or every seed is None."""
    return not any(needs.values()) or all(value is None for value in grads)


class _FieldCoupledDdAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable coupled double-diffraction transport (ADR-013 D4).

    Twin of :class:`_FieldCoupledRdAdFunction` for cid-7 (TX -> e1 -> e2 -> RX)
    rows. Differentiable inputs: eps_r / sigma_e / gain / thickness for the four
    wedge faces (16 scalars per path), frequency, and the tx/rx endpoints
    (source, target), whose gradients flow through the native per-leg
    re-anchoring. The frozen discovery seeds Q1/Q2 (edge1_position /
    edge2_position), the edge axes/normals/exterior angles, the edge bounds,
    mu_r, tx_power and the polarizations stay fixed; requesting their gradient
    fails loudly. Mesh-vertex gradients are refused one layer up (evaluation.py
    coupled block, ADR-013 D4).
    """

    @staticmethod
    def forward(
        source,
        target,
        edge1_position,
        edge1_direction,
        edge1_n0,
        edge1_n1,
        edge1_exterior,
        edge2_position,
        edge2_direction,
        edge2_n0,
        edge2_n1,
        edge2_exterior,
        tx_power,
        tx_polarization,
        rx_polarization,
        w1a_eps_r,
        w1a_sigma_e,
        w1a_mu_r,
        w1a_gain,
        w1a_thickness,
        w1b_eps_r,
        w1b_sigma_e,
        w1b_mu_r,
        w1b_gain,
        w1b_thickness,
        w2a_eps_r,
        w2a_sigma_e,
        w2a_mu_r,
        w2a_gain,
        w2a_thickness,
        w2b_eps_r,
        w2b_sigma_e,
        w2b_mu_r,
        w2b_gain,
        w2b_thickness,
        frequency,
        frequency_value,
        edge1_line_min,
        edge1_line_max,
        edge2_line_min,
        edge2_line_max,
    ):
        out = _required_native_op("field_coupled_dd")(
            source,
            target,
            edge1_position,
            edge1_direction,
            edge1_n0,
            edge1_n1,
            edge1_exterior,
            edge2_position,
            edge2_direction,
            edge2_n0,
            edge2_n1,
            edge2_exterior,
            tx_power,
            tx_polarization,
            rx_polarization,
            w1a_eps_r,
            w1a_sigma_e,
            w1a_mu_r,
            w1a_gain,
            w1a_thickness,
            w1b_eps_r,
            w1b_sigma_e,
            w1b_mu_r,
            w1b_gain,
            w1b_thickness,
            w2a_eps_r,
            w2a_sigma_e,
            w2a_mu_r,
            w2a_gain,
            w2a_thickness,
            w2b_eps_r,
            w2b_sigma_e,
            w2b_mu_r,
            w2b_gain,
            w2b_thickness,
            edge1_line_min,
            edge1_line_max,
            edge2_line_min,
            edge2_line_max,
            frequency_value,
        )
        return tuple(out[name] for name in _COUPLED_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[35]
        # The native backward/jvp take the 35 primal field tensors (source ..
        # wedge2 face1 thickness) followed by the four frozen edge bounds, then
        # the host frequency scalar. Q1/Q2/bounds are non-differentiable (ADR-013
        # D4) but saved so the companions forward them in the primal position.
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:35]
        )
        bounds = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (inputs[37], inputs[38], inputs[39], inputs[40])
        )
        ctx.frequency_value = inputs[36]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
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
        none_grads = (None,) * 41
        _ad_reject_fixed_inputs(
            "field_coupled_dd_ad",
            ctx.needs_input_grad,
            (
                (2, "edge1_position"),
                (3, "edge1_direction"),
                (4, "edge1_n0"),
                (5, "edge1_n1"),
                (6, "edge1_exterior"),
                (7, "edge2_position"),
                (8, "edge2_direction"),
                (9, "edge2_n0"),
                (10, "edge2_n1"),
                (11, "edge2_exterior"),
                (12, "tx_power"),
                (13, "tx_polarization"),
                (14, "rx_polarization"),
                (17, "wedge1_mu_r0"),
                (22, "wedge1_mu_r1"),
                (27, "wedge2_mu_r0"),
                (32, "wedge2_mu_r1"),
                (37, "edge1_line_min"),
                (38, "edge1_line_max"),
                (39, "edge2_line_min"),
                (40, "edge2_line_max"),
            ),
        )
        needs = _dd_backward_needs(ctx.needs_input_grad)
        grads = (grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain)
        if _dd_backward_is_noop(needs, grads):
            return none_grads
        saved = ctx.saved_tensors
        out = _required_native_op("field_coupled_dd_backward")(
            *saved,
            ctx.frequency_value,
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
            needs["eps"],
            needs["sigma"],
            needs["gain"],
            needs["thickness"],
            needs["frequency"],
            needs["geometry"],
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if needs["frequency"]
            else None
        )

        def material_column(name: str, column: int, index: int):
            if not ctx.needs_input_grad[index]:
                return None
            return out[name][:, column]

        # Material grad columns: wedge1 face0, wedge1 face1, wedge2 face0,
        # wedge2 face1 (mirrors the native (N, 4) slot layout).
        return (
            out["grad_source"] if ctx.needs_input_grad[0] else None,
            out["grad_target"] if ctx.needs_input_grad[1] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            material_column("grad_eps_r", 0, 15),
            material_column("grad_sigma_e", 0, 16),
            None,
            material_column("grad_gain", 0, 18),
            material_column("grad_thickness", 0, 19),
            material_column("grad_eps_r", 1, 20),
            material_column("grad_sigma_e", 1, 21),
            None,
            material_column("grad_gain", 1, 23),
            material_column("grad_thickness", 1, 24),
            material_column("grad_eps_r", 2, 25),
            material_column("grad_sigma_e", 2, 26),
            None,
            material_column("grad_gain", 2, 28),
            material_column("grad_thickness", 2, 29),
            material_column("grad_eps_r", 3, 30),
            material_column("grad_sigma_e", 3, 31),
            None,
            material_column("grad_gain", 3, 33),
            material_column("grad_thickness", 3, 34),
            grad_frequency,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        _ad_reject_fixed_tangents(
            "field_coupled_dd_ad",
            (
                (tangents[2], "edge1_position"),
                (tangents[3], "edge1_direction"),
                (tangents[4], "edge1_n0"),
                (tangents[5], "edge1_n1"),
                (tangents[6], "edge1_exterior"),
                (tangents[7], "edge2_position"),
                (tangents[8], "edge2_direction"),
                (tangents[9], "edge2_n0"),
                (tangents[10], "edge2_n1"),
                (tangents[11], "edge2_exterior"),
                (tangents[12], "tx_power"),
                (tangents[13], "tx_polarization"),
                (tangents[14], "rx_polarization"),
                (tangents[17], "wedge1_mu_r0"),
                (tangents[22], "wedge1_mu_r1"),
                (tangents[27], "wedge2_mu_r0"),
                (tangents[32], "wedge2_mu_r1"),
                (tangents[37], "edge1_line_min"),
                (tangents[38], "edge1_line_max"),
                (tangents[39], "edge2_line_min"),
                (tangents[40], "edge2_line_max"),
            ),
        )
        saved = ctx.saved_tensors
        scalar_shape = tuple(saved[15].shape)
        tangent_source = _ad_geometry_tangent(
            "field_coupled_dd_ad tangent_source", tangents[0], saved[0]
        )
        tangent_target = _ad_geometry_tangent(
            "field_coupled_dd_ad tangent_target", tangents[1], saved[1]
        )

        def material_pack(indices: tuple[int, int, int, int], name: str):
            columns = tuple(
                _ad_checked_tangent(
                    f"field_coupled_dd_ad tangent_{name}",
                    _ad_native_tangent_or_none(tangents[index]),
                    scalar_shape,
                )
                for index in indices
            )
            if all(column is None for column in columns):
                return None
            zero = torch.zeros(
                scalar_shape, device=saved[15].device, dtype=torch.float32
            )
            return torch.stack(
                [zero if column is None else column for column in columns], dim=1
            )

        tangent_eps = material_pack((15, 20, 25, 30), "eps_r")
        tangent_sigma = material_pack((16, 21, 26, 31), "sigma_e")
        tangent_gain = material_pack((18, 23, 28, 33), "gain")
        tangent_thickness = material_pack((19, 24, 29, 34), "thickness")
        tangent_frequency = _ad_frequency_tangent(tangents[35])
        if (
            tangent_source is None
            and tangent_target is None
            and tangent_eps is None
            and tangent_sigma is None
            and tangent_gain is None
            and tangent_thickness is None
            and tangent_frequency == 0.0
        ):
            return (None,) * 5
        with disable_functorch():
            out = _required_native_op("field_coupled_dd_jvp")(
                *(_ad_native_tensor(value) for value in saved),
                ctx.frequency_value,
                tangent_source,
                tangent_target,
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


def field_coupled_dd_ad(
    source: torch.Tensor,
    target: torch.Tensor,
    edge1_position: torch.Tensor,
    edge1_direction: torch.Tensor,
    edge1_n0: torch.Tensor,
    edge1_n1: torch.Tensor,
    edge1_exterior: torch.Tensor,
    edge2_position: torch.Tensor,
    edge2_direction: torch.Tensor,
    edge2_n0: torch.Tensor,
    edge2_n1: torch.Tensor,
    edge2_exterior: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    wedge1_material0: tuple[torch.Tensor, ...],
    wedge1_material1: tuple[torch.Tensor, ...],
    wedge2_material0: tuple[torch.Tensor, ...],
    wedge2_material1: tuple[torch.Tensor, ...],
    edge1_line_min: torch.Tensor,
    edge1_line_max: torch.Tensor,
    edge2_line_min: torch.Tensor,
    edge2_line_max: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_coupled_dd` (16 material scalars + frequency + tx/rx).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam); when not supplied it is read
    here, exactly once per apply.
    """

    if any(
        len(bundle) != 5
        for bundle in (
            wedge1_material0,
            wedge1_material1,
            wedge2_material0,
            wedge2_material1,
        )
    ):
        raise ValueError(
            "coupled dd material bundles must contain eps/sigma/mu/gain/thickness"
        )
    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldCoupledDdAdFunction.apply(
        source,
        target,
        edge1_position,
        edge1_direction,
        edge1_n0,
        edge1_n1,
        edge1_exterior,
        edge2_position,
        edge2_direction,
        edge2_n0,
        edge2_n1,
        edge2_exterior,
        tx_power,
        tx_polarization,
        rx_polarization,
        *wedge1_material0,
        *wedge1_material1,
        *wedge2_material0,
        *wedge2_material1,
        frequency,
        float(frequency_value),
        edge1_line_min,
        edge1_line_max,
        edge2_line_min,
        edge2_line_max,
    )
    return dict(zip(_COUPLED_OUTPUT_FIELDS, values, strict=True))


__all__ = [
    "_FieldCoupledDdAdFunction",
    "field_coupled_dd_ad",
]
