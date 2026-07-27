"""ADR-022 BDPT accumulate/finalize AD ``torch.autograd.Function`` wrappers.

Split out of :mod:`montecarlo.bdpt.autograd` to keep both modules within the
maintenance size budget. This module owns the two connection-reduction
companions (spec 6.4 accumulate, both power and coherent domains) and the linear
finalize map (spec 6.5/6.6, point components and component maps). The subpath
advance and endpoint connection wrappers stay in :mod:`montecarlo.bdpt.autograd`.

Every wrapper dispatches the SAME registered native forward symbol as
``ad_mode='none'`` so the primal values are bitwise identical, and routes
cotangents/tangents to the registered native ``_backward``/``_jvp`` companions
(never finite differences).
"""

from __future__ import annotations

import torch

from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.autograd_contracts import (
    _ad_first_order_only,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
)

from .kernels.maps import (
    bdpt_finalize_component_maps,
    bdpt_finalize_component_maps_backward,
    bdpt_finalize_component_maps_jvp,
    bdpt_finalize_point_components,
    bdpt_finalize_point_components_backward,
    bdpt_finalize_point_components_jvp,
)
from .paths_ad import (
    bdpt_accumulate_connection_samples_backward,
    bdpt_accumulate_connection_samples_forward_ad,
    bdpt_accumulate_connection_samples_jvp,
)


_FINALIZE_FIELDS = (
    "path_gain",
    "los_power",
    "reflection_power",
    "diffraction_power",
    "transmission_power",
    "scattering_power",
)


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
    @_ad_first_order_only
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
    @_ad_first_order_only
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
