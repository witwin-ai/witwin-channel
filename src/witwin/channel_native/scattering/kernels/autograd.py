"""Differentiable deterministic scattering ops (ADR-014).

Native ``torch.autograd.Function`` companions for the two forward scattering
ops (ADR-010 op 1 Kirchhoff ensemble rows, op 2 realization-coherent
phase-screen patch integral). Both mirror the plan-07 field companion pattern
in ``propagation/fields/kernels/rough_scale.py``: a plain forward, a
``once_differentiable`` VJP, and a forward-mode JVP that all dispatch registered
native kernels. Torch autograd may dispatch these companions but never
reconstructs the numerical operation.
"""

from __future__ import annotations

import torch

from witwin.channel_native.runtime import torch_compat
from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op
from witwin.channel_native.runtime.autograd_contracts import (
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_geometry_tangent,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
)

from .functional import (
    _ENSEMBLE_OUTPUT_FIELDS,
    _PATCH_OUTPUT_FIELDS,
    scattering_ensemble_eval,
    scattering_ensemble_eval_backward,
    scattering_ensemble_eval_jvp,
    scattering_patch_integral_eval_backward,
    scattering_patch_integral_eval_jvp,
)


def _grad_or_none(out: dict, key: str, needed: bool) -> torch.Tensor | None:
    return out[key] if needed else None


# Fixed inputs of the ensemble op: rejecting a gradient/tangent here fails
# loudly instead of silently detaching (ADR-014 differentiable-input set).
_ENSEMBLE_FIXED = (
    (12, "material_id"),
    (13, "backup_axis"),
    (14, "rx_pol"),
    (15, "rc_idx"),
    (16, "sc_idx"),
    (19, "table_offset"),
    (20, "table_dims"),
    (21, "material_slot"),
)
_PATCH_FIXED = (
    (0, "patch_tris"),
    (1, "patch_uvs"),
    (2, "rows"),
    (5, "n_rows"),
    (8, "pol_t"),
    (9, "pol_r"),
    (14, "quad_a"),
    (15, "quad_b"),
    (16, "quad_w"),
)


class _ScatteringEnsembleEvalAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable Kirchhoff ensemble rows (ADR-014 op 1).

    Differentiable inputs: the surviving-row geometry (``wo_rows``, ``r2_rows``,
    ``cos_o_rows``), the per-sample geometry/radiometry (``n_o``, ``t1r``,
    ``t2r``, ``wi_local``, ``cos_i``, ``r1``, ``a_te2``, ``a_tm2``,
    ``weights``), the resident BSDF tables (``f_te_flat``, ``f_tm_flat``) and
    the radiometric scale ``coef`` (a 0-d scalar tensor). ``material_id``,
    ``backup_axis``, ``rx_pol``, the ``rc``/``sc`` indices, the table metadata
    and ``threshold`` stay fixed; requesting their gradient fails loudly. The
    ``gain``/``amplitude``/``length`` outputs are differentiable; ``keep`` is
    marked non-differentiable.
    """

    @staticmethod
    def forward(
        wo_rows,
        r2_rows,
        cos_o_rows,
        n_o,
        t1r,
        t2r,
        wi_local,
        cos_i,
        r1,
        a_te2,
        a_tm2,
        weights,
        material_id,
        backup_axis,
        rx_pol,
        rc_idx,
        sc_idx,
        f_te_flat,
        f_tm_flat,
        table_offset,
        table_dims,
        material_slot,
        coef,
        threshold,
        coef_value,
    ):
        out = scattering_ensemble_eval(
            wo_rows,
            r2_rows,
            cos_o_rows,
            n_o,
            t1r,
            t2r,
            wi_local,
            cos_i,
            r1,
            a_te2,
            a_tm2,
            weights,
            material_id,
            backup_axis,
            rx_pol,
            rc_idx,
            sc_idx,
            f_te_flat,
            f_tm_flat,
            table_offset,
            table_dims,
            material_slot,
            coef=coef_value,
            threshold=threshold,
        )
        return tuple(out[name] for name in _ENSEMBLE_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        coef = inputs[22]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:22]
        )
        ctx.threshold = inputs[23]
        ctx.coef_value = inputs[24]
        ctx.coef_meta = (
            (coef.dtype, coef.device) if isinstance(coef, torch.Tensor) else None
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        ctx.mark_non_differentiable(output[3])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_gain, grad_amplitude, grad_length, _grad_keep):
        none_grads = (None,) * 25
        _ad_reject_fixed_inputs(
            "scattering_ensemble_eval_ad", ctx.needs_input_grad, _ENSEMBLE_FIXED
        )
        needed = tuple(bool(ctx.needs_input_grad[i]) for i in range(23))
        need_rows = any(needed[0:3])
        need_samples = any(needed[3:12])
        need_tables = needed[17] or needed[18]
        need_coef = needed[22]
        grads = (grad_gain, grad_amplitude, grad_length)
        if not (need_rows or need_samples or need_tables or need_coef) or all(
            value is None for value in grads
        ):
            return none_grads
        out = scattering_ensemble_eval_backward(
            *ctx.saved_tensors,
            coef=ctx.coef_value,
            threshold=ctx.threshold,
            grad_gain=grad_gain,
            grad_amplitude=grad_amplitude,
            grad_length=grad_length,
            need_grad_rows=need_rows,
            need_grad_samples=need_samples,
            need_grad_tables=need_tables,
            need_grad_coef=need_coef,
        )
        grad_coef = (
            _ad_frequency_grad(out["grad_coef"], ctx.coef_meta) if need_coef else None
        )
        return (
            _grad_or_none(out, "grad_wo_rows", needed[0]),
            _grad_or_none(out, "grad_r2_rows", needed[1]),
            _grad_or_none(out, "grad_cos_o_rows", needed[2]),
            _grad_or_none(out, "grad_n_o", needed[3]),
            _grad_or_none(out, "grad_t1r", needed[4]),
            _grad_or_none(out, "grad_t2r", needed[5]),
            _grad_or_none(out, "grad_wi_local", needed[6]),
            _grad_or_none(out, "grad_cos_i", needed[7]),
            _grad_or_none(out, "grad_r1", needed[8]),
            _grad_or_none(out, "grad_a_te2", needed[9]),
            _grad_or_none(out, "grad_a_tm2", needed[10]),
            _grad_or_none(out, "grad_weights", needed[11]),
            None,
            None,
            None,
            None,
            None,
            _grad_or_none(out, "grad_f_te", needed[17]),
            _grad_or_none(out, "grad_f_tm", needed[18]),
            None,
            None,
            None,
            grad_coef,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_wo_rows,
        t_r2_rows,
        t_cos_o_rows,
        t_n_o,
        t_t1r,
        t_t2r,
        t_wi_local,
        t_cos_i,
        t_r1,
        t_a_te2,
        t_a_tm2,
        t_weights,
        t_material_id,
        t_backup_axis,
        t_rx_pol,
        t_rc_idx,
        t_sc_idx,
        t_f_te_flat,
        t_f_tm_flat,
        t_table_offset,
        t_table_dims,
        t_material_slot,
        t_coef,
        _t_threshold,
        _t_coef_value,
    ):
        _ad_reject_fixed_tangents(
            "scattering_ensemble_eval_ad",
            (
                (t_material_id, "material_id"),
                (t_backup_axis, "backup_axis"),
                (t_rx_pol, "rx_pol"),
                (t_rc_idx, "rc_idx"),
                (t_sc_idx, "sc_idx"),
                (t_table_offset, "table_offset"),
                (t_table_dims, "table_dims"),
                (t_material_slot, "material_slot"),
            ),
        )
        saved = ctx.saved_tensors
        tangents = {
            "tangent_wo_rows": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_wo_rows", t_wo_rows, saved[0]
            ),
            "tangent_r2_rows": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_r2_rows", t_r2_rows, saved[1]
            ),
            "tangent_cos_o_rows": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_cos_o_rows",
                t_cos_o_rows,
                saved[2],
            ),
            "tangent_n_o": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_n_o", t_n_o, saved[3]
            ),
            "tangent_t1r": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_t1r", t_t1r, saved[4]
            ),
            "tangent_t2r": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_t2r", t_t2r, saved[5]
            ),
            "tangent_wi_local": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_wi_local", t_wi_local, saved[6]
            ),
            "tangent_cos_i": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_cos_i", t_cos_i, saved[7]
            ),
            "tangent_r1": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_r1", t_r1, saved[8]
            ),
            "tangent_a_te2": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_a_te2", t_a_te2, saved[9]
            ),
            "tangent_a_tm2": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_a_tm2", t_a_tm2, saved[10]
            ),
            "tangent_weights": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_weights", t_weights, saved[11]
            ),
            "tangent_f_te_flat": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_f_te_flat", t_f_te_flat, saved[17]
            ),
            "tangent_f_tm_flat": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_f_tm_flat", t_f_tm_flat, saved[18]
            ),
        }
        tangent_coef = _ad_frequency_tangent(t_coef)
        if tangent_coef == 0.0 and all(value is None for value in tangents.values()):
            return (None,) * len(_ENSEMBLE_OUTPUT_FIELDS)
        with torch_compat.disable_functorch():
            out = scattering_ensemble_eval_jvp(
                *(_ad_native_tensor(value) for value in saved),
                coef=ctx.coef_value,
                threshold=ctx.threshold,
                tangent_coef=tangent_coef,
                **tangents,
            )
        return (
            out["tangent_gain"],
            out["tangent_amplitude"],
            out["tangent_length"],
            None,
        )


def scattering_ensemble_eval_ad(
    wo_rows: torch.Tensor,
    r2_rows: torch.Tensor,
    cos_o_rows: torch.Tensor,
    n_o: torch.Tensor,
    t1r: torch.Tensor,
    t2r: torch.Tensor,
    wi_local: torch.Tensor,
    cos_i: torch.Tensor,
    r1: torch.Tensor,
    a_te2: torch.Tensor,
    a_tm2: torch.Tensor,
    weights: torch.Tensor,
    material_id: torch.Tensor,
    backup_axis: torch.Tensor,
    rx_pol: torch.Tensor,
    rc_idx: torch.Tensor,
    sc_idx: torch.Tensor,
    f_te_flat: torch.Tensor,
    f_tm_flat: torch.Tensor,
    table_offset: torch.Tensor,
    table_dims: torch.Tensor,
    material_slot: torch.Tensor,
    *,
    coef: torch.Tensor,
    threshold: float,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`scattering_ensemble_eval` (ADR-014 op 1).

    ``coef`` is the radiometric scale as a 0-d scalar tensor so frequency (and
    tx power) gradients flow through the ensemble rows; its host value is read
    once per apply at this seam (audit M3), never per row.
    """

    coef_value = _ad_frequency_value(coef)
    values = _ScatteringEnsembleEvalAdFunction.apply(
        wo_rows,
        r2_rows,
        cos_o_rows,
        n_o,
        t1r,
        t2r,
        wi_local,
        cos_i,
        r1,
        a_te2,
        a_tm2,
        weights,
        material_id,
        backup_axis,
        rx_pol,
        rc_idx,
        sc_idx,
        f_te_flat,
        f_tm_flat,
        table_offset,
        table_dims,
        material_slot,
        coef,
        float(threshold),
        float(coef_value),
    )
    return dict(zip(_ENSEMBLE_OUTPUT_FIELDS, values, strict=True))


class _ScatteringPatchIntegralEvalAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable phase-screen patch integral (ADR-014 op 2).

    Differentiable inputs: the phase-screen ``heights``, the smooth-stack Jones
    coefficients ``r_te``/``r_tm``, the per-row geometry (``d_i``, ``d_o``,
    ``r1_rows``, ``r2_rows``, ``centroids``) and the scalar ``k0`` (a 0-d
    tensor). The patch mesh (``patch_tris``, ``patch_uvs``), the row index,
    ``n_rows``, the polarizations and the quadrature nodes/weights stay fixed;
    requesting their gradient fails loudly. The complex ``total`` output is
    differentiable; ``integral`` and ``row_value`` are marked
    non-differentiable test buffers.
    """

    @staticmethod
    def forward(
        patch_tris,
        patch_uvs,
        rows,
        d_i,
        d_o,
        n_rows,
        r_te,
        r_tm,
        pol_t,
        pol_r,
        r1_rows,
        r2_rows,
        centroids,
        heights,
        quad_a,
        quad_b,
        quad_w,
        k0,
        k0_value,
    ):
        out = _required_patch_forward(
            patch_tris,
            patch_uvs,
            rows,
            d_i,
            d_o,
            n_rows,
            r_te,
            r_tm,
            pol_t,
            pol_r,
            r1_rows,
            r2_rows,
            centroids,
            heights,
            quad_a,
            quad_b,
            quad_w,
            k0_value,
        )
        return tuple(out[name] for name in _PATCH_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        k0 = inputs[17]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:17]
        )
        ctx.k0_value = inputs[18]
        ctx.k0_meta = (
            (k0.dtype, k0.device) if isinstance(k0, torch.Tensor) else None
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        ctx.mark_non_differentiable(output[1], output[2])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_total, _grad_integral, _grad_row_value):
        none_grads = (None,) * 19
        _ad_reject_fixed_inputs(
            "scattering_patch_integral_eval_ad", ctx.needs_input_grad, _PATCH_FIXED
        )
        needed = tuple(bool(ctx.needs_input_grad[i]) for i in range(18))
        need_heights = needed[13]
        need_jones = needed[6] or needed[7]
        need_geometry = any(needed[i] for i in (3, 4, 10, 11, 12))
        need_k0 = needed[17]
        if not (need_heights or need_jones or need_geometry or need_k0) or (
            grad_total is None
        ):
            return none_grads
        out = scattering_patch_integral_eval_backward(
            *ctx.saved_tensors[:14],
            k0=ctx.k0_value,
            grad_total=grad_total,
            need_grad_heights=need_heights,
            need_grad_jones=need_jones,
            need_grad_geometry=need_geometry,
            need_grad_k0=need_k0,
        )
        grad_k0 = (
            _ad_frequency_grad(out["grad_k0"], ctx.k0_meta) if need_k0 else None
        )
        return (
            None,
            None,
            None,
            _grad_or_none(out, "grad_d_i", needed[3]),
            _grad_or_none(out, "grad_d_o", needed[4]),
            None,
            _grad_or_none(out, "grad_r_te", needed[6]),
            _grad_or_none(out, "grad_r_tm", needed[7]),
            None,
            None,
            _grad_or_none(out, "grad_r1_rows", needed[10]),
            _grad_or_none(out, "grad_r2_rows", needed[11]),
            _grad_or_none(out, "grad_centroids", needed[12]),
            _grad_or_none(out, "grad_heights", needed[13]),
            None,
            None,
            None,
            grad_k0,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_patch_tris,
        t_patch_uvs,
        t_rows,
        t_d_i,
        t_d_o,
        t_n_rows,
        t_r_te,
        t_r_tm,
        t_pol_t,
        t_pol_r,
        t_r1_rows,
        t_r2_rows,
        t_centroids,
        t_heights,
        t_quad_a,
        t_quad_b,
        t_quad_w,
        t_k0,
        _t_k0_value,
    ):
        _ad_reject_fixed_tangents(
            "scattering_patch_integral_eval_ad",
            (
                (t_patch_tris, "patch_tris"),
                (t_patch_uvs, "patch_uvs"),
                (t_rows, "rows"),
                (t_n_rows, "n_rows"),
                (t_pol_t, "pol_t"),
                (t_pol_r, "pol_r"),
                (t_quad_a, "quad_a"),
                (t_quad_b, "quad_b"),
                (t_quad_w, "quad_w"),
            ),
        )
        saved = ctx.saved_tensors
        tangents = {
            "tangent_heights": _ad_geometry_tangent(
                "scattering_patch_integral_eval_ad tangent_heights",
                t_heights,
                saved[13],
            ),
            "tangent_r_te": _ad_geometry_tangent(
                "scattering_patch_integral_eval_ad tangent_r_te", t_r_te, saved[6]
            ),
            "tangent_r_tm": _ad_geometry_tangent(
                "scattering_patch_integral_eval_ad tangent_r_tm", t_r_tm, saved[7]
            ),
            "tangent_d_i": _ad_geometry_tangent(
                "scattering_patch_integral_eval_ad tangent_d_i", t_d_i, saved[3]
            ),
            "tangent_d_o": _ad_geometry_tangent(
                "scattering_patch_integral_eval_ad tangent_d_o", t_d_o, saved[4]
            ),
            "tangent_r1_rows": _ad_geometry_tangent(
                "scattering_patch_integral_eval_ad tangent_r1_rows",
                t_r1_rows,
                saved[10],
            ),
            "tangent_r2_rows": _ad_geometry_tangent(
                "scattering_patch_integral_eval_ad tangent_r2_rows",
                t_r2_rows,
                saved[11],
            ),
            "tangent_centroids": _ad_geometry_tangent(
                "scattering_patch_integral_eval_ad tangent_centroids",
                t_centroids,
                saved[12],
            ),
        }
        tangent_k0 = _ad_frequency_tangent(t_k0)
        if tangent_k0 == 0.0 and all(value is None for value in tangents.values()):
            return (None,) * len(_PATCH_OUTPUT_FIELDS)
        with torch_compat.disable_functorch():
            out = scattering_patch_integral_eval_jvp(
                *(_ad_native_tensor(value) for value in saved[:14]),
                k0=ctx.k0_value,
                tangent_k0=tangent_k0,
                **tangents,
            )
        return (out["tangent_total"], None, None)


def _required_patch_forward(
    patch_tris,
    patch_uvs,
    rows,
    d_i,
    d_o,
    n_rows,
    r_te,
    r_tm,
    pol_t,
    pol_r,
    r1_rows,
    r2_rows,
    centroids,
    heights,
    quad_a,
    quad_b,
    quad_w,
    k0_value,
):
    out = _required_native_op("scattering_patch_integral_eval")(
        patch_tris,
        patch_uvs,
        rows,
        d_i,
        d_o,
        n_rows,
        r_te,
        r_tm,
        pol_t,
        pol_r,
        r1_rows,
        r2_rows,
        centroids,
        heights,
        quad_a,
        quad_b,
        quad_w,
        float(k0_value),
    )
    expected = {"total", "integral", "row_value"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError(
            "_channel_native.scattering_patch_integral_eval returned invalid fields"
        )
    return out


def scattering_patch_integral_eval_ad(
    patch_tris: torch.Tensor,
    patch_uvs: torch.Tensor,
    rows: torch.Tensor,
    d_i: torch.Tensor,
    d_o: torch.Tensor,
    n_rows: torch.Tensor,
    r_te: torch.Tensor,
    r_tm: torch.Tensor,
    pol_t: torch.Tensor,
    pol_r: torch.Tensor,
    r1_rows: torch.Tensor,
    r2_rows: torch.Tensor,
    centroids: torch.Tensor,
    heights: torch.Tensor,
    *,
    k0: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`scattering_patch_integral_eval` (ADR-014 op 2).

    ``k0`` is the wavenumber as a 0-d scalar tensor so frequency gradients flow
    through the phase and prefactor; its host value is read once per apply at
    this seam (audit M3). The Duffy-mapped quadrature nodes/weights are gathered
    from the shared cache and threaded to the companions as fixed inputs.
    """

    from .functional import _duffy_nodes

    quad_a, quad_b, quad_w = _duffy_nodes(patch_tris.device)
    k0_value = _ad_frequency_value(k0)
    values = _ScatteringPatchIntegralEvalAdFunction.apply(
        patch_tris,
        patch_uvs,
        rows,
        d_i,
        d_o,
        n_rows,
        r_te,
        r_tm,
        pol_t,
        pol_r,
        r1_rows,
        r2_rows,
        centroids,
        heights,
        quad_a,
        quad_b,
        quad_w,
        k0,
        float(k0_value),
    )
    return dict(zip(_PATCH_OUTPUT_FIELDS, values, strict=True))


__all__ = [
    "_ScatteringEnsembleEvalAdFunction",
    "_ScatteringPatchIntegralEvalAdFunction",
    "scattering_ensemble_eval_ad",
    "scattering_patch_integral_eval_ad",
]
