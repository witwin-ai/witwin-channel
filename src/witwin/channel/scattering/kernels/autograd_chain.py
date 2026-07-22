"""Differentiable ADR-021 multi-bounce chain scattering ops (plan 10a).

Native ``torch.autograd.Function`` companions for the two forward chain
scattering ops (Op A ``scattering_chain_ensemble_eval`` power-domain rows, Op B
``scattering_chain_realization_eval`` coherent phase-screen realization). Both
mirror the plan-07 pattern already used by
``scattering/kernels/autograd.py``: a plain forward, a ``once_differentiable``
VJP and a forward-mode JVP that all dispatch registered native kernels. Torch
autograd may dispatch these companions but never reconstructs the numerical
operation.

The two AD-live scalars of each op cross the graph as 0-dim tensors (so their
gradients flow) while their numerical values cross the native ABI as ``double``
positionals (ADR-014 / ``field_free_space`` precedent): ``coef`` + ``frequency``
for Op A, ``k0`` + ``frequency`` for Op B.
"""

from __future__ import annotations

import torch

from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.symbols import required_symbol as _required_native_op
from witwin.channel.runtime.autograd_contracts import (
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_geometry_tangent,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
)

from .functional_chain import (
    _CHAIN_ENSEMBLE_PRIMAL_NAMES,
    _CHAIN_ENSEMBLE_OUTPUT_FIELDS,
    _CHAIN_REALIZATION_PRIMAL_NAMES,
    _CHAIN_REALIZATION_OUTPUT_FIELDS,
    _ordered_primal_args,
    scattering_chain_ensemble_eval,
    scattering_chain_ensemble_eval_backward,
    scattering_chain_ensemble_eval_jvp,
    scattering_chain_realization_eval_backward,
    scattering_chain_realization_eval_jvp,
)


def _grad_or_none(out: dict, key: str, needed: bool) -> torch.Tensor | None:
    return out[key] if needed else None


# ---------------------------------------------------------------------------
# Op A: multi-bounce ensemble (power domain).
# ---------------------------------------------------------------------------

# Fixed inputs of Op A (index into the apply arg list). Requesting a gradient on
# any of these in reverse mode fails loudly instead of silently detaching.
# ``_CHAIN_ENSEMBLE_FIXED`` is the reverse-mode set: it adds the continuous chain
# geometry (positions/normals/n_o/d_i/d_o/L1/L2/cos_i/cos_o) to the structurally
# frozen inputs, because reverse-mode chain geometry is a staged follow-up wave
# (the native backward rejects ``need_grad_geometry`` loudly). Forward-mode still
# forwards those geometry tangents (native ``_jvp`` supports them), so the JVP
# rejection set ``_CHAIN_ENSEMBLE_FIXED_TANGENTS`` excludes them.
_CHAIN_ENSEMBLE_FIXED_TANGENTS = (
    (0, "valid"),
    (1, "tx_pol"),
    (2, "rx_pol"),
    (3, "source"),
    (4, "vertex"),
    (5, "target"),
    (10, "c1_mu_r"),
    (13, "c1_depth"),
    (18, "c2_mu_r"),
    (21, "c2_depth"),
    (23, "t1r"),
    (24, "t2r"),
    (25, "backup_axis"),
    (26, "wi_local"),
    (33, "weights"),
    (34, "material_id"),
    (37, "table_offset"),
    (38, "table_dims"),
    (39, "material_slot"),
)
# Reverse-mode continuous geometry (fixed this wave; forward-mode tangent-able).
_CHAIN_ENSEMBLE_FIXED_GEOMETRY = (
    (6, "c1_positions"),
    (7, "c1_normals"),
    (14, "c2_positions"),
    (15, "c2_normals"),
    (22, "n_o"),
    (27, "cos_i"),
    (28, "cos_o"),
    (29, "d_i"),
    (30, "d_o"),
    (31, "l1"),
    (32, "l2"),
)
_CHAIN_ENSEMBLE_FIXED = _CHAIN_ENSEMBLE_FIXED_TANGENTS + _CHAIN_ENSEMBLE_FIXED_GEOMETRY


class _ScatteringChainEnsembleEvalAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable multi-bounce ensemble rows (ADR-021 Op A).

    Reverse-mode differentiable inputs: the two padded specular legs' Fresnel
    parameters (``c1/c2 eps_r/sigma_e/gain/thickness``), the resident BSDF tables
    (``f_te_flat``/``f_tm_flat``) and the two 0-dim AD scalars ``coef`` and
    ``frequency``. The continuous chain geometry
    (positions/normals/``n_o``/``d_i``/``d_o``/``l1``/``l2``/``cos_i``/``cos_o``)
    is a staged follow-up in reverse mode and fails loudly if requested; the
    forward-mode ``jvp`` still forwards its tangents. The endpoints
    (``source``/``vertex``/``target``), ``tx_pol``/``rx_pol``, the depths,
    ``mu_r``, the vertex frame axes ``t1r``/``t2r``/``backup_axis``,
    ``wi_local``, ``weights``, the material ids and the table metadata stay fixed
    in both modes. ``gain``/``amplitude``/``length`` are differentiable; ``keep``
    is marked non-differentiable.
    """

    @staticmethod
    def forward(
        valid,
        tx_pol,
        rx_pol,
        source,
        vertex,
        target,
        c1_positions,
        c1_normals,
        c1_eps_r,
        c1_sigma_e,
        c1_mu_r,
        c1_gain,
        c1_thickness,
        c1_depth,
        c2_positions,
        c2_normals,
        c2_eps_r,
        c2_sigma_e,
        c2_mu_r,
        c2_gain,
        c2_thickness,
        c2_depth,
        n_o,
        t1r,
        t2r,
        backup_axis,
        wi_local,
        cos_i,
        cos_o,
        d_i,
        d_o,
        l1,
        l2,
        weights,
        material_id,
        f_te_flat,
        f_tm_flat,
        table_offset,
        table_dims,
        material_slot,
        coef,
        frequency,
        threshold,
        coef_value,
        frequency_value,
    ):
        out = scattering_chain_ensemble_eval(
            valid,
            tx_pol,
            rx_pol,
            source,
            vertex,
            target,
            c1_positions,
            c1_normals,
            c1_eps_r,
            c1_sigma_e,
            c1_mu_r,
            c1_gain,
            c1_thickness,
            c1_depth,
            c2_positions,
            c2_normals,
            c2_eps_r,
            c2_sigma_e,
            c2_mu_r,
            c2_gain,
            c2_thickness,
            c2_depth,
            n_o,
            t1r,
            t2r,
            backup_axis,
            wi_local,
            cos_i,
            cos_o,
            d_i,
            d_o,
            l1,
            l2,
            weights,
            material_id,
            f_te_flat,
            f_tm_flat,
            table_offset,
            table_dims,
            material_slot,
            coef=coef_value,
            threshold=threshold,
            frequency_hz=frequency_value,
        )
        return tuple(out[name] for name in _CHAIN_ENSEMBLE_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        coef = inputs[40]
        frequency = inputs[41]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal for value in inputs[:40]
        )
        ctx.threshold = inputs[42]
        ctx.coef_value = inputs[43]
        ctx.frequency_value = inputs[44]
        ctx.coef_meta = (
            (coef.dtype, coef.device) if isinstance(coef, torch.Tensor) else None
        )
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        ctx.mark_non_differentiable(output[3])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_gain, grad_amplitude, grad_length, _grad_keep):
        none_grads = (None,) * 45
        # Rejects reverse-mode grads on both the structurally frozen inputs and
        # the continuous chain geometry (staged follow-up wave).
        _ad_reject_fixed_inputs(
            "scattering_chain_ensemble_eval_ad",
            ctx.needs_input_grad,
            _CHAIN_ENSEMBLE_FIXED,
        )
        needed = tuple(bool(ctx.needs_input_grad[i]) for i in range(42))
        need_chain1 = any(needed[i] for i in (8, 9, 11, 12))
        need_chain2 = any(needed[i] for i in (16, 17, 19, 20))
        need_tables = needed[35] or needed[36]
        need_coef = needed[40]
        need_frequency = needed[41]
        grads = (grad_gain, grad_amplitude, grad_length)
        if not (
            need_chain1
            or need_chain2
            or need_tables
            or need_coef
            or need_frequency
        ) or all(value is None for value in grads):
            return none_grads
        out = scattering_chain_ensemble_eval_backward(
            *ctx.saved_tensors,
            coef=ctx.coef_value,
            threshold=ctx.threshold,
            frequency_hz=ctx.frequency_value,
            grad_gain=grad_gain,
            grad_amplitude=grad_amplitude,
            grad_length=grad_length,
            need_grad_chain1=need_chain1,
            need_grad_chain2=need_chain2,
            need_grad_tables=need_tables,
            need_grad_geometry=False,
            need_grad_coef=need_coef,
            need_grad_frequency=need_frequency,
        )
        grad_coef = (
            _ad_frequency_grad(out["grad_coef"], ctx.coef_meta) if need_coef else None
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            None,  # valid
            None,  # tx_pol
            None,  # rx_pol
            None,  # source
            None,  # vertex
            None,  # target
            None,  # c1_positions (geometry: reverse staged)
            None,  # c1_normals
            _grad_or_none(out, "grad_c1_eps_r", needed[8]),
            _grad_or_none(out, "grad_c1_sigma_e", needed[9]),
            None,  # c1_mu_r
            _grad_or_none(out, "grad_c1_gain", needed[11]),
            _grad_or_none(out, "grad_c1_thickness", needed[12]),
            None,  # c1_depth
            None,  # c2_positions (geometry: reverse staged)
            None,  # c2_normals
            _grad_or_none(out, "grad_c2_eps_r", needed[16]),
            _grad_or_none(out, "grad_c2_sigma_e", needed[17]),
            None,  # c2_mu_r
            _grad_or_none(out, "grad_c2_gain", needed[19]),
            _grad_or_none(out, "grad_c2_thickness", needed[20]),
            None,  # c2_depth
            None,  # n_o (geometry: reverse staged)
            None,  # t1r
            None,  # t2r
            None,  # backup_axis
            None,  # wi_local
            None,  # cos_i (geometry: reverse staged)
            None,  # cos_o (geometry: reverse staged)
            None,  # d_i (geometry: reverse staged)
            None,  # d_o (geometry: reverse staged)
            None,  # l1 (geometry: reverse staged)
            None,  # l2 (geometry: reverse staged)
            None,  # weights
            None,  # material_id
            _grad_or_none(out, "grad_f_te", needed[35]),
            _grad_or_none(out, "grad_f_tm", needed[36]),
            None,  # table_offset
            None,  # table_dims
            None,  # material_slot
            grad_coef,
            grad_frequency,
            None,  # threshold
            None,  # coef_value
            None,  # frequency_value
        )

    @staticmethod
    def jvp(
        ctx,
        t_valid,
        t_tx_pol,
        t_rx_pol,
        t_source,
        t_vertex,
        t_target,
        t_c1_positions,
        t_c1_normals,
        t_c1_eps_r,
        t_c1_sigma_e,
        t_c1_mu_r,
        t_c1_gain,
        t_c1_thickness,
        t_c1_depth,
        t_c2_positions,
        t_c2_normals,
        t_c2_eps_r,
        t_c2_sigma_e,
        t_c2_mu_r,
        t_c2_gain,
        t_c2_thickness,
        t_c2_depth,
        t_n_o,
        t_t1r,
        t_t2r,
        t_backup_axis,
        t_wi_local,
        t_cos_i,
        t_cos_o,
        t_d_i,
        t_d_o,
        t_l1,
        t_l2,
        t_weights,
        t_material_id,
        t_f_te_flat,
        t_f_tm_flat,
        t_table_offset,
        t_table_dims,
        t_material_slot,
        t_coef,
        t_frequency,
        _t_threshold,
        _t_coef_value,
        _t_frequency_value,
    ):
        # Forward mode forwards geometry tangents (native jvp supports them);
        # only the structurally frozen inputs reject a tangent loudly.
        _ad_reject_fixed_tangents(
            "scattering_chain_ensemble_eval_ad",
            (
                (t_valid, "valid"),
                (t_tx_pol, "tx_pol"),
                (t_rx_pol, "rx_pol"),
                (t_source, "source"),
                (t_vertex, "vertex"),
                (t_target, "target"),
                (t_c1_mu_r, "c1_mu_r"),
                (t_c1_depth, "c1_depth"),
                (t_c2_mu_r, "c2_mu_r"),
                (t_c2_depth, "c2_depth"),
                (t_t1r, "t1r"),
                (t_t2r, "t2r"),
                (t_backup_axis, "backup_axis"),
                (t_wi_local, "wi_local"),
                (t_weights, "weights"),
                (t_material_id, "material_id"),
                (t_table_offset, "table_offset"),
                (t_table_dims, "table_dims"),
                (t_material_slot, "material_slot"),
            ),
        )
        saved = ctx.saved_tensors
        op = "scattering_chain_ensemble_eval_ad"
        tangents = {
            "tangent_c1_eps_r": _ad_geometry_tangent(f"{op} tangent_c1_eps_r", t_c1_eps_r, saved[8]),
            "tangent_c1_sigma_e": _ad_geometry_tangent(f"{op} tangent_c1_sigma_e", t_c1_sigma_e, saved[9]),
            "tangent_c1_gain": _ad_geometry_tangent(f"{op} tangent_c1_gain", t_c1_gain, saved[11]),
            "tangent_c1_thickness": _ad_geometry_tangent(f"{op} tangent_c1_thickness", t_c1_thickness, saved[12]),
            "tangent_c2_eps_r": _ad_geometry_tangent(f"{op} tangent_c2_eps_r", t_c2_eps_r, saved[16]),
            "tangent_c2_sigma_e": _ad_geometry_tangent(f"{op} tangent_c2_sigma_e", t_c2_sigma_e, saved[17]),
            "tangent_c2_gain": _ad_geometry_tangent(f"{op} tangent_c2_gain", t_c2_gain, saved[19]),
            "tangent_c2_thickness": _ad_geometry_tangent(f"{op} tangent_c2_thickness", t_c2_thickness, saved[20]),
            "tangent_f_te_flat": _ad_geometry_tangent(f"{op} tangent_f_te_flat", t_f_te_flat, saved[35]),
            "tangent_f_tm_flat": _ad_geometry_tangent(f"{op} tangent_f_tm_flat", t_f_tm_flat, saved[36]),
            "tangent_c1_positions": _ad_geometry_tangent(f"{op} tangent_c1_positions", t_c1_positions, saved[6]),
            "tangent_c1_normals": _ad_geometry_tangent(f"{op} tangent_c1_normals", t_c1_normals, saved[7]),
            "tangent_c2_positions": _ad_geometry_tangent(f"{op} tangent_c2_positions", t_c2_positions, saved[14]),
            "tangent_c2_normals": _ad_geometry_tangent(f"{op} tangent_c2_normals", t_c2_normals, saved[15]),
            "tangent_d_i": _ad_geometry_tangent(f"{op} tangent_d_i", t_d_i, saved[29]),
            "tangent_d_o": _ad_geometry_tangent(f"{op} tangent_d_o", t_d_o, saved[30]),
            "tangent_v_normal": _ad_geometry_tangent(f"{op} tangent_v_normal", t_n_o, saved[22]),
            "tangent_l1": _ad_geometry_tangent(f"{op} tangent_l1", t_l1, saved[31]),
            "tangent_l2": _ad_geometry_tangent(f"{op} tangent_l2", t_l2, saved[32]),
            "tangent_cos_i": _ad_geometry_tangent(f"{op} tangent_cos_i", t_cos_i, saved[27]),
            "tangent_cos_o": _ad_geometry_tangent(f"{op} tangent_cos_o", t_cos_o, saved[28]),
        }
        tangent_coef = _ad_frequency_tangent(t_coef)
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if (
            tangent_coef == 0.0
            and tangent_frequency == 0.0
            and all(value is None for value in tangents.values())
        ):
            return (None,) * len(_CHAIN_ENSEMBLE_OUTPUT_FIELDS)
        with torch_compat.disable_functorch():
            out = scattering_chain_ensemble_eval_jvp(
                *(_ad_native_tensor(value) for value in saved),
                coef=ctx.coef_value,
                threshold=ctx.threshold,
                frequency_hz=ctx.frequency_value,
                tangent_coef=tangent_coef,
                tangent_frequency=tangent_frequency,
                **tangents,
            )
        return (
            out["tangent_gain"],
            out["tangent_amplitude"],
            out["tangent_length"],
            None,
        )


def scattering_chain_ensemble_eval_ad(
    valid: torch.Tensor,
    tx_pol: torch.Tensor,
    rx_pol: torch.Tensor,
    source: torch.Tensor,
    vertex: torch.Tensor,
    target: torch.Tensor,
    c1_positions: torch.Tensor,
    c1_normals: torch.Tensor,
    c1_eps_r: torch.Tensor,
    c1_sigma_e: torch.Tensor,
    c1_mu_r: torch.Tensor,
    c1_gain: torch.Tensor,
    c1_thickness: torch.Tensor,
    c1_depth: torch.Tensor,
    c2_positions: torch.Tensor,
    c2_normals: torch.Tensor,
    c2_eps_r: torch.Tensor,
    c2_sigma_e: torch.Tensor,
    c2_mu_r: torch.Tensor,
    c2_gain: torch.Tensor,
    c2_thickness: torch.Tensor,
    c2_depth: torch.Tensor,
    n_o: torch.Tensor,
    t1r: torch.Tensor,
    t2r: torch.Tensor,
    backup_axis: torch.Tensor,
    wi_local: torch.Tensor,
    cos_i: torch.Tensor,
    cos_o: torch.Tensor,
    d_i: torch.Tensor,
    d_o: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    weights: torch.Tensor,
    material_id: torch.Tensor,
    f_te_flat: torch.Tensor,
    f_tm_flat: torch.Tensor,
    table_offset: torch.Tensor,
    table_dims: torch.Tensor,
    material_slot: torch.Tensor,
    *,
    coef: torch.Tensor,
    threshold: float,
    frequency: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`scattering_chain_ensemble_eval` (ADR-021 Op A).

    ``coef`` and ``frequency`` are the two AD-live scalars as 0-dim tensors so
    the radiometric-scale and frequency chains keep their gradients; their host
    values are read once per apply at this seam (audit M3), never per row.
    """

    primal_args = _ordered_primal_args(locals(), _CHAIN_ENSEMBLE_PRIMAL_NAMES)
    coef_value = _ad_frequency_value(coef)
    frequency_value = _ad_frequency_value(frequency)
    values = _ScatteringChainEnsembleEvalAdFunction.apply(
        *primal_args,
        coef,
        frequency,
        float(threshold),
        float(coef_value),
        float(frequency_value),
    )
    return dict(zip(_CHAIN_ENSEMBLE_OUTPUT_FIELDS, values, strict=True))


# ---------------------------------------------------------------------------
# Op B: coherent chain realization.
# ---------------------------------------------------------------------------

# Fixed inputs of Op B (index into the apply arg list, which appends the Duffy
# quadrature nodes as fixed inputs 42/43/44). ``source``/``vertex``/``target``
# are frozen structural endpoints (no tangent, no gradient in either mode).
_CHAIN_REALIZATION_FIXED = (
    (0, "valid"),
    (1, "patch_tris"),
    (2, "patch_uvs"),
    (3, "rows"),
    (6, "n_rows"),
    (7, "source"),
    (8, "vertex"),
    (9, "target"),
    (14, "c1_mu_r"),
    (17, "c1_depth"),
    (22, "c2_mu_r"),
    (25, "c2_depth"),
    (26, "tx_pol"),
    (27, "rx_pol"),
    (34, "cos_spec"),
    (35, "material_id"),
    (36, "layer_offset"),
    (37, "layer_count"),
    (41, "layer_mu_r"),
    (42, "quad_a"),
    (43, "quad_b"),
    (44, "quad_w"),
)


def _required_chain_realization_forward(
    valid,
    patch_tris,
    patch_uvs,
    rows,
    d_i,
    d_o,
    n_rows,
    source,
    vertex,
    target,
    c1_positions,
    c1_normals,
    c1_eps_r,
    c1_sigma_e,
    c1_mu_r,
    c1_gain,
    c1_thickness,
    c1_depth,
    c2_positions,
    c2_normals,
    c2_eps_r,
    c2_sigma_e,
    c2_mu_r,
    c2_gain,
    c2_thickness,
    c2_depth,
    tx_pol,
    rx_pol,
    L1,
    L2,
    sp1,
    sp2,
    centroids,
    heights,
    cos_spec,
    material_id,
    layer_offset,
    layer_count,
    layer_thickness_m,
    layer_eps_r,
    layer_sigma_e,
    layer_mu_r,
    quad_a,
    quad_b,
    quad_w,
    k0_value,
    frequency_value,
):
    """Raw Op B forward dispatch with explicit quadrature nodes (autograd seam).

    The Function holds the Duffy nodes as fixed inputs, so it dispatches the
    native op directly (the public facade appends the cached nodes itself).
    """

    out = _required_native_op("scattering_chain_realization_eval")(
        valid,
        patch_tris,
        patch_uvs,
        rows,
        d_i,
        d_o,
        n_rows,
        source,
        vertex,
        target,
        c1_positions,
        c1_normals,
        c1_eps_r,
        c1_sigma_e,
        c1_mu_r,
        c1_gain,
        c1_thickness,
        c1_depth,
        c2_positions,
        c2_normals,
        c2_eps_r,
        c2_sigma_e,
        c2_mu_r,
        c2_gain,
        c2_thickness,
        c2_depth,
        tx_pol,
        rx_pol,
        L1,
        L2,
        sp1,
        sp2,
        centroids,
        heights,
        cos_spec,
        material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        quad_a,
        quad_b,
        quad_w,
        float(k0_value),
        float(frequency_value),
    )
    expected = set(_CHAIN_REALIZATION_OUTPUT_FIELDS)
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError(
            "_channel.scattering_chain_realization_eval returned invalid fields"
        )
    return out


class _ScatteringChainRealizationEvalAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable coherent chain realization (ADR-021 Op B).

    Differentiable inputs: the phase-screen ``heights``, the CSR layer stack
    (``layer_thickness_m``/``layer_eps_r``/``layer_sigma_e``), the two padded
    specular legs' Fresnel parameters and geometry, the vertex directions
    ``d_i``/``d_o``, the unfolded lengths/spreading, the ``centroids`` and the
    two 0-dim AD scalars ``k0`` and ``frequency``. The patch mesh, ``rows``,
    ``n_rows``, the depths, ``mu_r``, ``cos_spec`` (derived-frozen), the
    material/CSR-index arrays, the endpoint polarizations and the quadrature
    nodes stay fixed; requesting their gradient fails loudly. ``total`` /
    ``path_field`` / ``path_gain`` are differentiable; ``integral`` and
    ``row_value`` are marked non-differentiable test buffers.
    """

    @staticmethod
    def forward(
        valid,
        patch_tris,
        patch_uvs,
        rows,
        d_i,
        d_o,
        n_rows,
        source,
        vertex,
        target,
        c1_positions,
        c1_normals,
        c1_eps_r,
        c1_sigma_e,
        c1_mu_r,
        c1_gain,
        c1_thickness,
        c1_depth,
        c2_positions,
        c2_normals,
        c2_eps_r,
        c2_sigma_e,
        c2_mu_r,
        c2_gain,
        c2_thickness,
        c2_depth,
        tx_pol,
        rx_pol,
        L1,
        L2,
        sp1,
        sp2,
        centroids,
        heights,
        cos_spec,
        material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        quad_a,
        quad_b,
        quad_w,
        k0,
        frequency,
        k0_value,
        frequency_value,
    ):
        out = _required_chain_realization_forward(
            valid,
            patch_tris,
            patch_uvs,
            rows,
            d_i,
            d_o,
            n_rows,
            source,
            vertex,
            target,
            c1_positions,
            c1_normals,
            c1_eps_r,
            c1_sigma_e,
            c1_mu_r,
            c1_gain,
            c1_thickness,
            c1_depth,
            c2_positions,
            c2_normals,
            c2_eps_r,
            c2_sigma_e,
            c2_mu_r,
            c2_gain,
            c2_thickness,
            c2_depth,
            tx_pol,
            rx_pol,
            L1,
            L2,
            sp1,
            sp2,
            centroids,
            heights,
            cos_spec,
            material_id,
            layer_offset,
            layer_count,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            quad_a,
            quad_b,
            quad_w,
            k0_value,
            frequency_value,
        )
        return tuple(out[name] for name in _CHAIN_REALIZATION_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        k0 = inputs[45]
        frequency = inputs[46]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal for value in inputs[:45]
        )
        ctx.k0_value = inputs[47]
        ctx.frequency_value = inputs[48]
        ctx.k0_meta = (k0.dtype, k0.device) if isinstance(k0, torch.Tensor) else None
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        ctx.mark_non_differentiable(output[3], output[4])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx, grad_total, grad_path_field, grad_path_gain, _grad_integral, _grad_row_value
    ):
        none_grads = (None,) * 49
        _ad_reject_fixed_inputs(
            "scattering_chain_realization_eval_ad",
            ctx.needs_input_grad,
            _CHAIN_REALIZATION_FIXED,
        )
        needed = tuple(bool(ctx.needs_input_grad[i]) for i in range(47))
        need_heights = needed[33]
        need_layers = any(needed[i] for i in (38, 39, 40))
        need_chain1 = any(needed[i] for i in (12, 13, 15, 16))
        need_chain2 = any(needed[i] for i in (20, 21, 23, 24))
        need_geometry = any(
            needed[i] for i in (4, 5, 10, 11, 18, 19, 28, 29, 30, 31, 32)
        )
        need_k0 = needed[45]
        need_frequency = needed[46]
        need_flags = (
            need_heights, need_layers, need_chain1, need_chain2,
            need_geometry, need_k0, need_frequency,
        )
        if not any(need_flags) or (
            grad_total is None and grad_path_field is None and grad_path_gain is None
        ):
            return none_grads
        saved = ctx.saved_tensors
        if grad_total is None:
            # path_field-only cotangents (D3 coherent combine) leave the scalar
            # total ungraded; the required ABI slot takes a zero cotangent.
            grad_total = torch.zeros((), dtype=torch.complex64, device=saved[1].device)
        out = scattering_chain_realization_eval_backward(
            *saved[:42],
            k0=ctx.k0_value,
            frequency_hz=ctx.frequency_value,
            grad_total=grad_total,
            grad_path_field=grad_path_field,
            grad_path_gain=grad_path_gain,
            need_grad_heights=need_heights,
            need_grad_layers=need_layers,
            need_grad_chain1=need_chain1,
            need_grad_chain2=need_chain2,
            need_grad_geometry=need_geometry,
            need_grad_k0=need_k0,
            need_grad_frequency=need_frequency,
        )
        grad_k0 = _ad_frequency_grad(out["grad_k0"], ctx.k0_meta) if need_k0 else None
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            None,  # valid
            None,  # patch_tris
            None,  # patch_uvs
            None,  # rows
            _grad_or_none(out, "grad_d_i", needed[4]),
            _grad_or_none(out, "grad_d_o", needed[5]),
            None,  # n_rows
            None,  # source
            None,  # vertex
            None,  # target
            _grad_or_none(out, "grad_c1_positions", needed[10]),
            _grad_or_none(out, "grad_c1_normals", needed[11]),
            _grad_or_none(out, "grad_c1_eps_r", needed[12]),
            _grad_or_none(out, "grad_c1_sigma_e", needed[13]),
            None,  # c1_mu_r
            _grad_or_none(out, "grad_c1_gain", needed[15]),
            _grad_or_none(out, "grad_c1_thickness", needed[16]),
            None,  # c1_depth
            _grad_or_none(out, "grad_c2_positions", needed[18]),
            _grad_or_none(out, "grad_c2_normals", needed[19]),
            _grad_or_none(out, "grad_c2_eps_r", needed[20]),
            _grad_or_none(out, "grad_c2_sigma_e", needed[21]),
            None,  # c2_mu_r
            _grad_or_none(out, "grad_c2_gain", needed[23]),
            _grad_or_none(out, "grad_c2_thickness", needed[24]),
            None,  # c2_depth
            None,  # tx_pol
            None,  # rx_pol
            _grad_or_none(out, "grad_L1", needed[28]),
            _grad_or_none(out, "grad_L2", needed[29]),
            _grad_or_none(out, "grad_sp1", needed[30]),
            _grad_or_none(out, "grad_sp2", needed[31]),
            _grad_or_none(out, "grad_centroids", needed[32]),
            _grad_or_none(out, "grad_heights", needed[33]),
            None,  # cos_spec
            None,  # material_id
            None,  # layer_offset
            None,  # layer_count
            _grad_or_none(out, "grad_layer_thickness", needed[38]),
            _grad_or_none(out, "grad_layer_eps_r", needed[39]),
            _grad_or_none(out, "grad_layer_sigma_e", needed[40]),
            None,  # layer_mu_r
            None,  # quad_a
            None,  # quad_b
            None,  # quad_w
            grad_k0,
            grad_frequency,
            None,  # k0_value
            None,  # frequency_value
        )

    @staticmethod
    def jvp(
        ctx,
        t_valid,
        t_patch_tris,
        t_patch_uvs,
        t_rows,
        t_d_i,
        t_d_o,
        t_n_rows,
        t_source,
        t_vertex,
        t_target,
        t_c1_positions,
        t_c1_normals,
        t_c1_eps_r,
        t_c1_sigma_e,
        t_c1_mu_r,
        t_c1_gain,
        t_c1_thickness,
        t_c1_depth,
        t_c2_positions,
        t_c2_normals,
        t_c2_eps_r,
        t_c2_sigma_e,
        t_c2_mu_r,
        t_c2_gain,
        t_c2_thickness,
        t_c2_depth,
        t_tx_pol,
        t_rx_pol,
        t_L1,
        t_L2,
        t_sp1,
        t_sp2,
        t_centroids,
        t_heights,
        t_cos_spec,
        t_material_id,
        t_layer_offset,
        t_layer_count,
        t_layer_thickness_m,
        t_layer_eps_r,
        t_layer_sigma_e,
        t_layer_mu_r,
        t_quad_a,
        t_quad_b,
        t_quad_w,
        t_k0,
        t_frequency,
        _t_k0_value,
        _t_frequency_value,
    ):
        _ad_reject_fixed_tangents(
            "scattering_chain_realization_eval_ad",
            (
                (t_valid, "valid"),
                (t_patch_tris, "patch_tris"),
                (t_patch_uvs, "patch_uvs"),
                (t_rows, "rows"),
                (t_n_rows, "n_rows"),
                (t_source, "source"),
                (t_vertex, "vertex"),
                (t_target, "target"),
                (t_c1_mu_r, "c1_mu_r"),
                (t_c1_depth, "c1_depth"),
                (t_c2_mu_r, "c2_mu_r"),
                (t_c2_depth, "c2_depth"),
                (t_tx_pol, "tx_pol"),
                (t_rx_pol, "rx_pol"),
                (t_cos_spec, "cos_spec"),
                (t_material_id, "material_id"),
                (t_layer_offset, "layer_offset"),
                (t_layer_count, "layer_count"),
                (t_layer_mu_r, "layer_mu_r"),
                (t_quad_a, "quad_a"),
                (t_quad_b, "quad_b"),
                (t_quad_w, "quad_w"),
            ),
        )
        saved = ctx.saved_tensors
        op = "scattering_chain_realization_eval_ad"
        tangents = {
            "tangent_heights": _ad_geometry_tangent(f"{op} tangent_heights", t_heights, saved[33]),
            "tangent_layer_thickness": _ad_geometry_tangent(f"{op} tangent_layer_thickness", t_layer_thickness_m, saved[38]),
            "tangent_layer_eps_r": _ad_geometry_tangent(f"{op} tangent_layer_eps_r", t_layer_eps_r, saved[39]),
            "tangent_layer_sigma_e": _ad_geometry_tangent(f"{op} tangent_layer_sigma_e", t_layer_sigma_e, saved[40]),
            "tangent_c1_eps_r": _ad_geometry_tangent(f"{op} tangent_c1_eps_r", t_c1_eps_r, saved[12]),
            "tangent_c1_sigma_e": _ad_geometry_tangent(f"{op} tangent_c1_sigma_e", t_c1_sigma_e, saved[13]),
            "tangent_c1_gain": _ad_geometry_tangent(f"{op} tangent_c1_gain", t_c1_gain, saved[15]),
            "tangent_c1_thickness": _ad_geometry_tangent(f"{op} tangent_c1_thickness", t_c1_thickness, saved[16]),
            "tangent_c2_eps_r": _ad_geometry_tangent(f"{op} tangent_c2_eps_r", t_c2_eps_r, saved[20]),
            "tangent_c2_sigma_e": _ad_geometry_tangent(f"{op} tangent_c2_sigma_e", t_c2_sigma_e, saved[21]),
            "tangent_c2_gain": _ad_geometry_tangent(f"{op} tangent_c2_gain", t_c2_gain, saved[23]),
            "tangent_c2_thickness": _ad_geometry_tangent(f"{op} tangent_c2_thickness", t_c2_thickness, saved[24]),
            "tangent_d_i": _ad_geometry_tangent(f"{op} tangent_d_i", t_d_i, saved[4]),
            "tangent_d_o": _ad_geometry_tangent(f"{op} tangent_d_o", t_d_o, saved[5]),
            "tangent_c1_positions": _ad_geometry_tangent(f"{op} tangent_c1_positions", t_c1_positions, saved[10]),
            "tangent_c1_normals": _ad_geometry_tangent(f"{op} tangent_c1_normals", t_c1_normals, saved[11]),
            "tangent_c2_positions": _ad_geometry_tangent(f"{op} tangent_c2_positions", t_c2_positions, saved[18]),
            "tangent_c2_normals": _ad_geometry_tangent(f"{op} tangent_c2_normals", t_c2_normals, saved[19]),
            "tangent_L1": _ad_geometry_tangent(f"{op} tangent_L1", t_L1, saved[28]),
            "tangent_L2": _ad_geometry_tangent(f"{op} tangent_L2", t_L2, saved[29]),
            "tangent_sp1": _ad_geometry_tangent(f"{op} tangent_sp1", t_sp1, saved[30]),
            "tangent_sp2": _ad_geometry_tangent(f"{op} tangent_sp2", t_sp2, saved[31]),
            "tangent_centroids": _ad_geometry_tangent(f"{op} tangent_centroids", t_centroids, saved[32]),
        }
        tangent_k0 = _ad_frequency_tangent(t_k0)
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if (
            tangent_k0 == 0.0
            and tangent_frequency == 0.0
            and all(value is None for value in tangents.values())
        ):
            return (None,) * len(_CHAIN_REALIZATION_OUTPUT_FIELDS)
        with torch_compat.disable_functorch():
            out = scattering_chain_realization_eval_jvp(
                *(_ad_native_tensor(value) for value in saved[:42]),
                k0=ctx.k0_value,
                frequency_hz=ctx.frequency_value,
                tangent_k0=tangent_k0,
                tangent_frequency=tangent_frequency,
                **tangents,
            )
        return (
            out["tangent_total"],
            out["tangent_path_field"],
            out["tangent_path_gain"],
            None,
            None,
        )


def scattering_chain_realization_eval_ad(
    valid: torch.Tensor,
    patch_tris: torch.Tensor,
    patch_uvs: torch.Tensor,
    rows: torch.Tensor,
    d_i: torch.Tensor,
    d_o: torch.Tensor,
    n_rows: torch.Tensor,
    source: torch.Tensor,
    vertex: torch.Tensor,
    target: torch.Tensor,
    c1_positions: torch.Tensor,
    c1_normals: torch.Tensor,
    c1_eps_r: torch.Tensor,
    c1_sigma_e: torch.Tensor,
    c1_mu_r: torch.Tensor,
    c1_gain: torch.Tensor,
    c1_thickness: torch.Tensor,
    c1_depth: torch.Tensor,
    c2_positions: torch.Tensor,
    c2_normals: torch.Tensor,
    c2_eps_r: torch.Tensor,
    c2_sigma_e: torch.Tensor,
    c2_mu_r: torch.Tensor,
    c2_gain: torch.Tensor,
    c2_thickness: torch.Tensor,
    c2_depth: torch.Tensor,
    tx_pol: torch.Tensor,
    rx_pol: torch.Tensor,
    L1: torch.Tensor,
    L2: torch.Tensor,
    sp1: torch.Tensor,
    sp2: torch.Tensor,
    centroids: torch.Tensor,
    heights: torch.Tensor,
    cos_spec: torch.Tensor,
    material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    k0: torch.Tensor,
    frequency: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`scattering_chain_realization_eval` (ADR-021 Op B).

    ``k0`` and ``frequency`` are the two AD-live scalars as 0-dim tensors so the
    carrier/prefactor and the in-kernel layer stack keep their gradients; their
    host values are read once per apply at this seam (audit M3). The Duffy
    quadrature nodes are gathered from the shared cache and threaded to the
    companions as fixed inputs.
    """

    from .functional import _duffy_nodes

    primal_args = _ordered_primal_args(locals(), _CHAIN_REALIZATION_PRIMAL_NAMES)
    quad_a, quad_b, quad_w = _duffy_nodes(patch_tris.device)
    k0_value = _ad_frequency_value(k0)
    frequency_value = _ad_frequency_value(frequency)
    values = _ScatteringChainRealizationEvalAdFunction.apply(
        *primal_args,
        quad_a,
        quad_b,
        quad_w,
        k0,
        frequency,
        float(k0_value),
        float(frequency_value),
    )
    return dict(zip(_CHAIN_REALIZATION_OUTPUT_FIELDS, values, strict=True))


__all__ = [
    "_ScatteringChainEnsembleEvalAdFunction",
    "_ScatteringChainRealizationEvalAdFunction",
    "scattering_chain_ensemble_eval_ad",
    "scattering_chain_realization_eval_ad",
]
