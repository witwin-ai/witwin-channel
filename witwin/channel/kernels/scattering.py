# Copyright Xingyu Chen.
# Native scattering kernel facades.

"""Native scattering kernel facades."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from witwin.channel.runtime import (
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
    validate_cuda_tensor,
)


__all__ = [
    "mc_scattering_pair_legs",
    "mc_scattering_pair_sum",
    "_KirchhoffTableBuildAdFunction",
    "_ScatteringEnsembleEvalAdFunction",
    "_ScatteringPatchIntegralEvalAdFunction",
    "kirchhoff_table_build_ad",
    "kirchhoff_table_build_backward",
    "kirchhoff_table_build_jvp",
    "scattering_chain_ensemble_eval",
    "scattering_ensemble_eval",
    "scattering_ensemble_eval_ad",
    "scattering_ensemble_eval_backward",
    "scattering_ensemble_eval_jvp",
    "scattering_patch_integral_eval",
    "scattering_patch_integral_eval_ad",
    "scattering_patch_integral_eval_backward",
    "scattering_patch_integral_eval_jvp",
    "scattering_table_pdf",
]


# ---------------------------------------------------------------------------
# functional
# ---------------------------------------------------------------------------
_PATCH_QUAD_ORDER = 16
_duffy_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _duffy_nodes(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Duffy-mapped 16x16 Gauss-Legendre nodes (float64 leggauss -> float32).

    The unit square ``(xi, eta)`` maps to barycentric ``(a, b) =
    (xi, eta * (1 - xi))`` with Jacobian ``(1 - xi)`` - the same construction
    as ``scattering.patch_phase_integral``.
    """

    cached = _duffy_cache.get(device)
    if cached is not None:
        return cached
    nodes, weights = np.polynomial.legendre.leggauss(_PATCH_QUAD_ORDER)
    xi = torch.from_numpy(0.5 * (nodes + 1.0)).to(device=device, dtype=torch.float32)
    w1 = torch.from_numpy(0.5 * weights).to(device=device, dtype=torch.float32)
    a = xi[:, None].expand(_PATCH_QUAD_ORDER, _PATCH_QUAD_ORDER)
    b = xi[None, :] * (1.0 - xi[:, None])
    w2d = (w1[:, None] * w1[None, :]) * (1.0 - xi[:, None])
    entry = (
        a.reshape(-1).contiguous(),
        b.reshape(-1).contiguous(),
        w2d.reshape(-1).contiguous(),
    )
    _duffy_cache[device] = entry
    return entry


_TABLE_EVAL_BACKWARD_FIELDS = ("grad_wi", "grad_wo", "grad_f_te", "grad_f_tm")
_TABLE_EVAL_JVP_FIELDS = ("tangent_f_te", "tangent_f_tm")


def scattering_table_pdf(
    valid: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor, sample_density: torch.Tensor, *,
    reverse: bool = False,
) -> torch.Tensor:
    """Native CUDA piecewise-constant Kirchhoff PDF; required op."""

    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=1)
    validate_cuda_tensor("wi", wi, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("wo", wo, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("sample_density", sample_density, dtype=torch.float32, ndim=4)
    if valid.shape != wi.shape[:1] or wo.shape != wi.shape:
        raise ValueError("valid, wi, and wo rows must match")
    return _required_native_op("scattering_table_pdf")(
        valid, wi, wo, sample_density, bool(reverse)
    )


def _validate_ensemble_inputs(scope: Mapping[str, torch.Tensor]) -> None:
    """Validate the shared ensemble primal inputs in ABI order."""

    validate_cuda_tensor("valid", scope["valid"], dtype=torch.bool, ndim=1)
    validate_cuda_tensor("wo_rows", scope["wo_rows"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("r2_rows", scope["r2_rows"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("cos_o_rows", scope["cos_o_rows"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("wi_local", scope["wi_local"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("rc_idx", scope["rc_idx"], dtype=torch.int64, ndim=1)
    validate_cuda_tensor("sc_idx", scope["sc_idx"], dtype=torch.int64, ndim=1)
    validate_cuda_tensor("material_id", scope["material_id"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("material_slot", scope["material_slot"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("table_dims", scope["table_dims"], dtype=torch.int32, ndim=2)
    if scope["valid"].shape != scope["wo_rows"].shape[:1]:
        raise ValueError("valid must match ensemble rows")


def scattering_ensemble_eval(
    valid: torch.Tensor, wo_rows: torch.Tensor, r2_rows: torch.Tensor, cos_o_rows: torch.Tensor,
    n_o: torch.Tensor, t1r: torch.Tensor, t2r: torch.Tensor, wi_local: torch.Tensor,
    cos_i: torch.Tensor, r1: torch.Tensor, a_te2: torch.Tensor, a_tm2: torch.Tensor,
    weights: torch.Tensor, material_id: torch.Tensor, backup_axis: torch.Tensor,
    rx_pol: torch.Tensor, rc_idx: torch.Tensor, sc_idx: torch.Tensor, f_te_flat: torch.Tensor,
    f_tm_flat: torch.Tensor, table_offset: torch.Tensor, table_dims: torch.Tensor,
    material_slot: torch.Tensor, *, coef: float, threshold: float,
) -> dict[str, torch.Tensor]:
    """Native Kirchhoff ensemble scattering row physics (rough-surface scattering).

    ``wo_rows`` / ``r2_rows`` / ``cos_o_rows`` are the surviving rows gathered
    from the Torch candidate grid (which stays Torch under the native contract). Per row the
    kernel owns wo_local, the stacked-table lookup, the outgoing s/p basis and
    receiver projections, and the radiometric gain/keep/amplitude/length.
    One launch per (tx, rx-chunk); required native op.
    """

    _validate_ensemble_inputs(locals())
    out = _required_native_op("scattering_ensemble_eval")(
        valid,
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
        float(coef),
        float(threshold),
    )
    expected = {"gain", "amplitude", "length", "keep"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError(
            "_channel.scattering_ensemble_eval returned invalid fields"
        )
    return out


def _validate_patch_inputs(scope: Mapping[str, torch.Tensor]) -> None:
    """Validate the shared patch-integral primal inputs in ABI order."""

    validate_cuda_tensor("valid", scope["valid"], dtype=torch.bool, ndim=1)
    validate_cuda_tensor("patch_tris", scope["patch_tris"], dtype=torch.float32, ndim=3)
    validate_cuda_tensor("patch_uvs", scope["patch_uvs"], dtype=torch.float32, ndim=3)
    validate_cuda_tensor("rows", scope["rows"], dtype=torch.int64, ndim=1)
    validate_cuda_tensor("d_i", scope["d_i"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("d_o", scope["d_o"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("r_te", scope["r_te"], dtype=torch.complex64, ndim=1)
    validate_cuda_tensor("r_tm", scope["r_tm"], dtype=torch.complex64, ndim=1)
    validate_cuda_tensor("heights", scope["heights"], dtype=torch.float32, ndim=2)
    if scope["valid"].shape != scope["rows"].shape:
        raise ValueError("valid must match patch rows")


def scattering_patch_integral_eval(
    valid: torch.Tensor, patch_tris: torch.Tensor, patch_uvs: torch.Tensor, rows: torch.Tensor,
    d_i: torch.Tensor, d_o: torch.Tensor, n_rows: torch.Tensor, r_te: torch.Tensor,
    r_tm: torch.Tensor, pol_t: torch.Tensor, pol_r: torch.Tensor, r1_rows: torch.Tensor,
    r2_rows: torch.Tensor, centroids: torch.Tensor, heights: torch.Tensor, *, k0: float,
) -> dict[str, torch.Tensor]:
    """Native realization-coherent phase-screen patch integral (rough-surface scattering).

    One fused launch per (tx, rx, structure) plus a fixed-order deterministic
    reduce: per selected patch row the kernel evaluates the Duffy-mapped 16x16
    Gauss-Legendre Kirchhoff phase integral over the phase-screen heights
    (half-texel edge-clamp bilinear sampling) and assembles the row
    coefficient (prefactor * jones * carrier / (r1 * r2)); the weighted total
    is a two-stage tree reduction (no float atomics). Returns the 0-dim
    complex64 ``total`` plus the per-row ``integral`` and ``row_value``
    buffers for tests. Required native op.
    """

    _validate_patch_inputs(locals())
    quad_a, quad_b, quad_w = _duffy_nodes(patch_tris.device)
    out = _required_native_op("scattering_patch_integral_eval")(
        valid,
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
        float(k0),
    )
    expected = {"total", "integral", "row_value"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError(
            "_channel.scattering_patch_integral_eval returned invalid fields"
        )
    return out


_ENSEMBLE_OUTPUT_FIELDS = ("gain", "amplitude", "length", "keep")
_ENSEMBLE_TANGENT_FIELDS = ("tangent_gain", "tangent_amplitude", "tangent_length")
_ENSEMBLE_BACKWARD_FIELDS = (
    "grad_wo_rows",
    "grad_r2_rows",
    "grad_cos_o_rows",
    "grad_n_o",
    "grad_t1r",
    "grad_t2r",
    "grad_wi_local",
    "grad_cos_i",
    "grad_r1",
    "grad_a_te2",
    "grad_a_tm2",
    "grad_weights",
    "grad_f_te",
    "grad_f_tm",
    "grad_coef",
)
_PATCH_OUTPUT_FIELDS = ("total", "integral", "row_value")
_PATCH_BACKWARD_FIELDS = (
    "grad_heights",
    "grad_r_te",
    "grad_r_tm",
    "grad_d_i",
    "grad_d_o",
    "grad_r1_rows",
    "grad_r2_rows",
    "grad_centroids",
    "grad_k0",
)


def scattering_ensemble_eval_backward(
    valid: torch.Tensor, wo_rows: torch.Tensor, r2_rows: torch.Tensor, cos_o_rows: torch.Tensor,
    n_o: torch.Tensor, t1r: torch.Tensor, t2r: torch.Tensor, wi_local: torch.Tensor,
    cos_i: torch.Tensor, r1: torch.Tensor, a_te2: torch.Tensor, a_tm2: torch.Tensor,
    weights: torch.Tensor, material_id: torch.Tensor, backup_axis: torch.Tensor,
    rx_pol: torch.Tensor, rc_idx: torch.Tensor, sc_idx: torch.Tensor, f_te_flat: torch.Tensor,
    f_tm_flat: torch.Tensor, table_offset: torch.Tensor, table_dims: torch.Tensor,
    material_slot: torch.Tensor, *, coef: float, threshold: float,
    grad_gain: torch.Tensor | None = None, grad_amplitude: torch.Tensor | None = None,
    grad_length: torch.Tensor | None = None, need_grad_rows: bool = False,
    need_grad_samples: bool = False, need_grad_tables: bool = False, need_grad_coef: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`scattering_ensemble_eval` (scattering AD)."""

    _validate_ensemble_inputs(locals())
    out = _required_native_op("scattering_ensemble_eval_backward")(
        valid,
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
        float(coef),
        float(threshold),
        grad_gain,
        grad_amplitude,
        grad_length,
        bool(need_grad_rows),
        bool(need_grad_samples),
        bool(need_grad_tables),
        bool(need_grad_coef),
    )
    if not isinstance(out, dict) or set(out) != set(_ENSEMBLE_BACKWARD_FIELDS):
        raise TypeError(
            "_channel.scattering_ensemble_eval_backward returned invalid fields"
        )
    return out


def scattering_ensemble_eval_jvp(
    valid: torch.Tensor, wo_rows: torch.Tensor, r2_rows: torch.Tensor, cos_o_rows: torch.Tensor,
    n_o: torch.Tensor, t1r: torch.Tensor, t2r: torch.Tensor, wi_local: torch.Tensor,
    cos_i: torch.Tensor, r1: torch.Tensor, a_te2: torch.Tensor, a_tm2: torch.Tensor,
    weights: torch.Tensor, material_id: torch.Tensor, backup_axis: torch.Tensor,
    rx_pol: torch.Tensor, rc_idx: torch.Tensor, sc_idx: torch.Tensor, f_te_flat: torch.Tensor,
    f_tm_flat: torch.Tensor, table_offset: torch.Tensor, table_dims: torch.Tensor,
    material_slot: torch.Tensor, *, coef: float, threshold: float,
    tangent_wo_rows: torch.Tensor | None = None, tangent_r2_rows: torch.Tensor | None = None,
    tangent_cos_o_rows: torch.Tensor | None = None, tangent_n_o: torch.Tensor | None = None,
    tangent_t1r: torch.Tensor | None = None, tangent_t2r: torch.Tensor | None = None,
    tangent_wi_local: torch.Tensor | None = None, tangent_cos_i: torch.Tensor | None = None,
    tangent_r1: torch.Tensor | None = None, tangent_a_te2: torch.Tensor | None = None,
    tangent_a_tm2: torch.Tensor | None = None, tangent_weights: torch.Tensor | None = None,
    tangent_f_te_flat: torch.Tensor | None = None, tangent_f_tm_flat: torch.Tensor | None = None,
    tangent_coef: float = 0.0,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`scattering_ensemble_eval` (scattering AD)."""

    _validate_ensemble_inputs(locals())
    out = _required_native_op("scattering_ensemble_eval_jvp")(
        valid,
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
        float(coef),
        float(threshold),
        tangent_wo_rows,
        tangent_r2_rows,
        tangent_cos_o_rows,
        tangent_n_o,
        tangent_t1r,
        tangent_t2r,
        tangent_wi_local,
        tangent_cos_i,
        tangent_r1,
        tangent_a_te2,
        tangent_a_tm2,
        tangent_weights,
        tangent_f_te_flat,
        tangent_f_tm_flat,
        float(tangent_coef),
    )
    if not isinstance(out, dict) or set(out) != set(_ENSEMBLE_TANGENT_FIELDS):
        raise TypeError(
            "_channel.scattering_ensemble_eval_jvp returned invalid fields"
        )
    return out


def scattering_patch_integral_eval_backward(
    valid: torch.Tensor, patch_tris: torch.Tensor, patch_uvs: torch.Tensor, rows: torch.Tensor,
    d_i: torch.Tensor, d_o: torch.Tensor, n_rows: torch.Tensor, r_te: torch.Tensor,
    r_tm: torch.Tensor, pol_t: torch.Tensor, pol_r: torch.Tensor, r1_rows: torch.Tensor,
    r2_rows: torch.Tensor, centroids: torch.Tensor, heights: torch.Tensor, *, k0: float,
    grad_total: torch.Tensor, need_grad_heights: bool = False, need_grad_jones: bool = False,
    need_grad_geometry: bool = False, need_grad_k0: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`scattering_patch_integral_eval` (scattering AD)."""

    _validate_patch_inputs(locals())
    quad_a, quad_b, quad_w = _duffy_nodes(patch_tris.device)
    out = _required_native_op("scattering_patch_integral_eval_backward")(
        valid,
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
        float(k0),
        grad_total,
        bool(need_grad_heights),
        bool(need_grad_jones),
        bool(need_grad_geometry),
        bool(need_grad_k0),
    )
    if not isinstance(out, dict) or set(out) != set(_PATCH_BACKWARD_FIELDS):
        raise TypeError(
            "_channel.scattering_patch_integral_eval_backward returned"
            " invalid fields"
        )
    return out


def scattering_patch_integral_eval_jvp(
    valid: torch.Tensor, patch_tris: torch.Tensor, patch_uvs: torch.Tensor, rows: torch.Tensor,
    d_i: torch.Tensor, d_o: torch.Tensor, n_rows: torch.Tensor, r_te: torch.Tensor,
    r_tm: torch.Tensor, pol_t: torch.Tensor, pol_r: torch.Tensor, r1_rows: torch.Tensor,
    r2_rows: torch.Tensor, centroids: torch.Tensor, heights: torch.Tensor, *, k0: float,
    tangent_heights: torch.Tensor | None = None, tangent_r_te: torch.Tensor | None = None,
    tangent_r_tm: torch.Tensor | None = None, tangent_d_i: torch.Tensor | None = None,
    tangent_d_o: torch.Tensor | None = None, tangent_r1_rows: torch.Tensor | None = None,
    tangent_r2_rows: torch.Tensor | None = None, tangent_centroids: torch.Tensor | None = None,
    tangent_k0: float = 0.0,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`scattering_patch_integral_eval` (scattering AD)."""

    _validate_patch_inputs(locals())
    quad_a, quad_b, quad_w = _duffy_nodes(patch_tris.device)
    out = _required_native_op("scattering_patch_integral_eval_jvp")(
        valid,
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
        float(k0),
        tangent_heights,
        tangent_r_te,
        tangent_r_tm,
        tangent_d_i,
        tangent_d_o,
        tangent_r1_rows,
        tangent_r2_rows,
        tangent_centroids,
        float(tangent_k0),
    )
    if not isinstance(out, dict) or set(out) != {"tangent_total"}:
        raise TypeError(
            "_channel.scattering_patch_integral_eval_jvp returned invalid fields"
        )
    return out


# ---------------------------------------------------------------------------
# functional_chain
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Coherent scattering-chain kernels.
#
# Op A (``scattering_chain_ensemble_eval``) generalizes op 1 to a specular
# reflection chain C1 -> diffuse vertex v_s -> chain C2 in the power domain;
# Op B (``scattering_chain_realization_eval``) generalizes op 2 to the coherent
# 2x2 Jones sandwich with a phase-screen realization at the vertex. Both carry
# two padded per-leg blocks ``[R, Dmax, ...]`` with ``Dmax = kMaxAdDepth = 8``
#. The facades are thin: validate the spec tables, dispatch
# the required native symbol, and assert the returned key-set exactly.
# ---------------------------------------------------------------------------

_K_MAX_AD_DEPTH = 8

_CHAIN_ENSEMBLE_OUTPUT_FIELDS = ("gain", "amplitude", "length", "keep")
_CHAIN_ENSEMBLE_TANGENT_FIELDS = (
    "tangent_gain",
    "tangent_amplitude",
    "tangent_length",
)
_CHAIN_ENSEMBLE_BACKWARD_FIELDS = (
    "grad_c1_eps_r",
    "grad_c1_sigma_e",
    "grad_c1_gain",
    "grad_c1_thickness",
    "grad_c2_eps_r",
    "grad_c2_sigma_e",
    "grad_c2_gain",
    "grad_c2_thickness",
    "grad_f_te",
    "grad_f_tm",
    "grad_coef",
    "grad_frequency",
)

_CHAIN_REALIZATION_OUTPUT_FIELDS = (
    "total",
    "path_field",
    "path_gain",
    "integral",
    "row_value",
)
_CHAIN_REALIZATION_TANGENT_FIELDS = (
    "tangent_total",
    "tangent_path_field",
    "tangent_path_gain",
)
_CHAIN_REALIZATION_BACKWARD_FIELDS = (
    "grad_heights",
    "grad_layer_thickness",
    "grad_layer_eps_r",
    "grad_layer_sigma_e",
    "grad_c1_eps_r",
    "grad_c1_sigma_e",
    "grad_c1_gain",
    "grad_c1_thickness",
    "grad_c2_eps_r",
    "grad_c2_sigma_e",
    "grad_c2_gain",
    "grad_c2_thickness",
    "grad_d_i",
    "grad_d_o",
    "grad_c1_positions",
    "grad_c1_normals",
    "grad_c2_positions",
    "grad_c2_normals",
    "grad_L1",
    "grad_L2",
    "grad_sp1",
    "grad_sp2",
    "grad_centroids",
    "grad_k0",
    "grad_frequency",
)

_CHAIN_ENSEMBLE_PRIMAL_NAMES = (
    "valid", "tx_pol", "rx_pol", "source", "vertex", "target",
    "c1_positions", "c1_normals", "c1_eps_r", "c1_sigma_e", "c1_mu_r",
    "c1_gain", "c1_thickness", "c1_depth",
    "c2_positions", "c2_normals", "c2_eps_r", "c2_sigma_e", "c2_mu_r",
    "c2_gain", "c2_thickness", "c2_depth",
    "n_o", "t1r", "t2r", "backup_axis", "wi_local", "cos_i", "cos_o",
    "d_i", "d_o", "l1", "l2", "weights", "material_id", "f_te_flat",
    "f_tm_flat", "table_offset", "table_dims", "material_slot",
)
_CHAIN_REALIZATION_PRIMAL_NAMES = (
    "valid", "patch_tris", "patch_uvs", "rows", "d_i", "d_o", "n_rows", "source",
    "vertex", "target", "c1_positions", "c1_normals", "c1_eps_r",
    "c1_sigma_e", "c1_mu_r", "c1_gain", "c1_thickness", "c1_depth",
    "c2_positions", "c2_normals", "c2_eps_r", "c2_sigma_e", "c2_mu_r",
    "c2_gain", "c2_thickness", "c2_depth", "tx_pol", "rx_pol", "L1", "L2",
    "sp1", "sp2", "centroids", "heights", "cos_spec", "material_id",
    "layer_offset", "layer_count", "layer_thickness_m", "layer_eps_r",
    "layer_sigma_e", "layer_mu_r",
)


def _ordered_primal_args(scope: dict[str, object], names: tuple[str, ...]) -> tuple[object, ...]:
    """Pack caller locals in the frozen typed native ABI order."""

    return tuple(scope[name] for name in names)


def _validate_chain_leg(
    prefix: str, positions: torch.Tensor, normals: torch.Tensor, eps_r: torch.Tensor,
    sigma_e: torch.Tensor, mu_r: torch.Tensor, gain: torch.Tensor, thickness: torch.Tensor,
    depth: torch.Tensor, rows: int,
) -> int:
    """Validate one padded specular leg block and return its ``Dmax`` width.

    The padded block is ``[R, Dmax, 3]`` positions/normals and ``[R, Dmax]``
    per-bounce Fresnel inputs with an ``[R]`` int32 per-row depth. ``Dmax`` is
    the static padded width and must not exceed ``kMaxAdDepth = 8``; the per-row ``depth`` values are trusted structural winners and
    are not read on the host (that would force a device sync).
    """

    validate_cuda_tensor(
        f"{prefix}_positions", positions, dtype=torch.float32, ndim=3, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        f"{prefix}_normals", normals, dtype=torch.float32, ndim=3, trailing_shape=(3,)
    )
    dmax = positions.shape[1]
    if dmax > _K_MAX_AD_DEPTH:
        raise ValueError(
            f"{prefix} leg depth {dmax} exceeds kMaxAdDepth={_K_MAX_AD_DEPTH}"
        )
    if tuple(positions.shape) != (rows, dmax, 3):
        raise ValueError(f"{prefix}_positions must have shape ({rows}, {dmax}, 3)")
    if tuple(normals.shape) != (rows, dmax, 3):
        raise ValueError(f"{prefix}_normals must have shape ({rows}, {dmax}, 3)")
    for name, tensor in (
        (f"{prefix}_eps_r", eps_r),
        (f"{prefix}_sigma_e", sigma_e),
        (f"{prefix}_mu_r", mu_r),
        (f"{prefix}_gain", gain),
        (f"{prefix}_thickness", thickness),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=2)
        if tuple(tensor.shape) != (rows, dmax):
            raise ValueError(f"{name} must have shape ({rows}, {dmax})")
    validate_cuda_tensor(f"{prefix}_depth", depth, dtype=torch.int32, ndim=1)
    if depth.shape[0] != rows:
        raise ValueError(f"{prefix}_depth must have {rows} rows")
    return dmax


def _validate_chain_valid(valid: torch.Tensor, rows: int, reference: torch.Tensor) -> None:
    """Validate the caller-owned device row mask without materializing one."""

    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=1)
    if valid.shape != (rows,):
        raise ValueError(f"valid must have shape ({rows},)")
    if valid.device != reference.device:
        raise ValueError("valid must share the chain row CUDA device")


def _require_same_device(named: tuple[tuple[str, torch.Tensor], ...]) -> None:
    """Assert every named tensor shares one CUDA device."""

    device = None
    for name, tensor in named:
        if tensor is None:
            continue
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            raise ValueError(
                f"{name} must share the CUDA device of the chain row block"
            )


def scattering_chain_ensemble_eval(
    valid: torch.Tensor, tx_pol: torch.Tensor, rx_pol: torch.Tensor, source: torch.Tensor,
    vertex: torch.Tensor, target: torch.Tensor, c1_positions: torch.Tensor,
    c1_normals: torch.Tensor, c1_eps_r: torch.Tensor, c1_sigma_e: torch.Tensor,
    c1_mu_r: torch.Tensor, c1_gain: torch.Tensor, c1_thickness: torch.Tensor,
    c1_depth: torch.Tensor, c2_positions: torch.Tensor, c2_normals: torch.Tensor,
    c2_eps_r: torch.Tensor, c2_sigma_e: torch.Tensor, c2_mu_r: torch.Tensor, c2_gain: torch.Tensor,
    c2_thickness: torch.Tensor, c2_depth: torch.Tensor, n_o: torch.Tensor, t1r: torch.Tensor,
    t2r: torch.Tensor, backup_axis: torch.Tensor, wi_local: torch.Tensor, cos_i: torch.Tensor,
    cos_o: torch.Tensor, d_i: torch.Tensor, d_o: torch.Tensor, l1: torch.Tensor, l2: torch.Tensor,
    weights: torch.Tensor, material_id: torch.Tensor, f_te_flat: torch.Tensor,
    f_tm_flat: torch.Tensor, table_offset: torch.Tensor, table_dims: torch.Tensor,
    material_slot: torch.Tensor, *, coef: float, threshold: float, frequency_hz: float,
) -> dict[str, torch.Tensor]:
    """Native multi-bounce ensemble scattering rows.

    Power-domain generalization of :func:`scattering_ensemble_eval`: C1 coherent
    Jones transport of ``tx_pol`` from ``source`` to the diffuse ``vertex`` yields
    the incident coherency diagonal, the resident Kirchhoff table gives the
    outgoing diagonal, and the C2 power-domain sandwich to ``target`` plus the
    receiver projection assemble the radiometric row gain (per-vertex ``weights``
    = ``A_patch`` and ``1/(L1^2 L2^2)`` spreading, op-1 convention). The incident
    coherency is computed in-kernel from the C1 transport (retained forward bin sums: no
    ``a_te2``/``a_tm2`` projection pair, no cross-pol slots). One launch per
    (tx, rx-chunk); required native op.
    """

    rows = int(tx_pol.shape[0])
    _validate_chain_valid(valid, rows, tx_pol)
    validate_cuda_tensor("tx_pol", tx_pol, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("rx_pol", rx_pol, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if rx_pol.shape[0] != rows:
        raise ValueError("rx_pol must match tx_pol rows")
    _validate_chain_leg(
        "c1", c1_positions, c1_normals, c1_eps_r, c1_sigma_e, c1_mu_r, c1_gain,
        c1_thickness, c1_depth, rows,
    )
    _validate_chain_leg(
        "c2", c2_positions, c2_normals, c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain,
        c2_thickness, c2_depth, rows,
    )
    for name, tensor in (
        ("source", source),
        ("vertex", vertex),
        ("target", target),
        ("n_o", n_o),
        ("t1r", t1r),
        ("t2r", t2r),
        ("backup_axis", backup_axis),
        ("wi_local", wi_local),
        ("d_i", d_i),
        ("d_o", d_o),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,))
        if tensor.shape[0] != rows:
            raise ValueError(f"{name} must have {rows} rows")
    for name, tensor in (
        ("cos_i", cos_i),
        ("cos_o", cos_o),
        ("l1", l1),
        ("l2", l2),
        ("weights", weights),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=1)
        if tensor.shape[0] != rows:
            raise ValueError(f"{name} must have {rows} rows")
    validate_cuda_tensor("material_id", material_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("f_te_flat", f_te_flat, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("f_tm_flat", f_tm_flat, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("table_offset", table_offset, dtype=torch.int64, ndim=1)
    validate_cuda_tensor("table_dims", table_dims, dtype=torch.int32, ndim=2, trailing_shape=(4,))
    validate_cuda_tensor("material_slot", material_slot, dtype=torch.int32, ndim=1)
    _require_same_device(
        (
            ("tx_pol", tx_pol),
            ("source", source),
            ("c1_positions", c1_positions),
            ("c2_positions", c2_positions),
            ("d_i", d_i),
            ("f_te_flat", f_te_flat),
            ("material_slot", material_slot),
        )
    )
    primal_args = _ordered_primal_args(locals(), _CHAIN_ENSEMBLE_PRIMAL_NAMES)
    out = _required_native_op("scattering_chain_ensemble_eval")(
        *primal_args,
        float(coef),
        float(threshold),
        float(frequency_hz),
    )
    if not isinstance(out, dict) or set(out) != set(_CHAIN_ENSEMBLE_OUTPUT_FIELDS):
        raise TypeError(
            "_channel.scattering_chain_ensemble_eval returned invalid fields"
        )
    return out


# ---------------------------------------------------------------------------
# autograd
# ---------------------------------------------------------------------------
def _grad_or_none(out: dict, key: str, needed: bool) -> torch.Tensor | None:
    return out[key] if needed else None


# Fixed inputs of the ensemble op: rejecting a gradient/tangent here fails
# loudly instead of silently detaching (scattering AD).
_ENSEMBLE_FIXED = (
    (0, "valid"),
    (13, "material_id"),
    (14, "backup_axis"),
    (15, "rx_pol"),
    (16, "rc_idx"),
    (17, "sc_idx"),
    (20, "table_offset"),
    (21, "table_dims"),
    (22, "material_slot"),
)
_PATCH_FIXED = (
    (0, "valid"),
    (1, "patch_tris"),
    (2, "patch_uvs"),
    (3, "rows"),
    (6, "n_rows"),
    (9, "pol_t"),
    (10, "pol_r"),
    (15, "quad_a"),
    (16, "quad_b"),
    (17, "quad_w"),
)


class _ScatteringEnsembleEvalAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable Kirchhoff ensemble rows (scattering AD).

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
        valid, wo_rows, r2_rows, cos_o_rows, n_o, t1r, t2r, wi_local, cos_i, r1, a_te2, a_tm2,
        weights, material_id, backup_axis, rx_pol, rc_idx, sc_idx, f_te_flat, f_tm_flat,
        table_offset, table_dims, material_slot, coef, threshold, coef_value,
    ):
        out = scattering_ensemble_eval(
            valid,
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
        coef = inputs[23]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:23]
        )
        ctx.threshold = inputs[24]
        ctx.coef_value = inputs[25]
        ctx.coef_meta = (
            (coef.dtype, coef.device) if isinstance(coef, torch.Tensor) else None
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        ctx.mark_non_differentiable(output[3])

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_gain, grad_amplitude, grad_length, _grad_keep):
        none_grads = (None,) * 26
        _ad_reject_fixed_inputs(
            "scattering_ensemble_eval_ad", ctx.needs_input_grad, _ENSEMBLE_FIXED
        )
        needed = tuple(bool(ctx.needs_input_grad[i]) for i in range(24))
        need_rows = any(needed[1:4])
        need_samples = any(needed[4:13])
        need_tables = needed[18] or needed[19]
        need_coef = needed[23]
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
            None,
            _grad_or_none(out, "grad_wo_rows", needed[1]),
            _grad_or_none(out, "grad_r2_rows", needed[2]),
            _grad_or_none(out, "grad_cos_o_rows", needed[3]),
            _grad_or_none(out, "grad_n_o", needed[4]),
            _grad_or_none(out, "grad_t1r", needed[5]),
            _grad_or_none(out, "grad_t2r", needed[6]),
            _grad_or_none(out, "grad_wi_local", needed[7]),
            _grad_or_none(out, "grad_cos_i", needed[8]),
            _grad_or_none(out, "grad_r1", needed[9]),
            _grad_or_none(out, "grad_a_te2", needed[10]),
            _grad_or_none(out, "grad_a_tm2", needed[11]),
            _grad_or_none(out, "grad_weights", needed[12]),
            None,
            None,
            None,
            None,
            None,
            _grad_or_none(out, "grad_f_te", needed[18]),
            _grad_or_none(out, "grad_f_tm", needed[19]),
            None,
            None,
            None,
            grad_coef,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx, t_valid, t_wo_rows, t_r2_rows, t_cos_o_rows, t_n_o, t_t1r, t_t2r, t_wi_local, t_cos_i,
        t_r1, t_a_te2, t_a_tm2, t_weights, t_material_id, t_backup_axis, t_rx_pol, t_rc_idx,
        t_sc_idx, t_f_te_flat, t_f_tm_flat, t_table_offset, t_table_dims, t_material_slot, t_coef,
        _t_threshold, _t_coef_value,
    ):
        _ad_reject_fixed_tangents(
            "scattering_ensemble_eval_ad",
            (
                (t_valid, "valid"),
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
                "scattering_ensemble_eval_ad tangent_wo_rows", t_wo_rows, saved[1]
            ),
            "tangent_r2_rows": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_r2_rows", t_r2_rows, saved[2]
            ),
            "tangent_cos_o_rows": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_cos_o_rows",
                t_cos_o_rows,
                saved[3],
            ),
            "tangent_n_o": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_n_o", t_n_o, saved[4]
            ),
            "tangent_t1r": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_t1r", t_t1r, saved[5]
            ),
            "tangent_t2r": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_t2r", t_t2r, saved[6]
            ),
            "tangent_wi_local": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_wi_local", t_wi_local, saved[7]
            ),
            "tangent_cos_i": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_cos_i", t_cos_i, saved[8]
            ),
            "tangent_r1": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_r1", t_r1, saved[9]
            ),
            "tangent_a_te2": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_a_te2", t_a_te2, saved[10]
            ),
            "tangent_a_tm2": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_a_tm2", t_a_tm2, saved[11]
            ),
            "tangent_weights": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_weights", t_weights, saved[12]
            ),
            "tangent_f_te_flat": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_f_te_flat", t_f_te_flat, saved[18]
            ),
            "tangent_f_tm_flat": _ad_geometry_tangent(
                "scattering_ensemble_eval_ad tangent_f_tm_flat", t_f_tm_flat, saved[19]
            ),
        }
        tangent_coef = _ad_frequency_tangent(t_coef)
        if tangent_coef == 0.0 and all(value is None for value in tangents.values()):
            return (None,) * len(_ENSEMBLE_OUTPUT_FIELDS)
        with disable_functorch():
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
    valid: torch.Tensor, wo_rows: torch.Tensor, r2_rows: torch.Tensor, cos_o_rows: torch.Tensor,
    n_o: torch.Tensor, t1r: torch.Tensor, t2r: torch.Tensor, wi_local: torch.Tensor,
    cos_i: torch.Tensor, r1: torch.Tensor, a_te2: torch.Tensor, a_tm2: torch.Tensor,
    weights: torch.Tensor, material_id: torch.Tensor, backup_axis: torch.Tensor,
    rx_pol: torch.Tensor, rc_idx: torch.Tensor, sc_idx: torch.Tensor, f_te_flat: torch.Tensor,
    f_tm_flat: torch.Tensor, table_offset: torch.Tensor, table_dims: torch.Tensor,
    material_slot: torch.Tensor, *, coef: torch.Tensor, threshold: float,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`scattering_ensemble_eval` (scattering AD).

    ``coef`` is the radiometric scale as a 0-d scalar tensor so frequency (and
    tx power) gradients flow through the ensemble rows; its host value is read
    once per apply at this seam, never per row.
    """

    coef_value = _ad_frequency_value(coef)
    values = _ScatteringEnsembleEvalAdFunction.apply(
        valid,
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


_TABLE_EVAL_OUTPUT_FIELDS = ("f_te", "f_tm")


class _ScatteringPatchIntegralEvalAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable phase-screen patch integral (scattering AD).

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
        valid, patch_tris, patch_uvs, rows, d_i, d_o, n_rows, r_te, r_tm, pol_t, pol_r, r1_rows,
        r2_rows, centroids, heights, quad_a, quad_b, quad_w, k0, k0_value,
    ):
        out = _required_patch_forward(
            valid,
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
        k0 = inputs[18]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:18]
        )
        ctx.k0_value = inputs[19]
        ctx.k0_meta = (
            (k0.dtype, k0.device) if isinstance(k0, torch.Tensor) else None
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        ctx.mark_non_differentiable(output[1], output[2])

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_total, _grad_integral, _grad_row_value):
        none_grads = (None,) * 20
        _ad_reject_fixed_inputs(
            "scattering_patch_integral_eval_ad", ctx.needs_input_grad, _PATCH_FIXED
        )
        needed = tuple(bool(ctx.needs_input_grad[i]) for i in range(19))
        need_heights = needed[14]
        need_jones = needed[7] or needed[8]
        need_geometry = any(needed[i] for i in (4, 5, 11, 12, 13))
        need_k0 = needed[18]
        if not (need_heights or need_jones or need_geometry or need_k0) or (
            grad_total is None
        ):
            return none_grads
        out = scattering_patch_integral_eval_backward(
            *ctx.saved_tensors[:15],
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
            None,
            _grad_or_none(out, "grad_d_i", needed[4]),
            _grad_or_none(out, "grad_d_o", needed[5]),
            None,
            _grad_or_none(out, "grad_r_te", needed[7]),
            _grad_or_none(out, "grad_r_tm", needed[8]),
            None,
            None,
            _grad_or_none(out, "grad_r1_rows", needed[11]),
            _grad_or_none(out, "grad_r2_rows", needed[12]),
            _grad_or_none(out, "grad_centroids", needed[13]),
            _grad_or_none(out, "grad_heights", needed[14]),
            None,
            None,
            None,
            grad_k0,
            None,
        )

    @staticmethod
    def jvp(
        ctx, t_valid, t_patch_tris, t_patch_uvs, t_rows, t_d_i, t_d_o, t_n_rows, t_r_te, t_r_tm,
        t_pol_t, t_pol_r, t_r1_rows, t_r2_rows, t_centroids, t_heights, t_quad_a, t_quad_b,
        t_quad_w, t_k0, _t_k0_value,
    ):
        _ad_reject_fixed_tangents(
            "scattering_patch_integral_eval_ad",
            (
                (t_valid, "valid"),
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
                saved[14],
            ),
            "tangent_r_te": _ad_geometry_tangent(
                "scattering_patch_integral_eval_ad tangent_r_te", t_r_te, saved[7]
            ),
            "tangent_r_tm": _ad_geometry_tangent(
                "scattering_patch_integral_eval_ad tangent_r_tm", t_r_tm, saved[8]
            ),
            "tangent_d_i": _ad_geometry_tangent(
                "scattering_patch_integral_eval_ad tangent_d_i", t_d_i, saved[4]
            ),
            "tangent_d_o": _ad_geometry_tangent(
                "scattering_patch_integral_eval_ad tangent_d_o", t_d_o, saved[5]
            ),
            "tangent_r1_rows": _ad_geometry_tangent(
                "scattering_patch_integral_eval_ad tangent_r1_rows",
                t_r1_rows,
                saved[11],
            ),
            "tangent_r2_rows": _ad_geometry_tangent(
                "scattering_patch_integral_eval_ad tangent_r2_rows",
                t_r2_rows,
                saved[12],
            ),
            "tangent_centroids": _ad_geometry_tangent(
                "scattering_patch_integral_eval_ad tangent_centroids",
                t_centroids,
                saved[13],
            ),
        }
        tangent_k0 = _ad_frequency_tangent(t_k0)
        if tangent_k0 == 0.0 and all(value is None for value in tangents.values()):
            return (None,) * len(_PATCH_OUTPUT_FIELDS)
        with disable_functorch():
            out = scattering_patch_integral_eval_jvp(
                *(_ad_native_tensor(value) for value in saved[:15]),
                k0=ctx.k0_value,
                tangent_k0=tangent_k0,
                **tangents,
            )
        return (out["tangent_total"], None, None)


def _required_patch_forward(
    valid, patch_tris, patch_uvs, rows, d_i, d_o, n_rows, r_te, r_tm, pol_t, pol_r, r1_rows,
    r2_rows, centroids, heights, quad_a, quad_b, quad_w, k0_value,
):
    out = _required_native_op("scattering_patch_integral_eval")(
        valid,
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
            "_channel.scattering_patch_integral_eval returned invalid fields"
        )
    return out


def scattering_patch_integral_eval_ad(
    valid: torch.Tensor, patch_tris: torch.Tensor, patch_uvs: torch.Tensor, rows: torch.Tensor,
    d_i: torch.Tensor, d_o: torch.Tensor, n_rows: torch.Tensor, r_te: torch.Tensor,
    r_tm: torch.Tensor, pol_t: torch.Tensor, pol_r: torch.Tensor, r1_rows: torch.Tensor,
    r2_rows: torch.Tensor, centroids: torch.Tensor, heights: torch.Tensor, *, k0: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`scattering_patch_integral_eval` (scattering AD).

    ``k0`` is the wavenumber as a 0-d scalar tensor so frequency gradients flow
    through the phase and prefactor; its host value is read once per apply at
    this seam. The Duffy-mapped quadrature nodes/weights are gathered
    from the shared cache and threaded to the companions as fixed inputs.
    """

    quad_a, quad_b, quad_w = _duffy_nodes(patch_tris.device)
    k0_value = _ad_frequency_value(k0)
    values = _ScatteringPatchIntegralEvalAdFunction.apply(
        valid,
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


# ---------------------------------------------------------------------------
# autograd_chain
#
# The chain companions share the ``_grad_or_none`` helper defined above; the
# two files this module merges carried byte-identical copies of it.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# table_build_ad
# ---------------------------------------------------------------------------
# Output field contracts of the two native companions.
_BACKWARD_FIELDS = (
    "grad_sigma_h",
    "grad_corr_x",
    "grad_corr_y",
    "grad_layer_thickness_m",
    "grad_layer_eps_r",
    "grad_layer_sigma_e",
    "grad_frequency",
)
_JVP_FIELDS = ("tangent_f_te", "tangent_f_tm")


def _check_table_tensor(name: str, tensor: torch.Tensor, ndim: int) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dim() != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    return tensor.contiguous()


def kirchhoff_table_build_backward(
    s_te: torch.Tensor, s_tm: torch.Tensor, a_te: torch.Tensor, a_tm: torch.Tensor,
    r_diff_te: torch.Tensor, r_diff_tm: torch.Tensor, cos_i: torch.Tensor, phi_i: torch.Tensor,
    cos_o: torch.Tensor, phi_o: torch.Tensor, layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor, layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, *,
    sigma_h: float, corr_x: float, corr_y: float, frequency_hz: float, grad_f_te: torch.Tensor,
    grad_f_tm: torch.Tensor, need_grad_rough: bool, need_grad_layers: bool,
    need_grad_frequency: bool,
) -> dict[str, torch.Tensor | None]:
    """Native table-build VJP facade (scattering AD)."""

    s_te = _check_table_tensor("s_te", s_te, 4)
    s_tm = _check_table_tensor("s_tm", s_tm, 4)
    grad_f_te = _check_table_tensor("grad_f_te", grad_f_te, 4)
    grad_f_tm = _check_table_tensor("grad_f_tm", grad_f_tm, 4)
    out = _required_native_op("kirchhoff_table_build_backward")(
        s_te,
        s_tm,
        a_te.contiguous(),
        a_tm.contiguous(),
        r_diff_te.contiguous(),
        r_diff_tm.contiguous(),
        cos_i.contiguous(),
        phi_i.contiguous(),
        cos_o.contiguous(),
        phi_o.contiguous(),
        layer_thickness_m.contiguous(),
        layer_eps_r.contiguous(),
        layer_sigma_e.contiguous(),
        layer_mu_r.contiguous(),
        float(sigma_h),
        float(corr_x),
        float(corr_y),
        float(frequency_hz),
        grad_f_te,
        grad_f_tm,
        bool(need_grad_rough),
        bool(need_grad_layers),
        bool(need_grad_frequency),
    )
    if not isinstance(out, dict) or set(out) != set(_BACKWARD_FIELDS):
        raise TypeError(
            "_channel.kirchhoff_table_build_backward returned invalid fields"
        )
    return out


def kirchhoff_table_build_jvp(
    s_te: torch.Tensor, s_tm: torch.Tensor, a_te: torch.Tensor, a_tm: torch.Tensor,
    r_diff_te: torch.Tensor, r_diff_tm: torch.Tensor, cos_i: torch.Tensor, phi_i: torch.Tensor,
    cos_o: torch.Tensor, phi_o: torch.Tensor, layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor, layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, *,
    sigma_h: float, corr_x: float, corr_y: float, frequency_hz: float,
    t_layer_thickness_m: torch.Tensor | None, t_layer_eps_r: torch.Tensor | None,
    t_layer_sigma_e: torch.Tensor | None, t_sigma_h: float, t_corr_x: float, t_corr_y: float,
    t_frequency: float,
) -> dict[str, torch.Tensor]:
    """Native table-build JVP facade (scattering AD)."""

    s_te = _check_table_tensor("s_te", s_te, 4)
    s_tm = _check_table_tensor("s_tm", s_tm, 4)
    out = _required_native_op("kirchhoff_table_build_jvp")(
        s_te,
        s_tm,
        a_te.contiguous(),
        a_tm.contiguous(),
        r_diff_te.contiguous(),
        r_diff_tm.contiguous(),
        cos_i.contiguous(),
        phi_i.contiguous(),
        cos_o.contiguous(),
        phi_o.contiguous(),
        layer_thickness_m.contiguous(),
        layer_eps_r.contiguous(),
        layer_sigma_e.contiguous(),
        layer_mu_r.contiguous(),
        float(sigma_h),
        float(corr_x),
        float(corr_y),
        float(frequency_hz),
        None if t_layer_thickness_m is None else t_layer_thickness_m.contiguous(),
        None if t_layer_eps_r is None else t_layer_eps_r.contiguous(),
        None if t_layer_sigma_e is None else t_layer_sigma_e.contiguous(),
        float(t_sigma_h),
        float(t_corr_x),
        float(t_corr_y),
        float(t_frequency),
    )
    if not isinstance(out, dict) or set(out) != set(_JVP_FIELDS):
        raise TypeError(
            "_channel.kirchhoff_table_build_jvp returned invalid fields"
        )
    return out


def _scalar_value(tensor: torch.Tensor) -> float:
    return float(_ad_native_tensor(tensor).reshape(()).detach())


def _scalar_tangent(tensor: torch.Tensor | None) -> float:
    value = _ad_native_tangent_or_none(tensor)
    if value is None:
        return 0.0
    return float(value.reshape(()).detach())


# Fixed inputs of the build op (scattering AD): grads/tangents here fail loudly
# instead of silently detaching.
_FIXED = (
    (6, "layer_mu_r"),
    (16, "cos_i"),
    (17, "phi_i"),
    (18, "cos_o"),
    (19, "phi_o"),
)


def _backward_need_flags(needed) -> tuple[bool, bool, bool]:
    """Resolve which native grad groups the VJP must compute."""

    need_rough = bool(needed[0] or needed[1] or needed[2])
    need_layers = bool(needed[3] or needed[4] or needed[5])
    need_frequency = bool(needed[7])
    return need_rough, need_layers, need_frequency


def _backward_is_noop(
    need_rough: bool, need_layers: bool, need_frequency: bool, grad_f_te: torch.Tensor | None,
    grad_f_tm: torch.Tensor | None,
) -> bool:
    """True when no differentiable input is requested or no upstream grad flows."""

    no_grad_requested = not (need_rough or need_layers or need_frequency)
    no_upstream = grad_f_te is None and grad_f_tm is None
    return no_grad_requested or no_upstream


def _backward_grad_tuple(out, ctx, needed, *, need_frequency: bool) -> tuple:
    """Pack the 21-slot input-gradient tuple from the native VJP output."""

    grad_sigma_h = (
        out["grad_sigma_h"].reshape(ctx.rough_shapes[0]) if needed[0] else None
    )
    grad_corr_x = (
        out["grad_corr_x"].reshape(ctx.rough_shapes[1]) if needed[1] else None
    )
    grad_corr_y = (
        out["grad_corr_y"].reshape(ctx.rough_shapes[2]) if needed[2] else None
    )
    grad_thickness = out["grad_layer_thickness_m"] if needed[3] else None
    grad_eps = out["grad_layer_eps_r"] if needed[4] else None
    grad_sigma = out["grad_layer_sigma_e"] if needed[5] else None
    grad_frequency = (
        _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
        if need_frequency
        else None
    )
    return (
        grad_sigma_h,
        grad_corr_x,
        grad_corr_y,
        grad_thickness,
        grad_eps,
        grad_sigma,
        None,  # layer_mu_r
        grad_frequency,
        None,  # f_te_built
        None,  # f_tm_built
        None,  # s_te
        None,  # s_tm
        None,  # a_te
        None,  # a_tm
        None,  # r_diff_te
        None,  # r_diff_tm
        None,  # cos_i
        None,  # phi_i
        None,  # cos_o
        None,  # phi_o
        None,  # frequency_value
    )


class _KirchhoffTableBuildAdFunction(torch.autograd.Function):
    """Differentiable Kirchhoff table build (scattering AD).

    Forward passes the numpy-built ``f_te``/``f_tm`` through unchanged;
    backward/jvp dispatch the native companions against the exported f32
    structural intermediates. Differentiable inputs: ``rough_sigma_h_m`` /
    ``rough_corr_x_m`` / ``rough_corr_y_m`` and the CSR ``layer_thickness_m`` /
    ``layer_eps_r`` / ``layer_sigma_e`` slices plus the frequency scalar tensor.
    ``layer_mu_r``, the directional grids and the balance/budget intermediates
    stay fixed; requesting a fixed gradient/tangent fails loudly.
    """

    @staticmethod
    def forward(
        rough_sigma_h, rough_corr_x, rough_corr_y, layer_thickness_m, layer_eps_r, layer_sigma_e,
        layer_mu_r, frequency, f_te_built, f_tm_built, s_te, s_tm, a_te, a_tm, r_diff_te, r_diff_tm,
        cos_i, phi_i, cos_o, phi_o, frequency_value,
    ):
        # The numpy build already produced f_te_built/f_tm_built; the forward is
        # a pure pass-through so the primal table stays bit-identical.
        return f_te_built, f_tm_built

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[7]
        ctx.sigma_h = _scalar_value(inputs[0])
        ctx.corr_x = _scalar_value(inputs[1])
        ctx.corr_y = _scalar_value(inputs[2])
        ctx.frequency_value = float(inputs[20])
        ctx.rough_shapes = (
            tuple(inputs[0].shape),
            tuple(inputs[1].shape),
            tuple(inputs[2].shape),
        )
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        # Physics tensors used by both backward and jvp (order matches the
        # facade signatures). Grids (16..19) and CSR params (3..6) included.
        saved = tuple(
            torch.autograd.forward_ad.unpack_dual(inputs[i]).primal
            for i in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 3, 4, 5, 6)
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_f_te, grad_f_tm):
        none_grads = (None,) * 21
        _ad_reject_fixed_inputs(
            "kirchhoff_table_build_ad", ctx.needs_input_grad, _FIXED
        )
        needed = ctx.needs_input_grad
        need_rough, need_layers, need_frequency = _backward_need_flags(needed)
        if _backward_is_noop(
            need_rough, need_layers, need_frequency, grad_f_te, grad_f_tm
        ):
            return none_grads
        saved = ctx.saved_tensors
        s_te = saved[0]
        if grad_f_te is None:
            grad_f_te = torch.zeros_like(s_te)
        if grad_f_tm is None:
            grad_f_tm = torch.zeros_like(s_te)
        out = kirchhoff_table_build_backward(
            *saved,
            sigma_h=ctx.sigma_h,
            corr_x=ctx.corr_x,
            corr_y=ctx.corr_y,
            frequency_hz=ctx.frequency_value,
            grad_f_te=grad_f_te,
            grad_f_tm=grad_f_tm,
            need_grad_rough=need_rough,
            need_grad_layers=need_layers,
            need_grad_frequency=need_frequency,
        )
        return _backward_grad_tuple(out, ctx, needed, need_frequency=need_frequency)

    @staticmethod
    def jvp(
        ctx, t_sigma_h, t_corr_x, t_corr_y, t_thickness, t_eps, t_sigma, t_mu_r, t_frequency,
        _t_f_te, _t_f_tm, _t_s_te, _t_s_tm, _t_a_te, _t_a_tm, _t_r_diff_te, _t_r_diff_tm, _t_cos_i,
        _t_phi_i, _t_cos_o, _t_phi_o, _t_frequency_value,
    ):
        _ad_reject_fixed_tangents(
            "kirchhoff_table_build_ad",
            (
                (t_mu_r, "layer_mu_r"),
                (_t_cos_i, "cos_i"),
                (_t_phi_i, "phi_i"),
                (_t_cos_o, "cos_o"),
                (_t_phi_o, "phi_o"),
            ),
        )
        saved = ctx.saved_tensors
        tangent_sigma_h = _scalar_tangent(t_sigma_h)
        tangent_corr_x = _scalar_tangent(t_corr_x)
        tangent_corr_y = _scalar_tangent(t_corr_y)
        tangent_frequency = _scalar_tangent(t_frequency)
        tangent_thickness = _ad_native_tangent_or_none(t_thickness)
        tangent_eps = _ad_native_tangent_or_none(t_eps)
        tangent_sigma = _ad_native_tangent_or_none(t_sigma)
        if (
            tangent_sigma_h == 0.0
            and tangent_corr_x == 0.0
            and tangent_corr_y == 0.0
            and tangent_frequency == 0.0
            and tangent_thickness is None
            and tangent_eps is None
            and tangent_sigma is None
        ):
            return None, None
        with disable_functorch():
            out = kirchhoff_table_build_jvp(
                *(_ad_native_tensor(value) for value in saved),
                sigma_h=ctx.sigma_h,
                corr_x=ctx.corr_x,
                corr_y=ctx.corr_y,
                frequency_hz=ctx.frequency_value,
                t_layer_thickness_m=tangent_thickness,
                t_layer_eps_r=tangent_eps,
                t_layer_sigma_e=tangent_sigma,
                t_sigma_h=tangent_sigma_h,
                t_corr_x=tangent_corr_x,
                t_corr_y=tangent_corr_y,
                t_frequency=tangent_frequency,
            )
        return out["tangent_f_te"], out["tangent_f_tm"]


def kirchhoff_table_build_ad(
    rough_sigma_h: torch.Tensor, rough_corr_x: torch.Tensor, rough_corr_y: torch.Tensor,
    layer_thickness_m: torch.Tensor, layer_eps_r: torch.Tensor, layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor, frequency: torch.Tensor, *, table,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attach the native build adjoint to a numpy-built ``KirchhoffTable``.

    ``table`` is the float64-numpy-built :class:`KirchhoffTable` (already carries
    the primal ``f_te``/``f_tm`` and the exported ``pre_balance_lobe_*`` /
    ``normalization_applied`` / ``r_diff_*`` intermediates). The build itself is
    not re-run: the returned ``f_te``/``f_tm`` are the primal values with a
    graph node connecting them to the differentiable leaf inputs.
    """

    if table.pre_balance_lobe_te is None or table.pre_balance_lobe_tm is None:
        raise ValueError(
            "kirchhoff_table_build_ad requires the AD-saved pre-balance lobes; "
            "build the table with witwin.channel.scene.resources.build_kirchhoff_table"
        )
    frequency_value = _ad_frequency_value(frequency)
    a_te = table.normalization_applied[..., 0].contiguous()
    a_tm = table.normalization_applied[..., 1].contiguous()
    f_te, f_tm = _KirchhoffTableBuildAdFunction.apply(
        rough_sigma_h,
        rough_corr_x,
        rough_corr_y,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        table.f_te,
        table.f_tm,
        table.pre_balance_lobe_te,
        table.pre_balance_lobe_tm,
        a_te,
        a_tm,
        table.r_diff_te,
        table.r_diff_tm,
        table.cos_theta_i,
        table.phi_i,
        table.cos_theta_o,
        table.phi_o,
        float(frequency_value),
    )
    return f_te, f_tm


def mc_scattering_pair_legs(
    samples: tuple[torch.Tensor, ...], tables: tuple[torch.Tensor, ...], receivers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Visibility legs ``(start, end, active)`` of every (receiver, sample) pair, receiver-major.

    ``samples`` = (points, normals, frame_t1, frame_t2, wi_local, incident, epsilon, material);
    ``tables`` = (material_slot, table_offset, table_dims, f_te_flat, f_tm_flat).
    """

    out = _required_native_op("mc_scattering_pair_legs")(*samples, *tables, receivers)
    if not isinstance(out, tuple) or len(out) != 3:
        raise TypeError("_channel.mc_scattering_pair_legs must return 3 tensors")
    return out[0], out[1], out[2]


def _mc_scattering_pair_sum(
    samples: tuple[torch.Tensor, ...], tables: tuple[torch.Tensor, ...], receivers: torch.Tensor,
    visible: torch.Tensor, value_te: torch.Tensor, value_tm: torch.Tensor, deposited: torch.Tensor,
) -> torch.Tensor:
    out = _required_native_op("mc_scattering_pair_sum")(
        *samples, *tables, receivers, visible, value_te, value_tm, deposited
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel.mc_scattering_pair_sum must return a tensor")
    return out


class _McScatteringPairSumFunction(torch.autograd.Function):
    """Per-receiver Kirchhoff deposit sum, differentiable in the flat table values.

    The samples, frames, receivers and visibility are frozen winners; the sum is linear in the
    table values, so the VJP spreads the cotangent over the interpolation corners and the JVP is
    the same sum over the tangent tables.
    """

    @staticmethod
    def forward(f_te, f_tm, samples, tables, receivers, visible, deposited):
        native_tables = (*tables[:3], f_te, f_tm)
        return _mc_scattering_pair_sum(
            samples, native_tables, receivers, visible, f_te, f_tm, deposited
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        f_te, f_tm, samples, tables, receivers, visible, _deposited = inputs
        ctx.samples = samples
        ctx.tables = (*tables[:3],)
        ctx.receivers = receivers
        ctx.visible = visible
        ctx.save_for_backward(f_te, f_tm)
        ctx.save_for_forward(f_te, f_tm)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_out):
        if grad_out is None:
            return (None,) * 7
        f_te, f_tm = ctx.saved_tensors
        out = _required_native_op("mc_scattering_pair_sum_backward")(
            *ctx.samples,
            *ctx.tables,
            _ad_native_tensor(f_te).detach(),
            _ad_native_tensor(f_tm).detach(),
            ctx.receivers,
            ctx.visible,
            _ad_native_tensor(grad_out).contiguous(),
        )
        return out[0], out[1], None, None, None, None, None

    @staticmethod
    def jvp(ctx, t_te, t_tm, *_rest):
        tangent_te = _ad_native_tangent_or_none(t_te)
        tangent_tm = _ad_native_tangent_or_none(t_tm)
        if tangent_te is None and tangent_tm is None:
            return None
        f_te, f_tm = ctx.saved_tensors
        zero = torch.zeros_like(_ad_native_tensor(f_te))
        with disable_functorch():
            return _mc_scattering_pair_sum(
                ctx.samples,
                (*ctx.tables, _ad_native_tensor(f_te), _ad_native_tensor(f_tm)),
                ctx.receivers,
                ctx.visible,
                tangent_te.contiguous() if tangent_te is not None else zero,
                tangent_tm.contiguous() if tangent_tm is not None else zero,
                zero.new_empty((0,), dtype=torch.int64),
            )


def mc_scattering_pair_sum(
    samples: tuple[torch.Tensor, ...], tables: tuple[torch.Tensor, ...], receivers: torch.Tensor,
    visible: torch.Tensor, *, deposited: torch.Tensor, ad: bool = False,
) -> torch.Tensor:
    """Per-receiver sum of ``0.5 (f_te + f_tm) cos_o cos_i / (r1^2 r2^2)`` over visible pairs.

    Each receiver sums its samples in a fixed order (reproducible). ``deposited`` (int64, one
    element) counts the positive deposits. Under ``ad`` the flat table values keep their graph.
    """

    if ad:
        return _McScatteringPairSumFunction.apply(  # type: ignore[no-any-return]
            tables[3], tables[4], samples, tables, receivers, visible, deposited
        )
    return _mc_scattering_pair_sum(
        samples, tables, receivers, visible, tables[3].detach(), tables[4].detach(), deposited
    )
