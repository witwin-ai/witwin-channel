from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor

_PATCH_QUAD_ORDER = 16
_duffy_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _duffy_nodes(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Duffy-mapped 16x16 Gauss-Legendre nodes (float64 leggauss -> float32).

    The unit square ``(xi, eta)`` maps to barycentric ``(a, b) =
    (xi, eta * (1 - xi))`` with Jacobian ``(1 - xi)`` - the same construction
    as ``scattering.phase_screen.patch_phase_integral``.
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


def scattering_table_eval(
    wi: torch.Tensor,
    wo: torch.Tensor,
    f_te: torch.Tensor,
    f_tm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native CUDA multilinear Kirchhoff-table evaluation; required op."""

    _validate_table_eval_inputs(wi, wo, f_te, f_tm)
    out = _required_native_op("scattering_table_eval")(wi, wo, f_te, f_tm)
    if not isinstance(out, dict) or set(out) != {"f_te", "f_tm"}:
        raise TypeError("_channel_native.scattering_table_eval returned invalid fields")
    return out["f_te"], out["f_tm"]


_TABLE_EVAL_BACKWARD_FIELDS = ("grad_wi", "grad_wo", "grad_f_te", "grad_f_tm")
_TABLE_EVAL_JVP_FIELDS = ("tangent_f_te", "tangent_f_tm")


def _validate_table_eval_inputs(
    wi: torch.Tensor,
    wo: torch.Tensor,
    f_te: torch.Tensor,
    f_tm: torch.Tensor,
) -> None:
    """Validate the shared table-lookup primal inputs in ABI order."""

    validate_cuda_tensor("wi", wi, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("wo", wo, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("f_te", f_te, dtype=torch.float32, ndim=4)
    validate_cuda_tensor("f_tm", f_tm, dtype=torch.float32, ndim=4)
    if wi.shape != wo.shape or wi.shape[1:] != (3,):
        raise ValueError("wi and wo must have matching shape (N, 3)")


def scattering_table_eval_backward(
    wi: torch.Tensor,
    wo: torch.Tensor,
    f_te: torch.Tensor,
    f_tm: torch.Tensor,
    *,
    grad_out_f_te: torch.Tensor | None = None,
    grad_out_f_tm: torch.Tensor | None = None,
    need_grad_dirs: bool = False,
    need_grad_tables: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`scattering_table_eval` (ADR-015 op 1).

    ``grad_wi``/``grad_wo`` are ``[N, 3]`` direct stores (``need_grad_dirs``);
    ``grad_f_te``/``grad_f_tm`` are the table-shaped 16-corner atomicAdd scatter
    (``need_grad_tables``). Entries are ``None`` when their owning flag is off.
    """

    _validate_table_eval_inputs(wi, wo, f_te, f_tm)
    out = _required_native_op("scattering_table_eval_backward")(
        wi,
        wo,
        f_te,
        f_tm,
        grad_out_f_te,
        grad_out_f_tm,
        bool(need_grad_dirs),
        bool(need_grad_tables),
    )
    if not isinstance(out, dict) or set(out) != set(_TABLE_EVAL_BACKWARD_FIELDS):
        raise TypeError(
            "_channel_native.scattering_table_eval_backward returned invalid fields"
        )
    return out


def scattering_table_eval_jvp(
    wi: torch.Tensor,
    wo: torch.Tensor,
    f_te: torch.Tensor,
    f_tm: torch.Tensor,
    *,
    tangent_wi: torch.Tensor | None = None,
    tangent_wo: torch.Tensor | None = None,
    tangent_f_te: torch.Tensor | None = None,
    tangent_f_tm: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`scattering_table_eval` (ADR-015 op 1).

    Elementwise per-row tangents ``tangent_f_te``/``tangent_f_tm`` ``[N]`` from
    the live tangents of ``wi``, ``wo`` and the two tables; a missing tangent is
    a zero tangent.
    """

    _validate_table_eval_inputs(wi, wo, f_te, f_tm)
    out = _required_native_op("scattering_table_eval_jvp")(
        wi,
        wo,
        f_te,
        f_tm,
        tangent_wi,
        tangent_wo,
        tangent_f_te,
        tangent_f_tm,
    )
    if not isinstance(out, dict) or set(out) != set(_TABLE_EVAL_JVP_FIELDS):
        raise TypeError(
            "_channel_native.scattering_table_eval_jvp returned invalid fields"
        )
    return out


def scattering_table_pdf(
    wi: torch.Tensor,
    wo: torch.Tensor,
    sample_density: torch.Tensor,
    *,
    reverse: bool = False,
) -> torch.Tensor:
    """Native CUDA piecewise-constant Kirchhoff PDF; required op."""

    validate_cuda_tensor("wi", wi, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("wo", wo, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("sample_density", sample_density, dtype=torch.float32, ndim=4)
    return _required_native_op("scattering_table_pdf")(
        wi, wo, sample_density, bool(reverse)
    )


def scattering_table_sample(
    wi: torch.Tensor,
    uniforms: torch.Tensor,
    marginal_cdf: torch.Tensor,
    conditional_cdf: torch.Tensor,
    sample_density: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Native CUDA CDF inversion plus forward/reverse PDFs; required op."""

    validate_cuda_tensor("wi", wi, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("uniforms", uniforms, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("marginal_cdf", marginal_cdf, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("conditional_cdf", conditional_cdf, dtype=torch.float32, ndim=4)
    validate_cuda_tensor("sample_density", sample_density, dtype=torch.float32, ndim=4)
    out = _required_native_op("scattering_table_sample")(
        wi, uniforms, marginal_cdf, conditional_cdf, sample_density
    )
    expected = {"wo", "pdf_forward", "pdf_reverse"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError("_channel_native.scattering_table_sample returned invalid fields")
    return out


def scattering_event_probabilities(
    cos_theta: torch.Tensor,
    material_id: torch.Tensor,
    cap_r_te: torch.Tensor,
    cap_r_tm: torch.Tensor,
    cap_t_te: torch.Tensor,
    cap_t_tm: torch.Tensor,
    rough_sigma_h_m: torch.Tensor,
    scatter_model_id: torch.Tensor,
    *,
    frequency_hz: float,
    probability_floor: float,
) -> dict[str, torch.Tensor]:
    """Fused native CUDA rough-event budgets and probabilities."""

    validate_cuda_tensor("cos_theta", cos_theta, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_id", material_id, dtype=torch.int32, ndim=1)
    out = _required_native_op("scattering_event_probabilities")(
        cos_theta,
        material_id,
        cap_r_te,
        cap_r_tm,
        cap_t_te,
        cap_t_tm,
        rough_sigma_h_m,
        scatter_model_id,
        float(frequency_hz),
        float(probability_floor),
    )
    expected = {"p_scatter", "p_transmit", "r_coh_amplitude", "rough"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError("_channel_native.scattering_event_probabilities returned invalid fields")
    return out


def _validate_ensemble_inputs(scope: Mapping[str, torch.Tensor]) -> None:
    """Validate the shared ensemble primal inputs in ABI order."""

    validate_cuda_tensor("wo_rows", scope["wo_rows"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("r2_rows", scope["r2_rows"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("cos_o_rows", scope["cos_o_rows"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("wi_local", scope["wi_local"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("rc_idx", scope["rc_idx"], dtype=torch.int64, ndim=1)
    validate_cuda_tensor("sc_idx", scope["sc_idx"], dtype=torch.int64, ndim=1)
    validate_cuda_tensor("material_id", scope["material_id"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("material_slot", scope["material_slot"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("table_dims", scope["table_dims"], dtype=torch.int32, ndim=2)


def scattering_ensemble_eval(
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
    coef: float,
    threshold: float,
) -> dict[str, torch.Tensor]:
    """Native Kirchhoff ensemble scattering row physics (ADR-010 op 1).

    ``wo_rows`` / ``r2_rows`` / ``cos_o_rows`` are the surviving rows gathered
    from the Torch candidate grid (which stays Torch per the ADR). Per row the
    kernel owns wo_local, the stacked-table lookup, the outgoing s/p basis and
    receiver projections, and the radiometric gain/keep/amplitude/length.
    One launch per (tx, rx-chunk); required native op.
    """

    _validate_ensemble_inputs(locals())
    out = _required_native_op("scattering_ensemble_eval")(
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
            "_channel_native.scattering_ensemble_eval returned invalid fields"
        )
    return out


def _validate_patch_inputs(scope: Mapping[str, torch.Tensor]) -> None:
    """Validate the shared patch-integral primal inputs in ABI order."""

    validate_cuda_tensor("patch_tris", scope["patch_tris"], dtype=torch.float32, ndim=3)
    validate_cuda_tensor("patch_uvs", scope["patch_uvs"], dtype=torch.float32, ndim=3)
    validate_cuda_tensor("rows", scope["rows"], dtype=torch.int64, ndim=1)
    validate_cuda_tensor("d_i", scope["d_i"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("d_o", scope["d_o"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("r_te", scope["r_te"], dtype=torch.complex64, ndim=1)
    validate_cuda_tensor("r_tm", scope["r_tm"], dtype=torch.complex64, ndim=1)
    validate_cuda_tensor("heights", scope["heights"], dtype=torch.float32, ndim=2)


def scattering_patch_integral_eval(
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
    k0: float,
) -> dict[str, torch.Tensor]:
    """Native realization-coherent phase-screen patch integral (ADR-010 op 2).

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
            "_channel_native.scattering_patch_integral_eval returned invalid fields"
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
    coef: float,
    threshold: float,
    grad_gain: torch.Tensor | None = None,
    grad_amplitude: torch.Tensor | None = None,
    grad_length: torch.Tensor | None = None,
    need_grad_rows: bool = False,
    need_grad_samples: bool = False,
    need_grad_tables: bool = False,
    need_grad_coef: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`scattering_ensemble_eval` (ADR-014 op 1)."""

    _validate_ensemble_inputs(locals())
    out = _required_native_op("scattering_ensemble_eval_backward")(
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
            "_channel_native.scattering_ensemble_eval_backward returned invalid fields"
        )
    return out


def scattering_ensemble_eval_jvp(
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
    coef: float,
    threshold: float,
    tangent_wo_rows: torch.Tensor | None = None,
    tangent_r2_rows: torch.Tensor | None = None,
    tangent_cos_o_rows: torch.Tensor | None = None,
    tangent_n_o: torch.Tensor | None = None,
    tangent_t1r: torch.Tensor | None = None,
    tangent_t2r: torch.Tensor | None = None,
    tangent_wi_local: torch.Tensor | None = None,
    tangent_cos_i: torch.Tensor | None = None,
    tangent_r1: torch.Tensor | None = None,
    tangent_a_te2: torch.Tensor | None = None,
    tangent_a_tm2: torch.Tensor | None = None,
    tangent_weights: torch.Tensor | None = None,
    tangent_f_te_flat: torch.Tensor | None = None,
    tangent_f_tm_flat: torch.Tensor | None = None,
    tangent_coef: float = 0.0,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`scattering_ensemble_eval` (ADR-014 op 1)."""

    _validate_ensemble_inputs(locals())
    out = _required_native_op("scattering_ensemble_eval_jvp")(
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
            "_channel_native.scattering_ensemble_eval_jvp returned invalid fields"
        )
    return out


def scattering_patch_integral_eval_backward(
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
    k0: float,
    grad_total: torch.Tensor,
    need_grad_heights: bool = False,
    need_grad_jones: bool = False,
    need_grad_geometry: bool = False,
    need_grad_k0: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`scattering_patch_integral_eval` (ADR-014 op 2)."""

    _validate_patch_inputs(locals())
    quad_a, quad_b, quad_w = _duffy_nodes(patch_tris.device)
    out = _required_native_op("scattering_patch_integral_eval_backward")(
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
            "_channel_native.scattering_patch_integral_eval_backward returned"
            " invalid fields"
        )
    return out


def scattering_patch_integral_eval_jvp(
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
    k0: float,
    tangent_heights: torch.Tensor | None = None,
    tangent_r_te: torch.Tensor | None = None,
    tangent_r_tm: torch.Tensor | None = None,
    tangent_d_i: torch.Tensor | None = None,
    tangent_d_o: torch.Tensor | None = None,
    tangent_r1_rows: torch.Tensor | None = None,
    tangent_r2_rows: torch.Tensor | None = None,
    tangent_centroids: torch.Tensor | None = None,
    tangent_k0: float = 0.0,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`scattering_patch_integral_eval` (ADR-014 op 2)."""

    _validate_patch_inputs(locals())
    quad_a, quad_b, quad_w = _duffy_nodes(patch_tris.device)
    out = _required_native_op("scattering_patch_integral_eval_jvp")(
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
            "_channel_native.scattering_patch_integral_eval_jvp returned invalid fields"
        )
    return out


__all__ = [
    "scattering_ensemble_eval",
    "scattering_ensemble_eval_backward",
    "scattering_ensemble_eval_jvp",
    "scattering_event_probabilities",
    "scattering_patch_integral_eval",
    "scattering_patch_integral_eval_backward",
    "scattering_patch_integral_eval_jvp",
    "scattering_table_eval",
    "scattering_table_eval_backward",
    "scattering_table_eval_jvp",
    "scattering_table_pdf",
    "scattering_table_sample",
]
