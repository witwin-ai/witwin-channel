# Copyright Xingyu Chen.
# Native scattering kernel facades.

"""Native scattering kernel facades."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict, cast

import numpy as np
import torch

from witwin.channel.kernels.spec import (
    AdInputs,
    Arg,
    NativeOps,
    flag,
    frequency_meta,
    real,
    save,
    tensor,
    opaque,
)
from witwin.channel.runtime import (
    _ad_first_order_only,
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    disable_functorch,
    required_symbol as _required_native_op,
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


_native = NativeOps(lambda symbol: _required_native_op(symbol))
_F32 = torch.float32

_TABLE_PDF = _native.op(
    "scattering_table_pdf", tensor("valid", torch.bool, 1), tensor("wi", _F32, 2),
    tensor("wo", _F32, 2), tensor("sample_density", _F32, 4), flag("reverse"),
)

# Shared ensemble primal inputs in ABI order with their facade checks and AD roles.
_ENSEMBLE_PRIMALS = (
    tensor("valid", torch.bool, 1, grad="fixed"),
    tensor("wo_rows", _F32, 2, grad="diff"),
    tensor("r2_rows", _F32, 1, grad="diff"),
    tensor("cos_o_rows", _F32, 1, grad="diff"),
    tensor("n_o", grad="diff"),
    tensor("t1r", grad="diff"),
    tensor("t2r", grad="diff"),
    tensor("wi_local", _F32, 2, grad="diff"),
    tensor("cos_i", grad="diff"),
    tensor("r1", grad="diff"),
    tensor("a_te2", grad="diff"),
    tensor("a_tm2", grad="diff"),
    tensor("weights", grad="diff"),
    tensor("material_id", torch.int32, 1, grad="fixed"),
    tensor("backup_axis", grad="fixed"),
    tensor("rx_pol", grad="fixed"),
    tensor("rc_idx", torch.int64, 1, grad="fixed"),
    tensor("sc_idx", torch.int64, 1, grad="fixed"),
    tensor("f_te_flat", grad="diff", abi="fte_flat"),
    tensor("f_tm_flat", grad="diff", abi="ftm_flat"),
    tensor("table_offset", grad="fixed"),
    tensor("table_dims", torch.int32, 2, grad="fixed"),
    tensor("material_slot", torch.int32, 1, grad="fixed"),
)
_ENSEMBLE_SCALARS = (real("coef"), real("threshold"))
# The order the ensemble facades check their inputs in (error precedence).
_ENSEMBLE_CHECKS = (
    "valid", "wo_rows", "r2_rows", "cos_o_rows", "wi_local", "rc_idx", "sc_idx", "material_id",
    "material_slot", "table_dims",
)
_ENSEMBLE_EVAL = _native.op("scattering_ensemble_eval", _ENSEMBLE_PRIMALS, _ENSEMBLE_SCALARS)
_ENSEMBLE_EVAL_BACKWARD = _native.op(
    "scattering_ensemble_eval_backward", _ENSEMBLE_PRIMALS, _ENSEMBLE_SCALARS,
    opaque("grad_gain"), opaque("grad_amplitude"), opaque("grad_length"), flag("need_grad_rows"),
    flag("need_grad_samples"), flag("need_grad_tables"), flag("need_grad_coef"),
)
_ENSEMBLE_EVAL_JVP = _native.op(
    "scattering_ensemble_eval_jvp", _ENSEMBLE_PRIMALS, _ENSEMBLE_SCALARS,
    *(
        opaque(f"t_{name}")
        for name in (
            "wo_rows", "r2_rows", "cos_o_rows", "n_o", "t1r", "t2r", "wi_local", "cos_i", "r1",
            "a_te2", "a_tm2", "weights", "fte_flat", "ftm_flat",
        )
    ),
    real("tangent_coef"),
)

# Shared patch-integral primal inputs in ABI order (the quadrature nodes follow ``heights``).
_PATCH_PRIMALS = (
    tensor("valid", torch.bool, 1, grad="fixed"),
    tensor("patch_tris", _F32, 3, grad="fixed"),
    tensor("patch_uvs", _F32, 3, grad="fixed"),
    tensor("rows", torch.int64, 1, grad="fixed"),
    tensor("d_i", _F32, 2, grad="diff"),
    tensor("d_o", _F32, 2, grad="diff"),
    tensor("n_rows", grad="fixed"),
    tensor("r_te", torch.complex64, 1, grad="diff"),
    tensor("r_tm", torch.complex64, 1, grad="diff"),
    tensor("pol_t", grad="fixed"),
    tensor("pol_r", grad="fixed"),
    tensor("r1_rows", grad="diff"),
    tensor("r2_rows", grad="diff"),
    tensor("centroids", grad="diff"),
    tensor("heights", _F32, 2, grad="diff"),
    tensor("quad_a", grad="fixed"),
    tensor("quad_b", grad="fixed"),
    tensor("quad_w", grad="fixed"),
)
_PATCH_CHECKS = (
    "valid", "patch_tris", "patch_uvs", "rows", "d_i", "d_o", "r_te", "r_tm", "heights",
)
_PATCH_EVAL = _native.op("scattering_patch_integral_eval", _PATCH_PRIMALS, real("k0"))
_PATCH_EVAL_BACKWARD = _native.op(
    "scattering_patch_integral_eval_backward", _PATCH_PRIMALS, real("k0"), tensor("grad_total"),
    flag("need_grad_heights"), flag("need_grad_jones"), flag("need_grad_geometry"),
    flag("need_grad_k0"),
)
_PATCH_EVAL_JVP = _native.op(
    "scattering_patch_integral_eval_jvp", _PATCH_PRIMALS, real("k0"),
    *(
        opaque(f"t_{name}")
        for name in ("heights", "r_te", "r_tm", "d_i", "d_o", "r1_rows", "r2_rows", "centroids")
    ),
    real("tangent_k0"),
)


def scattering_table_pdf(
    valid: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor, sample_density: torch.Tensor, *,
    reverse: bool = False,
) -> torch.Tensor:
    """Native CUDA piecewise-constant Kirchhoff PDF; required op."""

    values = (valid, wi, wo, sample_density, reverse)
    _TABLE_PDF.check(values, "valid", "wi", "wo", "sample_density")
    if valid.shape != wi.shape[:1] or wo.shape != wi.shape:
        raise ValueError("valid, wi, and wo rows must match")
    return cast(torch.Tensor, _TABLE_PDF(*values))


def _validate_ensemble_inputs(primals: tuple[torch.Tensor, ...]) -> None:
    """Validate the shared ensemble primal inputs."""

    _ENSEMBLE_EVAL.check(primals, *_ENSEMBLE_CHECKS)
    if primals[0].shape != primals[1].shape[:1]:
        raise ValueError("valid must match ensemble rows")


def _require_fields(out: object, expected: set[str] | tuple[str, ...], symbol: str) -> Any:
    if not isinstance(out, dict) or set(out) != set(expected):
        raise TypeError(f"_channel.{symbol} returned invalid fields")
    return out


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

    primals = (
        valid, wo_rows, r2_rows, cos_o_rows, n_o, t1r, t2r, wi_local, cos_i, r1, a_te2, a_tm2,
        weights, material_id, backup_axis, rx_pol, rc_idx, sc_idx, f_te_flat, f_tm_flat,
        table_offset, table_dims, material_slot,
    )
    _validate_ensemble_inputs(primals)
    out = _ENSEMBLE_EVAL(*primals, coef, threshold)
    return cast(
        dict[str, torch.Tensor],
        _require_fields(out, _ENSEMBLE_OUTPUT_FIELDS, _ENSEMBLE_EVAL.symbol),
    )


def _validate_patch_inputs(primals: tuple[torch.Tensor, ...]) -> None:
    """Validate the shared patch-integral primal inputs (``primals`` stops at ``heights``)."""

    _PATCH_EVAL.check(primals, *_PATCH_CHECKS)
    if primals[0].shape != primals[3].shape:
        raise ValueError("valid must match patch rows")


class PatchIntegralOutputs(TypedDict):
    """Phase-screen patch integral: per-row ``integral``/``row_value`` and their ``total``."""

    total: torch.Tensor
    integral: torch.Tensor
    row_value: torch.Tensor


def scattering_patch_integral_eval(
    valid: torch.Tensor, patch_tris: torch.Tensor, patch_uvs: torch.Tensor, rows: torch.Tensor,
    d_i: torch.Tensor, d_o: torch.Tensor, n_rows: torch.Tensor, r_te: torch.Tensor,
    r_tm: torch.Tensor, pol_t: torch.Tensor, pol_r: torch.Tensor, r1_rows: torch.Tensor,
    r2_rows: torch.Tensor, centroids: torch.Tensor, heights: torch.Tensor, *, k0: float,
) -> PatchIntegralOutputs:
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

    primals = (
        valid, patch_tris, patch_uvs, rows, d_i, d_o, n_rows, r_te, r_tm, pol_t, pol_r, r1_rows,
        r2_rows, centroids, heights,
    )
    _validate_patch_inputs(primals)
    out = _PATCH_EVAL(*primals, *_duffy_nodes(patch_tris.device), k0)
    return cast(
        PatchIntegralOutputs, _require_fields(out, _PATCH_OUTPUT_FIELDS, _PATCH_EVAL.symbol)
    )


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

    primals = (
        valid, wo_rows, r2_rows, cos_o_rows, n_o, t1r, t2r, wi_local, cos_i, r1, a_te2, a_tm2,
        weights, material_id, backup_axis, rx_pol, rc_idx, sc_idx, f_te_flat, f_tm_flat,
        table_offset, table_dims, material_slot,
    )
    _validate_ensemble_inputs(primals)
    out = _ENSEMBLE_EVAL_BACKWARD(
        *primals, coef, threshold, grad_gain, grad_amplitude, grad_length, need_grad_rows,
        need_grad_samples, need_grad_tables, need_grad_coef,
    )
    return cast(
        dict[str, torch.Tensor | None],
        _require_fields(out, _ENSEMBLE_BACKWARD_FIELDS, _ENSEMBLE_EVAL_BACKWARD.symbol),
    )


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

    primals = (
        valid, wo_rows, r2_rows, cos_o_rows, n_o, t1r, t2r, wi_local, cos_i, r1, a_te2, a_tm2,
        weights, material_id, backup_axis, rx_pol, rc_idx, sc_idx, f_te_flat, f_tm_flat,
        table_offset, table_dims, material_slot,
    )
    _validate_ensemble_inputs(primals)
    out = _ENSEMBLE_EVAL_JVP(
        *primals, coef, threshold, tangent_wo_rows, tangent_r2_rows, tangent_cos_o_rows,
        tangent_n_o, tangent_t1r, tangent_t2r, tangent_wi_local, tangent_cos_i, tangent_r1,
        tangent_a_te2, tangent_a_tm2, tangent_weights, tangent_f_te_flat, tangent_f_tm_flat,
        tangent_coef,
    )
    return cast(
        dict[str, torch.Tensor],
        _require_fields(out, _ENSEMBLE_TANGENT_FIELDS, _ENSEMBLE_EVAL_JVP.symbol),
    )


def scattering_patch_integral_eval_backward(
    valid: torch.Tensor, patch_tris: torch.Tensor, patch_uvs: torch.Tensor, rows: torch.Tensor,
    d_i: torch.Tensor, d_o: torch.Tensor, n_rows: torch.Tensor, r_te: torch.Tensor,
    r_tm: torch.Tensor, pol_t: torch.Tensor, pol_r: torch.Tensor, r1_rows: torch.Tensor,
    r2_rows: torch.Tensor, centroids: torch.Tensor, heights: torch.Tensor, *, k0: float,
    grad_total: torch.Tensor, need_grad_heights: bool = False, need_grad_jones: bool = False,
    need_grad_geometry: bool = False, need_grad_k0: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`scattering_patch_integral_eval` (scattering AD)."""

    primals = (
        valid, patch_tris, patch_uvs, rows, d_i, d_o, n_rows, r_te, r_tm, pol_t, pol_r, r1_rows,
        r2_rows, centroids, heights,
    )
    _validate_patch_inputs(primals)
    out = _PATCH_EVAL_BACKWARD(
        *primals, *_duffy_nodes(patch_tris.device), k0, grad_total, need_grad_heights,
        need_grad_jones, need_grad_geometry, need_grad_k0,
    )
    return cast(
        dict[str, torch.Tensor | None],
        _require_fields(out, _PATCH_BACKWARD_FIELDS, _PATCH_EVAL_BACKWARD.symbol),
    )


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

    primals = (
        valid, patch_tris, patch_uvs, rows, d_i, d_o, n_rows, r_te, r_tm, pol_t, pol_r, r1_rows,
        r2_rows, centroids, heights,
    )
    _validate_patch_inputs(primals)
    out = _PATCH_EVAL_JVP(
        *primals, *_duffy_nodes(patch_tris.device), k0, tangent_heights, tangent_r_te,
        tangent_r_tm, tangent_d_i, tangent_d_o, tangent_r1_rows, tangent_r2_rows,
        tangent_centroids, tangent_k0,
    )
    return cast(
        dict[str, torch.Tensor], _require_fields(out, ("tangent_total",), _PATCH_EVAL_JVP.symbol)
    )


# ---------------------------------------------------------------------------
# functional_chain
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Coherent scattering-chain kernels.
#
# ``scattering_chain_ensemble_eval`` generalizes op 1 to a specular reflection
# chain C1 -> diffuse vertex v_s -> chain C2 in the power domain. It carries two
# padded per-leg blocks ``[R, Dmax, ...]`` with ``Dmax = kMaxAdDepth = 8``. The
# facade is thin: validate the spec tables, dispatch the required native symbol,
# and assert the returned key-set exactly.
# ---------------------------------------------------------------------------

_K_MAX_AD_DEPTH = 8

_CHAIN_ENSEMBLE_OUTPUT_FIELDS = ("gain", "amplitude", "length", "keep")


def _vec3(name: str) -> Arg:
    return tensor(name, _F32, 2, (3,))



def _chain_leg(prefix: str) -> tuple[Arg, ...]:
    """One padded specular leg block: ``[R, Dmax, 3]`` geometry, ``[R, Dmax]`` Fresnel inputs."""

    return (
        tensor(f"{prefix}_positions", _F32, 3, (3,)),
        tensor(f"{prefix}_normals", _F32, 3, (3,)),
        *(
            tensor(f"{prefix}_{name}", _F32, 2)
            for name in ("eps_r", "sigma_e", "mu_r", "gain", "thickness")
        ),
        tensor(f"{prefix}_depth", torch.int32, 1),
    )


_CHAIN_ENSEMBLE_EVAL = _native.op(
    "scattering_chain_ensemble_eval",
    tensor("valid", torch.bool, 1),
    *(_vec3(name) for name in ("tx_pol", "rx_pol", "source", "vertex", "target")),
    _chain_leg("c1"),
    _chain_leg("c2"),
    *(_vec3(name) for name in ("n_o", "t1r", "t2r", "backup_axis", "wi_local")),
    tensor("cos_i", _F32, 1),
    tensor("cos_o", _F32, 1),
    _vec3("d_i"),
    _vec3("d_o"),
    *(tensor(name, _F32, 1) for name in ("l1", "l2", "weights")),
    tensor("material_id", torch.int32, 1),
    tensor("f_te_flat", _F32, 1, abi="fte_flat"),
    tensor("f_tm_flat", _F32, 1, abi="ftm_flat"),
    tensor("table_offset", torch.int64, 1),
    tensor("table_dims", torch.int32, 2, (4,)),
    tensor("material_slot", torch.int32, 1),
    real("coef"),
    real("threshold"),
    real("frequency_hz"),
)


def _validate_chain_leg(prefix: str, named: dict[str, torch.Tensor], rows: int) -> int:
    """Validate one padded specular leg block and return its ``Dmax`` width.

    The padded block is ``[R, Dmax, 3]`` positions/normals and ``[R, Dmax]``
    per-bounce Fresnel inputs with an ``[R]`` int32 per-row depth. ``Dmax`` is
    the static padded width and must not exceed ``kMaxAdDepth = 8``; the per-row ``depth`` values are trusted structural winners and
    are not read on the host (that would force a device sync).
    """

    positions, normals = named[f"{prefix}_positions"], named[f"{prefix}_normals"]
    _CHAIN_ENSEMBLE_EVAL.checks(named, f"{prefix}_positions", f"{prefix}_normals")
    dmax: int = positions.shape[1]
    if dmax > _K_MAX_AD_DEPTH:
        raise ValueError(
            f"{prefix} leg depth {dmax} exceeds kMaxAdDepth={_K_MAX_AD_DEPTH}"
        )
    if tuple(positions.shape) != (rows, dmax, 3):
        raise ValueError(f"{prefix}_positions must have shape ({rows}, {dmax}, 3)")
    if tuple(normals.shape) != (rows, dmax, 3):
        raise ValueError(f"{prefix}_normals must have shape ({rows}, {dmax}, 3)")
    fields = ("eps_r", "sigma_e", "mu_r", "gain", "thickness")
    for name in (f"{prefix}_{field}" for field in fields):
        _CHAIN_ENSEMBLE_EVAL.checks(named, name)
        if tuple(named[name].shape) != (rows, dmax):
            raise ValueError(f"{name} must have shape ({rows}, {dmax})")
    _CHAIN_ENSEMBLE_EVAL.checks(named, f"{prefix}_depth")
    if named[f"{prefix}_depth"].shape[0] != rows:
        raise ValueError(f"{prefix}_depth must have {rows} rows")
    return dmax


def _require_same_device(named: tuple[tuple[str, torch.Tensor], ...]) -> None:
    """Assert every named tensor shares one CUDA device."""

    device = None
    for name, tensor_value in named:
        if tensor_value is None:
            continue
        if device is None:
            device = tensor_value.device
        elif tensor_value.device != device:
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

    spec = _CHAIN_ENSEMBLE_EVAL
    primals = (
        valid, tx_pol, rx_pol, source, vertex, target, c1_positions, c1_normals, c1_eps_r,
        c1_sigma_e, c1_mu_r, c1_gain, c1_thickness, c1_depth, c2_positions, c2_normals, c2_eps_r,
        c2_sigma_e, c2_mu_r, c2_gain, c2_thickness, c2_depth, n_o, t1r, t2r, backup_axis,
        wi_local, cos_i, cos_o, d_i, d_o, l1, l2, weights, material_id, f_te_flat, f_tm_flat,
        table_offset, table_dims, material_slot,
    )
    named = dict(zip(spec.names, primals))
    rows = int(tx_pol.shape[0])
    spec.checks(named, "valid")
    if valid.shape != (rows,):
        raise ValueError(f"valid must have shape ({rows},)")
    if valid.device != tx_pol.device:
        raise ValueError("valid must share the chain row CUDA device")
    spec.checks(named, "tx_pol", "rx_pol")
    if rx_pol.shape[0] != rows:
        raise ValueError("rx_pol must match tx_pol rows")
    _validate_chain_leg("c1", named, rows)
    _validate_chain_leg("c2", named, rows)
    for name in (
        "source", "vertex", "target", "n_o", "t1r", "t2r", "backup_axis", "wi_local", "d_i",
        "d_o", "cos_i", "cos_o", "l1", "l2", "weights",
    ):
        spec.checks(named, name)
        if named[name].shape[0] != rows:
            raise ValueError(f"{name} must have {rows} rows")
    spec.checks(
        named, "material_id", "f_te_flat", "f_tm_flat", "table_offset", "table_dims",
        "material_slot",
    )
    _require_same_device(
        tuple(
            (name, named[name])
            for name in (
                "tx_pol", "source", "c1_positions", "c2_positions", "d_i", "f_te_flat",
                "material_slot",
            )
        )
    )
    out = spec(*primals, coef, threshold, frequency_hz)
    return cast(
        dict[str, torch.Tensor], _require_fields(out, _CHAIN_ENSEMBLE_OUTPUT_FIELDS, spec.symbol)
    )


# ---------------------------------------------------------------------------
# autograd
# ---------------------------------------------------------------------------
# Fixed inputs refuse a gradient or tangent instead of silently detaching (scattering AD).
_ENSEMBLE_AD = AdInputs(
    "scattering_ensemble_eval_ad", _ENSEMBLE_PRIMALS, opaque("coef", grad="diff"),
    opaque("threshold"), opaque("coef_value"),
)
_ENSEMBLE_ROW_INPUTS = ("wo_rows", "r2_rows", "cos_o_rows")
_ENSEMBLE_SAMPLE_INPUTS = (
    "n_o", "t1r", "t2r", "wi_local", "cos_i", "r1", "a_te2", "a_tm2", "weights",
)
_ENSEMBLE_TABLE_INPUTS = ("f_te_flat", "f_tm_flat")
# Native gradient key of each differentiable ensemble tensor input.
_ENSEMBLE_GRAD_KEYS = {
    **{name: f"grad_{name}" for name in _ENSEMBLE_ROW_INPUTS + _ENSEMBLE_SAMPLE_INPUTS},
    "f_te_flat": "grad_f_te",
    "f_tm_flat": "grad_f_tm",
}
_PATCH_AD = AdInputs(
    "scattering_patch_integral_eval_ad", _PATCH_PRIMALS, opaque("k0", grad="diff"),
    opaque("k0_value"),
)
_PATCH_GEOMETRY_INPUTS = ("d_i", "d_o", "r1_rows", "r2_rows", "centroids")
_PATCH_DIFF_INPUTS = ("heights", "r_te", "r_tm", "d_i", "d_o", "r1_rows", "r2_rows", "centroids")


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
        valid: Any, wo_rows: Any, r2_rows: Any, cos_o_rows: Any, n_o: Any, t1r: Any, t2r: Any,
        wi_local: Any, cos_i: Any, r1: Any, a_te2: Any, a_tm2: Any, weights: Any, material_id: Any,
        backup_axis: Any, rx_pol: Any, rc_idx: Any, sc_idx: Any, f_te_flat: Any, f_tm_flat: Any,
        table_offset: Any, table_dims: Any, material_slot: Any, coef: Any, threshold: Any,
        coef_value: Any,
    ) -> Any:
        out = scattering_ensemble_eval(
            valid, wo_rows, r2_rows, cos_o_rows, n_o, t1r, t2r, wi_local, cos_i, r1, a_te2,
            a_tm2, weights, material_id, backup_axis, rx_pol, rc_idx, sc_idx, f_te_flat,
            f_tm_flat, table_offset, table_dims, material_slot, coef=coef_value,
            threshold=threshold,
        )
        return tuple(out[name] for name in _ENSEMBLE_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        primals = _ENSEMBLE_AD.primals(inputs, *_ENSEMBLE_EVAL.names[:23])
        ctx.threshold = _ENSEMBLE_AD.get(inputs, "threshold")
        ctx.coef_value = _ENSEMBLE_AD.get(inputs, "coef_value")
        ctx.coef_meta = frequency_meta(_ENSEMBLE_AD.get(inputs, "coef"))
        save(ctx, *primals)
        ctx.mark_non_differentiable(output[3])

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx: Any, grad_gain: Any, grad_amplitude: Any, grad_length: Any, _grad_keep: Any,
    ) -> Any:
        _ENSEMBLE_AD.reject_fixed_grads(ctx)
        need_rows = _ENSEMBLE_AD.needs(ctx, *_ENSEMBLE_ROW_INPUTS)
        need_samples = _ENSEMBLE_AD.needs(ctx, *_ENSEMBLE_SAMPLE_INPUTS)
        need_tables = _ENSEMBLE_AD.needs(ctx, *_ENSEMBLE_TABLE_INPUTS)
        need_coef = _ENSEMBLE_AD.need(ctx, "coef")
        grads = (grad_gain, grad_amplitude, grad_length)
        if not (need_rows or need_samples or need_tables or need_coef) or all(
            value is None for value in grads
        ):
            return _ENSEMBLE_AD.none
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
        return _ENSEMBLE_AD.grads(
            ctx,
            **{
                name: out[key]
                for name, key in _ENSEMBLE_GRAD_KEYS.items()
                if _ENSEMBLE_AD.need(ctx, name)
            },
            coef=grad_coef,
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        _ENSEMBLE_AD.reject_fixed_tangents(tangents)
        saved = ctx.saved_tensors
        native_tangents = {
            f"tangent_{name}": _ENSEMBLE_AD.geometry_tangent(
                tangents, name, saved[_ENSEMBLE_AD.index(name)]
            )
            for name in _ENSEMBLE_ROW_INPUTS + _ENSEMBLE_SAMPLE_INPUTS + _ENSEMBLE_TABLE_INPUTS
        }
        tangent_coef = _ad_frequency_tangent(_ENSEMBLE_AD.tangent(tangents, "coef"))
        if tangent_coef == 0.0 and all(value is None for value in native_tangents.values()):
            return (None,) * len(_ENSEMBLE_OUTPUT_FIELDS)
        with disable_functorch():
            out = scattering_ensemble_eval_jvp(
                *(_ad_native_tensor(value) for value in saved),
                coef=ctx.coef_value,
                threshold=ctx.threshold,
                tangent_coef=tangent_coef,
                **native_tangents,
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
        valid, wo_rows, r2_rows, cos_o_rows, n_o, t1r, t2r, wi_local, cos_i, r1, a_te2, a_tm2,
        weights, material_id, backup_axis, rx_pol, rc_idx, sc_idx, f_te_flat, f_tm_flat,
        table_offset, table_dims, material_slot, coef, float(threshold), float(coef_value),
    )
    return dict(zip(_ENSEMBLE_OUTPUT_FIELDS, values, strict=True))


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
        valid: Any, patch_tris: Any, patch_uvs: Any, rows: Any, d_i: Any, d_o: Any, n_rows: Any,
        r_te: Any, r_tm: Any, pol_t: Any, pol_r: Any, r1_rows: Any, r2_rows: Any, centroids: Any,
        heights: Any, quad_a: Any, quad_b: Any, quad_w: Any, k0: Any, k0_value: Any,
    ) -> Any:
        out = _PATCH_EVAL(
            valid, patch_tris, patch_uvs, rows, d_i, d_o, n_rows, r_te, r_tm, pol_t, pol_r,
            r1_rows, r2_rows, centroids, heights, quad_a, quad_b, quad_w, k0_value,
        )
        _require_fields(out, _PATCH_OUTPUT_FIELDS, _PATCH_EVAL.symbol)
        return tuple(out[name] for name in _PATCH_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        primals = _PATCH_AD.primals(inputs, *_PATCH_EVAL.names[:18])
        ctx.k0_value = _PATCH_AD.get(inputs, "k0_value")
        ctx.k0_meta = frequency_meta(_PATCH_AD.get(inputs, "k0"))
        save(ctx, *primals)
        ctx.mark_non_differentiable(output[1], output[2])

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, grad_total: Any, _grad_integral: Any, _grad_row_value: Any) -> Any:
        _PATCH_AD.reject_fixed_grads(ctx)
        need_heights = _PATCH_AD.need(ctx, "heights")
        need_jones = _PATCH_AD.needs(ctx, "r_te", "r_tm")
        need_geometry = _PATCH_AD.needs(ctx, *_PATCH_GEOMETRY_INPUTS)
        need_k0 = _PATCH_AD.need(ctx, "k0")
        if not (need_heights or need_jones or need_geometry or need_k0) or (
            grad_total is None
        ):
            return _PATCH_AD.none
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
        return _PATCH_AD.grads(
            ctx,
            **{
                name: out[f"grad_{name}"]
                for name in _PATCH_DIFF_INPUTS
                if _PATCH_AD.need(ctx, name)
            },
            k0=grad_k0,
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        _PATCH_AD.reject_fixed_tangents(tangents)
        saved = ctx.saved_tensors
        native_tangents = {
            f"tangent_{name}": _PATCH_AD.geometry_tangent(
                tangents, name, saved[_PATCH_AD.index(name)]
            )
            for name in _PATCH_DIFF_INPUTS
        }
        tangent_k0 = _ad_frequency_tangent(_PATCH_AD.tangent(tangents, "k0"))
        if tangent_k0 == 0.0 and all(value is None for value in native_tangents.values()):
            return (None,) * len(_PATCH_OUTPUT_FIELDS)
        with disable_functorch():
            out = scattering_patch_integral_eval_jvp(
                *(_ad_native_tensor(value) for value in saved[:15]),
                k0=ctx.k0_value,
                tangent_k0=tangent_k0,
                **native_tangents,
            )
        return (out["tangent_total"], None, None)


def scattering_patch_integral_eval_ad(
    valid: torch.Tensor, patch_tris: torch.Tensor, patch_uvs: torch.Tensor, rows: torch.Tensor,
    d_i: torch.Tensor, d_o: torch.Tensor, n_rows: torch.Tensor, r_te: torch.Tensor,
    r_tm: torch.Tensor, pol_t: torch.Tensor, pol_r: torch.Tensor, r1_rows: torch.Tensor,
    r2_rows: torch.Tensor, centroids: torch.Tensor, heights: torch.Tensor, *, k0: torch.Tensor,
) -> PatchIntegralOutputs:
    """Differentiable :func:`scattering_patch_integral_eval` (scattering AD).

    ``k0`` is the wavenumber as a 0-d scalar tensor so frequency gradients flow
    through the phase and prefactor; its host value is read once per apply at
    this seam. The Duffy-mapped quadrature nodes/weights are gathered
    from the shared cache and threaded to the companions as fixed inputs.
    """

    quad_a, quad_b, quad_w = _duffy_nodes(patch_tris.device)
    k0_value = _ad_frequency_value(k0)
    values = _ScatteringPatchIntegralEvalAdFunction.apply(
        valid, patch_tris, patch_uvs, rows, d_i, d_o, n_rows, r_te, r_tm, pol_t, pol_r, r1_rows,
        r2_rows, centroids, heights, quad_a, quad_b, quad_w, k0, float(k0_value),
    )
    return cast(PatchIntegralOutputs, dict(zip(_PATCH_OUTPUT_FIELDS, values, strict=True)))


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

# The table-build physics inputs both companions take, in ABI order.
_TABLE_BUILD_TABLES = tuple(
    tensor(name)
    for name in (
        "s_te", "s_tm", "a_te", "a_tm", "r_diff_te", "r_diff_tm", "cos_i", "phi_i", "cos_o",
        "phi_o", "layer_thickness_m", "layer_eps_r", "layer_sigma_e", "layer_mu_r",
    )
)
_TABLE_BUILD_SCALARS = (real("sigma_h"), real("corr_x"), real("corr_y"), real("frequency_hz"))
_TABLE_BUILD_BACKWARD = _native.op(
    "kirchhoff_table_build_backward", _TABLE_BUILD_TABLES, _TABLE_BUILD_SCALARS,
    tensor("grad_f_te"), tensor("grad_f_tm"), flag("need_grad_rough"), flag("need_grad_layers"),
    flag("need_grad_frequency"),
)
_TABLE_BUILD_JVP = _native.op(
    "kirchhoff_table_build_jvp", _TABLE_BUILD_TABLES, _TABLE_BUILD_SCALARS,
    opaque("t_layer_thickness_m"), opaque("t_layer_eps_r"), opaque("t_layer_sigma_e"),
    real("t_sigma_h"), real("t_corr_x"), real("t_corr_y"), real("t_frequency"),
)


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
    tables = (
        a_te, a_tm, r_diff_te, r_diff_tm, cos_i, phi_i, cos_o, phi_o, layer_thickness_m,
        layer_eps_r, layer_sigma_e, layer_mu_r,
    )
    out = _TABLE_BUILD_BACKWARD(
        s_te, s_tm, *(table.contiguous() for table in tables), sigma_h, corr_x, corr_y,
        frequency_hz, grad_f_te, grad_f_tm, need_grad_rough, need_grad_layers,
        need_grad_frequency,
    )
    return cast(
        dict[str, torch.Tensor | None],
        _require_fields(out, _BACKWARD_FIELDS, _TABLE_BUILD_BACKWARD.symbol),
    )


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
    tables = (
        a_te, a_tm, r_diff_te, r_diff_tm, cos_i, phi_i, cos_o, phi_o, layer_thickness_m,
        layer_eps_r, layer_sigma_e, layer_mu_r,
    )
    layer_tangents = (t_layer_thickness_m, t_layer_eps_r, t_layer_sigma_e)
    out = _TABLE_BUILD_JVP(
        s_te, s_tm, *(table.contiguous() for table in tables), sigma_h, corr_x, corr_y,
        frequency_hz, *(None if t is None else t.contiguous() for t in layer_tangents),
        t_sigma_h, t_corr_x, t_corr_y, t_frequency,
    )
    return cast(dict[str, torch.Tensor], _require_fields(out, _JVP_FIELDS, _TABLE_BUILD_JVP.symbol))


def _scalar_value(tensor: torch.Tensor) -> float:
    return float(_ad_native_tensor(tensor).reshape(()).detach())


def _scalar_tangent(tensor: torch.Tensor | None) -> float:
    value = _ad_native_tangent_or_none(tensor)
    if value is None:
        return 0.0
    return float(value.reshape(()).detach())


# The apply inputs of the build AD; fixed inputs fail loudly instead of silently detaching.
_TABLE_BUILD_AD = AdInputs(
    "kirchhoff_table_build_ad",
    *(opaque(f"rough_{name}", grad="diff") for name in ("sigma_h", "corr_x", "corr_y")),
    *(opaque(name, grad="diff") for name in ("layer_thickness_m", "layer_eps_r", "layer_sigma_e")),
    opaque("layer_mu_r", grad="fixed"),
    opaque("frequency", grad="diff"),
    *(
        opaque(name)
        for name in (
            "f_te_built", "f_tm_built", "s_te", "s_tm", "a_te", "a_tm", "r_diff_te", "r_diff_tm",
        )
    ),
    *(opaque(name, grad="fixed") for name in ("cos_i", "phi_i", "cos_o", "phi_o")),
    opaque("frequency_value"),
)
_TABLE_BUILD_ROUGH = ("rough_sigma_h", "rough_corr_x", "rough_corr_y")
_TABLE_BUILD_LAYERS = ("layer_thickness_m", "layer_eps_r", "layer_sigma_e")
# The saved primals, in the order of the companion facades' positional inputs.
_TABLE_BUILD_SAVED = (
    "s_te", "s_tm", "a_te", "a_tm", "r_diff_te", "r_diff_tm", "cos_i", "phi_i", "cos_o", "phi_o",
    "layer_thickness_m", "layer_eps_r", "layer_sigma_e", "layer_mu_r",
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
        rough_sigma_h: Any, rough_corr_x: Any, rough_corr_y: Any, layer_thickness_m: Any,
        layer_eps_r: Any, layer_sigma_e: Any, layer_mu_r: Any, frequency: Any, f_te_built: Any,
        f_tm_built: Any, s_te: Any, s_tm: Any, a_te: Any, a_tm: Any, r_diff_te: Any, r_diff_tm: Any,
        cos_i: Any, phi_i: Any, cos_o: Any, phi_o: Any, frequency_value: Any,
    ) -> Any:
        # The numpy build already produced f_te_built/f_tm_built; the forward is
        # a pure pass-through so the primal table stays bit-identical.
        return f_te_built, f_tm_built

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        named = _TABLE_BUILD_AD.unpack(inputs)
        ctx.sigma_h = _scalar_value(named["rough_sigma_h"])
        ctx.corr_x = _scalar_value(named["rough_corr_x"])
        ctx.corr_y = _scalar_value(named["rough_corr_y"])
        ctx.frequency_value = float(named["frequency_value"])
        ctx.rough_shapes = tuple(tuple(named[name].shape) for name in _TABLE_BUILD_ROUGH)
        ctx.frequency_meta = frequency_meta(named["frequency"])
        save(ctx, *_TABLE_BUILD_AD.primals(inputs, *_TABLE_BUILD_SAVED))

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, grad_f_te: Any, grad_f_tm: Any) -> Any:
        _TABLE_BUILD_AD.reject_fixed_grads(ctx)
        need_rough = _TABLE_BUILD_AD.needs(ctx, *_TABLE_BUILD_ROUGH)
        need_layers = _TABLE_BUILD_AD.needs(ctx, *_TABLE_BUILD_LAYERS)
        need_frequency = _TABLE_BUILD_AD.need(ctx, "frequency")
        if not (need_rough or need_layers or need_frequency) or (
            grad_f_te is None and grad_f_tm is None
        ):
            return _TABLE_BUILD_AD.none
        saved = ctx.saved_tensors
        s_te = saved[0]
        if grad_f_te is None:
            grad_f_te = torch.zeros_like(s_te)
        if grad_f_tm is None:
            grad_f_tm = torch.zeros_like(s_te)
        out: dict[str, Any] = kirchhoff_table_build_backward(
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
        rough = {
            name: out[f"grad_{name.removeprefix('rough_')}"].reshape(shape)
            for name, shape in zip(_TABLE_BUILD_ROUGH, ctx.rough_shapes)
            if _TABLE_BUILD_AD.need(ctx, name)
        }
        layers = {
            name: out[f"grad_{name}"]
            for name in _TABLE_BUILD_LAYERS
            if _TABLE_BUILD_AD.need(ctx, name)
        }
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return _TABLE_BUILD_AD.place(**rough, **layers, frequency=grad_frequency)

    @staticmethod
    def jvp(
        ctx: Any, t_sigma_h: Any, t_corr_x: Any, t_corr_y: Any, t_thickness: Any, t_eps: Any,
        t_sigma: Any, t_mu_r: Any, t_frequency: Any, _t_f_te: Any, _t_f_tm: Any, _t_s_te: Any,
        _t_s_tm: Any, _t_a_te: Any, _t_a_tm: Any, _t_r_diff_te: Any, _t_r_diff_tm: Any,
        _t_cos_i: Any, _t_phi_i: Any, _t_cos_o: Any, _t_phi_o: Any, _t_frequency_value: Any,
    ) -> Any:
        tangents = (
            t_sigma_h, t_corr_x, t_corr_y, t_thickness, t_eps, t_sigma, t_mu_r, t_frequency,
            _t_f_te, _t_f_tm, _t_s_te, _t_s_tm, _t_a_te, _t_a_tm, _t_r_diff_te, _t_r_diff_tm,
            _t_cos_i, _t_phi_i, _t_cos_o, _t_phi_o, _t_frequency_value,
        )
        _TABLE_BUILD_AD.reject_fixed_tangents(tangents)
        saved = ctx.saved_tensors
        tangent_sigma_h, tangent_corr_x, tangent_corr_y, tangent_frequency = (
            _scalar_tangent(_TABLE_BUILD_AD.tangent(tangents, name))
            for name in (*_TABLE_BUILD_ROUGH, "frequency")
        )
        tangent_thickness, tangent_eps, tangent_sigma = (
            _ad_native_tangent_or_none(_TABLE_BUILD_AD.tangent(tangents, name))
            for name in _TABLE_BUILD_LAYERS
        )
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


class _KirchhoffTableBuild(Protocol):
    """The parts of a built Kirchhoff table that its AD replay reads."""

    @property
    def cos_theta_i(self) -> torch.Tensor: ...
    @property
    def phi_i(self) -> torch.Tensor: ...
    @property
    def cos_theta_o(self) -> torch.Tensor: ...
    @property
    def phi_o(self) -> torch.Tensor: ...
    @property
    def f_te(self) -> torch.Tensor: ...
    @property
    def f_tm(self) -> torch.Tensor: ...
    @property
    def r_diff_te(self) -> torch.Tensor: ...
    @property
    def r_diff_tm(self) -> torch.Tensor: ...
    @property
    def normalization_applied(self) -> torch.Tensor: ...
    @property
    def pre_balance_lobe_te(self) -> torch.Tensor | None: ...
    @property
    def pre_balance_lobe_tm(self) -> torch.Tensor | None: ...


def kirchhoff_table_build_ad(
    rough_sigma_h: torch.Tensor, rough_corr_x: torch.Tensor, rough_corr_y: torch.Tensor,
    layer_thickness_m: torch.Tensor, layer_eps_r: torch.Tensor, layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor, frequency: torch.Tensor, *, table: _KirchhoffTableBuild,
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


# Kirchhoff samples and table views of the pair kernels, in ABI order.
_PAIR_SAMPLES = tuple(
    tensor(name)
    for name in (
        "points", "normals", "frame_t1", "frame_t2", "wi_local", "incident", "epsilon", "material",
    )
)
_PAIR_TABLES = tuple(
    tensor(name) for name in ("material_slot", "table_offset", "table_dims", "f_te", "f_tm")
)
_PAIR_LEGS = _native.op("mc_scattering_pair_legs", _PAIR_SAMPLES, _PAIR_TABLES, tensor("receivers"))
_PAIR_SUM = _native.op(
    "mc_scattering_pair_sum", _PAIR_SAMPLES, _PAIR_TABLES, tensor("receivers"), tensor("visible"),
    tensor("value_te"), tensor("value_tm"), tensor("deposited"),
)
_PAIR_SUM_BACKWARD = _native.op(
    "mc_scattering_pair_sum_backward", _PAIR_SAMPLES, _PAIR_TABLES, tensor("receivers"),
    tensor("visible"), tensor("grad"),
)


def mc_scattering_pair_legs(
    samples: tuple[torch.Tensor, ...], tables: tuple[torch.Tensor, ...], receivers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Visibility legs ``(start, end, active)`` of every (receiver, sample) pair, receiver-major.

    ``samples`` = (points, normals, frame_t1, frame_t2, wi_local, incident, epsilon, material);
    ``tables`` = (material_slot, table_offset, table_dims, f_te_flat, f_tm_flat).
    """

    out = _PAIR_LEGS(*samples, *tables, receivers)
    if not isinstance(out, tuple) or len(out) != 3:
        raise TypeError("_channel.mc_scattering_pair_legs must return 3 tensors")
    return out[0], out[1], out[2]


def _mc_scattering_pair_sum(
    samples: tuple[torch.Tensor, ...], tables: tuple[torch.Tensor, ...], receivers: torch.Tensor,
    visible: torch.Tensor, value_te: torch.Tensor, value_tm: torch.Tensor, deposited: torch.Tensor,
) -> torch.Tensor:
    out = _PAIR_SUM(*samples, *tables, receivers, visible, value_te, value_tm, deposited)
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
    def forward(
        f_te: Any, f_tm: Any, samples: Any, tables: Any, receivers: Any, visible: Any,
        deposited: Any,
    ) -> Any:
        native_tables = (*tables[:3], f_te, f_tm)
        return _mc_scattering_pair_sum(
            samples, native_tables, receivers, visible, f_te, f_tm, deposited
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
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
    def backward(ctx: Any, grad_out: Any) -> Any:
        if grad_out is None:
            return (None,) * 7
        f_te, f_tm = ctx.saved_tensors
        out = _PAIR_SUM_BACKWARD(
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
    def jvp(ctx: Any, t_te: Any, t_tm: Any, *_rest: Any) -> Any:
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
        return _McScatteringPairSumFunction.apply(
            tables[3], tables[4], samples, tables, receivers, visible, deposited
        )
    return _mc_scattering_pair_sum(
        samples, tables, receivers, visible, tables[3].detach(), tables[4].detach(), deposited
    )
