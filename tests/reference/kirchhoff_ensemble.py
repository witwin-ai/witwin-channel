# Copyright Xingyu Chen.
# Implements kirchhoff ensemble.

"""Implements kirchhoff ensemble."""

from __future__ import annotations

import math

import torch

from witwin.core import normalize_vec3
from witwin.channel.scene.resources import eval_bsdf


def _sp_basis(
    n: torch.Tensor, d: torch.Tensor, backup_axis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    s = torch.cross(n, d, dim=-1)
    degenerate = torch.linalg.vector_norm(s, dim=-1, keepdim=True) < 1.0e-6
    s = torch.where(degenerate, backup_axis, normalize_vec3(s))
    p = torch.cross(s, d, dim=-1)
    return s, p


def kirchhoff_ensemble_rows(
    points: torch.Tensor,
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
    rx_positions: torch.Tensor,
    rx_pol: torch.Tensor,
    rc: torch.Tensor,
    sc: torch.Tensor,
    tables: dict[int, object],
    coef: float,
    threshold: float,
) -> dict[str, torch.Tensor]:
    """Per visibility-surviving row physics matching the removed Torch source."""

    to_rx = rx_positions[rc] - points[sc]
    r2_row = torch.linalg.vector_norm(to_rx, dim=-1).clamp_min(1.0e-6)
    wo_row = to_rx / r2_row[:, None]
    cos_o_row = (wo_row * n_o[sc]).sum(-1)
    wo_local = torch.stack(
        (
            (wo_row * t1r[sc]).sum(-1),
            (wo_row * t2r[sc]).sum(-1),
            cos_o_row,
        ),
        dim=-1,
    )
    f_te = torch.zeros_like(cos_o_row)
    f_tm = torch.zeros_like(cos_o_row)
    for material_index, table in tables.items():
        mask = material_id[sc] == material_index
        if not bool(mask.any()):
            continue
        wi_selected = wi_local[sc][mask].contiguous()
        valid = torch.ones(
            wi_selected.shape[0], dtype=torch.bool, device=wi_selected.device
        )
        te, tm = eval_bsdf(table, valid, wi_selected, wo_local[mask].contiguous())
        f_te[mask] = te
        f_tm[mask] = tm

    s_o, p_o = _sp_basis(n_o[sc], wo_row, backup_axis[sc])
    pol_r = rx_pol[rc]
    pol_r_perp = pol_r - (pol_r * wo_row).sum(-1, keepdim=True) * wo_row
    g_te2 = (pol_r_perp * s_o).sum(-1).square()
    g_tm2 = (pol_r_perp * p_o).sum(-1).square()
    f_eff = f_te * a_te2[sc] * g_te2 + f_tm * a_tm2[sc] * g_tm2

    gain = (
        coef
        * f_eff
        * cos_i[sc]
        * cos_o_row
        * weights[sc]
        / (r1[sc].square() * r2_row.square())
    )
    keep = gain > threshold
    amplitude = gain.clamp_min(0.0).sqrt()
    length = r1[sc] + r2_row
    return {
        "gain": gain,
        "amplitude": amplitude,
        "length": length,
        "direction": wo_row,
        "keep": keep,
    }


# ---------------------------------------------------------------------------
# Double-precision autograd oracle for the native scattering AD VJP/JVP.
#
# The block above reproduces the removed production Torch physics but leans on
# the *native* ``eval_bsdf`` table lookup (float32, non-differentiable), so it
# cannot serve as a gradient oracle. The functions below add a self-contained,
# float64-capable, fully differentiable re-derivation that consumes the native
# op's *gathered* differentiable inputs directly (``wo_rows``/``r2_rows``/
# ``cos_o_rows`` per row and ``n_o``/``t1r``/... per sample) plus dense
# per-material tables, so ``torch.autograd`` through it pins the exact VJP/JVP
# the native ``scattering_ensemble_eval_backward``/``_jvp`` companions must
# match. The multilinear table interpolation mirrors the cell-centered
# clamp/periodic conventions documented in scattering AD's derivative spec
# (``t = coord*n - 0.5`` clamp for the cos axes, ``n/(2*pi)`` for the periodic
# phi axes, ``npi == 1`` relative-azimuth coupling ``phi_o' = wrap(phi_o -
# phi_i)``). Test-only: MUST NOT be imported from production packages.
# ---------------------------------------------------------------------------


def _wrap_2pi(angle: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * math.pi
    return angle - two_pi * torch.floor(angle / two_pi)


def _axis_clamp_weights(
    coord: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cell-centered non-periodic axis (centers ``(k+0.5)/n``) with edge clamp.

 Returns ``(i0, i1, frac)`` so the interpolated value is
 ``(1-frac)*table[i0] + frac*table[i1]``. The gradient of ``frac`` w.r.t.
 ``coord`` is ``n`` when ``t = coord*n - 0.5`` lies in ``(0, n-1)`` and 0
 outside (the clamp saturates), matching the scattering AD spec. An axis with
 ``n == 1`` contributes a constant (zero partial).
 """

    if n == 1:
        index = torch.zeros_like(coord, dtype=torch.long)
        return index, index, torch.zeros_like(coord)
    t = coord * float(n) - 0.5
    tc = t.clamp(0.0, float(n - 1))
    base = torch.floor(tc.detach()).clamp(0.0, float(n - 2))
    i0 = base.long()
    frac = tc - base
    return i0, i0 + 1, frac


def _axis_periodic_weights(
    phi: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cell-centered periodic axis over ``[0, 2*pi)`` (centers ``(k+0.5)*2*pi/n``).

 ``d frac/d phi = n/(2*pi)`` everywhere (no clamp); indices wrap modulo ``n``.
 """

    t = phi * (float(n) / (2.0 * math.pi)) - 0.5
    base = torch.floor(t.detach())
    frac = t - base
    i0 = base.long() % n
    i1 = (i0 + 1) % n
    return i0, i1, frac


def bsdf_table_interp(
    wi_local: torch.Tensor,
    wo_local: torch.Tensor,
    f_te: torch.Tensor,
    f_tm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable multilinear Kirchhoff-table lookup (float64-capable).

 ``f_te``/``f_tm`` are dense ``[Nti, Npi, Nto, Npo]`` tables for one
 material; ``wi_local``/``wo_local`` are ``[N, 3]`` local-frame directions.
 Below-horizon pairs (``wi[2] <= 0`` or ``wo[2] <= 0``) return 0 with zero
 partials. For the production isotropic table (``Npi == 1``) the outgoing
 azimuth is measured relative to the incident azimuth.
 """

    nti, npi, nto, npo = (int(size) for size in f_te.shape)
    cos_i = wi_local[..., 2]
    cos_o = wo_local[..., 2]
    horizon = (cos_i > 0.0) & (cos_o > 0.0)
    phi_i = _wrap_2pi(torch.atan2(wi_local[..., 1], wi_local[..., 0]))
    phi_o = _wrap_2pi(torch.atan2(wo_local[..., 1], wo_local[..., 0]))

    ci0, ci1, cfi = _axis_clamp_weights(cos_i, nti)
    co0, co1, cfo = _axis_clamp_weights(cos_o, nto)
    if npi == 1:
        zero_idx = torch.zeros_like(ci0)
        pi0 = pi1 = zero_idx
        pfi = torch.zeros_like(cfi)
        phi_o_axis = _wrap_2pi(phi_o - phi_i)
    else:
        pi0, pi1, pfi = _axis_periodic_weights(phi_i, npi)
        phi_o_axis = phi_o
    po0, po1, pfo = _axis_periodic_weights(phi_o_axis, npo)

    result_te = torch.zeros_like(cos_i)
    result_tm = torch.zeros_like(cos_i)
    for ia, wa in ((ci0, 1.0 - cfi), (ci1, cfi)):
        for ipa, wpa in ((pi0, 1.0 - pfi), (pi1, pfi)):
            for ioa, woa in ((co0, 1.0 - cfo), (co1, cfo)):
                for ipoa, wpoa in ((po0, 1.0 - pfo), (po1, pfo)):
                    weight = wa * wpa * woa * wpoa
                    result_te = result_te + weight * f_te[ia, ipa, ioa, ipoa]
                    result_tm = result_tm + weight * f_tm[ia, ipa, ioa, ipoa]

    zero = torch.zeros_like(result_te)
    result_te = torch.where(horizon, result_te, zero)
    result_tm = torch.where(horizon, result_tm, zero)
    return result_te, result_tm


def kirchhoff_ensemble_gain_reference(
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
    backup_axis: torch.Tensor,
    rx_pol: torch.Tensor,
    rc_idx: torch.Tensor,
    sc_idx: torch.Tensor,
    f_te: torch.Tensor,
    f_tm: torch.Tensor,
    coef: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable re-derivation of scattering AD from *gathered* inputs.

 Mirrors ``scattering_ensemble_eval``'s per-row physics exactly, taking the
 already-gathered ``wo_rows``/``r2_rows``/``cos_o_rows`` and the per-sample
 ``n_o``/``t1r``/``t2r``/``wi_local``/``cos_i``/``r1``/``a_te2``/``a_tm2``/
 ``weights`` indexed by ``sc_idx``, the dense ``f_te``/``f_tm`` tables for
 the single test material, and the radiometric scalar ``coef``. Every listed
 tensor is a differentiable leaf; ``backup_axis``/``rx_pol``/``rc_idx``/
 ``sc_idx`` are fixed. ``keep``/``threshold`` are outside the AD contract.
 """

    sample = sc_idx.long()
    receiver = rc_idx.long()
    t1 = t1r[sample]
    t2 = t2r[sample]
    wo_local = torch.stack(
        (
            (wo_rows * t1).sum(-1),
            (wo_rows * t2).sum(-1),
            cos_o_rows,
        ),
        dim=-1,
    )
    f_te_row, f_tm_row = bsdf_table_interp(wi_local[sample], wo_local, f_te, f_tm)
    s_o, p_o = _sp_basis(n_o[sample], wo_rows, backup_axis[sample])
    pol_r = rx_pol[receiver]
    pol_r_perp = pol_r - (pol_r * wo_rows).sum(-1, keepdim=True) * wo_rows
    g_te2 = (pol_r_perp * s_o).sum(-1).square()
    g_tm2 = (pol_r_perp * p_o).sum(-1).square()
    f_eff = f_te_row * a_te2[sample] * g_te2 + f_tm_row * a_tm2[sample] * g_tm2

    base = coef * cos_i[sample] * cos_o_rows * weights[sample] / (
        r1[sample].square() * r2_rows.square()
    )
    gain = base * f_eff
    amplitude = gain.clamp_min(0.0).sqrt()
    length = r1[sample] + r2_rows
    return {"gain": gain, "amplitude": amplitude, "length": length}