"""Reference Torch realization-coherent patch-integral total (ADR-010 op 2).

The previous production per-row assembly and host per-patch loop of
``interactions/scattering.py::_realization_rows`` (jones, prefactor,
carrier, per-patch ``patch_phase_integral`` quadrature, weighted total).
``patch_phase_integral`` itself remains a public utility in
``witwin.channel.scene.resources`` (its only production caller was this
loop); this module reproduces the removed loop around it. Test-only: MUST NOT
be imported from production packages.
"""

from __future__ import annotations

import math

import torch

from witwin.core import normalize_vec3
from witwin.channel.scene.resources import patch_phase_integral


def _stable_tangent(n: torch.Tensor) -> torch.Tensor:
    axis = torch.zeros_like(n)
    pick = n.abs().argmin(dim=-1)
    axis.scatter_(-1, pick.unsqueeze(-1), 1.0)
    t = axis - (axis * n).sum(dim=-1, keepdim=True) * n
    return normalize_vec3(t)


def _sp_basis(
    n: torch.Tensor, d: torch.Tensor, backup_axis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    s = torch.cross(n, d, dim=-1)
    degenerate = torch.linalg.vector_norm(s, dim=-1, keepdim=True) < 1.0e-6
    s = torch.where(degenerate, backup_axis, normalize_vec3(s))
    p = torch.cross(s, d, dim=-1)
    return s, p


def realization_patch_total(
    runtime: object,
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
    k0: float,
    frequency_hz: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Removed production loop: returns (total, per-row integral buffer)."""

    device = patch_tris.device
    backup_axis = _stable_tangent(n_rows)
    s_i, p_i = _sp_basis(n_rows, d_i, backup_axis)
    s_o, p_o = _sp_basis(n_rows, d_o, backup_axis)
    pol_t = pol_t.reshape(1, 3)
    pol_t_perp = pol_t - (pol_t * d_i).sum(-1, keepdim=True) * d_i
    pol_r = pol_r.reshape(1, 3)
    pol_r_perp = pol_r - (pol_r * d_o).sum(-1, keepdim=True) * d_o
    a_te = (pol_t_perp * s_i).sum(-1)
    a_tm = (pol_t_perp * p_i).sum(-1)
    g_te = (pol_r_perp * s_o).sum(-1)
    g_tm = (pol_r_perp * p_o).sum(-1)
    jones = r_te * (a_te * g_te) + r_tm * (a_tm * g_tm)

    k_i_vec = d_i * k0
    k_s_vec = d_o * k0
    q = k_s_vec - k_i_vec
    q_norm = torch.linalg.vector_norm(q, dim=-1)
    q_n = (q * n_rows).sum(-1)
    prefactor = (
        1j * k0 * (q_norm.square() / (k0 * q_n.clamp_min(1.0e-9)))
        / (4.0 * math.pi)
    )
    carrier = torch.polar(
        torch.ones_like(q_n),
        -(k0 * (r1_rows + r2_rows) + (q * centroids).sum(-1)),
    ).to(torch.complex64)

    total = torch.zeros((), device=device, dtype=torch.complex64)
    integrals = torch.zeros(
        (int(rows.numel()),), device=device, dtype=torch.complex64
    )
    for slot, patch_index in enumerate(rows.tolist()):
        # Swapped wave vectors: patch_phase_integral integrates
        # exp(-j*(q'.x + q_n'*h)) with q' = -q, i.e. the physical +j
        # integrand of the production module docstring derivation.
        integral = patch_phase_integral(
            runtime,
            patch_tris[patch_index],
            patch_uvs[patch_index],
            k_s_vec[slot],
            k_i_vec[slot],
            frequency_hz,
        )
        integrals[slot] = integral
        total = total + (
            prefactor[slot]
            * jones[slot]
            * carrier[slot]
            / (r1_rows[slot] * r2_rows[slot])
        ) * integral
    return total, integrals


# ---------------------------------------------------------------------------
# Double-precision autograd oracle for the native ADR-014 op-2 VJP/JVP.
#
# ``realization_patch_total`` above is a faithful re-derivation but routes the
# quadrature through ``patch_phase_integral`` (float32, and it reads heights
# from a ``PhaseScreenRuntime`` rather than a leaf), so it cannot pin height or
# ``k0`` gradients in float64. The block below is a self-contained float64
# re-implementation that takes ``heights`` as a differentiable leaf and the
# same Duffy-mapped quadrature nodes the native op consumes, then reproduces
# the ADR-014 op-2 assembly (bilinear height sampling, integral, prefactor,
# jones, carrier) so ``torch.autograd`` through it pins the exact VJP/JVP the
# native ``scattering_patch_integral_eval_backward``/``_jvp`` companions must
# match. Test-only: MUST NOT be imported from production packages.
# ---------------------------------------------------------------------------


def _sample_height_bilinear(heights: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Cell-centered edge-clamp bilinear height sample (matches the runtime).

    ``heights`` is ``[H, W]`` (row = v/y axis); ``uv`` is ``[..., 2]`` with
    ``u`` along the columns. Differentiable w.r.t. ``heights``; the texel
    indexing is detached (piecewise constant), exactly as
    :meth:`PhaseScreenRuntime.sample_height`.
    """

    h_rows, w_cols = (int(size) for size in heights.shape)
    tx = (uv[..., 0] * w_cols - 0.5).clamp(0.0, float(w_cols - 1))
    ty = (uv[..., 1] * h_rows - 0.5).clamp(0.0, float(h_rows - 1))
    x0 = torch.floor(tx.detach())
    y0 = torch.floor(ty.detach())
    wx = tx - x0
    wy = ty - y0
    ix0 = x0.long()
    iy0 = y0.long()
    ix1 = (ix0 + 1).clamp(max=w_cols - 1)
    iy1 = (iy0 + 1).clamp(max=h_rows - 1)
    flat = heights.reshape(-1)

    def tex(iy: torch.Tensor, ix: torch.Tensor) -> torch.Tensor:
        return flat[iy * w_cols + ix]

    top = tex(iy0, ix0) * (1.0 - wx) + tex(iy0, ix1) * wx
    bot = tex(iy1, ix0) * (1.0 - wx) + tex(iy1, ix1) * wx
    return top * (1.0 - wy) + bot * wy


def realization_patch_eval_reference(
    heights: torch.Tensor,
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
    quad_a: torch.Tensor,
    quad_b: torch.Tensor,
    quad_w: torch.Tensor,
    k0: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable re-derivation of ADR-014 op 2 from a ``heights`` leaf.

    Live inputs: ``heights``, ``r_te``, ``r_tm``, ``d_i``, ``d_o``,
    ``r1_rows``, ``r2_rows``, ``centroids`` and the scalar ``k0``. Fixed:
    ``patch_tris``, ``patch_uvs``, ``rows``, ``n_rows``, ``pol_t``, ``pol_r``,
    and the quadrature nodes. Returns ``total`` (0-dim complex), the per-row
    ``integral`` and ``row_value`` buffers.
    """

    real_dtype = heights.dtype
    patch = rows.long()
    tri = patch_tris[patch].to(real_dtype)
    uv = patch_uvs[patch].to(real_dtype)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = torch.cross(e1, e2, dim=-1)
    a2 = torch.linalg.vector_norm(cross, dim=-1)  # |e1 x e2| = 2 * area
    n_hat = cross / a2.unsqueeze(-1).clamp_min(1.0e-30)

    a = quad_a.reshape(-1).to(real_dtype)
    b = quad_b.reshape(-1).to(real_dtype)
    w = quad_w.reshape(-1).to(real_dtype)
    pos = (
        tri[:, 0, None, :]
        + a[None, :, None] * e1[:, None, :]
        + b[None, :, None] * e2[:, None, :]
    )
    uv_pts = (
        uv[:, 0, None, :]
        + a[None, :, None] * (uv[:, 1] - uv[:, 0])[:, None, :]
        + b[None, :, None] * (uv[:, 2] - uv[:, 0])[:, None, :]
    )
    h = _sample_height_bilinear(heights, uv_pts)  # [R, Q]

    q = k0 * (d_o - d_i)  # [R, 3]
    q_int = -q
    q_int_n = (n_hat * q_int).sum(-1)  # [R]
    phase = (pos * q_int[:, None, :]).sum(-1) + q_int_n[:, None] * h  # [R, Q]
    phasor = torch.polar(torch.ones_like(phase), -phase)
    integral = a2.to(phasor.dtype) * (phasor * w[None, :]).sum(-1)  # [R] complex

    q_norm = torch.linalg.vector_norm(q, dim=-1)
    q_n = (q * n_rows.to(real_dtype)).sum(-1)
    prefactor = q_norm.square() / (4.0 * math.pi * q_n.clamp_min(1.0e-9))

    backup = _stable_tangent(n_rows.to(real_dtype))
    s_i, p_i = _sp_basis(n_rows.to(real_dtype), d_i, backup)
    s_o, p_o = _sp_basis(n_rows.to(real_dtype), d_o, backup)
    pol_t_ = pol_t.reshape(1, 3).to(real_dtype)
    pol_r_ = pol_r.reshape(1, 3).to(real_dtype)
    pol_t_perp = pol_t_ - (pol_t_ * d_i).sum(-1, keepdim=True) * d_i
    pol_r_perp = pol_r_ - (pol_r_ * d_o).sum(-1, keepdim=True) * d_o
    a_te = (pol_t_perp * s_i).sum(-1)
    a_tm = (pol_t_perp * p_i).sum(-1)
    g_te = (pol_r_perp * s_o).sum(-1)
    g_tm = (pol_r_perp * p_o).sum(-1)
    jones = r_te * (a_te * g_te) + r_tm * (a_tm * g_tm)

    cphase = -(k0 * (r1_rows + r2_rows) + (q * centroids).sum(-1))
    carrier = torch.polar(torch.ones_like(cphase), cphase)
    value = (1j * prefactor) * jones * carrier / (r1_rows * r2_rows)
    row_value = value * integral
    total = row_value.sum()
    return {"total": total, "integral": integral, "row_value": row_value}
