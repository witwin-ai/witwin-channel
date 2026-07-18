"""Reference Torch realization-coherent patch-integral total (ADR-010 op 2).

The previous production per-row assembly and host per-patch loop of
``propagation/enumerated/scattering.py::_realization_rows`` (jones, prefactor,
carrier, per-patch ``patch_phase_integral`` quadrature, weighted total).
``patch_phase_integral`` itself remains a public utility in
``witwin.channel_native.scattering`` (its only production caller was this
loop); this module reproduces the removed loop around it. Test-only: MUST NOT
be imported from production packages.
"""

from __future__ import annotations

import math

import torch

from witwin.channel_native.core.tensor_math import normalize_vec3
from witwin.channel_native.scattering import patch_phase_integral


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
