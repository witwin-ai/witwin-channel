# Copyright Xingyu Chen.
# Implements chain realization.

"""Reference float64 Torch multi-bounce coherent scattering (ADR-021 Op B).

Fully differentiable float64 re-derivation of the ADR-021 ``D2`` Op B math
(``scattering_chain_realization_eval``): the full 2x2 Jones sandwich

    E_rx = A_2 . S_patch(d_i, d_o; h) . A_1 . e_tx

with a phase-screen realization at the scatter vertex, specular chains ``A_1``
(C1) and ``A_2`` (C2) on either side, the carrier over the image-unfolded
lengths ``L1 + L2`` and the planar-chain spreading ``1/(L1 L2)`` (image
theory). This is the coherent (speckle) oracle and the gradient oracle the
future native Op B ``_backward`` / ``_jvp`` companions must match. The
degenerate ``d1 = d2 = 0`` case reduces symbol-for-symbol to the ADR-010 op-2
oracle ``phase_screen_realization.realization_patch_eval_reference`` (pinned by
the self-check gate). Test-only: MUST NOT be imported from production packages.

Convention sources of truth:

* The Duffy-mapped 16x16 Gauss-Legendre patch quadrature, the cell-centered
  bilinear height sample, the swapped-wave-vector integrand
  ``exp(-j (q'.x + q_n' h))`` with ``q' = -q``, the prefactor
  ``q_norm^2/(4pi q_n)`` and the carrier are ported from
  ``tests/reference/phase_screen_realization.py`` (ADR-010 op 2) and
  ``kernels/scattering.py::_duffy_nodes``.
* The per-bounce specular Jones chain transport (``A_1``/``A_2``) reuses
  :func:`chain_ensemble.transport_chain`, i.e. the ``reflection_chain_eval``
  conventions in ``field_transport_reflection.cu`` / ``field_transport.cuh``.
* ``PHASE_CONVENTION`` (``core/field_state.py``): world-Cartesian complex
  field, ``exp(-j k d)`` phasor, receiver projection by transverse dot.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from tests.reference.chain_ensemble import (
    _c3_dot_real,
    _c3_scale,
    _project_transverse,
    _safe_normalize,
    transport_chain,
)
from tests.reference.kirchhoff_ensemble import _sp_basis
from tests.reference.phase_screen_realization import (
    _sample_height_bilinear,
    _stable_tangent,
)

_C0 = 299792458.0
_PATCH_QUAD_ORDER = 16


def duffy_gl_nodes(
    device, dtype: torch.dtype = torch.float64, order: int = _PATCH_QUAD_ORDER
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Duffy-mapped GL nodes ``(a, b, w)``; mirrors ``functional._duffy_nodes``.

    The unit square ``(xi, eta)`` maps to barycentric ``(a, b) =
    (xi, eta*(1 - xi))`` with Jacobian ``(1 - xi)`` -- the same construction as
    ``scattering.patch_phase_integral`` and the native op. Built in
    float64 here (the oracle is double precision); the native op consumes the
    float32 cast of the identical nodes.
    """

    nodes, weights = np.polynomial.legendre.leggauss(order)
    xi = torch.from_numpy(0.5 * (nodes + 1.0)).to(device=device, dtype=dtype)
    w1 = torch.from_numpy(0.5 * weights).to(device=device, dtype=dtype)
    a = xi[:, None].expand(order, order)
    b = xi[None, :] * (1.0 - xi[:, None])
    w2d = (w1[:, None] * w1[None, :]) * (1.0 - xi[:, None])
    return a.reshape(-1).contiguous(), b.reshape(-1).contiguous(), w2d.reshape(-1).contiguous()


def _patch_integral(
    heights: torch.Tensor,
    patch_tris: torch.Tensor,
    patch_uvs: torch.Tensor,
    rows: torch.Tensor,
    d_i: torch.Tensor,
    d_o: torch.Tensor,
    quad_a: torch.Tensor,
    quad_b: torch.Tensor,
    quad_w: torch.Tensor,
    k0: torch.Tensor,
) -> torch.Tensor:
    """Per-row phase-screen aperture integral (ADR-010 op-2 quadrature).

    Verbatim port of the integral block of
    ``phase_screen_realization.realization_patch_eval_reference``: Duffy sample
    positions, cell-centered bilinear heights, swapped wave vector ``q' = -q``,
    ``integral = |e1 x e2| * sum_t w_t exp(-j (q'.x_t + q_n' h_t))``.
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
    return a2.to(phasor.dtype) * (phasor * w[None, :]).sum(-1)  # [R] complex


def chain_realization_eval(
    heights: torch.Tensor,
    patch_tris: torch.Tensor,
    patch_uvs: torch.Tensor,
    rows: torch.Tensor,
    source: torch.Tensor,
    tx_pol: torch.Tensor,
    c1: dict[str, torch.Tensor],
    vertex: torch.Tensor,
    n_rows: torch.Tensor,
    r_te: torch.Tensor,
    r_tm: torch.Tensor,
    d_i: torch.Tensor,
    d_o: torch.Tensor,
    c2: dict[str, torch.Tensor],
    target: torch.Tensor,
    rx_pol: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    centroids: torch.Tensor,
    quad_a: torch.Tensor,
    quad_b: torch.Tensor,
    quad_w: torch.Tensor,
    k0: torch.Tensor,
    frequency_hz: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable float64 ADR-021 Op B coherent per-row field and total.

    ``c1``/``c2`` are chain dicts (see :func:`chain_ensemble.transport_chain`;
    pass :func:`chain_ensemble._empty_chain` for the degenerate legs). ``d_i``,
    ``d_o`` are the LOCAL vertex directions (op-2 leaves) that drive the vertex
    operator basis, ``q``, prefactor and carrier; ``vertex`` positions the chain
    legs. ``r_te``/``r_tm`` are the smooth-stack Fresnel coefficients at the
    vertex local specular angle (``em_layer_stack_eval``); the chain Fresnel
    lives in the ``c1``/``c2`` material tensors.

    Live leaves per ADR-021 D5 Op B: ``heights``, ``r_te``/``r_tm``, chain
    geometry rows (``d_i``, ``d_o``, ``l1``, ``l2``, ``centroids``, chain
    positions/normals), chain material tensors, ``k0``, ``frequency_hz``. Fixed:
    ``patch_tris``/``patch_uvs``, ``rows``, ``quad_*`` nodes, ``n_rows`` and the
    polarization vectors. The ``d1 = d2 = 0`` reduction returns the op-2 total
    (self-check gate ``a``).

    Assembly: ``S_patch = (j * prefactor) [s_o p_o] diag(r_te, r_tm) [s_i p_i]^T
    * exp(-j k0 (L1+L2)) / (L1 L2) * integral``; the incident 2-vector is
    ``A_1 e_tx`` projected onto ``(s_i, p_i)`` and the receiver covector is
    ``A_2^dagger p_rx`` (built by transporting the outgoing world field through
    C2 and projecting onto ``p_rx``).
    """

    complex_dtype = torch.complex128 if heights.dtype == torch.float64 else torch.complex64
    backup = _stable_tangent(n_rows)
    s_i, p_i = _sp_basis(n_rows, d_i, backup)
    s_o, p_o = _sp_basis(n_rows, d_o, backup)

    # A_1 e_tx: transport the transmit field through C1 to the vertex, then
    # project onto the local incident s/p basis.
    first_leg = _safe_normalize(
        (c1["positions"][:, 0] if c1["positions"].shape[1] > 0 else vertex) - source,
        d_i,
    )
    e_tx = _project_transverse(tx_pol, first_leg).to(complex_dtype)
    e_in, _ = transport_chain(
        e_tx, source, vertex, c1["positions"], c1["normals"], c1["eps_r"],
        c1["sigma_e"], c1["mu_r"], c1["gain"], c1["thickness"], c1["sigma_b"],
        c1["rough"], frequency_hz,
    )
    e_s_in = _c3_dot_real(e_in, s_i)
    e_p_in = _c3_dot_real(e_in, p_i)

    # S diag(r_te, r_tm): outgoing world field before C2.
    e_out = _c3_scale(s_o, r_te * e_s_in) + _c3_scale(p_o, r_tm * e_p_in)

    # A_2: transport the outgoing world field through C2 to the receiver.
    e_rx_field, last_dir = transport_chain(
        e_out, vertex, target, c2["positions"], c2["normals"], c2["eps_r"],
        c2["sigma_e"], c2["mu_r"], c2["gain"], c2["thickness"], c2["sigma_b"],
        c2["rough"], frequency_hz,
    )
    rx_axis = _project_transverse(rx_pol, last_dir)
    e_rx = _c3_dot_real(e_rx_field, rx_axis)  # [R] complex jones scalar

    integral = _patch_integral(
        heights, patch_tris, patch_uvs, rows, d_i, d_o, quad_a, quad_b, quad_w, k0
    )

    q = k0 * (d_o - d_i)
    q_norm = torch.linalg.vector_norm(q, dim=-1)
    q_n = (q * n_rows).sum(-1)
    prefactor = q_norm.square() / (4.0 * math.pi * q_n.clamp_min(1.0e-9))
    cphase = -(k0 * (l1 + l2) + (q * centroids).sum(-1))
    carrier = torch.polar(torch.ones_like(cphase), cphase)
    value = (1j * prefactor) * e_rx * carrier / (l1 * l2)
    row_value = value * integral
    total = row_value.sum()
    return {
        "total": total,
        "integral": integral,
        "row_value": row_value,
        "e_rx": e_rx,
    }