# Copyright Xingyu Chen.
# Implements chain ensemble.

"""Reference float64 Torch multi-bounce ensemble scattering (ADR-021 Op A).

Fully differentiable float64 re-derivation of the ADR-021 ``D2`` Op A math
(``scattering_chain_ensemble_eval``): a diffuse-scatter vertex ``v_s`` with a
specular reflection chain ``C1`` before it and ``C2`` after it,

    TX --C1 (d1 reflections)--> v_s --C2 (d2 reflections)--> RX

in the power (coherency) domain. This module is the gradient oracle the future
native Op A ``_backward`` / ``_jvp`` companions must match, and the degenerate
``d1 = d2 = 0`` case reduces symbol-for-symbol to the ADR-010 op-1 oracle
``kirchhoff_ensemble.kirchhoff_ensemble_gain_reference`` (pinned by the
self-check gate). Test-only: MUST NOT be imported from production packages.

Convention sources of truth (every choice below cites the native line it
mirrors so the lockstep companion stays honest):

* Per-bounce specular Jones transport (frame, Fresnel, s/p projection, field
  update) mirrors ``reflection_chain_eval`` in
  ``native/channel/kernels/field_transport.cu`` and the
  device primitives ``reflect_frame`` / ``slab_fresnel`` / ``reflect_complex3``
  in ``native/channel/field_transport.cuh``.
* Rough per-bounce ``C_r = exp(-2*(k0*cos_b*sigma_b)^2)`` attenuation follows
  ``tests/reference/rough_reflection.py::rough_reflection_factor`` (ADR-010
  op 3), applied per specular bounce (ADR-021 section 2.6: the diffuse budget
  at ``v_s`` is separate and lives inside the table).
* The quadrilinear Kirchhoff BSDF table lookup and the transverse s/p basis are
  imported unchanged from ``tests/reference/kirchhoff_ensemble.py`` so the table
  interpolation is a single source of truth (ADR-014 cell-centered clamp /
  relative-azimuth conventions).
* ``PHASE_CONVENTION`` (``witwin/channel/constants.py``):
  world-Cartesian complex field, receiver projection by transverse dot. The
  ensemble object is a coherency diagonal, not a field, so no carrier appears
  here (zero-phase power rows, ADR-021 section 2.3).
"""

from __future__ import annotations

import math

import torch

from tests.reference.kirchhoff_ensemble import _sp_basis, bsdf_table_interp

_C0 = 299792458.0


# ---------------------------------------------------------------------------
# Vector primitives (float64). Real 3-vectors are ``[..., 3]`` real tensors;
# a Complex3 field is a ``[..., 3]`` complex tensor.
# ---------------------------------------------------------------------------


def _safe_normalize(vec: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """``safe_normalize`` (field_transport.cuh): unit ``vec`` else ``fallback``.

    The native helper falls back to a supplied direction when the length
    underflows; the fixtures keep every segment well above the threshold, so
    the fallback only guards the exact-degenerate configuration.
    """

    norm = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
    unit = vec / norm.clamp_min(1.0e-30)
    return torch.where(norm > 1.0e-12, unit, fallback)


def _project_transverse(pol: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """``project_to_wedge_plane``: unnormalized transverse projection (F1).

    ``field_transport.cuh`` deliberately does NOT normalize this projection so
    the short-dipole ``sin(theta)`` weight is preserved; the tx/rx polarization
    axes both use it.
    """

    return pol - (pol * direction).sum(-1, keepdim=True) * direction


def _c3_dot_real(value: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """``complex3_dot_real``: complex field dotted with a real axis -> scalar."""

    return (value * axis.to(value.dtype)).sum(-1)


def _c3_scale(axis: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
    """``cplx_scale_real``: real axis scaled by a complex per-row scalar."""

    return axis.to(scalar.dtype) * scalar.unsqueeze(-1)


# ---------------------------------------------------------------------------
# One specular reflection event (mirrors reflect_frame / slab_fresnel /
# reflect_complex3 in field_transport.cuh).
# ---------------------------------------------------------------------------


def _reflect_frame(
    incident: torch.Tensor, normal: torch.Tensor
) -> dict[str, torch.Tensor]:
    """``reflect_frame`` (field_transport.cuh:282): s/p basis of one bounce.

    Returns ``s_axis`` / ``p_in`` / ``p_out`` / ``reflected`` / ``cos_theta``
    with the same oriented-normal flip and cross-product order as the kernel.
    """

    ez = torch.zeros_like(incident)
    ez[..., 2] = 1.0
    incident = _safe_normalize(incident, ez)
    oriented = _safe_normalize(normal, ez)
    facing = (incident * oriented).sum(-1, keepdim=True) > 0.0
    oriented = torch.where(facing, -oriented, oriented)
    dot_in = (incident * oriented).sum(-1, keepdim=True)
    reflected = _safe_normalize(incident - 2.0 * dot_in * oriented, -incident)
    s_axis = _safe_normalize(torch.cross(oriented, incident, dim=-1), _fallback_perp(incident))
    p_in = _safe_normalize(torch.cross(s_axis, incident, dim=-1), _fallback_perp(incident))
    p_out = _safe_normalize(torch.cross(s_axis, reflected, dim=-1), _fallback_perp(reflected))
    return {
        "s_axis": s_axis,
        "p_in": p_in,
        "p_out": p_out,
        "reflected": reflected,
        "cos_theta": dot_in.abs().squeeze(-1),
    }


def _fallback_perp(direction: torch.Tensor) -> torch.Tensor:
    """A deterministic unit vector perpendicular to ``direction``.

    Stands in for ``stable_perp_basis``; only reached at the exact axial
    degeneracy the fixtures avoid, so its exact value is not load-bearing.
    """

    ex = torch.zeros_like(direction)
    ex[..., 0] = 1.0
    ey = torch.zeros_like(direction)
    ey[..., 1] = 1.0
    use_x = direction[..., 0].abs().unsqueeze(-1) < 0.9
    seed = torch.where(use_x, ex, ey)
    perp = seed - (seed * direction).sum(-1, keepdim=True) * direction
    return perp / torch.linalg.vector_norm(perp, dim=-1, keepdim=True).clamp_min(1.0e-30)


def _slab_fresnel(
    cos_theta: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    thickness: torch.Tensor,
    frequency_hz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``slab_fresnel`` (field_transport.cuh:209): thin-sheet (r_te, r_tm).

    Exact port of the device expression order (interface coefficients, the
    ``exp(-2j q)`` slab phase, the geometric-series denominator). The complex
    square root uses the principal branch (``torch.sqrt`` matches
    ``utd::cplx_sqrt``: real part >= 0, and the passive medium's negative
    imaginary permittivity lands on the decaying root).
    """

    two_pi = 2.0 * math.pi
    eps0 = 8.8541878128e-12
    small = 1.0e-6
    omega = (two_pi * frequency_hz).clamp_min(small)
    wavelength = _C0 / frequency_hz
    ct = cos_theta.abs().clamp(small, 1.0)
    sin2 = (1.0 - ct * ct).clamp_min(0.0)
    eta = torch.complex(
        eps_r.clamp_min(small), -sigma_e.clamp_min(0.0) / (omega * eps0)
    )
    mu = mu_r.clamp_min(small).to(eta.dtype)
    root = torch.sqrt(eta * mu - sin2.to(eta.dtype))
    mu_ct = (mu * ct.to(eta.dtype))
    eta_ct = eta * ct.to(eta.dtype)
    interface_te = (mu_ct - root) / (mu_ct + root)
    interface_tm = (eta_ct - root) / (eta_ct + root)
    q = root * (two_pi * thickness.clamp_min(0.0) / wavelength).to(eta.dtype)
    phase = torch.exp(-2j * q)
    one = torch.ones_like(phase)
    numerator = one - phase
    r_te = gain.to(eta.dtype) * (interface_te * numerator) / (
        one - interface_te * interface_te * phase
    )
    r_tm = gain.to(eta.dtype) * (interface_tm * numerator) / (
        one - interface_tm * interface_tm * phase
    )
    return r_te, r_tm


def _cr_factor(
    cos_theta: torch.Tensor, sigma_b: torch.Tensor, rough: torch.Tensor, k0: torch.Tensor
) -> torch.Tensor:
    """Per-bounce ``C_r = exp(-2*(k0*cos*sigma)^2)`` on rough bounces (1 else).

    ``rough_reflection.py`` uses ``cos_b = |dot(seg_dir, n)|`` which equals the
    reflect-frame ``cos_theta`` here.
    """

    attenuation = torch.exp(-2.0 * (k0 * cos_theta * sigma_b) ** 2)
    return torch.where(rough, attenuation, torch.ones_like(attenuation))


def transport_chain(
    value: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    positions: torch.Tensor,
    normals: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    thickness: torch.Tensor,
    sigma_b: torch.Tensor,
    rough: torch.Tensor,
    frequency_hz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transport a Complex3 field from ``start`` to ``end`` through the chain.

    ``positions``/``normals`` are ``[N, D, 3]`` (D may be 0 for an empty chain);
    the per-bounce material tensors are ``[N, D]``. Mirrors the bounce loop of
    ``reflection_chain_eval`` (frame -> Fresnel -> s/p decomposition -> field
    update), applying the rough ``C_r`` amplitude per bounce. Returns the field
    arriving at ``end`` and the propagation direction of the final leg
    (``end - previous`` normalized), i.e. the incident/outgoing direction the
    vertex operator needs. The propagation phase and ``1/r`` spreading are NOT
    applied here: they live in the radiometric carrier / ``L1,L2`` factors
    (ADR-021 Op A step 5), exactly as op-1 keeps the Jones transfer separate
    from the propagation scalar.
    """

    ez = torch.zeros_like(start)
    ez[..., 2] = 1.0
    k0 = 2.0 * math.pi * frequency_hz / _C0
    depth = positions.shape[1]
    previous = start
    first_target = positions[:, 0] if depth > 0 else end
    outgoing = _safe_normalize(first_target - start, ez)
    for bounce in range(depth):
        hit = positions[:, bounce]
        incident = _safe_normalize(hit - previous, outgoing)
        frame = _reflect_frame(incident, normals[:, bounce])
        r_te, r_tm = _slab_fresnel(
            frame["cos_theta"],
            eps_r[:, bounce],
            sigma_e[:, bounce],
            mu_r[:, bounce],
            gain[:, bounce],
            thickness[:, bounce],
            frequency_hz,
        )
        e_s = _c3_dot_real(value, frame["s_axis"])
        e_p = _c3_dot_real(value, frame["p_in"])
        value = _c3_scale(frame["s_axis"], r_te * e_s) + _c3_scale(
            frame["p_out"], r_tm * e_p
        )
        cr = _cr_factor(frame["cos_theta"], sigma_b[:, bounce], rough[:, bounce], k0)
        value = value * cr.to(value.dtype).unsqueeze(-1)
        outgoing = frame["reflected"]
        previous = hit
    last_dir = _safe_normalize(end - previous, outgoing)
    return value, last_dir


def _empty_chain(n: int, device, real_dtype) -> dict[str, torch.Tensor]:
    """Zero-depth chain tensors (the ``d = 0`` degenerate legs)."""

    return {
        "positions": torch.zeros((n, 0, 3), device=device, dtype=real_dtype),
        "normals": torch.zeros((n, 0, 3), device=device, dtype=real_dtype),
        "eps_r": torch.zeros((n, 0), device=device, dtype=real_dtype),
        "sigma_e": torch.zeros((n, 0), device=device, dtype=real_dtype),
        "mu_r": torch.ones((n, 0), device=device, dtype=real_dtype),
        "gain": torch.ones((n, 0), device=device, dtype=real_dtype),
        "thickness": torch.zeros((n, 0), device=device, dtype=real_dtype),
        "sigma_b": torch.zeros((n, 0), device=device, dtype=real_dtype),
        "rough": torch.zeros((n, 0), device=device, dtype=torch.bool),
    }


def chain_ensemble_gain_reference(
    source: torch.Tensor,
    tx_pol: torch.Tensor,
    c1: dict[str, torch.Tensor],
    vertex: torch.Tensor,
    n_o: torch.Tensor,
    t1r: torch.Tensor,
    t2r: torch.Tensor,
    backup_axis: torch.Tensor,
    f_te: torch.Tensor,
    f_tm: torch.Tensor,
    c2: dict[str, torch.Tensor],
    target: torch.Tensor,
    rx_pol: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    weights: torch.Tensor,
    coef: torch.Tensor,
    frequency_hz: torch.Tensor,
    threshold: float,
) -> dict[str, torch.Tensor]:
    """Differentiable float64 ADR-021 Op A per-row power gain.

    ``c1``/``c2`` are dicts of chain tensors (keys ``positions``, ``normals``,
    ``eps_r``, ``sigma_e``, ``mu_r``, ``gain``, ``thickness``, ``sigma_b``,
    ``rough``; see :func:`transport_chain`); pass :func:`_empty_chain` for the
    degenerate legs. ``f_te``/``f_tm`` are the dense ``[Nti, Npi, Nto, Npo]``
    Kirchhoff table for the single test material.

    Live (differentiable) leaves per ADR-021 D5 Op A: chain Fresnel inputs
    (``eps_r``/``sigma_e``/``gain``/``thickness``), ``sigma_b`` (``C_r``),
    per-row chain geometry (``positions``/``normals``, ``source``, ``vertex``,
    ``target``, ``l1``, ``l2``), table values, ``coef``, ``frequency_hz``, and
    the polarization vectors. Fixed: ``t1r``/``t2r`` (surface tangent frame),
    ``backup_axis``, ``threshold``. The ``d1 = d2 = 0`` reduction returns the
    op-1 gain (self-check gate ``a``).

    Assembly (ADR-021 Op A steps 1-5):

    1. Chain-1 transports the tx field to ``v_s``; the incident coherency
       diagonal is ``P_te = |E_s|^2``, ``P_tm = |E_p|^2`` in the local s/p basis
       of the last C1 leg ``d_i``.
    2. Quadrilinear table lookup ``(f_te, f_tm) = T(wi_local, wo_local)``.
    3. Outgoing coherency diagonal ``J_out = diag(f_te*P_te, f_tm*P_tm)``.
    4. Chain-2 sandwich ``J_rx = A_2 J_out A_2^dagger`` then ``p_rx^H J_rx p_rx``.
       Because ``J_out`` is diagonal this equals ``f_te*P_te*|c_s|^2 +
       f_tm*P_tm*|c_p|^2`` where ``c_s``/``c_p`` are the receiver responses to
       unit outgoing ``s_o``/``p_o`` fields transported through C2 (columns of
       ``A_2^H p_rx``); no cross term survives (v1 diagonal table contract).
    5. ``gain = coef * (p^H J p) * cos_i * cos_o * A_patch / (L1^2 L2^2)`` with
       ``coef`` bundling ``P_t*lambda^2/(4pi)^2`` and ``weights = A_patch`` so
       the degenerate case matches op-1's ``coef``/``weights`` factoring.
    """

    real_dtype = source.dtype
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64

    # Chain 1: transport the transmit field to the vertex.
    first_leg = _safe_normalize(
        (c1["positions"][:, 0] if c1["positions"].shape[1] > 0 else vertex) - source,
        _unit_z(source),
    )
    e_tx = _project_transverse(tx_pol, first_leg).to(complex_dtype)
    e_in, d_i = transport_chain(
        e_tx, source, vertex, c1["positions"], c1["normals"], c1["eps_r"],
        c1["sigma_e"], c1["mu_r"], c1["gain"], c1["thickness"], c1["sigma_b"],
        c1["rough"], frequency_hz,
    )
    s_i, p_i = _sp_basis(n_o, d_i, backup_axis)
    p_te = _c3_dot_real(e_in, s_i).abs().square()
    p_tm = _c3_dot_real(e_in, p_i).abs().square()

    # Table lookup in the fixed surface tangent frame (op-1 convention: wi/wo
    # point away from the surface, so the horizon gate sees positive cosines).
    wi_hat = -d_i
    cos_i = (wi_hat * n_o).sum(-1)
    wi_local = torch.stack(
        ((wi_hat * t1r).sum(-1), (wi_hat * t2r).sum(-1), cos_i), dim=-1
    )
    d_o = _safe_normalize(
        (c2["positions"][:, 0] if c2["positions"].shape[1] > 0 else target) - vertex,
        _unit_z(source),
    )
    cos_o = (d_o * n_o).sum(-1)
    wo_local = torch.stack(
        ((d_o * t1r).sum(-1), (d_o * t2r).sum(-1), cos_o), dim=-1
    )
    f_te_row, f_tm_row = bsdf_table_interp(wi_local, wo_local, f_te, f_tm)

    # Chain 2: receiver responses to the two outgoing basis fields. |c_s|^2 and
    # |c_p|^2 are the diagonal of A_2^H p_rx p_rx^H A_2 (op-1's g_te2/g_tm2 when
    # C2 is empty).
    s_o, p_o = _sp_basis(n_o, d_o, backup_axis)
    field_s, last_dir = transport_chain(
        s_o.to(complex_dtype), vertex, target, c2["positions"], c2["normals"],
        c2["eps_r"], c2["sigma_e"], c2["mu_r"], c2["gain"], c2["thickness"],
        c2["sigma_b"], c2["rough"], frequency_hz,
    )
    field_p, _ = transport_chain(
        p_o.to(complex_dtype), vertex, target, c2["positions"], c2["normals"],
        c2["eps_r"], c2["sigma_e"], c2["mu_r"], c2["gain"], c2["thickness"],
        c2["sigma_b"], c2["rough"], frequency_hz,
    )
    rx_axis = _project_transverse(rx_pol, last_dir)
    g_te2 = _c3_dot_real(field_s, rx_axis).abs().square()
    g_tm2 = _c3_dot_real(field_p, rx_axis).abs().square()

    p_j_p = f_te_row * p_te * g_te2 + f_tm_row * p_tm * g_tm2
    gain = coef * p_j_p * cos_i * cos_o * weights / (l1.square() * l2.square())
    amplitude = gain.clamp_min(0.0).sqrt()
    length = l1 + l2
    keep = gain > threshold
    return {
        "gain": gain,
        "amplitude": amplitude,
        "length": length,
        "keep": keep,
        "p_te": p_te,
        "p_tm": p_tm,
        "g_te2": g_te2,
        "g_tm2": g_tm2,
        "f_te": f_te_row,
        "f_tm": f_tm_row,
        "cos_i": cos_i,
        "cos_o": cos_o,
        "d_i": d_i,
        "d_o": d_o,
        "wi_local": wi_local,
        "wo_local": wo_local,
    }


def _unit_z(reference: torch.Tensor) -> torch.Tensor:
    ez = torch.zeros_like(reference)
    ez[..., 2] = 1.0
    return ez