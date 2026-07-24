"""Kirchhoff ensemble BSDF tables (contract sections 6, 7.2).

Conventions (fixed here, used by every consumer):

- The table lives in the LOCAL roughness frame: ``z`` is the mean-plane
  normal, ``x``/``y`` are the roughness principal axes (``corr_length_x_m``
  along ``x``). Callers rotate world directions into this frame using the
  stored ``principal_axis_rad`` before evaluating.
- ``wi``/``wo`` are unit vectors POINTING AWAY from the surface
  (``z > 0``): ``wi`` toward the source, ``wo`` toward the observation
  direction. The incident propagation direction is ``-wi``, so the
  scattering wave-vector transfer is ``q = k_s - k_i = k0*(wo + wi)`` with
  normal component ``q_n = k0*(cos_theta_o + cos_theta_i) > 0``.
- ``f_te``/``f_tm`` are co-pol POWER BSDF values per steradian, defined so
  that ``sum_hemisphere f * |cos_theta_o| dOmega = R_diff`` for each
  incidence bin (exact on the table's own outgoing grid, by construction).

Raw lobe shape (before normalization)::

    f_raw_q(wi, wo) = |q|^4 / (16*pi^2 * q_n^2 * cos_theta_i * cos_theta_o)
                      * |r_q_stack(cos_theta_h)|^2 * I(q_par, q_n)

where ``I`` is the production Beckmann series implemented in this module
and the polarized Fresnel factor is evaluated at the SPECULAR-EQUIVALENT
local incidence angle onto the half vector
``h = normalize(wo - wi_dir) = normalize(wo + wi)``::

    cos_theta_h = wi . h = wo . h = (1 + wi . wo) / |wo + wi|

Documented choice: ``wi . h`` is the angle between the incident ray and
the micro-plane normal that specularly maps ``wi`` to ``wo``; it equals
``cos_theta_i`` at the lobe peak, so the kernel reproduces the exact
stack reflectance ``R_bar_q(theta_i)`` there and tracks its angular /
Fabry-Perot structure across the lobe. (Evaluating at ``|h . n|`` instead
would sample the stack at NORMAL incidence for every near-specular pair
and misses the budget by ``R_bar(theta_i)/R_bar(0)``  -  measured up to 4x  - 
so it cannot satisfy the normalization tolerance band.) ``wi . h`` is
exactly reciprocal, as are ``q`` and the geometry prefactor.

Derivation of the prefactor (standard scalar Kirchhoff): the tangent-plane
aperture integral gives ``E_s = (j*k0/(4*pi*R)) * F * r *
Int exp(-j*q.x - j*q_n*h) dA`` with the Kirchhoff geometry factor
``F = |q|^2/(k0*q_n)`` (Beckmann's ``F`` rescaled so a smooth plate
conserves energy: ``F -> 2*cos_theta_i`` at specular). Ensemble-averaging
the diffuse part gives ``<|Int|^2> = A * I(q)``, and dividing the scattered
power density by ``A * cos_theta_i * cos_theta_o`` (BSDF flux convention)
yields the expression above. Its hemispheric energy: substituting
``d^2 q_par = k0^2 * cos_theta_o dOmega_o`` and Parseval
``(1/(2*pi)^2) Int I(q_par; q_n) d^2 q_par = 1 - exp(-g)`` shows
``Int f_raw*cos dOmega -> R_bar*(1 - C_r^2) = R_diff`` near specular
(``exp(-g) = C_r^2`` at ``q_n = 2*k0*cos_theta_i``). Residual horizon,
Fresnel-variation and grid errors are removed by reciprocal symmetric matrix
balancing. Its diagonal factor acts on both incident and outgoing states, so
the final production table retains Helmholtz reciprocity while matching every
directional diffuse-energy budget.

Energy accounting is exact on the discrete outgoing grid: the hemisphere
sum of ``f * cos * dOmega`` over the table bins equals ``r_diff`` bitwise
(up to float32 rounding), which is also the measure the sampling CDFs use.

Precompute runs in float64 numpy on CPU; the frozen table holds float32
torch tensors on the requested device. Runtime eval/sample/pdf are pure
batched torch (GPU-first).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch

from witwin.channel.materials.evaluation import layer_stack_rt
from witwin.channel.constants import C0

__all__ = [
    "KirchhoffTable",
    "build_kirchhoff_table",
    "eval_bsdf",
    "pdf",
    "pdf_reverse",
    "sample_directions",
]

# Grid resolution fixed by the implementation contract (section 6).
N_COS_THETA_I = 32
# An anisotropic reciprocal table must use the same directional state grid on
# both sides of f(wi, wo).  A coarser incidence azimuth grid followed by a
# per-incidence normalization cannot preserve reciprocity.  64 keeps the
# contract's outgoing resolution and makes the discrete transport matrix
# square, so symmetric balancing below can enforce both invariants exactly.
N_PHI_I_ANISO = 64
N_COS_THETA_O = 32
N_PHI_O = 64

# Applicability guards (contract section 6): tangent-plane approximation
# needs k0*l >= ~6 and moderate RMS slope sqrt(2)*sigma_h/l <= 0.5.
MIN_K0_CORR_LENGTH = 6.0
MAX_RMS_SLOPE = 0.5


def _kirchhoff_diffuse_lobe_series(
    q_par_x, q_par_y, q_n, sigma_h, lx, ly, n_terms: int = 64
):
    """Production Beckmann series for the Gaussian-correlation lobe."""

    qx, qy, qn = np.broadcast_arrays(
        np.asarray(q_par_x, dtype=np.float64),
        np.asarray(q_par_y, dtype=np.float64),
        np.asarray(q_n, dtype=np.float64),
    )
    g = (qn * float(sigma_h)) ** 2
    rho2 = (qx * lx) ** 2 + (qy * ly) ** 2
    m_flat = np.arange(1, n_terms + 1, dtype=np.float64)
    shape = (n_terms,) + (1,) * g.ndim
    m = m_flat.reshape(shape)
    log_fact = np.cumsum(np.log(m_flat)).reshape(shape)
    with np.errstate(divide="ignore"):
        log_g = np.log(g)
    log_term = m * log_g - log_fact - np.log(m) - rho2 / (4.0 * m) - g
    series = np.exp(log_term).sum(axis=0)
    result = np.pi * lx * ly * series
    return result if result.ndim else float(result)

@dataclass(frozen=True, slots=True)
class KirchhoffTable:
    """Precomputed Kirchhoff ensemble BSDF for one rough material.

    All tensors are float32 on one device. Axis tensors hold CELL CENTERS:
    ``cos_theta_i``/``cos_theta_o`` are uniform in cos over (0, 1],
    ``phi_i``/``phi_o`` uniform over [0, 2*pi). ``phi_i`` has one entry for
    isotropic roughness (``lx == ly``).
    """

    # Axes (cell centers).
    cos_theta_i: torch.Tensor  # [Nti]
    phi_i: torch.Tensor  # [Nphi_i] (1 for isotropic)
    cos_theta_o: torch.Tensor  # [Nto]
    phi_o: torch.Tensor  # [Npo]
    # Co-pol power BSDF channels [Nti, Nphi_i, Nto, Npo].
    f_te: torch.Tensor
    f_tm: torch.Tensor
    # Diffuse reflection budgets per incidence bin [Nti, Nphi_i].
    r_diff_te: torch.Tensor
    r_diff_tm: torch.Tensor
    r_diff_unpol: torch.Tensor
    # Symmetric matrix-balance factor a(w) [Nti, Nphi_i, 2], channel order
    # TE/TM.  The final table is f(wi,wo)=a(wi)*f_sym(wi,wo)*a(wo), which
    # preserves reciprocity while enforcing every row's energy budget.
    normalization_applied: torch.Tensor
    # Sampling tables built from the UNPOLARIZED mean lobe:
    # per-bin probability mass and the per-solid-angle density (mass/dOmega),
    # marginal CDF over cos_theta_o, conditional CDF over phi_o.
    sample_density: torch.Tensor  # [Nti, Nphi_i, Nto, Npo]
    marginal_cdf: torch.Tensor  # [Nti, Nphi_i, Nto]
    conditional_cdf: torch.Tensor  # [Nti, Nphi_i, Nto, Npo]
    # Domain metadata / validity flags.
    frequency_hz: float
    k0: float
    sigma_h_m: float
    corr_x_m: float
    corr_y_m: float
    principal_axis_rad: float
    anisotropic: bool
    k0_l_min: float
    rms_slope_max: float
    tangent_plane_ok: bool
    slope_ok: bool
    reciprocity_error: float
    # ADR-015 Part C differentiable-build intermediates. The float64 numpy
    # build is unchanged bit-for-bit; these are the exact f32 downcasts of the
    # structural quantities the native table-build adjoint recomputes against
    # (no f32 recompute drift). ``pre_balance_lobe_*`` are the reciprocity-
    # symmetrized raw lobes ``S`` BEFORE the diagonal energy balance
    # (``F = a S a``); the balance factors ``a`` and the diffuse budgets are
    # already exposed as ``normalization_applied`` and ``r_diff_te``/
    # ``r_diff_tm``. All default to ``None`` so a table built without the AD
    # path (e.g. a bare numpy import) carries no extra state.
    pre_balance_lobe_te: torch.Tensor | None = None  # [Nti, Npi, Nto, Npo]
    pre_balance_lobe_tm: torch.Tensor | None = None  # [Nti, Npi, Nto, Npo]

    @property
    def device(self) -> torch.device:
        return self.f_te.device

    @property
    def bin_solid_angle(self) -> float:
        """Solid angle of one outgoing bin: d(cos_theta) * d(phi)."""

        return (1.0 / N_COS_THETA_O) * (2.0 * math.pi / N_PHI_O)


def _cos_centers(n: int) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) + 0.5) / n


def _phi_centers(n: int) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) + 0.5) * (2.0 * np.pi / n)


def _stack_power_reflectances(
    layers: Sequence[tuple], cos_theta: np.ndarray, frequency_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(|r_te_stack|^2, |r_tm_stack|^2)`` at real incidence cosines."""

    rt = layer_stack_rt(layers, cos_theta, frequency_hz)
    return np.abs(rt.r_te) ** 2, np.abs(rt.r_tm) ** 2


def _raw_lobe_grid(
    layers: Sequence[tuple],
    frequency_hz: float,
    k0: float,
    sigma_h: float,
    lx: float,
    ly: float,
    n_terms: int,
    inc_cos: np.ndarray,
    inc_phi: np.ndarray,
    out_cos: np.ndarray,
    out_phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw (un-normalized) TE/TM lobe on an (incidence x outgoing) grid.

    Returns arrays ``[len(inc_cos), len(inc_phi), len(out_cos),
    len(out_phi)]``. Used twice per build: once on the table axes and once
    with incidence/outgoing axes swapped (the exact grid-resample of
    ``f(wo, wi)``  -  the analytic kernel is reciprocal, so re-evaluating at
    the swapped nodes is the lossless way to symmetrize; interpolating the
    coarse ``phi_i`` axis would smear the specular peak instead).
    """

    sin_inc = np.sqrt(np.maximum(0.0, 1.0 - inc_cos**2))
    sin_out = np.sqrt(np.maximum(0.0, 1.0 - out_cos**2))
    wo_x = sin_out[:, None] * np.cos(out_phi)[None, :]
    wo_y = sin_out[:, None] * np.sin(out_phi)[None, :]
    wo_z = np.broadcast_to(out_cos[:, None], wo_x.shape)
    f_te = np.empty((inc_cos.shape[0], inc_phi.shape[0], *wo_x.shape))
    f_tm = np.empty_like(f_te)
    for ti in range(inc_cos.shape[0]):
        for pi_idx in range(inc_phi.shape[0]):
            wi_x = sin_inc[ti] * np.cos(inc_phi[pi_idx])
            wi_y = sin_inc[ti] * np.sin(inc_phi[pi_idx])
            wi_z = inc_cos[ti]
            # q = k_s - k_i = k0*(wo + wi); components already in the
            # roughness principal frame (the table's local frame).
            qx = k0 * (wo_x + wi_x)
            qy = k0 * (wo_y + wi_y)
            qn = k0 * (wo_z + wi_z)
            lobe = _kirchhoff_diffuse_lobe_series(
                qx, qy, qn, sigma_h, lx, ly, n_terms=n_terms
            )
            # Specular-equivalent local incidence: cos_theta_h = wi.h with
            # h || (wo + wi) (module docstring); |wo + wi| = |q|/k0.
            q_sq = qx**2 + qy**2 + qn**2
            wi_dot_wo = wo_x * wi_x + wo_y * wi_y + wo_z * wi_z
            cos_h = np.clip((1.0 + wi_dot_wo) * k0 / np.sqrt(q_sq), 1e-6, 1.0)
            rr_te, rr_tm = _stack_power_reflectances(
                layers, cos_h.reshape(-1), frequency_hz
            )
            # Standard Kirchhoff geometry prefactor (see module docstring):
            # |q|^4 / (16*pi^2*q_n^2*cos_i*cos_o); reduces to k0^2/(4*pi^2)
            # at the specular direction and is symmetric under wi <-> wo.
            prefactor = q_sq**2 / (16.0 * np.pi**2 * qn**2 * wi_z * wo_z)
            shape = prefactor * lobe
            f_te[ti, pi_idx] = shape * rr_te.reshape(shape.shape)
            f_tm[ti, pi_idx] = shape * rr_tm.reshape(shape.shape)
    return f_te, f_tm


def _symmetric_energy_balance(
    symmetric_lobe: np.ndarray,
    target: np.ndarray,
    cos_o: np.ndarray,
    *,
    isotropic: bool,
    tolerance: float = 1e-11,
    max_iterations: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance a reciprocal nonnegative kernel without breaking symmetry.

    Finds positive diagonal factors ``a`` such that ``F_ij=a_i*S_ij*a_j``
    and every cosine-weighted row integral of ``F`` equals ``target_i``.
    This is the symmetric Sinkhorn fixed point with a half-step damping;
    unlike one-sided row normalization, reciprocity is invariant at every
    iteration.
    """

    n_ti, n_pi, n_to, n_po = symmetric_lobe.shape
    if n_ti != n_to:
        raise ValueError("reciprocal balance requires identical cos grids")
    d_omega = (1.0 / n_to) * (2.0 * np.pi / n_po)
    outgoing_weight = np.broadcast_to(
        cos_o[:, None] * d_omega, (n_to, n_po)
    ).reshape(-1)

    if isotropic:
        # Rotation invariance collapses the incident azimuth state.  The
        # reverse state factor depends only on cos(theta_o), while the full
        # relative-azimuth lobe remains in the row integral.
        s = symmetric_lobe[:, 0]
        rhs = target[:, 0]
        factor = np.ones(n_ti, dtype=np.float64)
        active = rhs > 0.0
        for _ in range(max_iterations):
            weighted = outgoing_weight * np.repeat(factor, n_po)
            denom = (s * weighted[None, :, None].reshape(1, n_to, n_po)).sum(
                axis=(1, 2)
            )
            ratio = np.ones_like(factor)
            ratio[active] = rhs[active] / np.maximum(
                factor[active] * denom[active], 1e-300
            )
            factor *= np.sqrt(ratio)
            factor[~active] = 0.0
            if np.max(np.abs(np.log(np.maximum(ratio[active], 1e-300))), initial=0.0) < tolerance:
                break
        else:
            raise ValueError("symmetric Kirchhoff energy balance did not converge")
        balanced = s * factor[:, None, None] * factor[None, :, None]
        return balanced[:, None], factor[:, None]

    if n_pi != n_po:
        raise ValueError("anisotropic reciprocal balance requires identical phi grids")
    states = n_ti * n_pi
    s = symmetric_lobe.reshape(states, states)
    rhs = target.reshape(states)
    weights = np.repeat(cos_o * d_omega, n_po)
    factor = np.ones(states, dtype=np.float64)
    active = rhs > 0.0
    for _ in range(max_iterations):
        denom = s @ (weights * factor)
        ratio = np.ones_like(factor)
        ratio[active] = rhs[active] / np.maximum(
            factor[active] * denom[active], 1e-300
        )
        factor *= np.sqrt(ratio)
        factor[~active] = 0.0
        if np.max(np.abs(np.log(np.maximum(ratio[active], 1e-300))), initial=0.0) < tolerance:
            break
    else:
        raise ValueError("symmetric Kirchhoff energy balance did not converge")
    balanced = s * factor[:, None] * factor[None, :]
    return balanced.reshape(symmetric_lobe.shape), factor.reshape(n_ti, n_pi)


def build_kirchhoff_table(
    roughness,
    layers: Sequence[tuple],
    frequency_hz: float,
    device: torch.device | str = "cuda",
) -> KirchhoffTable:
    """Precompute the Kirchhoff ensemble BSDF table for one material.

    ``roughness`` is a :class:`witwin.core.SurfaceRoughness`
    (or any object with the same fields); ``layers`` is the oracle layer
    list ``[(thickness_m, eps_r, sigma_e, mu_r), ...]`` in incidence order.
    Raises when the surface is outside the Kirchhoff applicability domain
    (``kirchhoff_domain_exceeded``) or reciprocal energy balancing fails to
    converge.
    """

    frequency_hz = float(frequency_hz)
    sigma_h = float(roughness.rms_height_m)
    lx = float(roughness.correlation_length_x_m)
    ly = float(roughness.correlation_length_y_m)
    axis_rad = float(roughness.principal_axis_rad)
    k0 = 2.0 * math.pi * frequency_hz / C0

    l_min = min(lx, ly)
    k0_l_min = k0 * l_min
    rms_slope_max = math.sqrt(2.0) * sigma_h / l_min
    tangent_plane_ok = k0_l_min >= MIN_K0_CORR_LENGTH
    slope_ok = rms_slope_max <= MAX_RMS_SLOPE
    if not (tangent_plane_ok and slope_ok):
        raise ValueError(
            "kirchhoff_domain_exceeded: ensemble Kirchhoff requires "
            f"k0*corr_length >= {MIN_K0_CORR_LENGTH:g} "
            f"(got {k0_l_min:.3g}) and RMS slope sqrt(2)*sigma_h/l <= "
            f"{MAX_RMS_SLOPE:g} (got {rms_slope_max:.3g})"
        )

    anisotropic = lx != ly
    cos_i = _cos_centers(N_COS_THETA_I)
    phi_i = _phi_centers(N_PHI_I_ANISO) if anisotropic else np.zeros(1)
    cos_o = _cos_centers(N_COS_THETA_O)
    phi_o = _phi_centers(N_PHI_O)
    n_pi = phi_i.shape[0]

    # 1) Smooth-stack budgets on the incidence grid (complex128 oracle).
    r_bar_te, r_bar_tm = _stack_power_reflectances(layers, cos_i, frequency_hz)
    # 2) Coherent part: |r_stack * C_r|^2 = R_bar * C_r^2.
    c_r = np.exp(-2.0 * (k0 * cos_i * sigma_h) ** 2)
    r_coh_te = r_bar_te * c_r**2
    r_coh_tm = r_bar_tm * c_r**2
    # 3) Diffuse budgets (>= 0 since C_r <= 1; max() guards fp rounding).
    r_diff_te = np.maximum(0.0, r_bar_te - r_coh_te)
    r_diff_tm = np.maximum(0.0, r_bar_tm - r_coh_tm)

    # 4) Raw lobe on the 4D grid.
    g_max = (2.0 * k0 * sigma_h) ** 2
    n_terms = int(max(64, g_max + 12.0 * math.sqrt(g_max) + 16.0))
    f_raw_te, f_raw_tm = _raw_lobe_grid(
        layers, frequency_hz, k0, sigma_h, lx, ly, n_terms,
        cos_i, phi_i, cos_o, phi_o,
    )

    # 5) Reciprocity symmetrization BEFORE normalization: f(wo, wi) is
    # obtained by exact re-evaluation on the swapped grid nodes (see
    # _raw_lobe_grid), then averaged with the forward evaluation. The
    # residual measures genuine wi<->wo asymmetry of the implementation
    # (the kernel is analytically reciprocal), so anything above float
    # rounding is a bug.
    swap_te, swap_tm = _raw_lobe_grid(
        layers, frequency_hz, k0, sigma_h, lx, ly, n_terms,
        cos_o, phi_o, cos_i, phi_i,
    )
    swap_te = np.transpose(swap_te, (2, 3, 0, 1))
    swap_tm = np.transpose(swap_tm, (2, 3, 0, 1))
    reciprocity_error = 0.0
    for forward, swapped in ((f_raw_te, swap_te), (f_raw_tm, swap_tm)):
        peak = float(forward.max())
        if peak > 0.0:
            err = float(np.abs(forward - swapped).max() / peak)
            reciprocity_error = max(reciprocity_error, err)
    if reciprocity_error >= 1e-3:
        raise ValueError(
            "kirchhoff table reciprocity error "
            f"{reciprocity_error:.3e} exceeds 1e-3 after symmetrization"
        )
    f_sym_te = 0.5 * (f_raw_te + swap_te)
    f_sym_tm = 0.5 * (f_raw_tm + swap_tm)

    # ADR-015 Part C: snapshot the pre-balance symmetrized lobes S before the
    # in-place diagonal energy balance below overwrites f_sym. The native
    # table-build adjoint consumes these (with a, r_diff) as its saved
    # intermediates; the numpy primal is unaffected.
    pre_balance_lobe_te = f_sym_te.copy()
    pre_balance_lobe_tm = f_sym_tm.copy()

    # 6) Symmetric energy balance on the discrete directional state matrix.
    # A one-sided row scale would make the energy exact but destroy
    # f(wi,wo)==f(wo,wi).  Diagonal scaling on both arguments preserves the
    # already-symmetric raw kernel and satisfies every row budget jointly.
    d_omega = (1.0 / N_COS_THETA_O) * (2.0 * np.pi / N_PHI_O)
    weight = cos_o[None, None, :, None] * d_omega  # broadcast over [.., to, po]
    r_diff_te_grid = np.broadcast_to(r_diff_te[:, None], (N_COS_THETA_I, n_pi)).copy()
    r_diff_tm_grid = np.broadcast_to(r_diff_tm[:, None], (N_COS_THETA_I, n_pi)).copy()
    scales = np.ones((N_COS_THETA_I, n_pi, 2))
    channels = (
        (0, f_sym_te, r_diff_te_grid),
        (1, f_sym_tm, r_diff_tm_grid),
    )
    for channel, f_sym, r_diff in channels:
        balanced, factor = _symmetric_energy_balance(
            f_sym, r_diff, cos_o, isotropic=not anisotropic
        )
        f_sym[...] = balanced
        integral = (f_sym * weight).sum(axis=(2, 3))
        active = r_diff > 0.0
        relative_error = np.zeros_like(integral)
        relative_error[active] = np.abs(integral[active] - r_diff[active]) / r_diff[active]
        if relative_error.max(initial=0.0) > 2e-9:
            raise ValueError(
                "symmetric Kirchhoff balance failed energy tolerance: "
                f"max relative error {relative_error.max():.3e}"
            )
        # Store the one-direction diagonal factor.  Unlike the obsolete
        # one-sided row scale, its magnitude alone is not a shape-error
        # metric: the physical correction on a pair is a(wi)*a(wo), and the
        # factors are jointly constrained by all directional energy rows.
        scales[:, :, channel] = factor

    # 7) Sampling tables from the UNPOLARIZED mean lobe.
    f_unpol = 0.5 * (f_sym_te + f_sym_tm)
    mass = f_unpol * weight  # [ti, pi, to, po] probability mass per bin
    total = mass.sum(axis=(2, 3), keepdims=True)
    uniform = np.full_like(mass, 1.0 / (N_COS_THETA_O * N_PHI_O))
    mass = np.where(total > 0.0, mass / np.where(total > 0.0, total, 1.0), uniform)
    density = mass / d_omega
    marginal = mass.sum(axis=3)  # [ti, pi, to]
    marginal_cdf = np.cumsum(marginal, axis=2)
    marginal_cdf /= marginal_cdf[..., -1:]
    cond = np.where(
        marginal[..., None] > 0.0,
        mass / np.where(marginal[..., None] > 0.0, marginal[..., None], 1.0),
        1.0 / N_PHI_O,
    )
    conditional_cdf = np.cumsum(cond, axis=3)
    conditional_cdf /= conditional_cdf[..., -1:]

    device = torch.device(device)

    def as32(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)

    return KirchhoffTable(
        cos_theta_i=as32(cos_i),
        phi_i=as32(phi_i),
        cos_theta_o=as32(cos_o),
        phi_o=as32(phi_o),
        f_te=as32(f_sym_te),
        f_tm=as32(f_sym_tm),
        r_diff_te=as32(r_diff_te_grid),
        r_diff_tm=as32(r_diff_tm_grid),
        r_diff_unpol=as32(0.5 * (r_diff_te_grid + r_diff_tm_grid)),
        normalization_applied=as32(scales),
        sample_density=as32(density),
        marginal_cdf=as32(marginal_cdf),
        conditional_cdf=as32(conditional_cdf),
        frequency_hz=frequency_hz,
        k0=k0,
        sigma_h_m=sigma_h,
        corr_x_m=lx,
        corr_y_m=ly,
        principal_axis_rad=axis_rad,
        anisotropic=anisotropic,
        k0_l_min=k0_l_min,
        rms_slope_max=rms_slope_max,
        tangent_plane_ok=tangent_plane_ok,
        slope_ok=slope_ok,
        reciprocity_error=reciprocity_error,
        pre_balance_lobe_te=as32(pre_balance_lobe_te),
        pre_balance_lobe_tm=as32(pre_balance_lobe_tm),
    )


def eval_bsdf(
    table: KirchhoffTable,
    valid: torch.Tensor,
    wi: torch.Tensor,
    wo: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear (multilinear) lookup of ``(f_te, f_tm)`` for batched pairs.

    ``wi``/``wo`` are [N, 3] local-frame unit vectors pointing away from the
    surface. Directions below the horizon return 0.
    """

    from witwin.channel.scattering.kernels.functional import (
        scattering_table_eval,
    )

    return scattering_table_eval(
        valid.contiguous(), wi.contiguous(), wo.contiguous(), table.f_te, table.f_tm
    )


def sample_directions(
    table: KirchhoffTable,
    valid: torch.Tensor,
    wi: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample outgoing directions by CDF inversion; returns ``(wo, pdf)``.

    Uses the nearest incidence bin (the same convention :func:`pdf` uses, so
    the sampler and its density are exactly consistent). ``u1`` inverts the
    marginal CDF over ``cos_theta_o``; ``u2`` the conditional CDF over
    ``phi_o``; both are linearly remapped inside the selected bin, i.e. the
    sampled density is piecewise constant per outgoing bin.
    """

    from witwin.channel.scattering.kernels.functional import (
        scattering_table_sample,
    )

    uniforms = torch.stack((u1, u2), dim=1).contiguous()
    out = scattering_table_sample(
        valid.contiguous(),
        wi.contiguous(),
        uniforms,
        table.marginal_cdf,
        table.conditional_cdf,
        table.sample_density,
    )
    return out["wo"], out["pdf_forward"]


def pdf(
    table: KirchhoffTable,
    valid: torch.Tensor,
    wi: torch.Tensor,
    wo: torch.Tensor,
) -> torch.Tensor:
    """Solid-angle sampling density of :func:`sample_directions`.

    Piecewise constant per outgoing bin and exactly consistent with the
    sampler (same nearest incidence bin, same mass table); integrates to 1
    over the hemisphere by construction. Zero below the horizon.
    """

    from witwin.channel.scattering.kernels.functional import scattering_table_pdf

    return scattering_table_pdf(
        valid.contiguous(),
        wi.contiguous(),
        wo.contiguous(),
        table.sample_density,
        reverse=False,
    )


def pdf_reverse(
    table: KirchhoffTable,
    valid: torch.Tensor,
    wo: torch.Tensor,
    wi: torch.Tensor,
) -> torch.Tensor:
    """Reverse-direction PDF: the SAME table evaluated with swapped args.

    BDPT evaluates the reverse strategy density by treating the outgoing
    direction as the incidence direction (contract section 5); no separate
    reverse table exists.
    """

    from witwin.channel.scattering.kernels.functional import scattering_table_pdf

    return scattering_table_pdf(
        valid.contiguous(),
        wi.contiguous(),
        wo.contiguous(),
        table.sample_density,
        reverse=True,
    )
