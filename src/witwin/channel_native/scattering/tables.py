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

where ``I`` is the Beckmann series
(:func:`witwin.channel_native.physics.oracle.kirchhoff_diffuse_lobe_series`)
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
(``exp(-g) = C_r^2`` at ``q_n = 2*k0*cos_theta_i``). The residual error
(lobe leakage below the horizon, ``q_n``/Fresnel variation across the
lobe, grid discretization) is removed by an explicit per-incidence-bin
normalization; the recorded scale factor must stay inside [0.25, 4] or the
build raises (a wrong SHAPE cannot hide behind renormalization).

The tolerance band is enforced where the split into coherent budget and
diffuse lobe is physically meaningful: incidence bins with
``cos_theta_i >= 0.15`` and ``r_diff > 5e-3``. Two documented exemptions:
(a) extreme grazing, where the tangent-plane kernel (no shadowing in v1)
over-predicts diffuse scatter while the coherent attenuation ``C_r -> 1``
sends the budget ``R_diff -> 0``; (b) bins whose diffuse budget is below
0.5% of the incident power  -  sharp Brewster/interference dips of the
smooth-stack budget that a lobe of finite angular width cannot follow
(and near-smooth surfaces whose lobe is narrower than a grid cell). Both
kinds of bins still receive the exact-energy normalization, so an O(1)
shape error there contributes well under 0.5% absolute energy error.
Energy stays budget-true by construction everywhere.

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

from witwin.channel_native.physics.oracle import (
    C0,
    kirchhoff_diffuse_lobe_series,
    layer_stack_rt,
)

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
N_PHI_I_ANISO = 16
N_COS_THETA_O = 32
N_PHI_O = 64

# Applicability guards (contract section 6): tangent-plane approximation
# needs k0*l >= ~6 and moderate RMS slope sqrt(2)*sigma_h/l <= 0.5.
MIN_K0_CORR_LENGTH = 6.0
MAX_RMS_SLOPE = 0.5

# Normalization tolerance band for the shape-only prefactor and the domain
# on which it is enforced (see module docstring): bins at extreme grazing
# incidence or with a negligible diffuse budget are exact-energy normalized
# but exempt from the shape check.
NORMALIZATION_BAND = (0.25, 4.0)
BAND_COS_THETA_MIN = 0.15
BAND_R_DIFF_MIN = 5e-3


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
    # Per-bin normalization scale applied to the raw lobe [Nti, Nphi_i, 2]
    # (channel order: TE, TM). Must lie in [0.25, 4] wherever r_diff is
    # above the floor; recorded for diagnostics and shape regression tests.
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
            lobe = kirchhoff_diffuse_lobe_series(
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


def build_kirchhoff_table(
    roughness,
    layers: Sequence[tuple],
    frequency_hz: float,
    device: torch.device | str = "cuda",
) -> KirchhoffTable:
    """Precompute the Kirchhoff ensemble BSDF table for one material.

    ``roughness`` is a :class:`witwin.channel_native.core.materials.Roughness`
    (or any object with the same fields); ``layers`` is the oracle layer
    list ``[(thickness_m, eps_r, sigma_e, mu_r), ...]`` in incidence order.
    Raises when the surface is outside the Kirchhoff applicability domain
    (``kirchhoff_domain_exceeded``) or when the shape-only prefactor needs a
    normalization outside [0.25, 4] (modeling bug, never silently rescaled).
    """

    frequency_hz = float(frequency_hz)
    sigma_h = float(roughness.rms_height_m)
    lx = float(roughness.corr_length_x_m)
    ly = float(roughness.corr_length_y_m)
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

    # 6) Exact per-incidence-bin energy normalization on the discrete grid.
    d_omega = (1.0 / N_COS_THETA_O) * (2.0 * np.pi / N_PHI_O)
    weight = cos_o[None, None, :, None] * d_omega  # broadcast over [.., to, po]
    r_diff_te_grid = np.broadcast_to(r_diff_te[:, None], (N_COS_THETA_I, n_pi)).copy()
    r_diff_tm_grid = np.broadcast_to(r_diff_tm[:, None], (N_COS_THETA_I, n_pi)).copy()
    scales = np.ones((N_COS_THETA_I, n_pi, 2))
    channels = (
        (0, f_sym_te, r_diff_te_grid),
        (1, f_sym_tm, r_diff_tm_grid),
    )
    band_cos = np.broadcast_to(cos_i[:, None], (N_COS_THETA_I, n_pi))
    for channel, f_sym, r_diff in channels:
        integral = (f_sym * weight).sum(axis=(2, 3))
        active = (r_diff > 0.0) & (integral > 0.0)
        scale = np.ones_like(integral)
        scale[active] = r_diff[active] / integral[active]
        # Shape check on the physically meaningful subdomain only (module
        # docstring); the exact-energy normalization itself applies to
        # every bin with a nonzero budget.
        checked = active & (band_cos >= BAND_COS_THETA_MIN) & (r_diff > BAND_R_DIFF_MIN)
        if checked.any():
            lo, hi = scale[checked].min(), scale[checked].max()
            if lo < NORMALIZATION_BAND[0] or hi > NORMALIZATION_BAND[1]:
                raise ValueError(
                    "kirchhoff lobe normalization outside the "
                    f"[{NORMALIZATION_BAND[0]:g}, {NORMALIZATION_BAND[1]:g}] "
                    f"tolerance band (range [{lo:.3g}, {hi:.3g}]): the "
                    "shape-only prefactor is wrong, refusing to rescale"
                )
        f_sym *= scale[:, :, None, None]
        # Zero the lobe where the budget is exactly zero so the table can
        # never return diffuse energy that the budget does not grant.
        f_sym *= (r_diff > 0.0)[:, :, None, None]
        scales[:, :, channel] = scale

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
    )


def _angles(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos_theta, phi in [0, 2*pi))`` of local-frame unit vectors [N, 3]."""

    cos_theta = w[:, 2]
    phi = torch.atan2(w[:, 1], w[:, 0])
    phi = torch.where(phi < 0.0, phi + 2.0 * math.pi, phi)
    return cos_theta, phi


def _linear_axis(
    coord: torch.Tensor, n: int, step: float, periodic: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear interpolation indices/weight on a cell-center axis.

    Centers sit at ``(i + 0.5)*step``. Non-periodic axes clamp to the outer
    centers (constant extrapolation); periodic axes wrap.
    """

    t = coord / step - 0.5
    if n == 1:
        zeros = torch.zeros_like(coord, dtype=torch.long)
        return zeros, zeros, torch.zeros_like(coord)
    if periodic:
        i0 = torch.floor(t)
        w = t - i0
        i0 = i0.long() % n
        i1 = (i0 + 1) % n
        return i0, i1, w
    t = t.clamp(0.0, float(n - 1))
    i0 = torch.floor(t).clamp(max=float(n - 2)).long()
    w = t - i0.to(t.dtype)
    return i0, i0 + 1, w


def _nearest_axis(coord: torch.Tensor, n: int, step: float, periodic: bool) -> torch.Tensor:
    """Nearest cell-center index on the same axis layout."""

    idx = torch.floor(coord / step).long()
    if periodic:
        return idx % n
    return idx.clamp(0, n - 1)


def _interp4(
    table: torch.Tensor,
    ti: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    pi_: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    to: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    po: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Multilinear interpolation of a [Nti, Npi, Nto, Npo] table (batched)."""

    n_pi, n_to, n_po = table.shape[1], table.shape[2], table.shape[3]
    flat = table.reshape(-1)
    out = torch.zeros_like(ti[2])
    for a, wa in ((ti[0], 1.0 - ti[2]), (ti[1], ti[2])):
        for b, wb in ((pi_[0], 1.0 - pi_[2]), (pi_[1], pi_[2])):
            for c, wc in ((to[0], 1.0 - to[2]), (to[1], to[2])):
                for d, wd in ((po[0], 1.0 - po[2]), (po[1], po[2])):
                    idx = ((a * n_pi + b) * n_to + c) * n_po + d
                    out = out + (wa * wb * wc * wd) * flat[idx]
    return out


def _lookup_angles(
    table: KirchhoffTable, wi: torch.Tensor, wo: torch.Tensor | None
):
    """Table-frame lookup angles shared by eval/sample/pdf.

    Returns ``(cos_i, phi_i, cos_o, phi_o_rel)``. Isotropic tables are
    built with incidence azimuth 0, so the outgoing azimuth axis stores the
    RELATIVE azimuth: lookups subtract ``phi_i`` (and the sampler adds it
    back), which makes the isotropic table exactly rotation-invariant.
    Anisotropic tables store absolute azimuths in the principal frame.
    """

    cos_i, phi_i = _angles(wi)
    if wo is None:
        return cos_i, phi_i, None, None
    cos_o, phi_o = _angles(wo)
    if table.phi_i.shape[0] == 1:
        phi_o = torch.remainder(phi_o - phi_i, 2.0 * math.pi)
    return cos_i, phi_i, cos_o, phi_o


def eval_bsdf(
    table: KirchhoffTable, wi: torch.Tensor, wo: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear (multilinear) lookup of ``(f_te, f_tm)`` for batched pairs.

    ``wi``/``wo`` are [N, 3] local-frame unit vectors pointing away from the
    surface. Directions below the horizon return 0.
    """

    n_pi = table.phi_i.shape[0]
    cos_i, phi_i, cos_o, phi_o = _lookup_angles(table, wi, wo)
    ti = _linear_axis(cos_i, N_COS_THETA_I, 1.0 / N_COS_THETA_I, False)
    pi_ = _linear_axis(phi_i, n_pi, 2.0 * math.pi / n_pi, True)
    to = _linear_axis(cos_o, N_COS_THETA_O, 1.0 / N_COS_THETA_O, False)
    po = _linear_axis(phi_o, N_PHI_O, 2.0 * math.pi / N_PHI_O, True)
    valid = (wi[:, 2] > 0.0) & (wo[:, 2] > 0.0)
    f_te = torch.where(valid, _interp4(table.f_te, ti, pi_, to, po), 0.0)
    f_tm = torch.where(valid, _interp4(table.f_tm, ti, pi_, to, po), 0.0)
    return f_te, f_tm


def sample_directions(
    table: KirchhoffTable,
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

    n_pi = table.phi_i.shape[0]
    cos_i, phi_i_val, _, _ = _lookup_angles(table, wi, None)
    ti = _nearest_axis(cos_i, N_COS_THETA_I, 1.0 / N_COS_THETA_I, False)
    pi_ = (
        torch.zeros_like(ti)
        if n_pi == 1
        else _nearest_axis(phi_i_val, n_pi, 2.0 * math.pi / n_pi, True)
    )
    marginal_rows = table.marginal_cdf[ti, pi_]  # [N, Nto]
    u1 = u1.clamp(0.0, 1.0 - 1e-7).unsqueeze(1)
    to = torch.searchsorted(marginal_rows, u1, right=True).clamp(
        max=N_COS_THETA_O - 1
    )
    cdf_hi = torch.gather(marginal_rows, 1, to)
    cdf_lo = torch.where(
        to > 0, torch.gather(marginal_rows, 1, (to - 1).clamp(min=0)), torch.zeros_like(cdf_hi)
    )
    bin_mass = cdf_hi - cdf_lo
    frac1 = torch.where(bin_mass > 0.0, (u1 - cdf_lo) / bin_mass, torch.full_like(bin_mass, 0.5))
    cos_o = ((to.to(u1.dtype) + frac1) / N_COS_THETA_O).squeeze(1).clamp(1e-6, 1.0)

    cond_rows = table.conditional_cdf[ti, pi_, to.squeeze(1)]  # [N, Npo]
    u2 = u2.clamp(0.0, 1.0 - 1e-7).unsqueeze(1)
    po = torch.searchsorted(cond_rows, u2, right=True).clamp(max=N_PHI_O - 1)
    ccdf_hi = torch.gather(cond_rows, 1, po)
    ccdf_lo = torch.where(
        po > 0, torch.gather(cond_rows, 1, (po - 1).clamp(min=0)), torch.zeros_like(ccdf_hi)
    )
    cmass = ccdf_hi - ccdf_lo
    frac2 = torch.where(cmass > 0.0, (u2 - ccdf_lo) / cmass, torch.full_like(cmass, 0.5))
    phi = ((po.to(u2.dtype) + frac2) * (2.0 * math.pi / N_PHI_O)).squeeze(1)
    if n_pi == 1:
        # Isotropic tables sample the RELATIVE azimuth; rotate back into the
        # caller's frame around the surface normal.
        phi = phi + phi_i_val

    sin_o = torch.sqrt((1.0 - cos_o * cos_o).clamp(min=0.0))
    wo = torch.stack((sin_o * torch.cos(phi), sin_o * torch.sin(phi), cos_o), dim=1)
    density = table.sample_density[ti, pi_, to.squeeze(1), po.squeeze(1)]
    return wo, density


def pdf(table: KirchhoffTable, wi: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
    """Solid-angle sampling density of :func:`sample_directions`.

    Piecewise constant per outgoing bin and exactly consistent with the
    sampler (same nearest incidence bin, same mass table); integrates to 1
    over the hemisphere by construction. Zero below the horizon.
    """

    n_pi = table.phi_i.shape[0]
    cos_i, phi_i_val, cos_o, phi_o = _lookup_angles(table, wi, wo)
    ti = _nearest_axis(cos_i, N_COS_THETA_I, 1.0 / N_COS_THETA_I, False)
    pi_ = (
        torch.zeros_like(ti)
        if n_pi == 1
        else _nearest_axis(phi_i_val, n_pi, 2.0 * math.pi / n_pi, True)
    )
    to = _nearest_axis(cos_o, N_COS_THETA_O, 1.0 / N_COS_THETA_O, False)
    po = _nearest_axis(phi_o, N_PHI_O, 2.0 * math.pi / N_PHI_O, True)
    density = table.sample_density[ti, pi_, to, po]
    valid = (wi[:, 2] > 0.0) & (wo[:, 2] > 0.0)
    return torch.where(valid, density, torch.zeros_like(density))


def pdf_reverse(table: KirchhoffTable, wo: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
    """Reverse-direction PDF: the SAME table evaluated with swapped args.

    BDPT evaluates the reverse strategy density by treating the outgoing
    direction as the incidence direction (contract section 5); no separate
    reverse table exists.
    """

    return pdf(table, wo, wi)
