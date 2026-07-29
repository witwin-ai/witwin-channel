# Copyright Xingyu Chen.
# Implements kirchhoff table build.

"""Implements kirchhoff table build."""

from __future__ import annotations

import math

import torch

_EPS0 = 8.8541878128e-12
_MU0 = 1.25663706212e-6
_C0 = 299792458.0


def _sqrt_passive(z: torch.Tensor) -> torch.Tensor:
    w = torch.sqrt(z)
    return torch.where(w.imag > 0.0, -w, w)


def _stack_power_reflectance(
    layers: torch.Tensor, cos_theta: torch.Tensor, frequency: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(|r_te|^2, |r_tm|^2)`` for the vacuum|layers|vacuum stack (float64).

 ``layers`` is ``[L, 4]`` (thickness_m, eps_r, sigma_e, mu_r); ``cos_theta``
 is a real tensor of incidence cosines. Mirrors ``materials``.
 """

    omega = 2.0 * math.pi * frequency
    cos_theta = cos_theta.to(torch.complex128)
    sin2 = 1.0 - cos_theta * cos_theta

    def medium(eps_r, sigma_e, mu_r):
        eps_rel = eps_r.to(torch.complex128) - 1j * sigma_e.to(torch.complex128) / (
            omega * _EPS0
        )
        mu_rel = mu_r.to(torch.complex128)
        k = (omega / _C0) * _sqrt_passive(eps_rel * mu_rel)
        return eps_rel * _EPS0, mu_rel * _MU0, k

    eps_out, mu_out, k_out = medium(
        torch.ones((), dtype=torch.float64),
        torch.zeros((), dtype=torch.float64),
        torch.ones((), dtype=torch.float64),
    )
    k_par2 = (k_out * k_out) * sin2

    def kz(k):
        return _sqrt_passive(k * k - k_par2)

    def admittances(eps, mu, k_z):
        return k_z / (omega * mu), omega * eps / k_z

    kz_out = kz(k_out)
    y_out_te, y_out_tm = admittances(eps_out, mu_out, kz_out)
    y_back_te, y_back_tm = y_out_te, y_out_tm

    def one_pol(y_out, y_layers, deltas, y_back):
        ones = torch.ones_like(cos_theta)
        m11 = ones.clone()
        m12 = torch.zeros_like(ones)
        m21 = torch.zeros_like(ones)
        m22 = ones.clone()
        log_scale = torch.zeros(cos_theta.shape, dtype=torch.float64)
        for y, delta in zip(y_layers, deltas):
            a = -delta.imag
            e_plus = torch.exp(1j * delta.real)
            e_minus = torch.exp(-2.0 * a - 1j * delta.real)
            cos_s = 0.5 * (e_plus + e_minus)
            sin_s = (e_plus - e_minus) / 2j
            l11 = cos_s
            l12 = 1j * sin_s / y
            l21 = 1j * y * sin_s
            l22 = cos_s
            m11, m12, m21, m22 = (
                m11 * l11 + m12 * l21,
                m11 * l12 + m12 * l22,
                m21 * l11 + m22 * l21,
                m21 * l12 + m22 * l22,
            )
            log_scale = log_scale + a
        b = m11 + m12 * y_back
        c = m21 + m22 * y_back
        denom = y_out * b + c
        r = (y_out * b - c) / denom
        return r

    y_te_layers, y_tm_layers, deltas = [], [], []
    for row in range(layers.shape[0]):
        eps_l, mu_l, k_l = medium(layers[row, 1], layers[row, 2], layers[row, 3])
        kz_l = kz(k_l)
        yte, ytm = admittances(eps_l, mu_l, kz_l)
        y_te_layers.append(yte)
        y_tm_layers.append(ytm)
        deltas.append(kz_l * layers[row, 0].to(torch.complex128))
    ones = torch.ones_like(cos_theta)
    r_te = one_pol(y_out_te * ones, y_te_layers, deltas, y_back_te)
    r_tm = one_pol(y_out_tm * ones, y_tm_layers, deltas, y_back_tm)
    return (r_te.abs() ** 2).real, (r_tm.abs() ** 2).real


def _lobe_series(qx, qy, qn, sigma_h, lx, ly, n_terms: int) -> torch.Tensor:
    g = (qn * sigma_h) ** 2
    rho2 = (qx * lx) ** 2 + (qy * ly) ** 2
    m = torch.arange(1, n_terms + 1, dtype=torch.float64).reshape(
        (n_terms,) + (1,) * g.ndim
    )
    log_fact = torch.lgamma(m + 1.0)
    log_g = torch.log(g.clamp_min(1e-300))
    log_term = m * log_g - log_fact - torch.log(m) - rho2 / (4.0 * m) - g
    series = torch.exp(log_term).sum(dim=0)
    # g == 0 -> smooth limit, all terms vanish.
    series = torch.where(g > 0.0, series, torch.zeros_like(series))
    return math.pi * lx * ly * series


def _raw_lobe_grid(
    layers, frequency, k0, sigma_h, lx, ly, n_terms,
    inc_cos, inc_phi, out_cos, out_phi,
) -> tuple[torch.Tensor, torch.Tensor]:
    sin_inc = torch.sqrt((1.0 - inc_cos**2).clamp_min(0.0))
    sin_out = torch.sqrt((1.0 - out_cos**2).clamp_min(0.0))
    wo_x = sin_out[:, None] * torch.cos(out_phi)[None, :]
    wo_y = sin_out[:, None] * torch.sin(out_phi)[None, :]
    wo_z = out_cos[:, None].expand_as(wo_x)
    n_ti, n_pi = inc_cos.shape[0], inc_phi.shape[0]
    f_te = []
    f_tm = []
    for ti in range(n_ti):
        row_te = []
        row_tm = []
        for pi in range(n_pi):
            wi_x = sin_inc[ti] * torch.cos(inc_phi[pi])
            wi_y = sin_inc[ti] * torch.sin(inc_phi[pi])
            wi_z = inc_cos[ti]
            qx = k0 * (wo_x + wi_x)
            qy = k0 * (wo_y + wi_y)
            qn = k0 * (wo_z + wi_z)
            lobe = _lobe_series(qx, qy, qn, sigma_h, lx, ly, n_terms)
            q_sq = qx**2 + qy**2 + qn**2
            wi_dot_wo = wo_x * wi_x + wo_y * wi_y + wo_z * wi_z
            cos_h = ((1.0 + wi_dot_wo) * k0 / torch.sqrt(q_sq)).clamp(1e-6, 1.0)
            rr_te, rr_tm = _stack_power_reflectance(
                layers, cos_h.reshape(-1), frequency
            )
            prefactor = q_sq**2 / (16.0 * math.pi**2 * qn**2 * wi_z * wo_z)
            shape = prefactor * lobe
            row_te.append(shape * rr_te.reshape(shape.shape))
            row_tm.append(shape * rr_tm.reshape(shape.shape))
        f_te.append(torch.stack(row_te, dim=0))
        f_tm.append(torch.stack(row_tm, dim=0))
    return torch.stack(f_te, dim=0), torch.stack(f_tm, dim=0)


def _sinkhorn_balance(
    s: torch.Tensor, r_diff: torch.Tensor, cos_o: torch.Tensor,
    *, isotropic: bool, iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unrolled symmetric Sinkhorn balance (differentiable, mirrors tables.py).

 Returns ``(balanced_lobe, factor)`` where ``factor`` has shape
 ``[n_ti, n_pi]``. ``iterations`` is fixed (tight convergence expected).
 """

    n_ti, n_pi, n_to, n_po = s.shape
    d_omega = (1.0 / n_to) * (2.0 * math.pi / n_po)
    if isotropic:
        sm = s[:, 0]  # [n_ti, n_to, n_po]
        rhs = r_diff[:, 0]
        active = rhs > 0.0
        factor = torch.ones(n_ti, dtype=torch.float64)
        w = cos_o * d_omega  # [n_to]
        for _ in range(iterations):
            denom = (sm * (w[None, :, None] * factor[None, :, None])).sum(dim=(1, 2))
            ratio = torch.where(
                active, rhs / (factor * denom).clamp_min(1e-300),
                torch.ones_like(factor),
            )
            factor = torch.where(active, factor * torch.sqrt(ratio), torch.zeros_like(factor))
        balanced = sm * factor[:, None, None] * factor[None, :, None]
        return balanced[:, None], factor[:, None]
    states = n_ti * n_pi
    sm = s.reshape(states, states)
    rhs = r_diff.reshape(states)
    active = rhs > 0.0
    weights = (cos_o * d_omega).repeat_interleave(n_po)  # [n_to*n_po]
    factor = torch.ones(states, dtype=torch.float64)
    for _ in range(iterations):
        denom = sm @ (weights * factor)
        ratio = torch.where(
            active, rhs / (factor * denom).clamp_min(1e-300), torch.ones_like(factor)
        )
        factor = torch.where(active, factor * torch.sqrt(ratio), torch.zeros_like(factor))
    balanced = sm * factor[:, None] * factor[None, :]
    return balanced.reshape(s.shape), factor.reshape(n_ti, n_pi)


def torch_build_table(
    sigma_h: torch.Tensor,
    lx: torch.Tensor,
    ly: torch.Tensor,
    layers: torch.Tensor,
    frequency: torch.Tensor,
    cos_i: torch.Tensor,
    phi_i: torch.Tensor,
    cos_o: torch.Tensor,
    phi_o: torch.Tensor,
    *,
    n_terms: int,
    isotropic: bool,
    sinkhorn_iters: int = 256,
) -> dict[str, torch.Tensor]:
    """Differentiable float64 Kirchhoff table build (oracle).

 Returns ``f_te``/``f_tm`` (balanced final tables) plus the pre-balance
 symmetrized lobes ``s_te``/``s_tm``, balance factors ``a_te``/``a_tm`` and
 diffuse budgets ``r_diff_te``/``r_diff_tm`` (the native saved intermediates).
 All tensors are float64 and carry the autograd graph to the parameters.
 """

    k0 = 2.0 * math.pi * frequency / _C0
    r_bar_te, r_bar_tm = _stack_power_reflectance(layers, cos_i, frequency)
    c_r = torch.exp(-2.0 * (k0 * cos_i * sigma_h) ** 2)
    r_diff_te = (r_bar_te - r_bar_te * c_r**2).clamp_min(0.0)
    r_diff_tm = (r_bar_tm - r_bar_tm * c_r**2).clamp_min(0.0)

    f_raw_te, f_raw_tm = _raw_lobe_grid(
        layers, frequency, k0, sigma_h, lx, ly, n_terms, cos_i, phi_i, cos_o, phi_o
    )
    swap_te, swap_tm = _raw_lobe_grid(
        layers, frequency, k0, sigma_h, lx, ly, n_terms, cos_o, phi_o, cos_i, phi_i
    )
    swap_te = swap_te.permute(2, 3, 0, 1)
    swap_tm = swap_tm.permute(2, 3, 0, 1)
    s_te = 0.5 * (f_raw_te + swap_te)
    s_tm = 0.5 * (f_raw_tm + swap_tm)

    n_ti, n_pi = cos_i.shape[0], phi_i.shape[0]
    r_diff_te_grid = r_diff_te[:, None].expand(n_ti, n_pi).contiguous()
    r_diff_tm_grid = r_diff_tm[:, None].expand(n_ti, n_pi).contiguous()
    bal_te, a_te = _sinkhorn_balance(
        s_te, r_diff_te_grid, cos_o, isotropic=isotropic, iterations=sinkhorn_iters
    )
    bal_tm, a_tm = _sinkhorn_balance(
        s_tm, r_diff_tm_grid, cos_o, isotropic=isotropic, iterations=sinkhorn_iters
    )
    return {
        "f_te": bal_te,
        "f_tm": bal_tm,
        "s_te": s_te,
        "s_tm": s_tm,
        "a_te": a_te,
        "a_tm": a_tm,
        "r_diff_te": r_diff_te_grid,
        "r_diff_tm": r_diff_tm_grid,
    }


def cos_centers(n: int) -> torch.Tensor:
    return (torch.arange(n, dtype=torch.float64) + 0.5) / n


def phi_centers(n: int) -> torch.Tensor:
    return (torch.arange(n, dtype=torch.float64) + 0.5) * (2.0 * math.pi / n)


def build_n_terms(k0: float, sigma_h: float) -> int:
    g_max = (2.0 * k0 * sigma_h) ** 2
    return int(max(64.0, g_max + 12.0 * math.sqrt(g_max) + 16.0))