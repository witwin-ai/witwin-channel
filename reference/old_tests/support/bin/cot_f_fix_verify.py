"""Verify: what should D look like if the cot*F product used its correct SB limit?"""
import drjit as dr
import witwin as wt
from witwin.channel import DEFAULT_VARIANT
from witwin.channel.trace.diffraction.utd import cot, f_utd, _compute_a_pm
import math
import cmath
n = 2.0
phi_prime = math.pi / 2
k_val = 2 * math.pi / 0.3
s_val = 3.0
s_prime_val = 5.0
L_val = s_val * s_prime_val / (s_val + s_prime_val)
kL = k_val * L_val

def cot_f_product_safe(cot_arg_val, a_val, kL_val):
    """Compute cot(cot_arg) * F(kL*a) with correct shadow boundary handling."""
    a = wt.Float(a_val)
    cot_arg = wt.Float(cot_arg_val)
    near_sb = a_val < 1e-4  # threshold
    if near_sb:
        # SB limit: sign(cos(cot_arg)) * sqrt(kL) * exp(j*pi/4)
        sign = 1.0 if math.cos(cot_arg_val) >= 0 else -1.0
        sb = cmath.sqrt(kL_val) * cmath.exp(1j * math.pi / 4)
        return wt.Complex2f(sign * sb.real, sign * sb.imag)
    else:
        c = cot(cot_arg)
        f = f_utd(wt.Float(kL_val * a_val))
        return float(c[0]) * f

def diffraction_coefficient_2d_fixed(phi_val, phi_prime_val, n_val, k, s, s_prime):
    """D coefficient with corrected shadow boundary limit."""
    L = s * s_prime / (s + s_prime)
    kL = k * L
    dif_phi = phi_val - phi_prime_val
    sum_phi = phi_val + phi_prime_val
    two_n = 2 * n_val

    a1v, a2v = _compute_a_pm(wt.Float(dif_phi), wt.Float(n_val))
    a3v, a4v = _compute_a_pm(wt.Float(sum_phi), wt.Float(n_val))

    d1 = cot_f_product_safe((math.pi + dif_phi) / two_n, float(a1v[0]), kL)
    d2 = cot_f_product_safe((math.pi - dif_phi) / two_n, float(a2v[0]), kL)
    d3 = cot_f_product_safe((math.pi + sum_phi) / two_n, float(a3v[0]), kL)
    d4 = cot_f_product_safe((math.pi - sum_phi) / two_n, float(a4v[0]), kL)

    factor = -cmath.exp(-1j * math.pi / 4) / (2 * n_val * math.sqrt(2 * math.pi * k))
    R0 = -1.0
    Rn = -1.0
    total = d1 + d2 + wt.Complex2f(R0, 0) * d3 + wt.Complex2f(Rn, 0) * d4
    return wt.Complex2f(factor.real, factor.imag) * total

print("=== Comparing D coefficient: current vs fixed ===")
print(f"{'delta':>10} | {'|D_current|':>12} | {'|D_fixed|':>12} | {'ratio':>8}")
print("-" * 55)

from witwin.channel.trace.diffraction.utd import diffraction_coefficient_2d
for delta in [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 0.0]:
    phi_val = math.pi + math.pi / 2 - delta

    D_current = diffraction_coefficient_2d(
        wt.Float(phi_val), wt.Float(phi_prime), wt.Float(n),
        wt.Float(k_val), wt.Float(s_val), wt.Float(s_prime_val)
    )
    D_fixed = diffraction_coefficient_2d_fixed(phi_val, phi_prime, n, k_val, s_val, s_prime_val)

    mag_current = float(dr.abs(D_current)[0])
    mag_fixed = float(dr.abs(D_fixed)[0])
    ratio = mag_current / mag_fixed if mag_fixed > 1e-20 else float('nan')
    print(f"{delta:10.1e} | {mag_current:12.6f} | {mag_fixed:12.6f} | {ratio:8.4f}")

# Shadow side
print(f"\n{'delta':>10} | {'|D_current|':>12} | {'|D_fixed|':>12} | {'ratio':>8} (shadow side)")
print("-" * 65)
for delta in [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 0.0]:
    phi_val = math.pi + math.pi / 2 + delta

    D_current = diffraction_coefficient_2d(
        wt.Float(phi_val), wt.Float(phi_prime), wt.Float(n),
        wt.Float(k_val), wt.Float(s_val), wt.Float(s_prime_val)
    )
    D_fixed = diffraction_coefficient_2d_fixed(phi_val, phi_prime, n, k_val, s_val, s_prime_val)

    mag_current = float(dr.abs(D_current)[0])
    mag_fixed = float(dr.abs(D_fixed)[0])
    ratio = mag_current / mag_fixed if mag_fixed > 1e-20 else float('nan')
    print(f"{delta:10.1e} | {mag_current:12.6f} | {mag_fixed:12.6f} | {ratio:8.4f}")
