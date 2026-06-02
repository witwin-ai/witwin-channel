"""Direct test of _cot_f_product vs clamped cot*F."""
import drjit as dr
import witwin as wt
from witwin.channel import DEFAULT_VARIANT
from witwin.channel.trace.diffraction.utd import _cot_f_product, cot, f_utd
import math
k_val = 2 * math.pi / 0.3
s, s_prime = 3.0, 5.0
L = s * s_prime / (s + s_prime)
kL = k_val * L
n = 2.0

print(f"kL = {kL:.4f}")
print(f"Expected |cot*F| limit at ISB ~= {n * math.sqrt(2*math.pi*kL):.4f}")

print("\n=== Direct _cot_f_product vs old cot*F ===")
print(f"{'delta':>10} | {'|old cot*F|':>12} | {'|new cotF|':>12} | {'sin(arg)':>12} | {'a':>12} | {'|F|':>12}")
print("-" * 90)
for delta in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0]:
    phi_val = math.pi + math.pi/2 - delta
    dif_phi = phi_val - math.pi/2
    cot_arg_val = (math.pi - dif_phi) / (2*n)

    # Compute a (minus branch for dif_phi)
    two_n_pi = 2*n*math.pi
    N_minus = round((dif_phi - math.pi)/(two_n_pi))
    phase_offset = two_n_pi * N_minus - dif_phi
    a_val = 2 * math.cos(0.5 * phase_offset)**2

    cot_arg = wt.Float(cot_arg_val)
    a = wt.Float(a_val)
    kL_f = wt.Float(kL)

    # Old: cot * F
    old_cot = cot(cot_arg)
    old_f = f_utd(kL_f * a)
    old_product = float(old_cot[0]) * old_f

    # New: _cot_f_product
    new_product = _cot_f_product(cot_arg, a, kL_f)

    sin_val = math.sin(cot_arg_val) if abs(cot_arg_val) > 1e-20 else 0.0
    print(f"{delta:10.1e} | {float(dr.abs(old_product)[0]):12.6f} | {float(dr.abs(new_product)[0]):12.6f} | {sin_val:12.4e} | {a_val:12.4e} | {float(dr.abs(old_f)[0]):12.8f}")

# Test with float64 math reference
print("\n=== Float64 reference ===")
import numpy as np
for delta in [1e-6, 1e-7, 1e-8, 0.0]:
    phi_val = math.pi + math.pi/2 - delta
    dif_phi = phi_val - math.pi/2
    cot_arg_val = (math.pi - dif_phi) / (2*n)
    two_n_pi = 2*n*math.pi
    N_minus = round((dif_phi - math.pi)/(two_n_pi))
    phase_offset = two_n_pi * N_minus - dif_phi
    a_val = 2 * np.cos(0.5 * phase_offset)**2

    if abs(np.sin(cot_arg_val)) > 0:
        cot_val_f64 = np.cos(cot_arg_val) / np.sin(cot_arg_val)
    else:
        cot_val_f64 = float('inf')

    x = kL * a_val
    # Simplified F(x) for very small x: F(x) ~= sqrt(pi*x/2) * (1+j)
    if x > 0:
        f_approx = np.sqrt(np.pi * x / 2) * (1 + 1j)
        product_f64 = cot_val_f64 * f_approx
    else:
        product_f64 = 0

    print(f"  delta={delta:8.1e}  cot_f64={cot_val_f64:14.2f}  a={a_val:14.4e}  kLa={kL*a_val:14.4e}  |cot*F|_f64={abs(product_f64):12.6f}")
