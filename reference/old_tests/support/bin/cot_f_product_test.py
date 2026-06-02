"""Test cot * f_utd product behavior near shadow boundary.

At the ISB, the d2 term has cot_arg ï¿?0 (pole) and a ï¿?0 simultaneously.
The product cot(eps) * F(kL*eps^2) should approach sqrt(kL)*exp(j*pi/4).
"""
import drjit as dr
import witwin as wt
from witwin.channel import DEFAULT_VARIANT
from witwin.channel.trace.diffraction.utd import cot, f_utd, _compute_a_pm, diffraction_coefficient_2d
import math
import cmath

print("=== cot() near pole (0) ===")
for eps_val in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0]:
    x = wt.Float(eps_val)
    c = cot(x)
    expected = math.cos(eps_val) / math.sin(eps_val) if eps_val > 0 else float('inf')
    print(f"  cot({eps_val:8.1e}) = {float(c[0]):14.4f}    expected={expected:14.4f}")

print("\n=== Half-plane (n=2) ISB test ===")
# Source at phi'=pi/2, observe along phi approaching the ISB
# For half-plane n=2, ISB is where phi = 2*pi - phi' = 3pi/2
# But standard ISB: phi that makes the d2 cot argument approach 0 or pi
# d2: cot((pi - (phi-phi'))/(2n))
# pole when (pi - dif_phi)/(2n) = m*pi, i.e. dif_phi = pi - 2n*m*pi
# For m=0: dif_phi = pi, i.e. phi = pi + phi'
# That's the shadow boundary: phi = pi + phi'
n = wt.Float(2.0)
phi_prime = wt.Float(math.pi / 2)
k_val = 2 * math.pi / 0.3  # 1 GHz
s_val = 3.0
s_prime_val = 5.0
kL = k_val * s_val * s_prime_val / (s_val + s_prime_val)

print(f"  phi' = pi/2 = {math.pi/2:.4f}")
print(f"  ISB at phi = pi + phi' = {math.pi + math.pi/2:.4f} = 3pi/2")
print(f"  kL = {kL:.4f}")
print(f"  sqrt(kL)*exp(j*pi/4) = ({cmath.sqrt(kL)*cmath.exp(1j*math.pi/4)}) |={abs(cmath.sqrt(kL)):.4f}")

print(f"\n  Approaching ISB from lit side (phi < pi+phi'):")
for delta in [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 0.0]:
    phi = wt.Float(math.pi + math.pi/2 - delta)
    D = diffraction_coefficient_2d(phi, phi_prime, n, wt.Float(k_val), wt.Float(s_val), wt.Float(s_prime_val))
    dif_phi_val = math.pi + math.pi/2 - delta - math.pi/2
    cot_arg_d2 = (math.pi - dif_phi_val) / (2 * 2.0)
    a1, a2 = _compute_a_pm(wt.Float(dif_phi_val), n)
    print(f"    delta={delta:8.1e}  phi={float(phi[0]):.6f}  D=({float(D.real[0]):12.6f}, {float(D.imag[0]):12.6f})  |D|={float(dr.abs(D)[0]):12.8f}  cot_arg_d2={cot_arg_d2:.6f}  a2={float(a2[0]):.8f}")

print(f"\n  Approaching ISB from shadow side (phi > pi+phi'):")
for delta in [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 0.0]:
    phi = wt.Float(math.pi + math.pi/2 + delta)
    D = diffraction_coefficient_2d(phi, phi_prime, n, wt.Float(k_val), wt.Float(s_val), wt.Float(s_prime_val))
    print(f"    delta={delta:8.1e}  phi={float(phi[0]):.6f}  D=({float(D.real[0]):12.6f}, {float(D.imag[0]):12.6f})  |D|={float(dr.abs(D)[0]):12.8f}")

# Also check individual d1,d2,d3,d4 terms
print(f"\n=== Individual d-terms at ISB ===")
for delta in [0.01, 0.001, 1e-4, 1e-5, 1e-6, 0.0]:
    phi_val = math.pi + math.pi/2 - delta
    dif_phi_val = phi_val - math.pi/2
    sum_phi_val = phi_val + math.pi/2
    two_n = 4.0
    L_val = s_val * s_prime_val / (s_val + s_prime_val)
    kL_val = k_val * L_val

    a1v, a2v = _compute_a_pm(wt.Float(dif_phi_val), n)
    a3v, a4v = _compute_a_pm(wt.Float(sum_phi_val), n)

    cot1 = cot(wt.Float((math.pi + dif_phi_val) / two_n))
    cot2 = cot(wt.Float((math.pi - dif_phi_val) / two_n))
    cot3 = cot(wt.Float((math.pi + sum_phi_val) / two_n))
    cot4 = cot(wt.Float((math.pi - sum_phi_val) / two_n))

    f1 = f_utd(wt.Float(kL_val * float(a1v[0])))
    f2 = f_utd(wt.Float(kL_val * float(a2v[0])))
    f3 = f_utd(wt.Float(kL_val * float(a3v[0])))
    f4 = f_utd(wt.Float(kL_val * float(a4v[0])))

    d1 = float(cot1[0]) * f1
    d2 = float(cot2[0]) * f2
    d3 = float(cot3[0]) * f3
    d4 = float(cot4[0]) * f4

    print(f"  delta={delta:8.1e}:")
    print(f"    cot1={float(cot1[0]):12.4f}  a1={float(a1v[0]):12.8f}  |F1|={float(dr.abs(f1)[0]):10.6f}  |d1|={float(dr.abs(d1)[0]):10.6f}")
    print(f"    cot2={float(cot2[0]):12.4f}  a2={float(a2v[0]):12.8f}  |F2|={float(dr.abs(f2)[0]):10.6f}  |d2|={float(dr.abs(d2)[0]):10.6f}")
    print(f"    cot3={float(cot3[0]):12.4f}  a3={float(a3v[0]):12.8f}  |F3|={float(dr.abs(f3)[0]):10.6f}  |d3|={float(dr.abs(d3)[0]):10.6f}")
    print(f"    cot4={float(cot4[0]):12.4f}  a4={float(a4v[0]):12.8f}  |F4|={float(dr.abs(f4)[0]):10.6f}  |d4|={float(dr.abs(d4)[0]):10.6f}")
