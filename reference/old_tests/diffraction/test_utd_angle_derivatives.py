"""Regression tests for analytic UTD angle derivatives."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import witwin as wt

import drjit as dr

from witwin.channel.trace.diffraction.utd import (
    DiffAngle,
    diffraction_coefficient_2d,
    diffraction_coefficient_2d_angle_derivative,
    slope_diffraction_coefficient_2d,
    slope_diffraction_coefficient_2d_angle_derivative,
)
from witwin.channel.utils import scalar
def _complex_abs(value):
    return scalar(dr.abs(value))


def _ad_angle_derivative(phi, phi_prime, wedge_n, k, s, s_prime, angle: DiffAngle, R0, Rn):
    phi_ad = wt.Float(phi)
    phi_prime_ad = wt.Float(phi_prime)
    if angle is DiffAngle.PHI:
        dr.enable_grad(phi_ad)
        dr.set_grad(phi_ad, wt.Float(1.0))
    elif angle is DiffAngle.PHI_PRIME:
        dr.enable_grad(phi_prime_ad)
        dr.set_grad(phi_prime_ad, wt.Float(1.0))
    else:
        raise ValueError(f"Unsupported angle selector: {angle}")
    value = diffraction_coefficient_2d(phi_ad, phi_prime_ad, wedge_n, k, s, s_prime, R0=R0, Rn=Rn)
    dr.forward_to(value.real, value.imag)
    return wt.Complex2f(dr.grad(value.real), dr.grad(value.imag))


def test_diffraction_angle_derivatives_match_ad_reference():
    phi = wt.Float(1.1)
    phi_prime = wt.Float(2.0)
    wedge_n = wt.Float(1.5)
    k = wt.Float(20.0)
    s = wt.Float(3.0)
    s_prime = wt.Float(4.0)
    R0 = wt.Complex2f(-0.25, 0.15)
    Rn = wt.Complex2f(-0.35, -0.05)
    analytic_phi = diffraction_coefficient_2d_angle_derivative(
        phi, phi_prime, wedge_n, k, s, s_prime, angle=DiffAngle.PHI, R0=R0, Rn=Rn
    )
    reference_phi = _ad_angle_derivative(
        phi, phi_prime, wedge_n, k, s, s_prime, angle=DiffAngle.PHI, R0=R0, Rn=Rn
    )
    assert _complex_abs(analytic_phi - reference_phi) < 5e-6

    analytic_phi_prime = diffraction_coefficient_2d_angle_derivative(
        phi, phi_prime, wedge_n, k, s, s_prime, angle=DiffAngle.PHI_PRIME, R0=R0, Rn=Rn
    )
    reference_phi_prime = _ad_angle_derivative(
        phi, phi_prime, wedge_n, k, s, s_prime, angle=DiffAngle.PHI_PRIME, R0=R0, Rn=Rn
    )
    assert _complex_abs(analytic_phi_prime - reference_phi_prime) < 5e-6


def test_slope_angle_derivatives_match_central_difference_reference():
    phi = wt.Float(1.2)
    phi_prime = wt.Float(1.9)
    wedge_n = wt.Float(1.5)
    k = wt.Float(20.0)
    s = wt.Float(3.5)
    s_prime = wt.Float(4.5)
    R0 = wt.Complex2f(-0.2, 0.1)
    Rn = wt.Complex2f(-0.3, -0.08)
    step = wt.Float(1e-3)

    analytic_phi = slope_diffraction_coefficient_2d_angle_derivative(
        phi, phi_prime, wedge_n, k, s, s_prime, angle=DiffAngle.PHI, R0=R0, Rn=Rn
    )
    reference_phi = (
        slope_diffraction_coefficient_2d(phi + step, phi_prime, wedge_n, k, s, s_prime, R0=R0, Rn=Rn)
        - slope_diffraction_coefficient_2d(phi - step, phi_prime, wedge_n, k, s, s_prime, R0=R0, Rn=Rn)
    ) * (0.5 / step)
    assert _complex_abs(analytic_phi - reference_phi) < 5e-5

    analytic_phi_prime = slope_diffraction_coefficient_2d_angle_derivative(
        phi, phi_prime, wedge_n, k, s, s_prime, angle=DiffAngle.PHI_PRIME, R0=R0, Rn=Rn
    )
    reference_phi_prime = (
        slope_diffraction_coefficient_2d(phi, phi_prime + step, wedge_n, k, s, s_prime, R0=R0, Rn=Rn)
        - slope_diffraction_coefficient_2d(phi, phi_prime - step, wedge_n, k, s, s_prime, R0=R0, Rn=Rn)
    ) * (0.5 / step)
    assert _complex_abs(analytic_phi_prime - reference_phi_prime) < 5e-5


