import math

import pytest

from witwin.channel.utils.utd import cot as shared_cot
from witwin.channel.utils.utd import f_utd as shared_f_utd
from witwin.channel.utils.utd import fresnel_integral as shared_fresnel_integral
from witwin.montecarlo import types as mc_wt
from witwin.montecarlo.path.diffraction_utd import UTD as mc_utd
def _complex_components(value) -> tuple[float, float]:
    return float(value.real[0]), float(value.imag[0])


def _complex_abs(value) -> float:
    real, imag = _complex_components(value)
    return math.hypot(real, imag)


@pytest.mark.parametrize("x", [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 50.0])
def test_monte_carlo_utd_fresnel_integral_matches_shared_reference(x: float) -> None:
    got = mc_utd.fresnel_integral(mc_wt.Float(x))
    expected = shared_fresnel_integral(mc_wt.Float(x))
    got_real, got_imag = _complex_components(got)
    exp_real, exp_imag = _complex_components(expected)
    assert got_real == pytest.approx(exp_real, rel=1e-7, abs=1e-9)
    assert got_imag == pytest.approx(exp_imag, rel=1e-7, abs=1e-9)


@pytest.mark.parametrize("x", [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 10.0, 50.0])
def test_monte_carlo_utd_transition_matches_shared_reference(x: float) -> None:
    got = mc_utd.f(mc_wt.Float(x))
    expected = shared_f_utd(mc_wt.Float(x))
    got_real, got_imag = _complex_components(got)
    exp_real, exp_imag = _complex_components(expected)
    assert got_real == pytest.approx(exp_real, rel=1e-7, abs=1e-9)
    assert got_imag == pytest.approx(exp_imag, rel=1e-7, abs=1e-9)


def test_monte_carlo_utd_transition_stays_small_near_shadow_boundary() -> None:
    magnitudes = [_complex_abs(mc_utd.f(mc_wt.Float(x))) for x in (1e-8, 1e-6, 1e-4, 1e-2)]
    assert magnitudes[0] < 1e-3
    assert magnitudes[1] < 1e-2
    assert magnitudes[2] < 5e-2
    assert magnitudes[3] < 0.2


def test_monte_carlo_utd_transition_weight_marks_shadow_boundary_proximity() -> None:
    near = float(mc_utd.transition_weight_from_argument(mc_wt.Float(1.0e-8))[0])
    far = float(mc_utd.transition_weight_from_argument(mc_wt.Float(50.0))[0])

    assert math.isfinite(near)
    assert math.isfinite(far)
    assert near > 0.99
    assert far < 0.05


def test_monte_carlo_utd_smooth_go_coefficient_follows_transition_response() -> None:
    near = mc_utd.f(mc_wt.Float(1.0e-8))
    far = mc_utd.f(mc_wt.Float(50.0))

    def coeff_abs2(response, side: float) -> float:
        real, imag = _complex_components(response)
        return 0.25 * ((1.0 + side * real) ** 2 + imag**2)

    near_lit = coeff_abs2(near, 1.0)
    near_shadow = coeff_abs2(near, -1.0)
    far_lit = coeff_abs2(far, 1.0)
    far_shadow = coeff_abs2(far, -1.0)

    assert near_lit == pytest.approx(0.25, rel=5.0e-3, abs=5.0e-3)
    assert near_shadow == pytest.approx(0.25, rel=5.0e-3, abs=5.0e-3)
    assert far_lit > 0.8
    assert far_shadow < 0.05


@pytest.mark.parametrize("x", [0.2, 0.5, 1.0, 2.0])
def test_monte_carlo_utd_cot_matches_shared_reference(x: float) -> None:
    got = float(mc_utd.cot(mc_wt.Float(x))[0])
    expected = float(shared_cot(mc_wt.Float(x))[0])
    assert got == pytest.approx(expected, rel=1e-7, abs=1e-9)
