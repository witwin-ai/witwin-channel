# Copyright Xingyu Chen.
# Implements constants.

"""Implements constants."""

from __future__ import annotations

import math

C0 = 299792458.0  # vacuum speed of light [m/s]
MU0 = 4.0e-7 * math.pi  # vacuum permeability [H/m]
EPS0 = 1.0 / (MU0 * C0 * C0)  # vacuum permittivity [F/m]
ETA0 = MU0 * C0  # vacuum impedance [ohm]

PHASOR = "exp(-j*k*d)"
TIME_DEPENDENCE = "exp(+j*2*pi*f*t)"
# The source-excited free-space amplitude: the ``path_field`` convention that
# the Deterministic and Monte Carlo results and the propagation consumer
# publish.
FREE_SPACE_AMPLITUDE = "sqrt(tx_power)*wavelength/(4*pi*distance)"
# The same amplitude at unit source excitation: the ``coefficient`` /
# ``field_xyz`` convention that the Path result publishes. Quoting the excited
# string next to ``coefficient_semantics="unit_excitation..."`` states two
# different amplitudes for one number, so each result quotes its own.
UNIT_EXCITATION_FREE_SPACE_AMPLITUDE = "wavelength/(4*pi*distance)"
POLARIZATION = "world_cartesian_complex3_then_receiver_projection"

PHASE_CONVENTION = {
    "phasor": PHASOR,
    "time_dependence": TIME_DEPENDENCE,
    "free_space_amplitude": FREE_SPACE_AMPLITUDE,
    "polarization": POLARIZATION,
}

UNIT_EXCITATION_PHASE_CONVENTION = {
    **PHASE_CONVENTION,
    "free_space_amplitude": UNIT_EXCITATION_FREE_SPACE_AMPLITUDE,
}

# Narrowband law for shifting an evaluated coefficient off the reference
# frequency. Nothing in this package applies it: a coefficient is always
# reported at the compiled reference frequency, and `delay_s` is published per
# row precisely so a caller can apply this itself. The sign follows the frozen
# phasor and time-dependence above, so it is stated here rather than left for
# each caller to rederive.
#
# It holds only while the coefficient may be treated as constant across the
# offset. Re-evaluating dispersive material response per frequency point is a
# different operation - N field evaluations rather than a post-multiply - and
# is not what this law describes.
NARROWBAND_FREQUENCY_OFFSET_LAW = (
    "H(f_ref+df) = C(f_ref)*exp(-j*2*pi*df*delay_s)"
)

# What the law above costs, quantified, so "narrowband" is a number rather than
# an adjective. Three independent terms:
#
# 1. spreading - the free-space lambda/(4*pi*d) amplitude, exactly
# f_ref/(f_ref+df) in magnitude with zero phase;
# 2. material - the frequency selectivity of the compiled material response,
# bounded by the slab fringe scale df_fringe below (a 0.1 m eps_r=4 slab at
# normal incidence fringes every 750 MHz);
# 3. dispersion - d(eps_r)/df from a witwin.core DispersionSpec, which the law
# drops entirely because a compiled record freezes it at the primal
# frequency.
#
# The wideband frequency-offset capability removes terms 1 and 2 exactly, by
# evaluating the same frozen rows natively at each absolute frequency, and
# REFUSES term 3 rather than approximating it (wideband evaluation).
NARROWBAND_FREQUENCY_OFFSET_ERROR_LAW = (
    "relative_error = O(df/f_ref) spreading"
    " + O(df/df_fringe) material"
    " + zeroth_order dispersion;"
    " df_fringe = c/(2*Re(sqrt(eps_r))*thickness_m*cos(theta_t))"
)

__all__ = [
    "C0",
    "EPS0",
    "ETA0",
    "FREE_SPACE_AMPLITUDE",
    "MU0",
    "NARROWBAND_FREQUENCY_OFFSET_ERROR_LAW",
    "NARROWBAND_FREQUENCY_OFFSET_LAW",
    "PHASE_CONVENTION",
    "PHASOR",
    "POLARIZATION",
    "TIME_DEPENDENCE",
    "UNIT_EXCITATION_FREE_SPACE_AMPLITUDE",
    "UNIT_EXCITATION_PHASE_CONVENTION",
]