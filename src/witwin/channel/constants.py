"""Electromagnetic constants and the package-wide phase convention.

This module holds values only, so every layer may depend on it. It is the
single source of truth for the sign and time conventions that solver metadata,
the propagation consumer contract, and the reference oracle all quote.

Note: Core carries its own ``witwin.core.material.VACUUM_PERMITTIVITY``
(8.8541878128e-12, the CODATA measured value) while ``EPS0`` here derives from
the pre-2019 exact definition ``1 / (MU0 * C0**2)``. The two differ in the
ninth significant digit. Reconciling them changes solver output and therefore
requires its own numerical ADR; do not silently align them here.
"""

from __future__ import annotations

import math

C0 = 299792458.0  # vacuum speed of light [m/s]
MU0 = 4.0e-7 * math.pi  # vacuum permeability [H/m]
EPS0 = 1.0 / (MU0 * C0 * C0)  # vacuum permittivity [F/m]
ETA0 = MU0 * C0  # vacuum impedance [ohm]

PHASOR = "exp(-j*k*d)"
TIME_DEPENDENCE = "exp(+j*2*pi*f*t)"
FREE_SPACE_AMPLITUDE = "sqrt(tx_power)*wavelength/(4*pi*distance)"
POLARIZATION = "world_cartesian_complex3_then_receiver_projection"

PHASE_CONVENTION = {
    "phasor": PHASOR,
    "time_dependence": TIME_DEPENDENCE,
    "free_space_amplitude": FREE_SPACE_AMPLITUDE,
    "polarization": POLARIZATION,
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

__all__ = [
    "C0",
    "EPS0",
    "ETA0",
    "FREE_SPACE_AMPLITUDE",
    "MU0",
    "NARROWBAND_FREQUENCY_OFFSET_LAW",
    "PHASE_CONVENTION",
    "PHASOR",
    "POLARIZATION",
    "TIME_DEPENDENCE",
]
