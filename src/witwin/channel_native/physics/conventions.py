"""Shared electromagnetic constants and phase conventions.

This module contains constants only.  Production solvers may depend on it;
the independent NumPy reference oracle remains a separate implementation.
"""

from __future__ import annotations

import math

C0 = 299792458.0  # vacuum speed of light [m/s]
MU0 = 4.0e-7 * math.pi  # vacuum permeability [H/m]
EPS0 = 1.0 / (MU0 * C0 * C0)  # vacuum permittivity [F/m]
ETA0 = MU0 * C0  # vacuum impedance [ohm]

__all__ = ["C0", "EPS0", "ETA0", "MU0"]
