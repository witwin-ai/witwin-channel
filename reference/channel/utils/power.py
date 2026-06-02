from __future__ import annotations

import drjit as dr
import witwin as wt

from .constants import POWER_DB_FLOOR


def to_power_db(a, imag=None):
    """Convert a complex field to power in dB while preserving gradients."""
    log10 = dr.log(wt.Float(10))
    if imag is not None:
        power = a * a + imag * imag
    else:
        power = dr.squared_norm(a)
    return 10 * dr.log(power + POWER_DB_FLOOR) / log10


__all__ = ["to_power_db"]
