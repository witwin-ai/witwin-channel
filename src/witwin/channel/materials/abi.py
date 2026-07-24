"""Channel-owned material ABI constants.

Logical material specifications are owned by :mod:`witwin.core`; this module
contains only the finite encodings consumed by native Channel kernels.
"""

from __future__ import annotations


MATERIAL_ABI_VERSION = 3
DIELECTRIC_MODEL_ID = 1
PEC_MODEL_ID = 2
PEC_EFFECTIVE_SIGMA_E = 1.0e9

__all__ = [
    "DIELECTRIC_MODEL_ID",
    "MATERIAL_ABI_VERSION",
    "PEC_EFFECTIVE_SIGMA_E",
    "PEC_MODEL_ID",
]
