"""Rough-surface scattering runtime (Kirchhoff tables + phase screens).

PyTorch-native GPU tensor code per the implementation contract (section 6):
tables are precomputed in float64 numpy at scene compile and evaluated /
sampled as float32 torch at runtime. This package deliberately stays torch
(no CUDA kernels)  -  table lookups are gather+FMA.
"""

from .energy import event_budget
from .phase_screen import (
    PhaseScreenRuntime,
    generate_gaussian_realization,
    patch_phase_integral,
    realization_seed,
)
from .tables import (
    KirchhoffTable,
    build_kirchhoff_table,
    eval_bsdf,
    pdf,
    pdf_reverse,
    sample_directions,
)

__all__ = [
    "KirchhoffTable",
    "PhaseScreenRuntime",
    "build_kirchhoff_table",
    "eval_bsdf",
    "event_budget",
    "generate_gaussian_realization",
    "patch_phase_integral",
    "pdf",
    "pdf_reverse",
    "realization_seed",
    "sample_directions",
]
