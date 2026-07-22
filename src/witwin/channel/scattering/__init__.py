"""Rough-surface scattering runtime (Kirchhoff tables + phase screens).

Tables are precomputed once in float64 at scene compile, uploaded once, and
all production eval/sample/PDF/event-budget operations require native CUDA.
PyTorch and CPU implementations are not production runtime alternatives.
"""

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
    "generate_gaussian_realization",
    "patch_phase_integral",
    "pdf",
    "pdf_reverse",
    "realization_seed",
    "sample_directions",
]
