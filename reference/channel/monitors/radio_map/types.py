from __future__ import annotations

from enum import StrEnum


class RadioMapMetric(StrEnum):
    PATH_GAIN = "path_gain"
    RSS = "rss"
    SINR = "sinr"


class CombineMode(StrEnum):
    INCOHERENT = "incoherent"
    COHERENT = "coherent"


class ReceiverModel(StrEnum):
    MATCHED_ISOTROPIC = "matched_isotropic"
    PROJECTED_POLARIZED = "projected_polarized"


class AccumulationBackend(StrEnum):
    AUTO = "auto"
    BASELINE = "baseline"
    NATIVE_COHERENT = "native_coherent"
    CELL_ACCUMULATION = "cell_accumulation"
    NATIVE_MONTE_CARLO = "native_monte_carlo"


class SamplingMode(StrEnum):
    DETERMINISTIC = "deterministic"
    MONTE_CARLO = "monte_carlo"


class ShadowBoundaryMode(StrEnum):
    NONE = "none"
    UTD_CROSS_TERM_SURROGATE = "utd_cross_term_surrogate"
    PROJECTED_ISB_COMPLETION = "projected_isb_completion"
    MATCHED_ISB_COMPLETION = "matched_isb_completion"


class SurfaceMode(StrEnum):
    AXIS_ALIGNED = "axis_aligned"
    ORIENTED = "oriented"


__all__ = [
    "AccumulationBackend",
    "CombineMode",
    "RadioMapMetric",
    "ReceiverModel",
    "SamplingMode",
    "ShadowBoundaryMode",
    "SurfaceMode",
]
