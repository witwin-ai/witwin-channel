from __future__ import annotations

from ..common import normalize_max_diffractions_override, normalize_ray_mode
from . import types as rm_types


def normalize_metric(metric: str) -> rm_types.RadioMapMetric:
    resolved = str(metric).lower()
    if resolved not in {
        rm_types.RadioMapMetric.PATH_GAIN,
        rm_types.RadioMapMetric.RSS,
        rm_types.RadioMapMetric.SINR,
    }:
        raise ValueError("metric must be 'path_gain', 'rss', or 'sinr'.")
    return rm_types.RadioMapMetric(resolved)


def normalize_combine_mode(value: str) -> rm_types.CombineMode:
    resolved = str(value).lower()
    if resolved not in {rm_types.CombineMode.INCOHERENT, rm_types.CombineMode.COHERENT}:
        raise ValueError("combine_mode must be 'incoherent' or 'coherent'.")
    return rm_types.CombineMode(resolved)


def normalize_receiver_model(
    value: str | None,
    *,
    combine_mode: rm_types.CombineMode,
    surface_mode: rm_types.SurfaceMode,
) -> rm_types.ReceiverModel:
    if value is None:
        if (
            combine_mode == rm_types.CombineMode.INCOHERENT
            and surface_mode == rm_types.SurfaceMode.AXIS_ALIGNED
        ):
            return rm_types.ReceiverModel.MATCHED_ISOTROPIC
        return rm_types.ReceiverModel.PROJECTED_POLARIZED
    resolved = str(value).lower()
    if resolved not in {
        rm_types.ReceiverModel.MATCHED_ISOTROPIC,
        rm_types.ReceiverModel.PROJECTED_POLARIZED,
    }:
        raise ValueError(
            "receiver_model must be 'matched_isotropic' or 'projected_polarized'."
        )
    return rm_types.ReceiverModel(resolved)


def normalize_accumulation_backend(value: str) -> rm_types.AccumulationBackend:
    resolved = str(value).lower()
    if resolved not in {
        rm_types.AccumulationBackend.AUTO,
        rm_types.AccumulationBackend.BASELINE,
        rm_types.AccumulationBackend.NATIVE_COHERENT,
        rm_types.AccumulationBackend.CELL_ACCUMULATION,
        rm_types.AccumulationBackend.NATIVE_MONTE_CARLO,
    }:
        raise ValueError(
            "accumulation_backend must be 'auto', 'baseline', "
            "'native_coherent', 'cell_accumulation', or 'native_monte_carlo'."
        )
    return rm_types.AccumulationBackend(resolved)


def normalize_shadow_boundary_mode(value: str) -> rm_types.ShadowBoundaryMode:
    resolved = str(value).lower()
    if resolved not in {
        rm_types.ShadowBoundaryMode.NONE,
        rm_types.ShadowBoundaryMode.UTD_CROSS_TERM_SURROGATE,
        rm_types.ShadowBoundaryMode.PROJECTED_ISB_COMPLETION,
        rm_types.ShadowBoundaryMode.MATCHED_ISB_COMPLETION,
    }:
        raise ValueError(
            "shadow_boundary_mode must be 'none', 'utd_cross_term_surrogate', "
            "'projected_isb_completion', or 'matched_isb_completion'."
        )
    return rm_types.ShadowBoundaryMode(resolved)


def normalize_shadow_support_cutoff_db(value) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if resolved < 0.0:
        raise ValueError("shadow_support_cutoff_db must be >= 0 when provided.")
    return resolved


def normalize_positive_power(value, *, name: str, allow_zero: bool = True) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if resolved < 0.0 or (not allow_zero and resolved <= 0.0):
        comparator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {comparator}.")
    return resolved


def normalize_sampling_mode(value: str) -> rm_types.SamplingMode:
    resolved = str(value).lower()
    if resolved not in {rm_types.SamplingMode.DETERMINISTIC, rm_types.SamplingMode.MONTE_CARLO}:
        raise ValueError("sampling_mode must be 'deterministic' or 'monte_carlo'.")
    return rm_types.SamplingMode(resolved)


def normalize_positive_int(value, *, name: str) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be > 0.")
    return resolved


def normalize_nonnegative_int(value, *, name: str) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be >= 0.")
    return resolved


def normalize_probability(value, *, name: str) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if resolved <= 0.0 or resolved > 1.0:
        raise ValueError(f"{name} must be in the range (0, 1].")
    return resolved


def normalize_nonnegative_threshold(value, *, name: str) -> float:
    if value is None:
        return 0.0
    resolved = float(value)
    if resolved < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return resolved


def normalize_seed(value) -> int:
    if value is None:
        return 0
    return int(value)


def normalize_optional_bool(value, *, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a bool or None.")


__all__ = [
    "normalize_accumulation_backend",
    "normalize_combine_mode",
    "normalize_max_diffractions_override",
    "normalize_metric",
    "normalize_nonnegative_int",
    "normalize_nonnegative_threshold",
    "normalize_optional_bool",
    "normalize_positive_int",
    "normalize_positive_power",
    "normalize_probability",
    "normalize_ray_mode",
    "normalize_receiver_model",
    "normalize_sampling_mode",
    "normalize_seed",
    "normalize_shadow_boundary_mode",
    "normalize_shadow_support_cutoff_db",
]
