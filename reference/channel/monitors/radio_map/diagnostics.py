from __future__ import annotations

from typing import Mapping

import drjit as dr
import numpy as np
import witwin as wt

from .monitor import RadioMapMonitor
from ...kernels.monitors.field.radio_map_accumulate import radiomap_vector_power
from ...utils import scalar
from ...utils.drjit_ops import ArrayInit, complex_abs_sqr
from ...utils.polarization import (
    project_real_polarization_to_ray,
    vector_zero,
    vector_from_scalar_and_real_direction,
)
from ..orchestration import ResolvedTraceConfig


PROJECTED_ISB_COMPLETION_RATIO_TARGET = 0.55
PROJECTED_ISB_COMPLETION_GAIN = 1.0
MATCHED_ISB_COMPLETION_BOUNDARY_HALF_FIELD = 0.5


def _gather_positions(positions, index_array: np.ndarray):
    safe_index = wt.UInt32(index_array.astype(np.uint32, copy=False))
    return wt.Point3f(
        dr.gather(wt.Float, positions.x, safe_index),
        dr.gather(wt.Float, positions.y, safe_index),
        dr.gather(wt.Float, positions.z, safe_index),
    )


def _raw_path_count(raw: Mapping[str, object]) -> int:
    return int(dr.width(raw["rx_index"]))


def _remap_raw_rx_index(raw: dict[str, object], group_indices: np.ndarray):
    if _raw_path_count(raw) == 0:
        return
    mapping = wt.UInt32(group_indices.astype(np.uint32, copy=False))
    raw["rx_index"] = dr.gather(wt.UInt32, mapping, raw["rx_index"])
    if "local_rx_index" in raw:
        raw["local_rx_index"] = dr.gather(wt.UInt32, mapping, raw["local_rx_index"])


def _zero_float(width: int):
    return dr.zeros(wt.Float, int(width))


def _shadow_boundary_mode(monitor: RadioMapMonitor) -> str:
    return str(getattr(monitor, "shadow_boundary_mode", "none"))


def _utd_cross_term_surrogate_enabled(monitor: RadioMapMonitor) -> bool:
    return _shadow_boundary_mode(monitor) == "utd_cross_term_surrogate"


def _projected_isb_completion_enabled(monitor: RadioMapMonitor) -> bool:
    return _shadow_boundary_mode(monitor) == "projected_isb_completion"


def _matched_isb_completion_enabled(monitor: RadioMapMonitor) -> bool:
    return _shadow_boundary_mode(monitor) == "matched_isb_completion"


def _ensure_utd_shadow_boundary_diagnostics(weighted_diagnostics, *, n_rx: int):
    for key in (
        "utd_surrogate_incident_cross",
        "utd_surrogate_reflection_cross",
        "utd_surrogate_total",
    ):
        weighted_diagnostics["incoherent"].setdefault(key, _zero_float(n_rx))
    return weighted_diagnostics


def _accumulate_complex_by_rx(raw: Mapping[str, object], *, n_rx: int):
    result = ArrayInit.complex_zero(n_rx)
    if _raw_path_count(raw) == 0:
        return result
    rx_index = raw["rx_index"]
    dr.scatter_reduce(dr.ReduceOp.Add, result.real, raw["a"].real, rx_index)
    dr.scatter_reduce(dr.ReduceOp.Add, result.imag, raw["a"].imag, rx_index)
    return result


def _accumulate_power_by_rx(raw: Mapping[str, object], *, n_rx: int):
    result = _zero_float(n_rx)
    if _raw_path_count(raw) == 0:
        return result
    dr.scatter_reduce(
        dr.ReduceOp.Add,
        result,
        complex_abs_sqr(raw["a"]),
        raw["rx_index"],
    )
    return result


def _add_complex(lhs, rhs):
    return wt.Complex2f(lhs.real + rhs.real, lhs.imag + rhs.imag)


def _scale_complex(value, scale: float):
    return wt.Complex2f(value.real * float(scale), value.imag * float(scale))


def _add_complex_vector(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return {axis: _add_complex(lhs[axis], rhs[axis]) for axis in ("x", "y", "z")}


def _scale_complex_vector(value, scale: float):
    if value is None:
        return None
    return {axis: _scale_complex(value[axis], scale) for axis in ("x", "y", "z")}


def _add_float(lhs, rhs):
    return lhs + rhs


def _scale_float(value, scale: float):
    return value * float(scale)


def _scatter_float(target, value, rx_idx):
    if target is None or value is None or dr.width(rx_idx) == 0:
        return target
    dr.scatter_reduce(dr.ReduceOp.Add, target, value, rx_idx)
    return target


def _single_tx_sinr(rss, *, noise_power: float):
    n_rx = int(dr.width(rss))
    if noise_power > 0.0:
        return rss / float(noise_power)
    inf = dr.full(wt.Float, float("inf"), n_rx)
    return dr.select(rss > 0.0, inf, _zero_float(n_rx))


def _vector_power(field_vector):
    return radiomap_vector_power(field_vector)


def _vector_power_symbolic(field_vector):
    return (
        complex_abs_sqr(field_vector["x"])
        + complex_abs_sqr(field_vector["y"])
        + complex_abs_sqr(field_vector["z"])
    )


def _diffraction_anchor_coordinate(axis: str, tx_pos, position: float) -> float:
    return float(position) if str(axis) == "z" else float(scalar(tx_pos.z))


def _empty_radio_map_diagnostics(n_rx: int, *, include_vector_coherent: bool = False):
    diagnostics = {
        "coherent": {
            "los": ArrayInit.complex_zero(n_rx),
            "reflection": ArrayInit.complex_zero(n_rx),
            "diffraction": ArrayInit.complex_zero(n_rx),
            "total": ArrayInit.complex_zero(n_rx),
        },
        "incoherent": {
            "los": _zero_float(n_rx),
            "reflection": _zero_float(n_rx),
            "diffraction": _zero_float(n_rx),
            "total": _zero_float(n_rx),
        },
        "coherent_power": {
            "los": _zero_float(n_rx),
            "reflection": _zero_float(n_rx),
            "diffraction": _zero_float(n_rx),
            "total": _zero_float(n_rx),
        },
    }
    if include_vector_coherent:
        diagnostics["vector_coherent"] = {
            "los": vector_zero(n_rx),
            "reflection": vector_zero(n_rx),
            "diffraction": vector_zero(n_rx),
            "total": vector_zero(n_rx),
        }
    return diagnostics


def _finalize_radio_map_component_totals(weighted_diagnostics):
    weighted_diagnostics["coherent"]["total"] = _add_complex(
        weighted_diagnostics["coherent"]["los"],
        _add_complex(
            weighted_diagnostics["coherent"]["reflection"],
            weighted_diagnostics["coherent"]["diffraction"],
        ),
    )
    weighted_diagnostics["incoherent"]["total"] = _add_float(
        weighted_diagnostics["incoherent"]["los"],
        _add_float(
            weighted_diagnostics["incoherent"]["reflection"],
            weighted_diagnostics["incoherent"]["diffraction"],
        ),
    )
    if "vector_coherent" in weighted_diagnostics:
        weighted_diagnostics["vector_coherent"]["total"] = _add_complex_vector(
            weighted_diagnostics["vector_coherent"]["los"],
            _add_complex_vector(
                weighted_diagnostics["vector_coherent"]["reflection"],
                weighted_diagnostics["vector_coherent"]["diffraction"],
            ),
        )
    return weighted_diagnostics


def _accumulate_sample_diagnostics_no_diff_matched_isotropic(
    weighted_diagnostics,
    *,
    sample_weight: float,
    los_coherent,
    reflection_coherent,
    los_power,
    reflection_power,
    los_field_vector,
    reflection_vector_coherent,
):
    weighted_diagnostics["coherent"]["los"] = _add_complex(
        weighted_diagnostics["coherent"]["los"],
        _scale_complex(los_coherent, sample_weight),
    )
    weighted_diagnostics["coherent"]["reflection"] = _add_complex(
        weighted_diagnostics["coherent"]["reflection"],
        _scale_complex(reflection_coherent, sample_weight),
    )
    weighted_diagnostics["incoherent"]["los"] = _add_float(
        weighted_diagnostics["incoherent"]["los"],
        _scale_float(los_power, sample_weight),
    )
    weighted_diagnostics["incoherent"]["reflection"] = _add_float(
        weighted_diagnostics["incoherent"]["reflection"],
        _scale_float(reflection_power, sample_weight),
    )
    weighted_diagnostics["coherent_power"]["los"] = _add_float(
        weighted_diagnostics["coherent_power"]["los"],
        _scale_float(los_power, sample_weight),
    )
    weighted_diagnostics["coherent_power"]["reflection"] = _add_float(
        weighted_diagnostics["coherent_power"]["reflection"],
        _scale_float(reflection_power, sample_weight),
    )
    vector_diagnostics = weighted_diagnostics.get("vector_coherent")
    if vector_diagnostics is not None:
        vector_diagnostics["los"] = _add_complex_vector(
            vector_diagnostics["los"],
            _scale_complex_vector(los_field_vector, sample_weight),
        )
        vector_diagnostics["reflection"] = _add_complex_vector(
            vector_diagnostics["reflection"],
            _scale_complex_vector(reflection_vector_coherent, sample_weight),
        )
    return weighted_diagnostics


def _finalize_no_diff_matched_isotropic_totals(
    weighted_diagnostics,
    *,
    compute_total_power: bool,
):
    total_coherent = _add_complex(
        weighted_diagnostics["coherent"]["los"],
        weighted_diagnostics["coherent"]["reflection"],
    )
    total_incoherent = _add_float(
        weighted_diagnostics["incoherent"]["los"],
        weighted_diagnostics["incoherent"]["reflection"],
    )
    weighted_diagnostics["coherent"]["total"] = total_coherent
    weighted_diagnostics["incoherent"]["total"] = total_incoherent
    total_vector = None
    if "vector_coherent" in weighted_diagnostics:
        total_vector = _add_complex_vector(
            weighted_diagnostics["vector_coherent"]["los"],
            weighted_diagnostics["vector_coherent"]["reflection"],
        )
        weighted_diagnostics["vector_coherent"]["total"] = total_vector
    total_power = None
    if compute_total_power and total_vector is not None:
        total_power = _vector_power_symbolic(total_vector)
        weighted_diagnostics["coherent_power"]["total"] = total_power
    return total_power


def _finalize_utd_shadow_boundary_surrogate_total(weighted_diagnostics):
    incident_cross = weighted_diagnostics["incoherent"]["utd_surrogate_incident_cross"]
    reflection_cross = weighted_diagnostics["incoherent"]["utd_surrogate_reflection_cross"]
    surrogate_total = weighted_diagnostics["incoherent"]["total"] + incident_cross + reflection_cross
    surrogate_total = dr.maximum(surrogate_total, wt.Float(0.0))
    weighted_diagnostics["incoherent"]["utd_surrogate_total"] = surrogate_total
    return surrogate_total


def _finalize_projected_isb_completion_total(weighted_diagnostics, completion_payload):
    completion = completion_payload["coherent"]
    completion_power = completion_payload["power"]
    weighted_diagnostics["coherent"]["projected_isb_completion"] = completion
    weighted_diagnostics["coherent_power"]["projected_isb_completion"] = completion_power
    weighted_diagnostics["incoherent"]["projected_isb_completion_weight"] = completion_payload[
        "incident_weight"
    ]
    weighted_diagnostics["incoherent"]["projected_isb_completion_deficiency"] = completion_payload[
        "deficiency"
    ]
    weighted_diagnostics["incoherent"]["projected_isb_continued_direct_power"] = completion_payload[
        "continued_direct_power"
    ]
    weighted_diagnostics["incoherent"]["projected_isb_amplitude_ratio"] = completion_payload[
        "amplitude_ratio"
    ]
    smoothed_diffraction = _add_complex(
        weighted_diagnostics["coherent"]["diffraction"],
        completion,
    )
    surrogate_total = _add_complex(weighted_diagnostics["coherent"]["total"], completion)
    surrogate_total_power = complex_abs_sqr(surrogate_total)
    weighted_diagnostics["coherent"]["diffraction"] = smoothed_diffraction
    weighted_diagnostics["coherent"]["total"] = surrogate_total
    weighted_diagnostics["coherent_power"]["diffraction"] = complex_abs_sqr(
        smoothed_diffraction
    )
    weighted_diagnostics["coherent_power"]["total"] = surrogate_total_power
    weighted_diagnostics["coherent"]["projected_isb_surrogate_total"] = surrogate_total
    weighted_diagnostics["coherent_power"]["projected_isb_surrogate_total"] = surrogate_total_power
    return surrogate_total_power


def _ensure_diffraction_breakdown_diagnostics(weighted_diagnostics):
    diffraction_power = weighted_diagnostics["coherent_power"]["diffraction"]
    weighted_diagnostics["coherent_power"].setdefault(
        "raw_diffraction",
        diffraction_power,
    )
    weighted_diagnostics["coherent_power"].setdefault(
        "matched_isb_completion_only",
        _scale_float(diffraction_power, 0.0),
    )
    weighted_diagnostics["coherent_power"]["folded_diffraction"] = weighted_diagnostics[
        "coherent_power"
    ]["diffraction"]


def _finalize_matched_isb_completion_total(weighted_diagnostics, completion_payload):
    raw_diffraction_power = weighted_diagnostics["coherent_power"]["diffraction"]
    completion = completion_payload["coherent"]
    completion_vector = completion_payload["vector_coherent"]
    completion_power = completion_payload["power"]
    weighted_diagnostics["coherent"]["matched_isb_completion"] = completion
    weighted_diagnostics["coherent_power"]["matched_isb_completion"] = completion_power
    weighted_diagnostics["coherent_power"]["matched_isb_completion_only"] = completion_power
    weighted_diagnostics["incoherent"]["matched_isb_completion_weight"] = completion_payload[
        "incident_weight"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_sum_incident_weight"] = completion_payload[
        "sum_incident_weight"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_max_incident_weight"] = completion_payload[
        "max_incident_weight"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_argmax_margin"] = completion_payload[
        "argmax_margin"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_support_edge_count"] = completion_payload[
        "support_edge_count"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_argmax_edge_idx"] = completion_payload[
        "argmax_edge_idx"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_continued_direct_power"] = completion_payload[
        "continued_direct_power"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_hard_visibility"] = completion_payload[
        "hard_visibility"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_transition_magnitude"] = completion_payload[
        "transition_magnitude"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_transition_phase"] = completion_payload[
        "transition_phase"
    ]
    if "vector_coherent" not in weighted_diagnostics:
        raise RuntimeError(
            "matched_isb_completion requires vector_coherent diagnostics buffers."
        )
    weighted_diagnostics["vector_coherent"]["matched_isb_completion"] = completion_vector
    smoothed_diffraction = _add_complex(
        weighted_diagnostics["coherent"]["diffraction"],
        completion,
    )
    smoothed_diffraction_vector = _add_complex_vector(
        weighted_diagnostics["vector_coherent"]["diffraction"],
        completion_vector,
    )
    surrogate_total = _add_complex(weighted_diagnostics["coherent"]["total"], completion)
    surrogate_total_vector = _add_complex_vector(
        weighted_diagnostics["vector_coherent"]["total"],
        completion_vector,
    )
    surrogate_total_power = _vector_power(surrogate_total_vector)
    weighted_diagnostics["coherent_power"]["raw_diffraction"] = raw_diffraction_power
    weighted_diagnostics["coherent"]["diffraction"] = smoothed_diffraction
    weighted_diagnostics["coherent"]["total"] = surrogate_total
    weighted_diagnostics["coherent_power"]["diffraction"] = _vector_power(
        smoothed_diffraction_vector
    )
    weighted_diagnostics["coherent_power"]["folded_diffraction"] = weighted_diagnostics[
        "coherent_power"
    ]["diffraction"]
    weighted_diagnostics["coherent_power"]["total"] = surrogate_total_power
    weighted_diagnostics["vector_coherent"]["diffraction"] = smoothed_diffraction_vector
    weighted_diagnostics["vector_coherent"]["total"] = surrogate_total_vector
    weighted_diagnostics["coherent"]["matched_isb_surrogate_total"] = surrogate_total
    weighted_diagnostics["coherent_power"]["matched_isb_surrogate_total"] = surrogate_total_power
    weighted_diagnostics["vector_coherent"]["matched_isb_surrogate_total"] = surrogate_total_vector
    return surrogate_total_power


def _finalize_matched_isb_completion_total_no_diff(weighted_diagnostics, completion_payload):
    raw_diffraction_power = weighted_diagnostics["coherent_power"]["diffraction"]
    completion = completion_payload["coherent"]
    completion_vector = completion_payload["vector_coherent"]
    completion_power = completion_payload["power"]
    weighted_diagnostics["coherent"]["matched_isb_completion"] = completion
    weighted_diagnostics["coherent_power"]["matched_isb_completion"] = completion_power
    weighted_diagnostics["coherent_power"]["matched_isb_completion_only"] = completion_power
    weighted_diagnostics["incoherent"]["matched_isb_completion_weight"] = completion_payload[
        "incident_weight"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_sum_incident_weight"] = completion_payload[
        "sum_incident_weight"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_max_incident_weight"] = completion_payload[
        "max_incident_weight"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_argmax_margin"] = completion_payload[
        "argmax_margin"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_support_edge_count"] = completion_payload[
        "support_edge_count"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_argmax_edge_idx"] = completion_payload[
        "argmax_edge_idx"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_continued_direct_power"] = completion_payload[
        "continued_direct_power"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_hard_visibility"] = completion_payload[
        "hard_visibility"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_transition_magnitude"] = completion_payload[
        "transition_magnitude"
    ]
    weighted_diagnostics["incoherent"]["matched_isb_transition_phase"] = completion_payload[
        "transition_phase"
    ]
    if "vector_coherent" not in weighted_diagnostics:
        raise RuntimeError(
            "matched_isb_completion requires vector_coherent diagnostics buffers."
        )
    weighted_diagnostics["vector_coherent"]["matched_isb_completion"] = completion_vector
    weighted_diagnostics["coherent_power"]["raw_diffraction"] = raw_diffraction_power
    weighted_diagnostics["coherent"]["diffraction"] = completion
    weighted_diagnostics["coherent_power"]["diffraction"] = completion_power
    weighted_diagnostics["coherent_power"]["folded_diffraction"] = completion_power
    weighted_diagnostics["vector_coherent"]["diffraction"] = completion_vector
    surrogate_total = _add_complex(weighted_diagnostics["coherent"]["total"], completion)
    surrogate_total_vector = _add_complex_vector(
        weighted_diagnostics["vector_coherent"]["total"],
        completion_vector,
    )
    surrogate_total_power = _vector_power_symbolic(surrogate_total_vector)
    weighted_diagnostics["coherent"]["total"] = surrogate_total
    weighted_diagnostics["coherent_power"]["total"] = surrogate_total_power
    weighted_diagnostics["vector_coherent"]["total"] = surrogate_total_vector
    weighted_diagnostics["coherent"]["matched_isb_surrogate_total"] = surrogate_total
    weighted_diagnostics["coherent_power"]["matched_isb_surrogate_total"] = surrogate_total_power
    weighted_diagnostics["vector_coherent"]["matched_isb_surrogate_total"] = surrogate_total_vector
    return surrogate_total_power


def _accumulate_sample_diagnostics(
    weighted_diagnostics,
    *,
    monitor: RadioMapMonitor,
    n_rx: int,
    sample_weight: float,
    sample_updates_component_buffers: bool,
    los_coherent,
    reflection_coherent,
    diffraction_coherent,
    los_power,
    reflection_power,
    diffraction_power,
    los_field_vector=None,
    reflection_vector_coherent=None,
    diffraction_vector_coherent=None,
    diffraction_incident_cross=None,
    diffraction_reflection_cross=None,
):
    if sample_updates_component_buffers:
        return weighted_diagnostics

    weighted_diagnostics["coherent"]["los"] = _add_complex(
        weighted_diagnostics["coherent"]["los"],
        _scale_complex(los_coherent, sample_weight),
    )
    weighted_diagnostics["coherent"]["reflection"] = _add_complex(
        weighted_diagnostics["coherent"]["reflection"],
        _scale_complex(reflection_coherent, sample_weight),
    )
    weighted_diagnostics["coherent"]["diffraction"] = _add_complex(
        weighted_diagnostics["coherent"]["diffraction"],
        _scale_complex(diffraction_coherent, sample_weight),
    )
    weighted_diagnostics["incoherent"]["los"] = _add_float(
        weighted_diagnostics["incoherent"]["los"],
        _scale_float(los_power, sample_weight),
    )
    weighted_diagnostics["incoherent"]["reflection"] = _add_float(
        weighted_diagnostics["incoherent"]["reflection"],
        _scale_float(reflection_power, sample_weight),
    )
    weighted_diagnostics["incoherent"]["diffraction"] = _add_float(
        weighted_diagnostics["incoherent"]["diffraction"],
        _scale_float(diffraction_power, sample_weight),
    )
    vector_diagnostics = weighted_diagnostics.get("vector_coherent")
    if vector_diagnostics is not None:
        vector_diagnostics["los"] = _add_complex_vector(
            vector_diagnostics["los"],
            _scale_complex_vector(los_field_vector, sample_weight),
        )
        vector_diagnostics["reflection"] = _add_complex_vector(
            vector_diagnostics["reflection"],
            _scale_complex_vector(reflection_vector_coherent, sample_weight),
        )
        vector_diagnostics["diffraction"] = _add_complex_vector(
            vector_diagnostics["diffraction"],
            _scale_complex_vector(diffraction_vector_coherent, sample_weight),
        )
    if _utd_cross_term_surrogate_enabled(monitor):
        weighted_diagnostics["incoherent"]["utd_surrogate_incident_cross"] = _add_float(
            weighted_diagnostics["incoherent"]["utd_surrogate_incident_cross"],
            _scale_float(diffraction_incident_cross, sample_weight),
        )
        weighted_diagnostics["incoherent"]["utd_surrogate_reflection_cross"] = _add_float(
            weighted_diagnostics["incoherent"]["utd_surrogate_reflection_cross"],
            _scale_float(diffraction_reflection_cross, sample_weight),
        )

    matched_isotropic_vector_coherent = (
        str(monitor.combine_mode) == "coherent"
        and str(monitor.receiver_model) == "matched_isotropic"
    )
    total_coherent = _add_complex(
        _add_complex(los_coherent, reflection_coherent),
        diffraction_coherent,
    )
    if matched_isotropic_vector_coherent:
        reflection_coherent_power = (
            _zero_float(n_rx)
            if reflection_vector_coherent is None
            else _vector_power(reflection_vector_coherent)
        )
        diffraction_coherent_power = (
            _zero_float(n_rx)
            if diffraction_vector_coherent is None
            else _vector_power(diffraction_vector_coherent)
        )
        total_vector = _add_complex_vector(
            _add_complex_vector(los_field_vector, reflection_vector_coherent),
            diffraction_vector_coherent,
        )
        total_coherent_power = (
            _zero_float(n_rx) if total_vector is None else _vector_power(total_vector)
        )
        weighted_diagnostics["coherent_power"]["los"] = _add_float(
            weighted_diagnostics["coherent_power"]["los"],
            _scale_float(los_power, sample_weight),
        )
        weighted_diagnostics["coherent_power"]["reflection"] = _add_float(
            weighted_diagnostics["coherent_power"]["reflection"],
            _scale_float(reflection_coherent_power, sample_weight),
        )
        weighted_diagnostics["coherent_power"]["diffraction"] = _add_float(
            weighted_diagnostics["coherent_power"]["diffraction"],
            _scale_float(diffraction_coherent_power, sample_weight),
        )
        weighted_diagnostics["coherent_power"]["total"] = _add_float(
            weighted_diagnostics["coherent_power"]["total"],
            _scale_float(total_coherent_power, sample_weight),
        )
        return weighted_diagnostics

    weighted_diagnostics["coherent_power"]["los"] = _add_float(
        weighted_diagnostics["coherent_power"]["los"],
        _scale_float(complex_abs_sqr(los_coherent), sample_weight),
    )
    weighted_diagnostics["coherent_power"]["reflection"] = _add_float(
        weighted_diagnostics["coherent_power"]["reflection"],
        _scale_float(complex_abs_sqr(reflection_coherent), sample_weight),
    )
    weighted_diagnostics["coherent_power"]["diffraction"] = _add_float(
        weighted_diagnostics["coherent_power"]["diffraction"],
        _scale_float(complex_abs_sqr(diffraction_coherent), sample_weight),
    )
    weighted_diagnostics["coherent_power"]["total"] = _add_float(
        weighted_diagnostics["coherent_power"]["total"],
        _scale_float(complex_abs_sqr(total_coherent), sample_weight),
    )
    return weighted_diagnostics


def _baseline_los_power(
    *,
    monitor: RadioMapMonitor,
    sample_positions,
    tx_pos,
    config: ResolvedTraceConfig,
    los_coherent,
):
    if str(monitor.receiver_model) == "matched_isotropic":
        ray_dir = sample_positions - tx_pos
        tx_pol_dir = project_real_polarization_to_ray(config.tx_polarization, ray_dir)
        los_field_vector = vector_from_scalar_and_real_direction(los_coherent, tx_pol_dir)
        return _vector_power(los_field_vector), los_field_vector
    return complex_abs_sqr(los_coherent), None


__all__ = [
    "MATCHED_ISB_COMPLETION_BOUNDARY_HALF_FIELD",
    "PROJECTED_ISB_COMPLETION_GAIN",
    "PROJECTED_ISB_COMPLETION_RATIO_TARGET",
    "_accumulate_complex_by_rx",
    "_accumulate_power_by_rx",
    "_accumulate_sample_diagnostics",
    "_add_complex",
    "_add_complex_vector",
    "_add_float",
    "_baseline_los_power",
    "_diffraction_anchor_coordinate",
    "_empty_radio_map_diagnostics",
    "_ensure_diffraction_breakdown_diagnostics",
    "_ensure_utd_shadow_boundary_diagnostics",
    "_finalize_matched_isb_completion_total",
    "_finalize_projected_isb_completion_total",
    "_finalize_radio_map_component_totals",
    "_finalize_utd_shadow_boundary_surrogate_total",
    "_gather_positions",
    "_matched_isb_completion_enabled",
    "_projected_isb_completion_enabled",
    "_raw_path_count",
    "_remap_raw_rx_index",
    "_scale_complex",
    "_scale_complex_vector",
    "_scale_float",
    "_scatter_float",
    "_shadow_boundary_mode",
    "_single_tx_sinr",
    "_utd_cross_term_surrogate_enabled",
    "_vector_power",
    "_zero_float",
]
