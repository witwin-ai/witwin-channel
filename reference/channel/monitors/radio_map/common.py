from __future__ import annotations

from typing import Mapping

import drjit as dr
import numpy as np
import witwin as wt

from .monitor import RadioMapMonitor
from ..._native import native_extension_available
from ...kernels.monitors.field.radio_map_accumulate import radiomap_vector_power
from ...scene import Scene
from ...trace.materials import ReflectionTraceDetail
from ...utils import scalar
from ...utils.drjit_ops import ArrayInit, complex_abs_sqr
from ...utils.polarization import (
    project_real_polarization_to_ray,
    vector_zero,
    vector_from_scalar_and_real_direction,
)
from ..orchestration import ResolvedTraceConfig
from ..profiler import capture_cuda_memory_report


PROJECTED_ISB_COMPLETION_RATIO_TARGET = 0.55
PROJECTED_ISB_COMPLETION_GAIN = 1.0
MATCHED_ISB_COMPLETION_BOUNDARY_HALF_FIELD = 0.5


def _normalize_radio_map_accumulation_backend(value: str) -> str:
    resolved = str(value).lower()
    if resolved not in {
        "auto",
        "baseline",
        "native_coherent",
        "cell_accumulation",
        "native_monte_carlo",
    }:
        raise ValueError(
            "radio_map_accumulation_backend must be 'auto', 'baseline', "
            "'native_coherent', 'cell_accumulation', or 'native_monte_carlo'."
        )
    return resolved


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


def _resolve_noise_power(scene: Scene, monitor: RadioMapMonitor) -> tuple[float, str]:
    if monitor.noise_power is not None:
        return float(monitor.noise_power), "monitor_override"
    metadata = getattr(scene, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("noise_power", "thermal_noise_power"):
            if key in metadata:
                return float(metadata[key]), f"scene_metadata:{key}"
    for key in ("noise_power", "thermal_noise_power"):
        value = getattr(scene, key, None)
        if value is not None:
            return float(value), f"scene_attribute:{key}"
    return 0.0, "default_zero"


def _single_tx_sinr(rss, *, noise_power: float):
    n_rx = int(dr.width(rss))
    if noise_power > 0.0:
        return rss / float(noise_power)
    inf = dr.full(wt.Float, float("inf"), n_rx)
    return dr.select(rss > 0.0, inf, _zero_float(n_rx))


def _count_nonzero_complex(value) -> int:
    array = np.asarray(value, dtype=np.complex64)
    return int(np.count_nonzero(np.abs(array) > 0.0))


def _count_reflection_paths(reflection_detail: ReflectionTraceDetail | None) -> int:
    if reflection_detail is None:
        return 0
    groups = tuple(reflection_detail.source_paths_per_bounce)
    total = 0
    for group in groups:
        if group is None:
            continue
        total += int(group.get("n_paths", 0))
    return int(total)


def _vector_power(field_vector):
    return radiomap_vector_power(field_vector)


def _diffraction_anchor_coordinate(axis: str, tx_pos, position: float) -> float:
    return float(position) if str(axis) == "z" else float(scalar(tx_pos.z))


def _radio_map_diffraction_state_layout(accumulation_backend: str) -> str:
    from ...trace.diffraction.state import PATH_EXPORT_REDUCED_STATE_LAYOUT

    return (
        "full"
        if str(accumulation_backend) == "native_coherent"
        else PATH_EXPORT_REDUCED_STATE_LAYOUT
    )


def _radio_map_diffraction_cache_key(base_key, *, state_layout: str):
    if base_key is None:
        return None
    return tuple(base_key) + ("radio_map_state_layout", str(state_layout))


def _radio_map_native_coherent_supported(
    monitor: RadioMapMonitor,
    grid,
    config: ResolvedTraceConfig,
    *,
    scene: Scene | None = None,
) -> bool:
    return (
        grid.surface_mode == "axis_aligned"
        and monitor.combine_mode == "coherent"
        and str(monitor.receiver_model) == "projected_polarized"
        and native_extension_available()
        and str(config.reflection_field_backend) == "native"
        and str(config.diffraction_execution.suffix_backend) == "native"
    )


def _point_grad_enabled(point) -> bool:
    if point is None:
        return False
    for axis in ("x", "y", "z"):
        component = getattr(point, axis, None)
        if component is None:
            continue
        try:
            if bool(dr.grad_enabled(component)):
                return True
        except Exception:
            continue
    return False


def _scene_geometry_grad_enabled(scene) -> bool:
    if scene is None:
        return False
    vertices = getattr(scene, "vertices", None)
    if vertices is not None and _point_grad_enabled(vertices):
        return True
    tri_data = getattr(scene, "tri_data_gpu", None)
    if isinstance(tri_data, dict):
        for key in ("v0", "v1", "v2"):
            value = tri_data.get(key)
            if value is not None and _point_grad_enabled(value):
                return True
    return False


def _scene_material_grad_enabled(scene) -> bool:
    if scene is None:
        return False
    tri_data = getattr(scene, "tri_data_gpu", None)
    if isinstance(tri_data, dict):
        for key in ("material_eps_r", "material_sigma_e"):
            value = tri_data.get(key)
            if value is None:
                continue
            try:
                if bool(dr.grad_enabled(value)):
                    return True
            except Exception:
                continue
    return False


def _radio_map_cell_accumulation_supported(
    monitor: RadioMapMonitor,
    grid,
    config: ResolvedTraceConfig,
    *,
    tx_pos,
    scene: Scene,
) -> bool:
    matched_isotropic_vector_coherent = (
        str(monitor.combine_mode) == "coherent"
        and str(monitor.receiver_model) == "matched_isotropic"
    )
    if (
        grid.surface_mode != "axis_aligned"
        or (
            str(monitor.combine_mode) != "incoherent"
            and not matched_isotropic_vector_coherent
        )
    ):
        return False
    if str(monitor.receiver_model) not in {"projected_polarized", "matched_isotropic"}:
        return False
    if _point_grad_enabled(tx_pos):
        return False
    if _scene_geometry_grad_enabled(scene):
        return False
    if (
        bool(config.use_scene_materials_for_reflection)
        or bool(config.use_scene_materials_for_diffraction)
    ) and _scene_material_grad_enabled(scene):
        return False
    return True


def _radio_map_native_monte_carlo_supported(
    monitor: RadioMapMonitor,
    grid,
    config: ResolvedTraceConfig,
    *,
    tx_pos,
    scene: Scene,
) -> bool:
    if (
        str(getattr(monitor, "sampling_mode", "deterministic")) != "monte_carlo"
        or grid.surface_mode != "axis_aligned"
        or str(monitor.combine_mode) != "incoherent"
        or str(monitor.receiver_model) != "matched_isotropic"
        or not native_extension_available()
    ):
        return False
    return True


def _resolve_radio_map_accumulation_backend(
    *,
    requested_backend: str,
    monitor: RadioMapMonitor,
    grid,
    config: ResolvedTraceConfig,
    tx_pos,
    scene: Scene,
) -> str:
    resolved_requested = _normalize_radio_map_accumulation_backend(requested_backend)
    if str(getattr(monitor, "sampling_mode", "deterministic")) == "monte_carlo":
        if resolved_requested == "auto":
            if _radio_map_native_monte_carlo_supported(
                monitor,
                grid,
                config,
                tx_pos=tx_pos,
                scene=scene,
            ):
                return "native_monte_carlo"
            raise RuntimeError(
                "sampling_mode='monte_carlo' requires the bundled native extension and an "
                "axis-aligned matched-isotropic radio-map workload."
            )
        if resolved_requested != "native_monte_carlo":
            raise RuntimeError(
                "sampling_mode='monte_carlo' only supports "
                "radio_map_accumulation_backend='auto' or 'native_monte_carlo'."
            )
        if not _radio_map_native_monte_carlo_supported(
            monitor,
            grid,
            config,
            tx_pos=tx_pos,
            scene=scene,
        ):
            raise RuntimeError(
                "radio_map_accumulation_backend='native_monte_carlo' requires the bundled "
                "native extension and an axis-aligned matched-isotropic Monte Carlo "
                "radio-map workload."
            )
        return resolved_requested
    if resolved_requested == "auto":
        if _radio_map_native_coherent_supported(monitor, grid, config, scene=scene):
            return "native_coherent"
        if _radio_map_cell_accumulation_supported(
            monitor,
            grid,
            config,
            tx_pos=tx_pos,
            scene=scene,
        ):
            return "cell_accumulation"
        return "baseline"
    if resolved_requested == "native_coherent" and not _radio_map_native_coherent_supported(
        monitor,
        grid,
        config,
        scene=scene,
    ):
        raise RuntimeError(
            "radio_map_accumulation_backend='native_coherent' requires an axis-aligned "
            "RadioMapMonitor with combine_mode='coherent', native_extension_available()==True, "
            "and trace.reflection_field_backend='native', diffraction_execution.suffix_backend='native'."
        )
    if resolved_requested == "cell_accumulation" and not _radio_map_cell_accumulation_supported(
        monitor,
        grid,
        config,
        tx_pos=tx_pos,
        scene=scene,
    ):
        raise RuntimeError(
            "radio_map_accumulation_backend='cell_accumulation' requires an axis-aligned "
            "RadioMapMonitor with either combine_mode='incoherent' or "
            "combine_mode='coherent' plus receiver_model='matched_isotropic', and a "
            "gradient-disabled workload."
        )
    return resolved_requested


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


def _finalize_matched_isb_completion_total(weighted_diagnostics, completion_payload):
    completion = completion_payload["coherent"]
    completion_vector = completion_payload["vector_coherent"]
    completion_power = completion_payload["power"]
    weighted_diagnostics["coherent"]["matched_isb_completion"] = completion
    weighted_diagnostics["coherent_power"]["matched_isb_completion"] = completion_power
    weighted_diagnostics["incoherent"]["matched_isb_completion_weight"] = completion_payload[
        "incident_weight"
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
    weighted_diagnostics["coherent"]["diffraction"] = smoothed_diffraction
    weighted_diagnostics["coherent"]["total"] = surrogate_total
    weighted_diagnostics["coherent_power"]["diffraction"] = _vector_power(
        smoothed_diffraction_vector
    )
    weighted_diagnostics["coherent_power"]["total"] = surrogate_total_power
    weighted_diagnostics["vector_coherent"]["diffraction"] = smoothed_diffraction_vector
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
        total_coherent_power = _zero_float(n_rx) if total_vector is None else _vector_power(total_vector)
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


def _build_radio_map_metadata(
    *,
    monitor: RadioMapMonitor,
    grid,
    scene: Scene,
    solver_controls,
    path_counts,
    sample_metadata,
    aggregate_runtime_reuse,
    aggregate_runtime_backends,
    reflection_detail: ReflectionTraceDetail | None,
    radio_map_accumulation_backend: str,
    resolved_accumulation_backend: str,
    noise_power: float,
    noise_power_source: str,
):
    matched_isb_completion = _matched_isb_completion_enabled(monitor)
    projected_isb_completion = _projected_isb_completion_enabled(monitor)
    utd_cross_term_surrogate = _utd_cross_term_surrogate_enabled(monitor)
    matched_isotropic_vector_coherent = (
        str(monitor.combine_mode) == "coherent"
        and str(monitor.receiver_model) == "matched_isotropic"
    )
    if projected_isb_completion:
        coherent_diag_definition = (
            "sample-weighted projected path-coefficient sums with shadow-boundary completion "
            "folded into diffraction and total diagnostics while explicit completion-only "
            "diagnostics remain available"
        )
        coherent_power_definition = (
            "sample-weighted squared magnitude of coherent projected sums with shadow-boundary "
            "completion folded into diffraction and total diagnostics"
        )
    elif matched_isb_completion:
        coherent_diag_definition = (
            "sample-weighted projected component diagnostics with ISB-completed diffraction "
            "and total retained alongside matched-isotropic vector coherent path gain"
        )
        coherent_power_definition = (
            "sample-weighted matched-isotropic vector coherent powers per component with "
            "ISB-completed diffraction and total, while explicit completion-only diagnostics "
            "remain separate"
        )
    elif str(monitor.receiver_model) == "projected_polarized":
        coherent_diag_definition = "sample-weighted projected path-coefficient sums"
        coherent_power_definition = "sample-weighted squared magnitude of coherent projected sums"
    elif matched_isotropic_vector_coherent:
        coherent_diag_definition = (
            "sample-weighted projected component diagnostics retained for parity debugging while "
            "path_gain uses matched-isotropic vector coherent accumulation"
        )
        coherent_power_definition = (
            "sample-weighted matched-isotropic vector coherent powers per component and total"
        )
    else:
        coherent_diag_definition = (
            "sample-weighted projected component diagnostics retained for parity debugging while "
            "path_gain uses matched-isotropic power accumulation"
        )
        coherent_power_definition = (
            "sample-weighted squared magnitude of projected coherent diagnostics retained alongside "
            "matched-isotropic incoherent path-gain accumulation"
        )
    metadata = {
        "receiver_sampling": {
            "monitor_name": monitor.name,
            "monitor_kind": monitor.kind,
            "sampling_mode": str(getattr(monitor, "sampling_mode", "deterministic")),
            "surface_mode": grid.surface_mode,
            "axis": grid.axis,
            "position": grid.position,
            "bounds": grid.bounds,
            "center": grid.center,
            "orientation": grid.orientation,
            "size": grid.size,
            "tangential_axes": grid.tangential_axes,
            "grid_shape": grid.grid_shape,
            "cell_size": grid.cell_size,
            "cell_centered": True,
            "quadrature_mode": monitor.quadrature_mode,
            "samples_per_cell": monitor.samples_per_cell,
            "sample_offsets_local": tuple(sample.offset_local for sample in grid.sample_sets),
        },
        "metric_contract": {
            "path_gain": (
                (
                    "squared_norm_of_matched_isotropic_vector_coherent_sum_plus_isb_visibility_"
                    "completion_weighted_by_fixed_cell_quadrature"
                )
                if matched_isb_completion
                else (
                    "squared_norm_of_matched_isotropic_vector_coherent_sum_weighted_by_fixed_cell_quadrature"
                    if matched_isotropic_vector_coherent
                    else (
                        (
                            "squared_norm_of_receiver_model_matched_coherent_sum_plus_projected_isb_"
                            "shadow_completion_weighted_by_fixed_cell_quadrature"
                        )
                        if projected_isb_completion
                        else "squared_norm_of_receiver_model_matched_coherent_sum_weighted_by_fixed_cell_quadrature"
                    )
                    if monitor.combine_mode == "coherent"
                    else (
                        (
                            "sum_vector_component_powers_plus_utd_shadow_boundary_surrogate_cross_terms_"
                            "weighted_by_fixed_cell_quadrature"
                        )
                        if utd_cross_term_surrogate
                        else "sum_vector_component_powers_weighted_by_fixed_cell_quadrature"
                    )
                    if str(monitor.receiver_model) == "matched_isotropic"
                    else "sum_squared_projected_path_amplitudes_weighted_by_fixed_cell_quadrature"
                )
            ),
            "rss": "tx_power_times_path_gain",
            "sinr": "rss_over_noise_plus_other_tx_rss",
        },
        "metric": monitor.metric,
        "combine_mode": monitor.combine_mode,
        "receiver_model": monitor.receiver_model,
        "shadow_boundary_mode": getattr(monitor, "shadow_boundary_mode", "none"),
        "shadow_support_cutoff_db": getattr(monitor, "shadow_support_cutoff_db", None),
        "tx_power": float(monitor.tx_power),
        "noise_power": float(noise_power),
        "noise_power_source": str(noise_power_source),
        "path_counts": path_counts,
        "sample_evaluations": tuple(sample_metadata),
        "runtime_reuse": {
            "diffraction_state_prep_cache": {
                "mode": str(aggregate_runtime_reuse["cache_mode"]),
                "hits": int(aggregate_runtime_reuse["state_preparation_hits"]),
                "misses": int(aggregate_runtime_reuse["state_preparation_misses"]),
                "state_layout": str(aggregate_runtime_reuse["state_layout"]),
            },
        },
        "accumulation_backend": {
            "requested": _normalize_radio_map_accumulation_backend(radio_map_accumulation_backend),
            "resolved": resolved_accumulation_backend,
            "cell_accumulation_mode": (
                "direct_in_loop_scatter"
                if resolved_accumulation_backend in {"cell_accumulation", "native_monte_carlo"}
                else (
                    "sample_reduction"
                )
            ),
        },
        "solver_mode": solver_controls,
        "execution_intent": dict(solver_controls["execution_intent"]),
        "performance_memory": {
            "torch_cuda": capture_cuda_memory_report(),
        },
        "coherent_diagnostics": {
            "definition": coherent_diag_definition,
            "components": ("los", "reflection", "diffraction", "total"),
        },
        "coherent_power_diagnostics": {
            "definition": coherent_power_definition,
            "components": ("los", "reflection", "diffraction", "total"),
        },
        "shadow_boundary_surrogate": {
            "mode": _shadow_boundary_mode(monitor),
            "enabled": bool(_shadow_boundary_mode(monitor) != "none"),
            "support_cutoff_db": getattr(monitor, "shadow_support_cutoff_db", None),
            "cross_term_model": (
                "matched_isotropic_vector_inner_product"
                if utd_cross_term_surrogate
                else None
            ),
            "completion_model": (
                "projected_shadow_side_direct_continuation"
                if projected_isb_completion
                else (
                    "matched_isotropic_isb_scene_edge_complex_transition_residual_completion"
                    if matched_isb_completion
                    else None
                )
            ),
            "visibility_model": (
                "scene_edge_incident_transition_weighted_average_with_incident_gated_direct_mode_residual_matching"
                if matched_isb_completion
                else None
            ),
            "diagnostic_components": (
                "utd_surrogate_incident_cross",
                "utd_surrogate_reflection_cross",
                "utd_surrogate_total",
            )
            if utd_cross_term_surrogate
            else (
                (
                    "projected_isb_completion_weight",
                    "projected_isb_completion_deficiency",
                    "projected_isb_continued_direct_power",
                    "projected_isb_amplitude_ratio",
                    "projected_isb_completion",
                    "projected_isb_surrogate_total",
                )
                if projected_isb_completion
                else (
                    (
                        "matched_isb_completion_weight",
                        "matched_isb_continued_direct_power",
                        "matched_isb_hard_visibility",
                        "matched_isb_transition_magnitude",
                        "matched_isb_transition_phase",
                        "matched_isb_completion",
                        "matched_isb_surrogate_total",
                    )
                    if matched_isb_completion
                    else ()
                )
            ),
            "parameters": (
                {
                    "ratio_target": float(PROJECTED_ISB_COMPLETION_RATIO_TARGET),
                    "completion_gain": float(PROJECTED_ISB_COMPLETION_GAIN),
                }
                if projected_isb_completion
                else (
                    {
                        "shadow_boundary_half_field_limit": float(
                            MATCHED_ISB_COMPLETION_BOUNDARY_HALF_FIELD
                        ),
                    }
                    if matched_isb_completion
                    else {}
                )
            ),
        },
    }
    if reflection_detail is not None:
        metadata["reflection_sampling"] = dict(reflection_detail.get("reflection_sampling", {}))
        metadata["reflection_backend"] = dict(reflection_detail.get("dda_stats", {}))
    reflection_runtime_backend = {}
    if metadata.get("reflection_backend"):
        reflection_runtime_backend.update(dict(metadata["reflection_backend"]))
    reflection_runtime_backend.update(dict(aggregate_runtime_backends["reflection"]))
    metadata["runtime_backends"] = {
        "reflection": reflection_runtime_backend,
        "diffraction": dict(aggregate_runtime_backends["diffraction"]),
        "suffix": dict(aggregate_runtime_backends["suffix"]),
    }
    return metadata


def _build_radio_map_result_payload(
    *,
    monitor: RadioMapMonitor,
    grid,
    weighted_diagnostics,
    metadata,
    path_gain,
    rss,
    sinr,
    tx_pos,
    noise_power: float,
    sample_payload_positions,
    timing,
):
    n_rx = int(grid.n_cells)
      = {
        "name": monitor.name,
        "kind": monitor.kind,
        "metric": monitor.metric,
        "combine_mode": monitor.combine_mode,
        "receiver_model": monitor.receiver_model,
        "grid_shape": grid.grid_shape,
        "cell_size": grid.cell_size,
        "surface": grid.surface_descriptor(),
        "coords": {
            "grid_x": grid.grid_x,
            "grid_y": grid.grid_y,
            "x": grid.x_coords,
            "y": grid.y_coords,
            "axis_x": grid.tangential_axes[0],
            "axis_y": grid.tangential_axes[1],
            "cell_centers": grid.cell_centers,
            "sample_positions": tuple(sample_payload_positions),
        },
        "metrics": {
            "path_gain": path_gain,
            "rss": rss,
            "sinr": sinr,
            "tx_association": dr.zeros(wt.Int32, n_rx),
        },
        "diagnostics": {
            "coherent": weighted_diagnostics["coherent"],
            "incoherent": weighted_diagnostics["incoherent"],
            "coherent_power": weighted_diagnostics["coherent_power"],
        },
        "metadata": metadata,
        "tx_pos": (scalar(tx_pos.x), scalar(tx_pos.y), scalar(tx_pos.z)),
        "tx_power": float(monitor.tx_power),
        "noise_power": float(noise_power),
    }
    if timing is not None:
        result["timing"] = timing
    return result


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
    "_build_radio_map_metadata",
    "_build_radio_map_result_payload",
    "_count_nonzero_complex",
    "_count_reflection_paths",
    "_diffraction_anchor_coordinate",
    "_empty_radio_map_diagnostics",
    "_ensure_utd_shadow_boundary_diagnostics",
    "_finalize_matched_isb_completion_total",
    "_finalize_projected_isb_completion_total",
    "_finalize_radio_map_component_totals",
    "_finalize_utd_shadow_boundary_surrogate_total",
    "_gather_positions",
    "_normalize_radio_map_accumulation_backend",
    "_matched_isb_completion_enabled",
    "_point_grad_enabled",
    "_projected_isb_completion_enabled",
    "_radio_map_diffraction_cache_key",
    "_radio_map_diffraction_state_layout",
    "_raw_path_count",
    "_remap_raw_rx_index",
    "_resolve_noise_power",
    "_resolve_radio_map_accumulation_backend",
    "_scale_complex",
    "_scale_complex_vector",
    "_scale_float",
    "_scatter_float",
    "_scene_geometry_grad_enabled",
    "_scene_material_grad_enabled",
    "_shadow_boundary_mode",
    "_single_tx_sinr",
    "_utd_cross_term_surrogate_enabled",
    "_vector_power",
    "_zero_float",
]
