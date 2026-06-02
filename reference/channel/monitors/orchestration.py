"""Shared helpers for monitor-specific trace orchestration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from ..config import DiffractionExecutionConfig, ReflectionFieldBackend, TraceConfig
from ..utils import scalar
from ..utils.constants import SPEED_OF_LIGHT

FAST_APPROXIMATE_REFLECTION_RAYS_CAP = 1024
FAST_APPROXIMATE_REFLECTION_BOUNCES_CAP = 1
FAST_APPROXIMATE_MAX_DIFFRACTIONS_CAP = 3
FAST_APPROXIMATE_TOTAL_STATE_BUDGET = 768
FAST_APPROXIMATE_INSERTED_STATE_BUDGET = 256
FAST_APPROXIMATE_INSERTED_REFLECTION_DEPTH_CAP = 1
MEMORY_SAFE_TOTAL_STATE_BUDGET = 2048
MEMORY_SAFE_INSERTED_STATE_BUDGET = 512
MEMORY_SAFE_INSERTED_REFLECTION_DEPTH_CAP = 1

TraceExecutionIntent = Literal[
    "field",
    "field_scalar_only",
    "path_export",
    "radio_map_coherent",
    "radio_map_incoherent",
]


@dataclass(frozen=True)
class ResolvedTraceConfig:
    """Pre-resolved trace parameters shared across monitor solvers."""

    frequency: float
    wavelength: float
    k: float
    reflection_n_rays: int
    reflection_max_bounces: int
    reflection_coef: float
    min_ray_contribution_threshold: float
    reflection_relative_permittivity: float
    reflection_conductivity: float
    reflection_material: Mapping[str, float] | None
    diffraction_material: Mapping[str, float] | None
    use_scene_materials_for_reflection: bool
    use_scene_materials_for_diffraction: bool
    resolution_wavelength: float
    enable_rd_diffraction: bool
    max_diffractions: int
    diffraction_state_budget: int | None
    inserted_reflection_state_budget: int | None
    max_inserted_reflections_per_path: int | None
    solver_mode: str
    memory_profile: str
    reflection_field_backend: ReflectionFieldBackend
    tx_polarization: tuple[float, float, float]
    rx_polarization: tuple[float, float, float] | None
    diffraction_execution: DiffractionExecutionConfig
    cell_size: float


def resolve_material(
    raw: Mapping[str, Any] | None,
    *,
    relative_permittivity: float,
    conductivity: float,
    gain: float,
) -> Mapping[str, float] | None:
    if raw is None:
        return None
    return {
        "relative_permittivity": float(raw.get("relative_permittivity", relative_permittivity)),
        "conductivity": float(raw.get("conductivity", conductivity)),
        "gain": float(raw.get("gain", gain)),
    }


def resolve_trace_config(*, frequency: float, config: TraceConfig) -> ResolvedTraceConfig:
    wavelength = SPEED_OF_LIGHT / float(frequency)
    reflection_coef = float(config.reflection_coef)
    resolution_wavelength = float(config.resolution_wavelength)
    return ResolvedTraceConfig(
        frequency=float(frequency),
        wavelength=wavelength,
        k=2.0 * math.pi / wavelength,
        reflection_n_rays=int(config.reflection_n_rays),
        reflection_max_bounces=int(config.reflection_max_bounces),
        reflection_coef=reflection_coef,
        min_ray_contribution_threshold=float(config.min_ray_contribution_threshold),
        reflection_relative_permittivity=float(config.reflection_relative_permittivity),
        reflection_conductivity=float(config.reflection_conductivity),
        reflection_material=resolve_material(
            config.reflection_material,
            relative_permittivity=float(config.reflection_relative_permittivity),
            conductivity=float(config.reflection_conductivity),
            gain=reflection_coef,
        ),
        diffraction_material=resolve_material(
            config.diffraction_material,
            relative_permittivity=5.0,
            conductivity=0.0,
            gain=reflection_coef,
        ),
        use_scene_materials_for_reflection=bool(config.use_scene_materials_for_reflection),
        use_scene_materials_for_diffraction=bool(config.use_scene_materials_for_diffraction),
        resolution_wavelength=resolution_wavelength,
        enable_rd_diffraction=bool(config.enable_rd_diffraction),
        max_diffractions=int(config.max_diffractions),
        diffraction_state_budget=(
            None if config.diffraction_state_budget is None else int(config.diffraction_state_budget)
        ),
        inserted_reflection_state_budget=(
            None
            if config.inserted_reflection_state_budget is None
            else int(config.inserted_reflection_state_budget)
        ),
        max_inserted_reflections_per_path=(
            None
            if config.max_inserted_reflections_per_path is None
            else int(config.max_inserted_reflections_per_path)
        ),
        solver_mode=str(config.solver_mode),
        memory_profile=str(config.memory_profile),
        reflection_field_backend=config.reflection_field_backend,
        tx_polarization=tuple(float(value) for value in config.tx_polarization),
        rx_polarization=(
            None
            if config.rx_polarization is None
            else tuple(float(value) for value in config.rx_polarization)
        ),
        diffraction_execution=config.diffraction_execution,
        cell_size=resolution_wavelength * wavelength,
    )


def build_execution_intent(
    kind: TraceExecutionIntent,
    *,
    return_geometry: bool = False,
    return_diffraction_audit: bool = False,
) -> dict[str, object]:
    if kind not in {
        "field",
        "field_scalar_only",
        "path_export",
        "radio_map_coherent",
        "radio_map_incoherent",
    }:
        raise ValueError(f"Unsupported execution intent: {kind}")
    is_radio_map = kind in {"radio_map_coherent", "radio_map_incoherent"}
    return {
        "kind": str(kind),
        "path_export_enabled": bool(kind == "path_export"),
        "radio_map_enabled": bool(is_radio_map),
        "geometry_enabled": bool(return_geometry),
        "field_payload": (
            "full"
            if kind == "field"
            else "total_only"
            if kind == "field_scalar_only"
            else "none"
        ),
        "radio_map_combine_mode": (
            "coherent"
            if kind == "radio_map_coherent"
            else "incoherent"
            if kind == "radio_map_incoherent"
            else None
        ),
        "diffraction_audit_enabled": bool(return_diffraction_audit),
    }


def resolve_solver_controls(
    config: TraceConfig,
    *,
    execution_intent: TraceExecutionIntent = "field",
    max_diffractions_override: int | None = None,
) -> dict[str, object]:
    mode = str(config.solver_mode)
    if mode not in {"accuracy", "fast_approximate"}:
        raise ValueError(f"Unsupported solver_mode: {mode}")
    memory_profile = str(config.memory_profile)
    if memory_profile not in {"default", "memory_safe"}:
        raise ValueError(f"Unsupported memory_profile: {memory_profile}")

    requested = {
        "reflection_n_rays": int(config.reflection_n_rays),
        "reflection_max_bounces": int(config.reflection_max_bounces),
        "max_diffractions": (
            int(config.max_diffractions)
            if max_diffractions_override is None
            else int(max_diffractions_override)
        ),
        "diffraction_state_budget": (
            None if config.diffraction_state_budget is None else int(config.diffraction_state_budget)
        ),
        "inserted_reflection_state_budget": (
            None
            if config.inserted_reflection_state_budget is None
            else int(config.inserted_reflection_state_budget)
        ),
        "max_inserted_reflections_per_path": (
            None
            if config.max_inserted_reflections_per_path is None
            else int(config.max_inserted_reflections_per_path)
        ),
        "memory_profile": memory_profile,
    }
    effective = dict(requested)
    changes = []

    def _set_effective(key, value, reason):
        if effective[key] != value:
            changes.append({
                "parameter": key,
                "requested": effective[key],
                "effective": value,
                "reason": reason,
            })
            effective[key] = value

    if mode == "fast_approximate":
        _set_effective(
            "reflection_n_rays",
            min(effective["reflection_n_rays"], FAST_APPROXIMATE_REFLECTION_RAYS_CAP),
            "Cap sampled reflection rays to keep mixed-path tracing bounded in fast approximate mode.",
        )
        _set_effective(
            "reflection_max_bounces",
            min(effective["reflection_max_bounces"], FAST_APPROXIMATE_REFLECTION_BOUNCES_CAP),
            "Cap reflection bounces to keep prefix/suffix tracing bounded in fast approximate mode.",
        )
        _set_effective(
            "max_diffractions",
            min(effective["max_diffractions"], FAST_APPROXIMATE_MAX_DIFFRACTIONS_CAP),
            "Cap diffraction order to keep recursive state growth bounded in fast approximate mode.",
        )
        requested_inserted_depth = effective["max_inserted_reflections_per_path"]
        if requested_inserted_depth is None:
            requested_inserted_depth = FAST_APPROXIMATE_INSERTED_REFLECTION_DEPTH_CAP
        _set_effective(
            "max_inserted_reflections_per_path",
            min(requested_inserted_depth, FAST_APPROXIMATE_INSERTED_REFLECTION_DEPTH_CAP),
            "Restrict inserted mixed depth in fast approximate mode.",
        )
        requested_total_budget = effective["diffraction_state_budget"]
        if requested_total_budget is None:
            requested_total_budget = FAST_APPROXIMATE_TOTAL_STATE_BUDGET
        _set_effective(
            "diffraction_state_budget",
            min(requested_total_budget, FAST_APPROXIMATE_TOTAL_STATE_BUDGET),
            "Apply an explicit per-order total-state budget in fast approximate mode.",
        )
        requested_inserted_budget = effective["inserted_reflection_state_budget"]
        if requested_inserted_budget is None:
            requested_inserted_budget = FAST_APPROXIMATE_INSERTED_STATE_BUDGET
        _set_effective(
            "inserted_reflection_state_budget",
            min(requested_inserted_budget, FAST_APPROXIMATE_INSERTED_STATE_BUDGET),
            "Apply an explicit per-order inserted-state budget in fast approximate mode.",
        )

    if memory_profile == "memory_safe":
        requested_inserted_depth = effective["max_inserted_reflections_per_path"]
        if requested_inserted_depth is None:
            requested_inserted_depth = MEMORY_SAFE_INSERTED_REFLECTION_DEPTH_CAP
        _set_effective(
            "max_inserted_reflections_per_path",
            min(requested_inserted_depth, MEMORY_SAFE_INSERTED_REFLECTION_DEPTH_CAP),
            "Apply an explicit inserted-reflection depth guardrail in memory-safe mode.",
        )
        requested_total_budget = effective["diffraction_state_budget"]
        if requested_total_budget is None:
            requested_total_budget = MEMORY_SAFE_TOTAL_STATE_BUDGET
        _set_effective(
            "diffraction_state_budget",
            min(requested_total_budget, MEMORY_SAFE_TOTAL_STATE_BUDGET),
            "Apply an explicit per-order total-state budget in memory-safe mode.",
        )
        requested_inserted_budget = effective["inserted_reflection_state_budget"]
        if requested_inserted_budget is None:
            requested_inserted_budget = MEMORY_SAFE_INSERTED_STATE_BUDGET
        _set_effective(
            "inserted_reflection_state_budget",
            min(requested_inserted_budget, MEMORY_SAFE_INSERTED_STATE_BUDGET),
            "Apply an explicit per-order inserted-state budget in memory-safe mode.",
        )

    if (
        effective["diffraction_state_budget"] is not None
        and effective["inserted_reflection_state_budget"] is not None
        and effective["inserted_reflection_state_budget"] > effective["diffraction_state_budget"]
    ):
        _set_effective(
            "inserted_reflection_state_budget",
            effective["diffraction_state_budget"],
            "Inserted-state budget cannot exceed the total per-order state budget.",
        )

    return {
        "selected": mode,
        "execution_intent": build_execution_intent(execution_intent),
        "requested": requested,
        "effective": effective,
        "changes": changes,
        "guarantees": (
            "Enumerate all currently implemented diffraction families allowed by the requested depth limits; "
            "do not apply additional automatic pruning beyond explicit user budgets."
            if mode == "accuracy"
            else "Apply explicit guardrails to reflection sampling, mixed depth, and per-order state counts; "
            "may prune weak source states before higher-order expansion and may omit deeper mixed families "
            "to keep runtime bounded."
        ),
    }


def reflection_discovery_key_for_field(monitor, tx_pos):
    if monitor.ray_mode == "2d":
        return ("2d", "circle")
    if monitor.ray_sampling == "full_sphere":
        return ("3d", "full_sphere")
    if monitor.ray_sampling == "hemisphere":
        return ("3d", "hemisphere", monitor.axis, float(monitor.position))
    plane_distance = abs(float(monitor.position) - float(scalar(getattr(tx_pos, monitor.axis))))
    span_0 = float(monitor.bounds[0][1] - monitor.bounds[0][0])
    span_1 = float(monitor.bounds[1][1] - monitor.bounds[1][0])
    near_plane_threshold = 0.25 * min(span_0, span_1)
    if plane_distance <= near_plane_threshold:
        return ("3d", "full_sphere")
    return ("3d", "hemisphere", monitor.axis, float(monitor.position))


def reflection_discovery_key_for_path(monitor):
    if monitor.ray_mode == "2d":
        return ("2d", "circle")
    return ("3d", "full_sphere")


def reflection_discovery_key_for_radio_map(monitor):
    if monitor.ray_mode == "2d":
        return ("2d", "circle")
    return ("3d", "full_sphere")


__all__ = [
    "TraceExecutionIntent",
    "ResolvedTraceConfig",
    "build_execution_intent",
    "reflection_discovery_key_for_path",
    "reflection_discovery_key_for_field",
    "reflection_discovery_key_for_radio_map",
    "resolve_material",
    "resolve_solver_controls",
    "resolve_trace_config",
]
