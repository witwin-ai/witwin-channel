"""Shared internal configuration primitives for channel solver configs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from witwin.channel.core.numerics.constants import SPEED_OF_LIGHT


SolverMode = Literal["accuracy", "fast_approximate"]
ReflectionFieldBackend = Literal["drjit", "native"]
MemoryProfile = Literal["default", "memory_safe"]

SOLVER_MODES = {"accuracy", "fast_approximate"}
REFLECTION_FIELD_BACKENDS = {"drjit", "native"}
MEMORY_PROFILES = {"default", "memory_safe"}

ACCURACY_GUARANTEES = (
    "Enumerate all currently implemented diffraction families allowed by the requested depth limits; "
    "do not apply additional automatic pruning beyond explicit user budgets."
)
FAST_GUARANTEES = (
    "Apply explicit guardrails to reflection sampling, mixed depth, and per-order state counts; "
    "may prune weak source states before higher-order expansion and may omit deeper mixed families "
    "to keep runtime bounded."
)

FAST_REFLECTION_RAYS_CAP = 1024
FAST_REFLECTION_BOUNCES_CAP = 1
FAST_MAX_DIFFRACTIONS_CAP = 3
FAST_TOTAL_STATE_BUDGET = 768
FAST_INSERTED_STATE_BUDGET = 256
FAST_INSERTED_DEPTH_CAP = 1
MEMORY_TOTAL_STATE_BUDGET = 2048
MEMORY_INSERTED_STATE_BUDGET = 512
MEMORY_INSERTED_DEPTH_CAP = 1


def validate_literal(value: str, *, name: str, allowed: set[str]) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}; got {value!r}.")
    return value


@dataclass(slots=True)
class CommonTraceTuning:
    """Base marker and validation owner for shared trace tuning fields."""

    def validate_common_trace_tuning(self) -> None:
        if not 0.0 <= float(self.min_ray_contribution_threshold) < 1.0:
            raise ValueError("min_ray_contribution_threshold must satisfy 0.0 <= value < 1.0.")
        if float(self.resolution_wavelength) <= 0.0:
            raise ValueError("resolution_wavelength must be > 0.")
        validate_literal(str(self.solver_mode), name="solver_mode", allowed=SOLVER_MODES)
        validate_literal(str(self.memory_profile), name="memory_profile", allowed=MEMORY_PROFILES)
        validate_literal(
            str(self.reflection_field_backend),
            name="reflection_field_backend",
            allowed=REFLECTION_FIELD_BACKENDS,
        )
        for name in ("diffraction_state_budget", "inserted_reflection_state_budget"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be > 0 when provided.")
        if (
            self.max_inserted_reflections_per_path is not None
            and int(self.max_inserted_reflections_per_path) < 0
        ):
            raise ValueError("max_inserted_reflections_per_path must be >= 0 when provided.")


@dataclass(frozen=True)
class ResolvedTraceBase:
    """Base marker and wave-parameter owner for resolved trace configs."""

    @staticmethod
    def wave_parameters(*, frequency: float, resolution_wavelength: float) -> tuple[float, float, float]:
        wavelength = SPEED_OF_LIGHT / float(frequency)
        return (
            wavelength,
            2.0 * math.pi / wavelength,
            float(resolution_wavelength) * wavelength,
        )


@dataclass(frozen=True, slots=True)
class GuardrailRule:
    key: str
    cap: int
    fill_when_none: bool
    reason: str


FAST_GUARDRAILS = (
    GuardrailRule(
        "reflection_n_rays",
        FAST_REFLECTION_RAYS_CAP,
        False,
        "Cap sampled reflection rays to keep mixed-path tracing bounded in fast approximate mode.",
    ),
    GuardrailRule(
        "reflection_max_bounces",
        FAST_REFLECTION_BOUNCES_CAP,
        False,
        "Cap reflection bounces to keep prefix/suffix tracing bounded in fast approximate mode.",
    ),
    GuardrailRule(
        "max_diffractions",
        FAST_MAX_DIFFRACTIONS_CAP,
        False,
        "Cap diffraction order to keep recursive state growth bounded in fast approximate mode.",
    ),
    GuardrailRule(
        "max_inserted_reflections_per_path",
        FAST_INSERTED_DEPTH_CAP,
        True,
        "Restrict inserted mixed depth in fast approximate mode.",
    ),
    GuardrailRule(
        "diffraction_state_budget",
        FAST_TOTAL_STATE_BUDGET,
        True,
        "Apply an explicit per-order total-state budget in fast approximate mode.",
    ),
    GuardrailRule(
        "inserted_reflection_state_budget",
        FAST_INSERTED_STATE_BUDGET,
        True,
        "Apply an explicit per-order inserted-state budget in fast approximate mode.",
    ),
)
MEMORY_GUARDRAILS = (
    GuardrailRule(
        "max_inserted_reflections_per_path",
        MEMORY_INSERTED_DEPTH_CAP,
        True,
        "Apply an explicit inserted-reflection depth guardrail in memory-safe mode.",
    ),
    GuardrailRule(
        "diffraction_state_budget",
        MEMORY_TOTAL_STATE_BUDGET,
        True,
        "Apply an explicit per-order total-state budget in memory-safe mode.",
    ),
    GuardrailRule(
        "inserted_reflection_state_budget",
        MEMORY_INSERTED_STATE_BUDGET,
        True,
        "Apply an explicit per-order inserted-state budget in memory-safe mode.",
    ),
)


def _apply_guardrails(effective: dict, changes: list, guardrails) -> None:
    for rule in guardrails:
        current = effective[rule.key]
        baseline = rule.cap if (current is None and rule.fill_when_none) else current
        if baseline is None:
            continue
        new_value = min(baseline, rule.cap)
        if new_value != current:
            changes.append({
                "parameter": rule.key,
                "requested": current,
                "effective": new_value,
                "reason": rule.reason,
            })
            effective[rule.key] = new_value


def resolve_common_solver_controls(
    config,
    *,
    max_diffractions_override: int | None = None,
) -> dict[str, object]:
    mode = validate_literal(str(config.solver_mode), name="solver_mode", allowed=SOLVER_MODES)
    memory_profile = validate_literal(
        str(config.memory_profile),
        name="memory_profile",
        allowed=MEMORY_PROFILES,
    )

    requested = {
        "reflection_n_rays": int(config.num_samples),
        "reflection_max_bounces": int(config.max_bounces),
        "max_diffractions": int(
            config.max_diffraction_order
            if max_diffractions_override is None
            else max_diffractions_override
        ),
        "diffraction_state_budget": (
            None if config.diffraction_state_budget is None
            else int(config.diffraction_state_budget)
        ),
        "inserted_reflection_state_budget": (
            None if config.inserted_reflection_state_budget is None
            else int(config.inserted_reflection_state_budget)
        ),
        "max_inserted_reflections_per_path": (
            None if config.max_inserted_reflections_per_path is None
            else int(config.max_inserted_reflections_per_path)
        ),
        "memory_profile": memory_profile,
    }
    effective = dict(requested)
    changes: list = []

    if mode == "fast_approximate":
        _apply_guardrails(effective, changes, FAST_GUARDRAILS)
    if memory_profile == "memory_safe":
        _apply_guardrails(effective, changes, MEMORY_GUARDRAILS)

    total = effective["diffraction_state_budget"]
    inserted = effective["inserted_reflection_state_budget"]
    if total is not None and inserted is not None and inserted > total:
        changes.append({
            "parameter": "inserted_reflection_state_budget",
            "requested": inserted,
            "effective": total,
            "reason": "Inserted-state budget cannot exceed the total per-order state budget.",
        })
        effective["inserted_reflection_state_budget"] = total

    return {
        "selected": mode,
        "requested": requested,
        "effective": effective,
        "changes": changes,
        "guarantees": ACCURACY_GUARANTEES if mode == "accuracy" else FAST_GUARANTEES,
    }


__all__ = [
    "ACCURACY_GUARANTEES",
    "FAST_GUARANTEES",
    "CommonTraceTuning",
    "MemoryProfile",
    "ReflectionFieldBackend",
    "ResolvedTraceBase",
    "SolverMode",
    "resolve_common_solver_controls",
    "validate_literal",
]
