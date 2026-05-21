from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal

from witwin.channel.core.numerics.constants import SPEED_OF_LIGHT
from witwin.channel.core.results.ray_mode import RayMode
from witwin.channel.deterministic.config import (
    DiffractionExecutionConfig,
    MemoryProfile,
    ReflectionFieldBackend,
    SolverMode,
    coerce_diffraction_execution,
    resolve_solver_controls as _resolve_solver_controls,
)
from witwin.channel.core.scene.edge_policy import EdgePolicy, coerce_edge_policy


_SOLVER_MODES = {"accuracy", "fast_approximate"}
_MEMORY_PROFILES = {"default", "memory_safe"}
_REFLECTION_FIELD_BACKENDS = {"drjit", "native"}


def _check_literal(value: str, *, name: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}; got {value!r}.")


@dataclass(slots=True)
class Tuning:
    """Advanced runtime controls for standalone path solving."""

    min_ray_contribution_threshold: float = 0.0
    resolution_wavelength: float = 0.125
    enable_rd_diffraction: bool = False
    diffraction_state_budget: int | None = None
    inserted_reflection_state_budget: int | None = None
    max_inserted_reflections_per_path: int | None = None
    shadow_support_cutoff_db: float | None = None
    solver_mode: SolverMode = "accuracy"
    memory_profile: MemoryProfile = "default"
    reflection_field_backend: ReflectionFieldBackend = "native"
    diffraction_execution: DiffractionExecutionConfig | Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.min_ray_contribution_threshold) < 1.0:
            raise ValueError("min_ray_contribution_threshold must satisfy 0.0 <= value < 1.0.")
        if float(self.resolution_wavelength) <= 0.0:
            raise ValueError("resolution_wavelength must be > 0.")
        if (
            self.shadow_support_cutoff_db is not None
            and float(self.shadow_support_cutoff_db) < 0.0
        ):
            raise ValueError("shadow_support_cutoff_db must be >= 0.")
        for name in ("diffraction_state_budget", "inserted_reflection_state_budget"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be > 0 when provided.")
        if (
            self.max_inserted_reflections_per_path is not None
            and int(self.max_inserted_reflections_per_path) < 0
        ):
            raise ValueError("max_inserted_reflections_per_path must be >= 0 when provided.")
        for value, name, allowed in (
            (str(self.solver_mode), "solver_mode", _SOLVER_MODES),
            (str(self.memory_profile), "memory_profile", _MEMORY_PROFILES),
            (str(self.reflection_field_backend), "reflection_field_backend", _REFLECTION_FIELD_BACKENDS),
        ):
            _check_literal(value, name=name, allowed=allowed)
        self.diffraction_execution = coerce_diffraction_execution(self.diffraction_execution)


def _coerce_tuning(tuning: Tuning | Mapping[str, object] | None) -> Tuning:
    if tuning is None:
        return Tuning()
    if isinstance(tuning, Tuning):
        return tuning
    return Tuning(**dict(tuning))


@dataclass(slots=True)
class Config:
    """User-facing configuration for standalone path solving."""

    max_num_paths: int | None = None
    max_diffraction_order: int = 1
    return_geometry: bool = False
    num_samples: int = 10000
    max_bounces: int = 2
    synthetic_array: bool = True
    edge_policy: EdgePolicy | None = None
    tuning: Tuning | Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.max_num_paths is not None and int(self.max_num_paths) <= 0:
            raise ValueError("max_num_paths must be > 0 when provided.")
        if int(self.max_diffraction_order) < 0:
            raise ValueError("max_diffraction_order must be >= 0.")
        if int(self.num_samples) <= 0:
            raise ValueError("num_samples must be > 0.")
        if int(self.max_bounces) < 0:
            raise ValueError("max_bounces must be >= 0.")
        if not isinstance(self.synthetic_array, bool):
            raise TypeError("synthetic_array must be a bool.")
        self.edge_policy = coerce_edge_policy(self.edge_policy)
        self.tuning = _coerce_tuning(self.tuning)


def derive_wave_params(*, frequency: float) -> tuple[float, float]:
    """Return ``(wavelength, k)`` derived from the carrier ``frequency``."""
    wavelength = SPEED_OF_LIGHT / float(frequency)
    return wavelength, 2.0 * math.pi / wavelength


@dataclass(frozen=True, slots=True)
class PathSolveSpec:
    ray_mode: RayMode
    name: str = "path"
    kind: Literal["path"] = "path"


def resolve_solver_controls(
    config: Config, *, max_diffractions_override: int | None = None,
) -> dict[str, object]:
    tuning = config.tuning
    internal_config = SimpleNamespace(
        num_samples=int(config.num_samples),
        max_bounces=int(config.max_bounces),
        max_diffraction_order=int(config.max_diffraction_order),
        tuning=SimpleNamespace(
            diffraction_state_budget=tuning.diffraction_state_budget,
            inserted_reflection_state_budget=tuning.inserted_reflection_state_budget,
            max_inserted_reflections_per_path=tuning.max_inserted_reflections_per_path,
            solver_mode=tuning.solver_mode,
            memory_profile=tuning.memory_profile,
        ),
    )
    return _resolve_solver_controls(
        internal_config,
        execution_intent="path_export",
        max_diffractions_override=max_diffractions_override,
    )


__all__ = [
    "Config",
    "DiffractionExecutionConfig",
    "EdgePolicy",
    "PathSolveSpec",
    "Tuning",
    "derive_wave_params",
    "resolve_solver_controls",
]
