"""System-level immutable configuration objects for channel execution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, Protocol

from . import types as rm_types
from witwin.channel._config.base import (
    CommonTraceTuning,
    MemoryProfile,
    ReflectionFieldBackend,
    ResolvedTraceBase,
    SolverMode,
    resolve_common_solver_controls,
    validate_literal as _validate_literal,
)
from witwin.channel.core.results.ray_mode import DEFAULT_RAY_MODE, RayMode, normalize_ray_mode
from witwin.channel.core.scene.edge_policy import EdgePolicy, coerce_edge_policy


class ReceiverGridLike(Protocol):
    n_cells: int


SuffixBackend = Literal["drjit", "native"]
AccumulatePrimalMode = Literal["auto", "drjit", "rayd_optix", "rayd_exact_coherent"]
ShadowBoundaryBackend = Literal["auto", "dense_native", "native_candidate"]
ReflectionTransitionMode = Literal["hard", "f_weight_reference", "f_weight_native"]
ReflectionSecondaryVisibilityMode = Literal["hard", "f_weight"]
TraceExecutionIntent = Literal["field", "field_scalar_only", "path_export", "coherent"]

_SUFFIX_BACKENDS = {"drjit", "native"}
_ACCUMULATE_PRIMAL_MODES = {"auto", "drjit", "rayd_optix", "rayd_exact_coherent"}
_SHADOW_BOUNDARY_BACKENDS = {"auto", "dense_native", "native_candidate"}
_REFLECTION_TRANSITION_MODES = {"hard", "f_weight_reference", "f_weight_native"}
_REFLECTION_SECONDARY_VISIBILITY_MODES = {"hard", "f_weight"}
_METRICS = {"path_gain", "rss", "sinr"}
_QUADRATURE_MODES = {"center", "stratified_fixed_n"}
_EXECUTION_INTENTS = {"field", "field_scalar_only", "path_export", "coherent"}


def _point2(value) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        v = float(value)
        return (v, v)
    return (float(value[0]), float(value[1]))


def _optional_point2(value) -> tuple[float, float] | None:
    return None if value is None else _point2(value)


def _quadrature(quadrature_mode: str, samples_per_cell: int | None) -> tuple[str, int]:
    mode = str(quadrature_mode).lower()
    if mode == "center":
        return ("center", 1)
    if mode == "stratified_fixed_n":
        return ("stratified_fixed_n", int(samples_per_cell))
    raise ValueError("quadrature_mode must be 'center' or 'stratified_fixed_n'.")


@dataclass(slots=True)
class DiffractionExecutionConfig:
    accumulate_primal: AccumulatePrimalMode = "auto"
    suffix_backend: SuffixBackend = "native"
    suffix_russian_roulette: bool = False

    def __post_init__(self):
        self.accumulate_primal = _validate_literal(
            self.accumulate_primal,
            name="accumulate_primal",
            allowed=_ACCUMULATE_PRIMAL_MODES,
        )
        self.suffix_backend = _validate_literal(
            self.suffix_backend, name="suffix_backend", allowed=_SUFFIX_BACKENDS,
        )
        self.suffix_russian_roulette = bool(self.suffix_russian_roulette)

    def to_dict(self) -> dict[str, object]:
        return {
            "accumulate_primal": self.accumulate_primal,
            "suffix_backend": self.suffix_backend,
            "suffix_russian_roulette": self.suffix_russian_roulette,
        }


def coerce_diffraction_execution(
    execution: DiffractionExecutionConfig | Mapping[str, object] | None,
) -> DiffractionExecutionConfig:
    if execution is None:
        return DiffractionExecutionConfig()
    if isinstance(execution, DiffractionExecutionConfig):
        return execution
    return DiffractionExecutionConfig(
        accumulate_primal=execution.get("accumulate_primal", "auto"),
        suffix_backend=execution.get("suffix_backend", "native"),
        suffix_russian_roulette=execution.get("suffix_russian_roulette", False),
    )


@dataclass(slots=True)
class ReflectionSuffixConfig:
    """Bundled parameters for diffraction-suffix reflection tracing."""
    n_rays: int = 0
    max_bounces: int = 0
    coef: float = 0.7
    mode: str = "2d"
    detail: Mapping[str, object] | None = None
    grid: ReceiverGridLike | None = None
    grid_data: dict[str, object] | None = None
    rx_z: float | None = None

    @property
    def enabled(self) -> bool:
        return (
            self.grid is not None
            and self.grid_data is not None
            and self.max_bounces > 0
            and self.n_rays > 0
        )


@dataclass(frozen=True, slots=True)
class FieldSolveSpec:
    """Normalized per-solve field metadata handed to path tracers."""

    axis: str
    position: float
    bounds: tuple[tuple[float, float], tuple[float, float]]
    grid_shape: tuple[int, int]
    ray_mode: RayMode
    ray_sampling: str
    shadow_support_cutoff_db: float | None = None
    name: str = "field"
    kind: str = "field"


@dataclass(slots=True)
class SolveSpec:
    """Bundle the normalized deterministic radiomap runtime inputs."""

    axis: str
    position: float
    bounds: tuple[tuple[float, float], tuple[float, float]]
    grid_shape: tuple[int, int] | None
    cell_size: tuple[float, float] | None
    metric: rm_types.Metric
    tx_power: float
    noise_power: float | None
    ray_mode: RayMode
    quadrature_mode: str
    samples_per_cell: int
    shadow_boundary_correction: bool
    shadow_boundary_backend: str
    shadow_boundary_tile_shape: tuple[int, int]
    shadow_boundary_band_width_wavelengths: float
    shadow_boundary_max_candidate_factor: float
    shadow_support_cutoff_db: float | None
    name: str = "deterministic"
    kind: str = "radio_map"
    surface_mode: rm_types.SurfaceMode = rm_types.SurfaceMode.AXIS_ALIGNED

    @classmethod
    def from_public(
        cls,
        *,
        grid: "GridSpec",
        config: "Config",
        ray_mode: str = DEFAULT_RAY_MODE,
    ) -> "SolveSpec":
        tuning = config.tuning
        resolved_quadrature_mode, resolved_samples_per_cell = _quadrature(
            config.quadrature_mode, config.samples_per_cell,
        )
        if bool(config.shadow_boundary_correction) and resolved_quadrature_mode != "center":
            raise ValueError("shadow_boundary_correction=True requires quadrature_mode='center'.")
        return cls(
            axis=str(grid.axis),
            position=float(grid.position),
            bounds=tuple(tuple(float(v) for v in pair) for pair in grid.bounds),
            grid_shape=None if grid.grid_shape is None else tuple(int(v) for v in grid.grid_shape),
            cell_size=_optional_point2(grid.cell_size),
            metric=rm_types.Metric(str(config.metric).lower()),
            tx_power=1.0,
            noise_power=None,
            ray_mode=normalize_ray_mode(ray_mode),
            quadrature_mode=resolved_quadrature_mode,
            samples_per_cell=resolved_samples_per_cell,
            shadow_boundary_correction=bool(config.shadow_boundary_correction),
            shadow_boundary_backend=str(tuning.shadow_boundary_backend),
            shadow_boundary_tile_shape=tuple(
                int(v) for v in tuning.shadow_boundary_tile_shape
            ),
            shadow_boundary_band_width_wavelengths=float(
                tuning.shadow_boundary_band_width_wavelengths
            ),
            shadow_boundary_max_candidate_factor=float(tuning.shadow_boundary_max_candidate_factor),
            shadow_support_cutoff_db=(
                None if tuning.shadow_support_cutoff_db is None
                else float(tuning.shadow_support_cutoff_db)
            ),
        )

    @property
    def spans(self) -> tuple[float, float]:
        return (
            self.bounds[0][1] - self.bounds[0][0],
            self.bounds[1][1] - self.bounds[1][0],
        )

    def resolve_grid_shape(
        self,
        *,
        default_cell_size: float | tuple[float, float] | None = None,
    ) -> tuple[int, int]:
        if self.grid_shape is not None:
            return self.grid_shape
        resolved_default = _optional_point2(default_cell_size)
        requested_cell_size = self.cell_size if self.cell_size is not None else resolved_default
        if requested_cell_size is None:
            raise ValueError(
                "GridSpec requires grid_shape or cell_size, or solve must provide a default cell_size."
            )
        span_0, span_1 = self.spans
        nx = max(1, int(math.ceil(span_0 / requested_cell_size[0])))
        ny = max(1, int(math.ceil(span_1 / requested_cell_size[1])))
        return (nx, ny)

    def resolve_cell_size(
        self,
        *,
        default_cell_size: float | tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        span_0, span_1 = self.spans
        grid_shape = self.resolve_grid_shape(default_cell_size=default_cell_size)
        return (span_0 / float(grid_shape[0]), span_1 / float(grid_shape[1]))


@dataclass(slots=True)
class Tuning(CommonTraceTuning):
    """Advanced runtime controls for deterministic radiomap solving."""

    min_ray_contribution_threshold: float = 0.0
    resolution_wavelength: float = 0.125
    enable_rd_diffraction: bool = False
    shadow_support_cutoff_db: float | None = None
    diffraction_state_budget: int | None = None
    inserted_reflection_state_budget: int | None = None
    max_inserted_reflections_per_path: int | None = None
    solver_mode: SolverMode = "accuracy"
    memory_profile: MemoryProfile = "default"
    reflection_field_backend: ReflectionFieldBackend = "native"
    shadow_boundary_backend: ShadowBoundaryBackend = "auto"
    shadow_boundary_tile_shape: tuple[int, int] = (8, 8)
    shadow_boundary_band_width_wavelengths: float = 3.0
    shadow_boundary_max_candidate_factor: float = 96.0
    reflection_transition_mode: ReflectionTransitionMode = "hard"
    reflection_f_weight_boundary_radius_wavelengths: float = 2.0
    reflection_f_weight_max_edges_per_slot: int = 1
    reflection_secondary_visibility_mode: ReflectionSecondaryVisibilityMode = "hard"
    diffraction_execution: DiffractionExecutionConfig | Mapping[str, object] | None = None

    def __post_init__(self):
        self.validate_common_trace_tuning()
        if (
            self.shadow_support_cutoff_db is not None
            and float(self.shadow_support_cutoff_db) < 0.0
        ):
            raise ValueError("shadow_support_cutoff_db must be >= 0.")
        _validate_literal(
            str(self.shadow_boundary_backend),
            name="shadow_boundary_backend",
            allowed=_SHADOW_BOUNDARY_BACKENDS,
        )
        tile_shape = tuple(int(v) for v in self.shadow_boundary_tile_shape)
        if len(tile_shape) != 2 or tile_shape[0] <= 0 or tile_shape[1] <= 0:
            raise ValueError("shadow_boundary_tile_shape must contain two positive integers.")
        if float(self.shadow_boundary_band_width_wavelengths) <= 0.0:
            raise ValueError("shadow_boundary_band_width_wavelengths must be > 0.")
        if float(self.shadow_boundary_max_candidate_factor) <= 0.0:
            raise ValueError("shadow_boundary_max_candidate_factor must be > 0.")
        transition_mode = _validate_literal(
            str(self.reflection_transition_mode),
            name="reflection_transition_mode",
            allowed=_REFLECTION_TRANSITION_MODES,
        )
        secondary_visibility_mode = _validate_literal(
            str(self.reflection_secondary_visibility_mode),
            name="reflection_secondary_visibility_mode",
            allowed=_REFLECTION_SECONDARY_VISIBILITY_MODES,
        )
        if float(self.reflection_f_weight_boundary_radius_wavelengths) <= 0.0:
            raise ValueError("reflection_f_weight_boundary_radius_wavelengths must be > 0.")
        max_edges = int(self.reflection_f_weight_max_edges_per_slot)
        if max_edges <= 0:
            raise ValueError("reflection_f_weight_max_edges_per_slot must be > 0.")
        self.shadow_boundary_tile_shape = tile_shape
        self.reflection_transition_mode = transition_mode
        self.reflection_secondary_visibility_mode = secondary_visibility_mode
        self.reflection_f_weight_boundary_radius_wavelengths = float(
            self.reflection_f_weight_boundary_radius_wavelengths
        )
        self.reflection_f_weight_max_edges_per_slot = max_edges
        self.diffraction_execution = coerce_diffraction_execution(self.diffraction_execution)


def _coerce_tuning(tuning: Tuning | Mapping[str, object] | None) -> Tuning:
    if tuning is None:
        return Tuning()
    if isinstance(tuning, Tuning):
        return tuning
    return Tuning(**dict(tuning))


@dataclass(slots=True)
class Config:
    """User-facing configuration for the deterministic radiomap solver."""

    metric: str = "path_gain"
    quadrature_mode: str = "center"
    samples_per_cell: int | None = None
    shadow_boundary_correction: bool = False
    num_samples: int = 10000
    max_bounces: int = 2
    max_diffraction_order: int = 1
    synthetic_array: bool = True
    edge_policy: EdgePolicy | None = None
    tuning: Tuning | Mapping[str, object] | None = None

    def __post_init__(self):
        if int(self.num_samples) <= 0:
            raise ValueError("num_samples must be > 0.")
        if int(self.max_bounces) < 0:
            raise ValueError("max_bounces must be >= 0.")
        if int(self.max_diffraction_order) < 0:
            raise ValueError("max_diffraction_order must be >= 0.")
        if not isinstance(self.synthetic_array, bool):
            raise TypeError("synthetic_array must be a bool.")
        _validate_literal(str(self.metric), name="metric", allowed=_METRICS)
        _validate_literal(str(self.quadrature_mode), name="quadrature_mode", allowed=_QUADRATURE_MODES)
        if not isinstance(self.shadow_boundary_correction, bool):
            raise TypeError("shadow_boundary_correction must be a bool.")
        if self.samples_per_cell is not None and int(self.samples_per_cell) <= 0:
            raise ValueError("samples_per_cell must be > 0 when provided.")
        self.edge_policy = coerce_edge_policy(self.edge_policy)
        self.tuning = _coerce_tuning(self.tuning)


@dataclass(frozen=True)
class ResolvedTraceConfig(ResolvedTraceBase):
    """Pre-resolved trace parameters shared across monitor solvers."""

    frequency: float
    wavelength: float
    k: float
    reflection_n_rays: int
    reflection_max_bounces: int
    min_ray_contribution_threshold: float
    resolution_wavelength: float
    enable_rd_diffraction: bool
    max_diffractions: int
    diffraction_state_budget: int | None
    inserted_reflection_state_budget: int | None
    max_inserted_reflections_per_path: int | None
    shadow_support_cutoff_db: float | None
    solver_mode: str
    memory_profile: str
    reflection_field_backend: ReflectionFieldBackend
    tx_polarization: tuple[float, float, float]
    rx_polarization: tuple[float, float, float] | None
    diffraction_execution: DiffractionExecutionConfig
    reflection_transition_mode: ReflectionTransitionMode
    reflection_f_weight_boundary_radius_wavelengths: float
    reflection_f_weight_max_edges_per_slot: int
    reflection_secondary_visibility_mode: ReflectionSecondaryVisibilityMode
    cell_size: float


def resolve_trace_config(*, frequency: float, config: Config) -> ResolvedTraceConfig:
    tuning = config.tuning
    if (
        bool(config.shadow_boundary_correction)
        and (
            tuning.reflection_transition_mode != "hard"
            or tuning.reflection_secondary_visibility_mode != "hard"
        )
    ):
        raise ValueError(
            "shadow_boundary_correction=True cannot be combined with "
            "reflection F-weighting modes. Disable shadow_boundary_correction "
            "while reflection transition or secondary visibility F-weighting is active."
        )
    wavelength, k, cell_size = ResolvedTraceBase.wave_parameters(
        frequency=frequency,
        resolution_wavelength=float(tuning.resolution_wavelength),
    )
    return ResolvedTraceConfig(
        frequency=float(frequency),
        wavelength=wavelength,
        k=k,
        reflection_n_rays=int(config.num_samples),
        reflection_max_bounces=int(config.max_bounces),
        min_ray_contribution_threshold=float(tuning.min_ray_contribution_threshold),
        resolution_wavelength=float(tuning.resolution_wavelength),
        enable_rd_diffraction=bool(tuning.enable_rd_diffraction),
        max_diffractions=int(config.max_diffraction_order),
        diffraction_state_budget=(
            None if tuning.diffraction_state_budget is None
            else int(tuning.diffraction_state_budget)
        ),
        inserted_reflection_state_budget=(
            None if tuning.inserted_reflection_state_budget is None
            else int(tuning.inserted_reflection_state_budget)
        ),
        max_inserted_reflections_per_path=(
            None if tuning.max_inserted_reflections_per_path is None
            else int(tuning.max_inserted_reflections_per_path)
        ),
        shadow_support_cutoff_db=(
            None if tuning.shadow_support_cutoff_db is None
            else float(tuning.shadow_support_cutoff_db)
        ),
        solver_mode=str(tuning.solver_mode),
        memory_profile=str(tuning.memory_profile),
        reflection_field_backend=tuning.reflection_field_backend,
        tx_polarization=(1.0, 0.0, 0.0),
        rx_polarization=None,
        diffraction_execution=tuning.diffraction_execution,
        reflection_transition_mode=tuning.reflection_transition_mode,
        reflection_f_weight_boundary_radius_wavelengths=float(
            tuning.reflection_f_weight_boundary_radius_wavelengths
        ),
        reflection_f_weight_max_edges_per_slot=int(tuning.reflection_f_weight_max_edges_per_slot),
        reflection_secondary_visibility_mode=tuning.reflection_secondary_visibility_mode,
        cell_size=cell_size,
    )


def build_execution_intent(
    kind: TraceExecutionIntent,
    *,
    return_geometry: bool = False,
) -> dict[str, object]:
    _validate_literal(kind, name="execution_intent", allowed=_EXECUTION_INTENTS)
    return {
        "kind": str(kind),
        "path_export_enabled": bool(kind == "path_export"),
        "map_enabled": bool(kind == "coherent"),
        "geometry_enabled": bool(return_geometry),
        "field_payload": (
            "full" if kind == "field"
            else "total_only" if kind == "field_scalar_only"
            else "none"
        ),
    }


def resolve_solver_controls(
    config: Config,
    *,
    execution_intent: TraceExecutionIntent = "field",
    max_diffractions_override: int | None = None,
) -> dict[str, object]:
    tuning = config.tuning
    controls = resolve_common_solver_controls(
        SimpleNamespace(
            num_samples=config.num_samples,
            max_bounces=config.max_bounces,
            max_diffraction_order=config.max_diffraction_order,
            diffraction_state_budget=tuning.diffraction_state_budget,
            inserted_reflection_state_budget=tuning.inserted_reflection_state_budget,
            max_inserted_reflections_per_path=tuning.max_inserted_reflections_per_path,
            solver_mode=tuning.solver_mode,
            memory_profile=tuning.memory_profile,
        ),
        max_diffractions_override=max_diffractions_override,
    )
    return {
        **controls,
        "execution_intent": build_execution_intent(execution_intent),
    }


__all__ = [
    "Config",
    "DiffractionExecutionConfig",
    "FieldSolveSpec",
    "MemoryProfile",
    "ReflectionFieldBackend",
    "ReflectionSecondaryVisibilityMode",
    "ReflectionTransitionMode",
    "ReflectionSuffixConfig",
    "ResolvedTraceConfig",
    "SolveSpec",
    "SolverMode",
    "SuffixBackend",
    "Tuning",
    "TraceExecutionIntent",
    "build_execution_intent",
    "coerce_diffraction_execution",
    "resolve_solver_controls",
    "resolve_trace_config",
]
