"""System-level immutable configuration objects for Monte Carlo radiomap execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from witwin.channel._config.base import (
    CommonTraceTuning,
    MemoryProfile,
    ReflectionFieldBackend,
    ResolvedTraceBase,
    SolverMode,
    resolve_common_solver_controls,
)
from witwin.channel.core.scene.edge_policy import EdgePolicy, coerce_edge_policy


class ReceiverGridLike(Protocol):
    n_cells: int


AccumulatePrimalMode = Literal["auto", "drjit", "rayd_optix"]
SuffixBackend = Literal["drjit", "native"]
SuffixDdaMode = Literal["symbolic"]
ShadowBoundaryMode = Literal["none", "utd_power_smoothing"]
ShadowBoundaryBackend = Literal["auto", "drjit", "native_candidate"]
IntegratorName = Literal["basic", "bdpt"]
AccumulationBackend = Literal["auto", "native_monte_carlo", "rayd_reflection_accumulation"]
BDPTDiffractionSampling = Literal["hash", "sobol"]
FilterMethod = Literal["gaussian", "bilateral"]
TraceExecutionIntent = Literal[
    "field", "field_scalar_only", "path_export", "radio_map_incoherent",
]


def _coerce_component(value):
    if value is None or isinstance(value, ComponentFilterConfig):
        return value
    return ComponentFilterConfig(**dict(value))


def _coerce_filter(value):
    if value is None or isinstance(value, FilterConfig):
        return value
    return FilterConfig(**dict(value))


@dataclass(slots=True)
class ComponentFilterConfig:
    """Configure one differentiable power-domain component filter."""

    method: FilterMethod = "gaussian"
    radius: int = 1
    sigma: float = 1.0
    range_sigma: float | None = None
    blend: float = 1.0

    def __post_init__(self):
        if self.method not in ("gaussian", "bilateral"):
            raise ValueError(f"filtering method must be 'gaussian' or 'bilateral'; got {self.method!r}.")
        if self.radius < 0:
            raise ValueError("filter radius must be >= 0.")
        if self.sigma <= 0.0:
            raise ValueError("filter sigma must be > 0.")
        if self.range_sigma is None and self.method == "bilateral":
            self.range_sigma = self.sigma
        if self.range_sigma is not None and self.range_sigma <= 0.0:
            raise ValueError("filter range_sigma must be > 0 when provided.")
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("filter blend must be in the range [0, 1].")


@dataclass(slots=True)
class FilterConfig:
    """Configure opt-in differentiable Monte Carlo power filtering."""

    reflection: ComponentFilterConfig | Mapping[str, object] | None = None
    diffraction: ComponentFilterConfig | Mapping[str, object] | None = None

    def __post_init__(self):
        self.reflection = _coerce_component(self.reflection)
        self.diffraction = _coerce_component(self.diffraction)

    @property
    def enabled(self) -> bool:
        return self.reflection is not None or self.diffraction is not None


@dataclass(slots=True)
class DiffractionExecutionConfig:
    accumulate_primal: AccumulatePrimalMode = "auto"
    suffix_backend: SuffixBackend = "native"
    suffix_dda: SuffixDdaMode = "symbolic"
    suffix_russian_roulette: bool = False

    def __post_init__(self):
        if self.accumulate_primal not in _ALLOWED_DIFFRACTION_ACCUMULATE_PRIMAL:
            raise ValueError(
                "accumulate_primal must be one of "
                f"{sorted(_ALLOWED_DIFFRACTION_ACCUMULATE_PRIMAL)}; "
                f"got {self.accumulate_primal!r}."
            )


def _coerce_diffraction_execution(
    execution: DiffractionExecutionConfig | Mapping[str, object] | None,
) -> DiffractionExecutionConfig:
    if execution is None:
        return DiffractionExecutionConfig()
    if isinstance(execution, DiffractionExecutionConfig):
        return execution
    return DiffractionExecutionConfig(**dict(execution))


@dataclass(slots=True)
class TraceConfig:
    num_samples: int = 10000
    max_bounces: int = 2
    min_ray_contribution_threshold: float = 0.0
    resolution_wavelength: float = 0.125
    enable_rd_diffraction: bool = False
    max_diffraction_order: int = 2
    enable_bdpt_reflection_coupled_diffraction: bool = True
    shadow_boundary_backend: ShadowBoundaryBackend = "auto"
    shadow_boundary_tile_shape: tuple[int, int] = (8, 8)
    shadow_boundary_band_width_wavelengths: float = 3.0
    shadow_boundary_max_candidate_factor: float = 64.0
    diffraction_state_budget: int | None = None
    inserted_reflection_state_budget: int | None = None
    max_inserted_reflections_per_path: int | None = None
    solver_mode: SolverMode = "accuracy"
    memory_profile: MemoryProfile = "default"
    reflection_field_backend: ReflectionFieldBackend = "native"
    diffraction_execution: DiffractionExecutionConfig = field(default_factory=DiffractionExecutionConfig)


_ALLOWED_INTEGRATORS = {"basic", "bdpt"}
_ALLOWED_ACCUMULATION = {"auto", "native_monte_carlo", "rayd_reflection_accumulation"}
_ALLOWED_DIFFRACTION_ACCUMULATE_PRIMAL = {"auto", "drjit", "rayd_optix"}
_ALLOWED_SHADOW_MODES = {"none", "utd_power_smoothing"}
_ALLOWED_SHADOW_BACKENDS = {"auto", "drjit", "native_candidate"}
_ALLOWED_BDPT_SAMPLING = {"hash", "sobol"}
@dataclass(slots=True)
class Tuning(CommonTraceTuning):
    """Advanced runtime controls for Monte Carlo radiomap solving."""

    min_ray_contribution_threshold: float = 0.0
    resolution_wavelength: float = 0.125
    enable_rd_diffraction: bool = False
    enable_bdpt_reflection_coupled_diffraction: bool = True
    shadow_boundary_mode: ShadowBoundaryMode = "utd_power_smoothing"
    shadow_boundary_backend: ShadowBoundaryBackend = "auto"
    shadow_boundary_tile_shape: tuple[int, int] = (8, 8)
    shadow_boundary_band_width_wavelengths: float = 3.0
    shadow_boundary_max_candidate_factor: float = 64.0
    diffraction_state_budget: int | None = None
    inserted_reflection_state_budget: int | None = None
    max_inserted_reflections_per_path: int | None = None
    solver_mode: SolverMode = "accuracy"
    memory_profile: MemoryProfile = "default"
    reflection_field_backend: ReflectionFieldBackend = "native"
    diffraction_execution: DiffractionExecutionConfig | Mapping[str, object] | None = None

    def __post_init__(self):
        self.validate_common_trace_tuning()
        if self.shadow_boundary_mode not in _ALLOWED_SHADOW_MODES:
            raise ValueError(f"shadow_boundary_mode must be one of {sorted(_ALLOWED_SHADOW_MODES)}; got {self.shadow_boundary_mode!r}.")
        if self.shadow_boundary_backend not in _ALLOWED_SHADOW_BACKENDS:
            raise ValueError(f"shadow_boundary_backend must be one of {sorted(_ALLOWED_SHADOW_BACKENDS)}; got {self.shadow_boundary_backend!r}.")
        tile = tuple(self.shadow_boundary_tile_shape)
        if len(tile) != 2 or int(tile[0]) <= 0 or int(tile[1]) <= 0:
            raise ValueError("shadow_boundary_tile_shape must contain exactly two positive integers.")
        self.shadow_boundary_tile_shape = (int(tile[0]), int(tile[1]))
        if self.shadow_boundary_band_width_wavelengths <= 0.0:
            raise ValueError("shadow_boundary_band_width_wavelengths must be > 0.")
        if self.shadow_boundary_max_candidate_factor <= 0.0:
            raise ValueError("shadow_boundary_max_candidate_factor must be > 0.")
        self.diffraction_execution = _coerce_diffraction_execution(self.diffraction_execution)


@dataclass(frozen=True, slots=True)
class IntegratorOptions:
    """Integrator-specific Monte Carlo runtime controls."""

    samples_per_tx: int = 65536
    seed: int = 0
    rr_depth: int | None = None
    rr_prob: float | None = None
    stop_threshold: float | None = None
    bdpt_diffraction_sampling: BDPTDiffractionSampling = "sobol"
    integrator: IntegratorName = "basic"
    accumulation_backend: AccumulationBackend = "auto"
    ad: bool | None = None

    def __post_init__(self):
        if self.samples_per_tx <= 0:
            raise ValueError("samples_per_tx must be > 0.")
        if self.rr_depth is not None and self.rr_depth < 0:
            raise ValueError("rr_depth must be >= 0 when provided.")
        if self.rr_prob is not None and not (0.0 < self.rr_prob <= 1.0):
            raise ValueError("rr_prob must be in the range (0, 1].")
        if self.stop_threshold is not None and self.stop_threshold < 0.0:
            raise ValueError("stop_threshold must be >= 0 when provided.")
        if self.integrator not in _ALLOWED_INTEGRATORS:
            raise ValueError("integrator must be 'basic' or 'bdpt'.")
        if self.accumulation_backend not in _ALLOWED_ACCUMULATION:
            raise ValueError(
                "accumulation_backend must be 'auto', 'native_monte_carlo', "
                "or 'rayd_reflection_accumulation'."
            )
        if self.bdpt_diffraction_sampling not in _ALLOWED_BDPT_SAMPLING:
            raise ValueError(f"bdpt_diffraction_sampling must be one of {sorted(_ALLOWED_BDPT_SAMPLING)}; got {self.bdpt_diffraction_sampling!r}.")
        if self.ad is not None and not isinstance(self.ad, bool):
            raise TypeError("ad must be a bool or None.")


def _coerce_tuning(tuning: Tuning | Mapping[str, object] | None) -> Tuning:
    if tuning is None:
        return Tuning()
    if isinstance(tuning, Tuning):
        return tuning
    return Tuning(**dict(tuning))


def _coerce_integrator_options(
    integrator_options: IntegratorOptions | Mapping[str, object] | None,
) -> IntegratorOptions:
    if integrator_options is None:
        return IntegratorOptions()
    if isinstance(integrator_options, IntegratorOptions):
        return integrator_options
    return IntegratorOptions(**dict(integrator_options))


@dataclass(slots=True)
class Config:
    """Configure the public Monte Carlo radiomap solver contract."""

    num_samples: int = 10000
    max_bounces: int = 2
    max_diffraction_order: int = 1
    synthetic_array: bool = True
    edge_policy: EdgePolicy | None = None
    filtering: FilterConfig | Mapping[str, object] | None = None
    tuning: Tuning | Mapping[str, object] | None = None
    integrator_options: IntegratorOptions | Mapping[str, object] | None = None

    def __post_init__(self):
        if self.num_samples <= 0:
            raise ValueError("num_samples must be > 0.")
        if self.max_bounces < 0:
            raise ValueError("max_bounces must be >= 0.")
        if not 0 <= self.max_diffraction_order <= 3:
            raise ValueError("Monte Carlo radiomap currently supports max_diffraction_order in the range [0, 3].")
        if not isinstance(self.synthetic_array, bool):
            raise TypeError("synthetic_array must be a bool.")
        integrator_options = _coerce_integrator_options(self.integrator_options)
        if integrator_options.integrator == "basic" and self.max_diffraction_order > 1:
            raise ValueError("integrator='basic' supports max_diffraction_order in {0, 1}; use integrator='bdpt' for depths up to 3.")
        self.filtering = _coerce_filter(self.filtering)
        self.edge_policy = coerce_edge_policy(self.edge_policy)
        self.tuning = _coerce_tuning(self.tuning)
        self.integrator_options = integrator_options

    def to_trace_config(self) -> TraceConfig:
        tuning = self.tuning
        return TraceConfig(
            num_samples=self.num_samples,
            max_bounces=self.max_bounces,
            min_ray_contribution_threshold=tuning.min_ray_contribution_threshold,
            resolution_wavelength=tuning.resolution_wavelength,
            enable_rd_diffraction=tuning.enable_rd_diffraction,
            max_diffraction_order=self.max_diffraction_order,
            enable_bdpt_reflection_coupled_diffraction=tuning.enable_bdpt_reflection_coupled_diffraction,
            shadow_boundary_backend=tuning.shadow_boundary_backend,
            shadow_boundary_tile_shape=tuning.shadow_boundary_tile_shape,
            shadow_boundary_band_width_wavelengths=tuning.shadow_boundary_band_width_wavelengths,
            shadow_boundary_max_candidate_factor=tuning.shadow_boundary_max_candidate_factor,
            diffraction_state_budget=tuning.diffraction_state_budget,
            inserted_reflection_state_budget=tuning.inserted_reflection_state_budget,
            max_inserted_reflections_per_path=tuning.max_inserted_reflections_per_path,
            solver_mode=tuning.solver_mode,
            memory_profile=tuning.memory_profile,
            reflection_field_backend=tuning.reflection_field_backend,
            diffraction_execution=tuning.diffraction_execution,
        )


def build_execution_intent(
    kind: TraceExecutionIntent,
    *,
    return_geometry: bool = False,
    return_diffraction_audit: bool = False,
) -> dict[str, object]:
    if kind not in {"field", "field_scalar_only", "path_export", "radio_map_incoherent"}:
        raise ValueError(f"Unsupported execution intent: {kind}")
    is_radio_map = kind == "radio_map_incoherent"
    field_payload = (
        "full" if kind == "field"
        else "total_only" if kind == "field_scalar_only"
        else "none"
    )
    return {
        "kind": str(kind),
        "path_export_enabled": kind == "path_export",
        "radio_map_enabled": is_radio_map,
        "geometry_enabled": bool(return_geometry),
        "field_payload": field_payload,
        "radio_map_combine_mode": "incoherent" if is_radio_map else None,
        "diffraction_audit_enabled": bool(return_diffraction_audit),
    }


def resolve_solver_controls(
    config: TraceConfig,
    *,
    execution_intent: TraceExecutionIntent = "field",
    max_diffractions_override: int | None = None,
) -> dict[str, object]:
    controls = resolve_common_solver_controls(
        config,
        max_diffractions_override=max_diffractions_override,
    )
    return {
        **controls,
        "execution_intent": build_execution_intent(execution_intent),
    }


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
    enable_bdpt_reflection_coupled_diffraction: bool
    shadow_boundary_backend: ShadowBoundaryBackend
    shadow_boundary_tile_shape: tuple[int, int]
    shadow_boundary_band_width_wavelengths: float
    shadow_boundary_max_candidate_factor: float
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

    @classmethod
    def from_config(cls, *, frequency: float, config: TraceConfig) -> ResolvedTraceConfig:
        resolution_wavelength = float(config.resolution_wavelength)
        wavelength, k, cell_size = ResolvedTraceBase.wave_parameters(
            frequency=frequency,
            resolution_wavelength=resolution_wavelength,
        )
        tx_polarization = getattr(config, "tx_polarization", (1.0, 0.0, 0.0))
        rx_polarization = getattr(config, "rx_polarization", None)
        return cls(
            frequency=float(frequency),
            wavelength=wavelength,
            k=k,
            reflection_n_rays=int(config.num_samples),
            reflection_max_bounces=int(config.max_bounces),
            min_ray_contribution_threshold=float(config.min_ray_contribution_threshold),
            resolution_wavelength=resolution_wavelength,
            enable_rd_diffraction=bool(config.enable_rd_diffraction),
            max_diffractions=int(config.max_diffraction_order),
            enable_bdpt_reflection_coupled_diffraction=bool(config.enable_bdpt_reflection_coupled_diffraction),
            shadow_boundary_backend=config.shadow_boundary_backend,
            shadow_boundary_tile_shape=tuple(config.shadow_boundary_tile_shape),
            shadow_boundary_band_width_wavelengths=float(config.shadow_boundary_band_width_wavelengths),
            shadow_boundary_max_candidate_factor=float(config.shadow_boundary_max_candidate_factor),
            diffraction_state_budget=(
                None if config.diffraction_state_budget is None else int(config.diffraction_state_budget)
            ),
            inserted_reflection_state_budget=(
                None if config.inserted_reflection_state_budget is None
                else int(config.inserted_reflection_state_budget)
            ),
            max_inserted_reflections_per_path=(
                None if config.max_inserted_reflections_per_path is None
                else int(config.max_inserted_reflections_per_path)
            ),
            solver_mode=str(config.solver_mode),
            memory_profile=str(config.memory_profile),
            reflection_field_backend=config.reflection_field_backend,
            tx_polarization=tuple(float(value) for value in tx_polarization),
            rx_polarization=(
                None if rx_polarization is None
                else tuple(float(value) for value in rx_polarization)
            ),
            diffraction_execution=config.diffraction_execution,
            cell_size=cell_size,
        )


__all__ = [
    "AccumulatePrimalMode",
    "AccumulationBackend",
    "BDPTDiffractionSampling",
    "ComponentFilterConfig",
    "Config",
    "DiffractionExecutionConfig",
    "FilterConfig",
    "FilterMethod",
    "IntegratorName",
    "IntegratorOptions",
    "MemoryProfile",
    "ReflectionFieldBackend",
    "ResolvedTraceConfig",
    "ShadowBoundaryBackend",
    "ShadowBoundaryMode",
    "SolverMode",
    "SuffixBackend",
    "SuffixDdaMode",
    "Tuning",
    "TraceConfig",
    "TraceExecutionIntent",
    "build_execution_intent",
    "resolve_solver_controls",
]
