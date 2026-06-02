"""System-level immutable configuration objects for channel execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol


class ReceiverGridLike(Protocol):
    n_cells: int


SolverMode = Literal["accuracy", "fast_approximate"]
AccumulatePrimalMode = Literal["drjit"]
AccumulateJvpMode = Literal["drjit_replay"]
AccumulateBackwardMode = Literal["drjit_replay"]
ReflectionFieldBackend = Literal["drjit", "native"]
SuffixBackend = Literal["drjit", "native"]
SuffixDdaMode = Literal["symbolic"]
MemoryProfile = Literal["default", "memory_safe"]

_SOLVER_MODES = {"accuracy", "fast_approximate"}
_ACCUMULATE_PRIMAL_MODES = {"drjit"}
_ACCUMULATE_JVP_MODES = {"drjit_replay"}
_ACCUMULATE_BACKWARD_MODES = {"drjit_replay"}
_REFLECTION_FIELD_BACKENDS = {"drjit", "native"}
_SUFFIX_BACKENDS = {"drjit", "native"}
_SUFFIX_DDA_MODES = {"symbolic"}
_MEMORY_PROFILES = {"default", "memory_safe"}


def _ensure_mapping(data: Mapping[str, object] | None, *, name: str) -> Mapping[str, object]:
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return data


def _coerce_polarization(
    value: Sequence[float] | None,
    *,
    name: str = "tx_polarization",
) -> tuple[float, float, float]:
    if value is None:
        return (1.0, 0.0, 0.0)
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three components.")
    return (float(value[0]), float(value[1]), float(value[2]))


def _validate_literal(value: str, *, name: str, allowed: set[str]) -> str:
    if value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of {allowed_text}; got {value!r}.")
    return value


@dataclass(slots=True)
class DiffractionExecutionConfig:
    accumulate_primal: AccumulatePrimalMode = "drjit"
    accumulate_jvp: AccumulateJvpMode = "drjit_replay"
    accumulate_backward: AccumulateBackwardMode = "drjit_replay"
    suffix_backend: SuffixBackend = "native"
    suffix_dda: SuffixDdaMode = "symbolic"
    suffix_russian_roulette: bool = False

    def __post_init__(self):
        self.accumulate_primal = _validate_literal(
            self.accumulate_primal,
            name="accumulate_primal",
            allowed=_ACCUMULATE_PRIMAL_MODES,
        )
        self.accumulate_jvp = _validate_literal(
            self.accumulate_jvp,
            name="accumulate_jvp",
            allowed=_ACCUMULATE_JVP_MODES,
        )
        self.accumulate_backward = _validate_literal(
            self.accumulate_backward,
            name="accumulate_backward",
            allowed=_ACCUMULATE_BACKWARD_MODES,
        )
        self.suffix_backend = _validate_literal(
            self.suffix_backend,
            name="suffix_backend",
            allowed=_SUFFIX_BACKENDS,
        )
        self.suffix_dda = _validate_literal(
            self.suffix_dda,
            name="suffix_dda",
            allowed=_SUFFIX_DDA_MODES,
        )
        self.suffix_russian_roulette = bool(self.suffix_russian_roulette)

    @classmethod
    def default(cls):
        return cls(
            accumulate_primal="drjit",
            accumulate_jvp="drjit_replay",
            accumulate_backward="drjit_replay",
            suffix_backend="native",
            suffix_dda="symbolic",
        )

    @classmethod
    def strict_drjit(cls):
        return cls(
            accumulate_primal="drjit",
            accumulate_jvp="drjit_replay",
            accumulate_backward="drjit_replay",
            suffix_backend="drjit",
            suffix_dda="symbolic",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None):
        data = _ensure_mapping(data, name="diffraction execution config")
        if not data:
            return cls.default()
        return cls(
            accumulate_primal=data.get("accumulate_primal", "drjit"),
            accumulate_jvp=data.get("accumulate_jvp", "drjit_replay"),
            accumulate_backward=data.get("accumulate_backward", "drjit_replay"),
            suffix_backend=data.get("suffix_backend", "native"),
            suffix_dda=data.get("suffix_dda", "symbolic"),
            suffix_russian_roulette=data.get("suffix_russian_roulette", False),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "accumulate_primal": self.accumulate_primal,
            "accumulate_jvp": self.accumulate_jvp,
            "accumulate_backward": self.accumulate_backward,
            "suffix_backend": self.suffix_backend,
            "suffix_dda": self.suffix_dda,
            "suffix_russian_roulette": self.suffix_russian_roulette,
        }


@dataclass(slots=True)
class TraceConfig:
    reflection_n_rays: int = 10000
    reflection_max_bounces: int = 2
    reflection_coef: float = 0.7
    min_ray_contribution_threshold: float = 0.0
    reflection_relative_permittivity: float = 5.0
    reflection_conductivity: float = 0.0
    reflection_material: Mapping[str, object] | None = None
    diffraction_material: Mapping[str, object] | None = None
    use_scene_materials_for_reflection: bool = True
    use_scene_materials_for_diffraction: bool = True
    resolution_wavelength: float = 0.125
    enable_rd_diffraction: bool = False
    max_diffractions: int = 2
    diffraction_state_budget: int | None = None
    inserted_reflection_state_budget: int | None = None
    max_inserted_reflections_per_path: int | None = None
    solver_mode: SolverMode = "accuracy"
    memory_profile: MemoryProfile = "default"
    reflection_field_backend: ReflectionFieldBackend = "native"
    tx_polarization: tuple[float, float, float] = (1.0, 0.0, 0.0)
    rx_polarization: tuple[float, float, float] | None = None
    diffraction_execution: DiffractionExecutionConfig = field(default_factory=DiffractionExecutionConfig.default)

    def __post_init__(self):
        self.reflection_n_rays = int(self.reflection_n_rays)
        self.reflection_max_bounces = int(self.reflection_max_bounces)
        self.reflection_coef = float(self.reflection_coef)
        self.min_ray_contribution_threshold = float(self.min_ray_contribution_threshold)
        self.reflection_relative_permittivity = float(self.reflection_relative_permittivity)
        self.reflection_conductivity = float(self.reflection_conductivity)
        self.use_scene_materials_for_reflection = bool(self.use_scene_materials_for_reflection)
        self.use_scene_materials_for_diffraction = bool(self.use_scene_materials_for_diffraction)
        self.resolution_wavelength = float(self.resolution_wavelength)
        self.enable_rd_diffraction = bool(self.enable_rd_diffraction)
        self.max_diffractions = int(self.max_diffractions)
        self.diffraction_state_budget = (
            None if self.diffraction_state_budget is None else int(self.diffraction_state_budget)
        )
        self.inserted_reflection_state_budget = (
            None
            if self.inserted_reflection_state_budget is None
            else int(self.inserted_reflection_state_budget)
        )
        self.max_inserted_reflections_per_path = (
            None
            if self.max_inserted_reflections_per_path is None
            else int(self.max_inserted_reflections_per_path)
        )
        self.tx_polarization = _coerce_polarization(
            self.tx_polarization,
            name="tx_polarization",
        )
        self.rx_polarization = (
            None
            if self.rx_polarization is None
            else _coerce_polarization(self.rx_polarization, name="rx_polarization")
        )
        self.solver_mode = _validate_literal(
            self.solver_mode,
            name="solver_mode",
            allowed=_SOLVER_MODES,
        )
        self.memory_profile = _validate_literal(
            self.memory_profile,
            name="memory_profile",
            allowed=_MEMORY_PROFILES,
        )
        self.reflection_field_backend = _validate_literal(
            self.reflection_field_backend,
            name="reflection_field_backend",
            allowed=_REFLECTION_FIELD_BACKENDS,
        )
        if not 0.0 <= self.min_ray_contribution_threshold < 1.0:
            raise ValueError("min_ray_contribution_threshold must satisfy 0.0 <= value < 1.0.")
        self.reflection_material = (
            None if self.reflection_material is None else dict(self.reflection_material)
        )
        self.diffraction_material = (
            None if self.diffraction_material is None else dict(self.diffraction_material)
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None):
        data = _ensure_mapping(data, name="trace config")
        return cls(
            reflection_n_rays=data.get("reflection_n_rays", 10000),
            reflection_max_bounces=data.get("reflection_max_bounces", 2),
            reflection_coef=data.get("reflection_coef", 0.7),
            min_ray_contribution_threshold=data.get("min_ray_contribution_threshold", 0.0),
            reflection_relative_permittivity=data.get("reflection_relative_permittivity", 5.0),
            reflection_conductivity=data.get("reflection_conductivity", 0.0),
            reflection_material=data.get("reflection_material"),
            diffraction_material=data.get("diffraction_material"),
            use_scene_materials_for_reflection=data.get("use_scene_materials_for_reflection", True),
            use_scene_materials_for_diffraction=data.get("use_scene_materials_for_diffraction", True),
            resolution_wavelength=data.get("resolution_wavelength", 0.125),
            enable_rd_diffraction=data.get("enable_rd_diffraction", False),
            max_diffractions=data.get("max_diffractions", 2),
            diffraction_state_budget=data.get("diffraction_state_budget"),
            inserted_reflection_state_budget=data.get("inserted_reflection_state_budget"),
            max_inserted_reflections_per_path=data.get("max_inserted_reflections_per_path"),
            solver_mode=data.get("solver_mode", "accuracy"),
            memory_profile=data.get("memory_profile", "default"),
            reflection_field_backend=data.get("reflection_field_backend", "native"),
            tx_polarization=data.get("tx_polarization", (1.0, 0.0, 0.0)),
            rx_polarization=data.get("rx_polarization"),
            diffraction_execution=DiffractionExecutionConfig.from_dict(data.get("diffraction_execution")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "reflection_n_rays": self.reflection_n_rays,
            "reflection_max_bounces": self.reflection_max_bounces,
            "reflection_coef": self.reflection_coef,
            "min_ray_contribution_threshold": self.min_ray_contribution_threshold,
            "reflection_relative_permittivity": self.reflection_relative_permittivity,
            "reflection_conductivity": self.reflection_conductivity,
            "reflection_material": None if self.reflection_material is None else dict(self.reflection_material),
            "diffraction_material": None if self.diffraction_material is None else dict(self.diffraction_material),
            "use_scene_materials_for_reflection": self.use_scene_materials_for_reflection,
            "use_scene_materials_for_diffraction": self.use_scene_materials_for_diffraction,
            "resolution_wavelength": self.resolution_wavelength,
            "enable_rd_diffraction": self.enable_rd_diffraction,
            "max_diffractions": self.max_diffractions,
            "diffraction_state_budget": self.diffraction_state_budget,
            "inserted_reflection_state_budget": self.inserted_reflection_state_budget,
            "max_inserted_reflections_per_path": self.max_inserted_reflections_per_path,
            "solver_mode": self.solver_mode,
            "memory_profile": self.memory_profile,
            "reflection_field_backend": self.reflection_field_backend,
            "tx_polarization": list(self.tx_polarization),
            "rx_polarization": None if self.rx_polarization is None else list(self.rx_polarization),
            "diffraction_execution": self.diffraction_execution.to_dict(),
        }


@dataclass(slots=True)
class ChannelConfig:
    trace: TraceConfig = field(default_factory=TraceConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None):
        data = _ensure_mapping(data, name="channel config")
        return cls(
            trace=TraceConfig.from_dict(data.get("trace")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "trace": self.trace.to_dict(),
        }


def coerce_channel_config(config: ChannelConfig | Mapping[str, object] | None) -> ChannelConfig:
    if config is None:
        return ChannelConfig()
    if isinstance(config, ChannelConfig):
        return config
    return ChannelConfig.from_dict(config)


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
        return self.grid is not None and self.grid_data is not None and self.max_bounces > 0 and self.n_rays > 0


def coerce_diffraction_execution(
    execution: DiffractionExecutionConfig | Mapping[str, object] | None,
) -> DiffractionExecutionConfig:
    if execution is None:
        return DiffractionExecutionConfig.default()
    if isinstance(execution, DiffractionExecutionConfig):
        return execution
    return DiffractionExecutionConfig.from_dict(execution)


__all__ = [
    "AccumulateBackwardMode",
    "AccumulateJvpMode",
    "AccumulatePrimalMode",
    "ChannelConfig",
    "DiffractionExecutionConfig",
    "MemoryProfile",
    "ReflectionFieldBackend",
    "ReflectionSuffixConfig",
    "SolverMode",
    "SuffixBackend",
    "SuffixDdaMode",
    "TraceConfig",
    "coerce_channel_config",
    "coerce_diffraction_execution",
]
