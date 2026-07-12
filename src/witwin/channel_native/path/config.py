from __future__ import annotations

from dataclasses import dataclass

from witwin.channel_native.core.components import (
    DEFAULT_COMPONENTS as _DEFAULT_COMPONENTS,
    NO_AD_MODES as _VALID_AD_MODES,
    validated_components,
)


# Public component set. transmission exports specular wall-penetration paths
# (wave 2); scattering exports single-bounce incoherent Kirchhoff patch paths
# (wave 3). transmission depth is capped like reflection (chains count wall
# penetrations); scattering is single-bounce in v1.
# Default component set is unchanged: the new components are strictly opt-in.
# Components whose chain length is bounded by max_depth (public cap of 5).
_DEPTH_CAPPED_COMPONENTS = frozenset({"reflection", "transmission"})
_VALID_MAX_PATHS_SCOPES = frozenset({"per_pair"})
_MAX_COUPLED_CANDIDATES = 1_000_000


@dataclass(frozen=True, slots=True)
class Config:
    max_depth: int = 1
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str] = (
        _DEFAULT_COMPONENTS
    )
    max_paths: int | None = None
    max_paths_scope: str = "per_pair"
    ad_mode: str = "none"
    coupled_paths: bool = False
    coupled_candidate_limit: int = 1_000_000
    # Rough-surface scattering quadrature (wave 3): fixed per-area sample
    # density (per-face cap of 4096 samples), a per-pair strongest-paths cap,
    # and an absolute path_gain floor for exported patch paths.
    scattering_samples_per_m2: float = 8.0
    scattering_max_paths_per_pair: int = 4096
    scattering_power_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.scattering_samples_per_m2 <= 0.0:
            raise ValueError("scattering_samples_per_m2 must be positive")
        if self.scattering_max_paths_per_pair <= 0:
            raise ValueError("scattering_max_paths_per_pair must be positive")
        if self.scattering_power_threshold < 0.0:
            raise ValueError("scattering_power_threshold must be non-negative")
        components = validated_components(
            self.components, error_message="components must be a subset of {valid}"
        )
        if self.max_depth > 5 and components & _DEPTH_CAPPED_COMPONENTS:
            raise RuntimeError("path reflection/transmission support max_depth <= 5")
        if self.coupled_paths:
            if self.max_depth < 2:
                raise RuntimeError(
                    "coupled reflection-diffraction paths require max_depth >= 2"
                )
            if not {"reflection", "diffraction"}.issubset(components):
                raise RuntimeError(
                    "coupled paths require both reflection and diffraction components"
                )
        if self.max_paths is not None and self.max_paths <= 0:
            raise ValueError("max_paths must be positive when set")
        if self.max_paths_scope not in _VALID_MAX_PATHS_SCOPES:
            raise ValueError("path max_paths_scope must be 'per_pair'")
        if self.coupled_candidate_limit <= 0:
            raise ValueError("coupled_candidate_limit must be positive")
        if self.coupled_candidate_limit > _MAX_COUPLED_CANDIDATES:
            raise ValueError(
                "coupled_candidate_limit cannot exceed the hard limit of 1000000"
            )
        if self.ad_mode not in _VALID_AD_MODES:
            raise ValueError(
                "path supports_ad=False in the first replacement release; "
                "ad_mode must be 'none'"
            )
        object.__setattr__(self, "components", components)
