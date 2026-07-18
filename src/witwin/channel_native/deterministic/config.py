from __future__ import annotations

from dataclasses import dataclass

from witwin.channel_native.core.components import (
    AD_MODES as _VALID_AD_MODES,
    DEFAULT_COMPONENTS as _DEFAULT_COMPONENTS,
    validated_components,
)


# Public component set. transmission carries specular wall-penetration paths
# (wave 2); scattering carries single-bounce Kirchhoff rough-surface paths
# (wave 3). transmission depth is capped like reflection (chains count wall
# penetrations); scattering is single-bounce in v1.
# Default component set is unchanged: the new components are strictly opt-in.
# Components whose chain length is bounded by max_depth (public cap of 5).
_DEPTH_CAPPED_COMPONENTS = frozenset({"reflection", "transmission"})
_VALID_SORT_KEYS = frozenset({"receiver_transmitter_depth_component"})
_VALID_MAX_PATHS_SCOPES = frozenset({"global", "per_pair"})
_MAX_COUPLED_CANDIDATES = 1_000_000


def _validate_coupled_config(
    *,
    coupled_paths: bool,
    max_depth: int,
    components: frozenset[str],
    coupled_candidate_limit: int,
) -> None:
    """Validate the coupled reflection-diffraction gate (ADR-011)."""

    if coupled_paths:
        if max_depth < 2:
            raise RuntimeError(
                "coupled reflection-diffraction paths require max_depth >= 2"
            )
        if not {"reflection", "diffraction"}.issubset(components):
            raise RuntimeError(
                "coupled paths require both reflection and diffraction components"
            )
    if coupled_candidate_limit <= 0:
        raise ValueError("coupled_candidate_limit must be positive")
    if coupled_candidate_limit > _MAX_COUPLED_CANDIDATES:
        raise ValueError(
            "coupled_candidate_limit cannot exceed the hard limit of 1000000"
        )


@dataclass(frozen=True, slots=True)
class Config:
    max_depth: int = 1
    max_diffraction_order: int = 1
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str] = (
        _DEFAULT_COMPONENTS
    )
    coherent: bool = True
    return_field: bool = True
    export_paths: bool = False
    max_paths: int | None = None
    max_paths_scope: str = "global"
    sort_key: str = "receiver_transmitter_depth_component"
    diagnostics: bool = False
    ad_mode: str = "none"
    # Coupled reflection-diffraction paths (ADR-011). Opt-in; when set the
    # deterministic grid solver enumerates the R->D / D->R compensator rows
    # (component ids 3/4) and accumulates their coherent contribution into a
    # dedicated coupled field slot. coupled_candidate_limit is a per-receiver-
    # block work/safety budget: the deterministic engine streams coupled
    # discovery over receiver blocks sized so each block stays under it.
    coupled_paths: bool = False
    coupled_candidate_limit: int = 1_000_000
    # Rough-surface scattering quadrature (wave 3). The patch density is a
    # fixed per-area sample count with a documented per-face cap of 4096
    # samples; per (tx, rx) pair only the strongest samples up to
    # scattering_max_paths_per_pair survive (dropped power is reported in the
    # scattering metadata, never silently redistributed) and samples below
    # scattering_power_threshold (absolute path_gain floor) are discarded.
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
        if self.max_diffraction_order < 0:
            raise ValueError("max_diffraction_order must be 0 or 1")
        if self.max_diffraction_order > 1:
            raise RuntimeError("max_diffraction_order above 1 is not supported yet")

        components = validated_components(
            self.components,
            error_message="components must be a non-empty subset of {valid}",
        )
        if self.max_depth > 5 and components & _DEPTH_CAPPED_COMPONENTS:
            raise RuntimeError(
                "deterministic reflection/transmission currently support max_depth <= 5"
            )
        _validate_coupled_config(
            coupled_paths=self.coupled_paths,
            max_depth=self.max_depth,
            components=components,
            coupled_candidate_limit=self.coupled_candidate_limit,
        )
        if self.max_paths is not None and self.max_paths <= 0:
            raise ValueError("max_paths must be positive when set")
        if self.max_paths_scope not in _VALID_MAX_PATHS_SCOPES:
            raise ValueError("max_paths_scope must be 'global' or 'per_pair'")
        if self.sort_key not in _VALID_SORT_KEYS:
            raise ValueError(f"sort_key must be one of {sorted(_VALID_SORT_KEYS)}")
        if self.ad_mode not in _VALID_AD_MODES:
            raise RuntimeError(
                f"deterministic ad_mode must be one of {sorted(_VALID_AD_MODES)}"
            )

        object.__setattr__(self, "components", components)
