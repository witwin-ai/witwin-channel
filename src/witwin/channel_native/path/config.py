from __future__ import annotations

from dataclasses import dataclass

from witwin.channel_native.core.components import (
    AD_MODES as _VALID_AD_MODES,
    DEFAULT_COMPONENTS as _DEFAULT_COMPONENTS,
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
# ADR-021 D1 chain-depth cap: each specular leg is bounded by the native
# kMaxAdDepth = 8, so the public cap on d1 + d2 is 2 * 8 = 16.
_MAX_SCATTER_CHAIN_DEPTH = 16


def _validate_scatter_chain(
    *,
    max_depth: int,
    samples_per_m2: float,
    max_rows: int,
    components: frozenset[str],
) -> None:
    """Validate the ADR-021 D1 enumerated scatter-chain config (shared)."""

    if max_depth < 0:
        raise ValueError("scattering_chain_max_depth must be non-negative")
    if max_depth > _MAX_SCATTER_CHAIN_DEPTH:
        raise ValueError(
            "scattering_chain_max_depth cannot exceed 16 (2 * kMaxAdDepth); each "
            "specular leg is bounded by the native kMaxAdDepth = 8"
        )
    if samples_per_m2 <= 0.0:
        raise ValueError("scattering_chain_samples_per_m2 must be positive")
    if max_rows <= 0:
        raise ValueError("scattering_chain_max_rows must be positive")
    if max_depth >= 1 and "scattering" not in components:
        raise RuntimeError(
            "scattering_chain_max_depth >= 1 requires the 'scattering' component "
            "(ADR-021 D1 appends component_id=6 scatter-chain rows)"
        )


def _validate_isb_boundary_taper(width: float) -> None:
    """Validate the ADR-017 ISB boundary taper width bound."""

    if not (0.0 < width <= 4.0):
        raise ValueError("isb_boundary_taper_width must be in (0, 4]")


def _validate_capacity_config(
    *,
    path_capacity_per_pair: int | None,
    diffraction_state_capacity: int | None,
    max_paths: int | None,
) -> None:
    """Validate the ADR-029 host-known capacity fields."""

    for name, value in (
        ("path_capacity_per_pair", path_capacity_per_pair),
        ("diffraction_state_capacity", diffraction_state_capacity),
    ):
        if value is None:
            continue
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer when set")
        if value < 0:
            raise ValueError(f"{name} must be non-negative when set")
    if (
        max_paths is not None
        and path_capacity_per_pair is not None
        and max_paths > path_capacity_per_pair
    ):
        raise ValueError(
            "max_paths cannot exceed path_capacity_per_pair when "
            "max_paths_scope='per_pair'"
        )


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
    # Enumerated scatter-chain path class (ADR-021 D1). DEFAULT-OFF opt-in:
    # scattering_chain_max_depth = 0 disables chain discovery so exported paths
    # are byte-identical to today. When >= 1 it caps d1 + d2, the combined
    # specular reflection depth of the two legs around the single diffuse vertex
    # (TX --C1(d1)--> v_s --C2(d2)--> RX). Each leg is bounded by the native
    # kMaxAdDepth = 8, so the public cap is 2 * 8 = 16. Chain vertices are drawn
    # at a documented lower density (scattering_chain_samples_per_m2) and only
    # the strongest scattering_chain_max_rows joined rows per (tx, rx) survive.
    # Requires the 'scattering' component.
    scattering_chain_max_depth: int = 0
    scattering_chain_samples_per_m2: float = 2.0
    scattering_chain_max_rows: int = 256
    # ISB boundary taper (ADR-017). DEFAULT-OFF visual-continuity heuristic: the
    # hard LoS occlusion gate becomes a C1 membership taper tau(c / (width * w_F))
    # and the compensating order-1 diffraction odd step spreads over the same
    # congruent window. OFF (the default) is bit-identical to the hard gate and
    # the unchanged diffraction window for every existing caller (enforced by a
    # bitwise regression test); the switch must never default ON. The width
    # scales the Fresnel penumbra w_F of the grazed silhouette edge; the
    # projection-validated optimum is 0.5 (artifacts/isb-taper/report.json).
    isb_boundary_taper: bool = False
    isb_boundary_taper_width: float = 0.5
    # ADR-029 host-known capacity contracts. None remains constructible during
    # staged activation; solve rejects it when the corresponding capacity is
    # required. Capacity is not a path-selection/truncation policy.
    path_capacity_per_pair: int | None = None
    diffraction_state_capacity: int | None = None

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        _validate_isb_boundary_taper(self.isb_boundary_taper_width)
        if self.scattering_samples_per_m2 <= 0.0:
            raise ValueError("scattering_samples_per_m2 must be positive")
        if self.scattering_max_paths_per_pair <= 0:
            raise ValueError("scattering_max_paths_per_pair must be positive")
        if self.scattering_power_threshold < 0.0:
            raise ValueError("scattering_power_threshold must be non-negative")
        components = validated_components(
            self.components, error_message="components must be a subset of {valid}"
        )
        _validate_scatter_chain(
            max_depth=self.scattering_chain_max_depth,
            samples_per_m2=self.scattering_chain_samples_per_m2,
            max_rows=self.scattering_chain_max_rows,
            components=components,
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
        _validate_capacity_config(
            path_capacity_per_pair=self.path_capacity_per_pair,
            diffraction_state_capacity=self.diffraction_state_capacity,
            max_paths=self.max_paths,
        )
        if self.coupled_candidate_limit <= 0:
            raise ValueError("coupled_candidate_limit must be positive")
        if self.coupled_candidate_limit > _MAX_COUPLED_CANDIDATES:
            raise ValueError(
                "coupled_candidate_limit cannot exceed the hard limit of 1000000"
            )
        if self.ad_mode not in _VALID_AD_MODES:
            raise ValueError(
                f"path ad_mode must be one of {sorted(_VALID_AD_MODES)}"
            )
        object.__setattr__(self, "components", components)
