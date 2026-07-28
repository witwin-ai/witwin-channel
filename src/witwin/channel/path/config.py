from __future__ import annotations

from dataclasses import dataclass

from witwin.channel.components import (
    AD_MODES as _VALID_AD_MODES,
    DEFAULT_COMPONENTS as _DEFAULT_COMPONENTS,
    DEPTH_CAPPED_COMPONENTS,
    validate_coupled_candidate_limit,
    validate_coupled_gate,
    validate_isb_boundary_taper,
    validate_max_depth,
    validate_scatter_chain,
    validated_components,
)


# Public component set. transmission exports specular wall-penetration paths
# (wave 2); scattering exports single-bounce incoherent Kirchhoff patch paths
# (wave 3). transmission depth is capped like reflection (chains count wall
# penetrations); scattering is single-bounce in v1.
# Default component set is unchanged: the new components are strictly opt-in.
_VALID_MAX_PATHS_SCOPES = frozenset({"per_pair"})


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
    def __post_init__(self) -> None:
        validate_max_depth(self.max_depth)
        validate_isb_boundary_taper(self.isb_boundary_taper_width)
        if self.scattering_samples_per_m2 <= 0.0:
            raise ValueError("scattering_samples_per_m2 must be positive")
        if self.scattering_max_paths_per_pair <= 0:
            raise ValueError("scattering_max_paths_per_pair must be positive")
        if self.scattering_power_threshold < 0.0:
            raise ValueError("scattering_power_threshold must be non-negative")
        components = validated_components(
            self.components, error_message="components must be a subset of {valid}"
        )
        validate_scatter_chain(
            max_depth=self.scattering_chain_max_depth,
            samples_per_m2=self.scattering_chain_samples_per_m2,
            max_rows=self.scattering_chain_max_rows,
            components=components,
        )
        if self.max_depth > 5 and components & DEPTH_CAPPED_COMPONENTS:
            raise RuntimeError("path reflection/transmission support max_depth <= 5")
        validate_coupled_gate(
            coupled_paths=self.coupled_paths,
            max_depth=self.max_depth,
            components=components,
        )
        if self.max_paths is not None and self.max_paths <= 0:
            raise ValueError("max_paths must be positive when set")
        if self.max_paths_scope not in _VALID_MAX_PATHS_SCOPES:
            raise ValueError("path max_paths_scope must be 'per_pair'")
        validate_coupled_candidate_limit(self.coupled_candidate_limit)
        if self.ad_mode not in _VALID_AD_MODES:
            raise ValueError(
                f"path ad_mode must be one of {sorted(_VALID_AD_MODES)}"
            )
        object.__setattr__(self, "components", components)
