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


# Public component set. transmission carries specular wall-penetration paths
# (wave 2); scattering carries single-bounce Kirchhoff rough-surface paths
# (wave 3). transmission depth is capped like reflection (chains count wall
# penetrations); scattering is single-bounce in v1.
# Default component set is unchanged: the new components are strictly opt-in.
_VALID_SORT_KEYS = frozenset({"receiver_transmitter_depth_component"})
_VALID_MAX_PATHS_SCOPES = frozenset({"global", "per_pair"})


def _validate_scattering_coherent(
    *, scattering_coherent: bool, components: frozenset[str]
) -> None:
    """Validate the ADR-021 D3 coherent-scattering combine precondition."""

    if scattering_coherent and "scattering" not in components:
        # ADR-021 D3: the coherent combine only applies to scattering rows.
        # The scene-level requirement (realization-coherent phase screens,
        # not ensemble surfaces) is enforced at solve time where the scene
        # is known; here we reject the config-level precondition loudly.
        raise RuntimeError(
            "scattering_coherent=True requires the 'scattering' component "
            "(ADR-021 D3 combines scattering rows coherently and has no "
            "effect on any other component)"
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
    # Coherent scattering combine (ADR-021 D3). DEFAULT-OFF opt-in. OFF keeps
    # the scattering slot an incoherent POWER sum bit-identical to today; ON
    # sums the complex path_field of scattering rows per (tx, rx) and finalizes
    # |sum|^2 (the ADR-019 per-component phasor precedent). It is physical only
    # for realization-coherent phase-screen rows, which carry a true complex
    # field; ensemble rows are zero-phase power rows, so the pipeline refuses an
    # ensemble-only solve loudly. Requires the 'scattering' component.
    scattering_coherent: bool = False
    # Enumerated scatter-chain path class (ADR-021 D1). DEFAULT-OFF opt-in:
    # scattering_chain_max_depth = 0 disables chain discovery so the pipeline is
    # byte-identical to today. When >= 1 it is the cap on d1 + d2, the combined
    # specular reflection depth of the two legs around the single diffuse vertex
    # (TX --C1(d1)--> v_s --C2(d2)--> RX with 1 <= d1 + d2 <= cap). Each leg is
    # independently bounded by the native kMaxAdDepth = 8, so the public cap is
    # 2 * 8 = 16. The chain-sample vertices are drawn at a documented lower
    # density (scattering_chain_samples_per_m2) than the single-bounce sampler,
    # and only the strongest scattering_chain_max_rows joined rows per (tx, rx)
    # survive. Requires the 'scattering' component.
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
        if self.max_diffraction_order < 0:
            raise ValueError("max_diffraction_order must be 0 or 1")
        if self.max_diffraction_order > 1:
            raise RuntimeError("max_diffraction_order above 1 is not supported yet")

        components = validated_components(
            self.components,
            error_message="components must be a non-empty subset of {valid}",
        )
        _validate_scattering_coherent(
            scattering_coherent=self.scattering_coherent, components=components
        )
        validate_scatter_chain(
            max_depth=self.scattering_chain_max_depth,
            samples_per_m2=self.scattering_chain_samples_per_m2,
            max_rows=self.scattering_chain_max_rows,
            components=components,
        )
        if self.max_depth > 5 and components & DEPTH_CAPPED_COMPONENTS:
            raise RuntimeError(
                "deterministic reflection/transmission currently support max_depth <= 5"
            )
        validate_coupled_gate(
            coupled_paths=self.coupled_paths,
            max_depth=self.max_depth,
            components=components,
        )
        validate_coupled_candidate_limit(self.coupled_candidate_limit)
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
