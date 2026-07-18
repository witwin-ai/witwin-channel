from __future__ import annotations

from dataclasses import dataclass

from witwin.channel_native.core.components import (
    BOUNCE_COMPONENTS as _BOUNCE_COMPONENTS,
    DEFAULT_COMPONENTS as _DEFAULT_COMPONENTS,
    NO_AD_MODES as _VALID_AD_MODES,
    validated_components,
)


# Public component set. transmission runs straight endpoint chains plus the
# event-selected shooting sampler for mixed chains; scattering is accepted
# plumbing that emits zero maps until its wave lands. Both are surface events
# that require at least one bounce (max_depth >= 1); transmission chains count
# wall penetrations, scattering is single-bounce in v1. component_mask bits:
# 1=los, 2=reflection, 4=diffraction, 8=transmission, 16=scattering.
# Default component set is unchanged: the new components are strictly opt-in.
# Components that require at least one interaction bounce to contribute.
_VALID_MIS = frozenset({"balance", "power_heuristic", "none"})
_VALID_RECEIVER_STRATEGIES = frozenset({"grid_area", "point_sphere"})
_VALID_ACCUMULATION_STRATEGIES = frozenset({"auto", "atomic", "staged", "compact"})


def _validate_coherent_combine(
    coherent: bool, components: frozenset[str], ad_mode: str
) -> None:
    """ADR-019: coherent combine is only defined for the enumerable delta/UTD
    family that carries a complex field. Refuse it loudly for the stochastic
    transmission/scattering samplers rather than silently combining Monte
    Carlo power samples as phasors. The AD refusal mirrors ADR-017: the
    coherent path refuses AD until its native companions exist (subsumed by
    the release-wide ad_mode gate, kept explicit for the ADR-019 record)."""
    if not coherent:
        return
    unsupported = components & {"transmission", "scattering"}
    if unsupported:
        raise RuntimeError(
            "coherent combine supports only {los, reflection, diffraction} "
            f"components; refused for {sorted(unsupported)}"
        )
    if ad_mode != "none":
        raise RuntimeError("coherent combine does not support ad_mode != 'none'")


@dataclass(frozen=True, slots=True)
class Config:
    samples: int = 4096
    seed: int = 0
    max_depth: int = 3
    max_light_depth: int | None = None
    max_diffraction_order: int = 1
    coupled_paths: bool = False
    coupled_candidate_limit: int = 1_000_000
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str] = (
        _DEFAULT_COMPONENTS
    )
    # Coherent combine (ADR-019). DEFAULT-OFF opt-in switch. OFF (the default)
    # keeps today's power-domain incoherent accumulation BIT-IDENTICAL (enforced
    # by a bitwise regression test). ON sums the complex projected field
    # coefficient of the enumerated delta/UTD discrete connections per
    # (tx, rx, component) and finalizes |sum|^2, so paths within a component
    # interfere coherently (matching the deterministic per-component coherent
    # power). Only the enumerable delta/UTD family carries a coherent field, so
    # coherent is scoped to components subset of {los, reflection, diffraction}
    # (coupled folds into diffraction); BDPT's stochastic transmission/scattering
    # samplers have no coherent field and are refused under coherent.
    coherent: bool = False
    mis: str = "power_heuristic"
    power_heuristic_beta: float = 2.0
    receiver_strategy: str = "grid_area"
    accumulation_strategy: str = "auto"
    sample_streams: int = 1
    diagnostics: bool = False
    export_paths: bool = False
    max_exported_paths: int | None = None
    ad_mode: str = "none"
    workspace_limit_bytes: int | None = 1 << 30

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        max_light_depth = (
            self.max_depth if self.max_light_depth is None else self.max_light_depth
        )
        if max_light_depth < 0:
            raise ValueError("max_light_depth must be non-negative")
        if self.max_diffraction_order not in {0, 1}:
            raise ValueError("max_diffraction_order must be 0 or 1")
        components = validated_components(
            self.components,
            error_message="components must be a non-empty subset of {valid}",
        )
        if self.max_depth < 1 and components & _BOUNCE_COMPONENTS:
            raise RuntimeError("BDPT scattering requires max_depth >= 1")
        if "diffraction" in components and self.max_diffraction_order == 0:
            raise RuntimeError("diffraction requires max_diffraction_order > 0")
        if self.coupled_paths:
            if self.max_depth < 2:
                raise RuntimeError("coupled paths require max_depth >= 2")
            if not {"reflection", "diffraction"}.issubset(components):
                raise RuntimeError(
                    "coupled paths require reflection and diffraction components"
                )
        if (
            self.coupled_candidate_limit <= 0
            or self.coupled_candidate_limit > 1_000_000
        ):
            raise ValueError("coupled_candidate_limit must be in [1, 1000000]")
        if self.mis not in _VALID_MIS:
            raise ValueError(f"mis must be one of {sorted(_VALID_MIS)}")
        if self.power_heuristic_beta <= 0.0:
            raise ValueError("power_heuristic_beta must be positive")
        if self.receiver_strategy not in _VALID_RECEIVER_STRATEGIES:
            raise ValueError(
                f"receiver_strategy must be one of {sorted(_VALID_RECEIVER_STRATEGIES)}"
            )
        if self.accumulation_strategy not in _VALID_ACCUMULATION_STRATEGIES:
            raise ValueError(
                f"accumulation_strategy must be one of {sorted(_VALID_ACCUMULATION_STRATEGIES)}"
            )
        if self.sample_streams <= 0:
            raise ValueError("sample_streams must be positive")
        if self.max_exported_paths is not None and self.max_exported_paths < 0:
            raise ValueError("max_exported_paths must be non-negative")
        if self.ad_mode not in _VALID_AD_MODES:
            raise ValueError(
                "montecarlo_bdpt supports_ad=False in the first replacement release; "
                "ad_mode must be 'none'"
            )
        _validate_coherent_combine(self.coherent, components, self.ad_mode)
        if self.workspace_limit_bytes is not None and self.workspace_limit_bytes < 0:
            raise ValueError("workspace_limit_bytes must be non-negative")

        object.__setattr__(self, "max_light_depth", max_light_depth)
        object.__setattr__(self, "components", components)
