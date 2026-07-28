from __future__ import annotations

from dataclasses import dataclass

from witwin.channel.components import (
    AD_MODES as _VALID_AD_MODES,
    DEFAULT_COMPONENTS as _DEFAULT_COMPONENTS,
    validate_bounce_depth,
    validate_max_depth,
    validate_samples,
    validate_seed,
    validate_workspace_limit_bytes,
    validated_components,
)


# Public component set. transmission traces straight penetration chains
# through up to max_depth walls (grid radiomaps only); scattering is accepted
# plumbing that emits zero maps until its wave lands. Both are surface events
# that require at least one bounce (max_depth >= 1).
# Default component set is unchanged: the new components are strictly opt-in.
@dataclass(frozen=True, slots=True)
class Config:
    samples: int = 4096
    max_depth: int = 1
    seed: int = 0
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str] = _DEFAULT_COMPONENTS
    diagnostics: bool = False
    ad_mode: str = "none"
    workspace_limit_bytes: int | None = 1 << 30

    def __post_init__(self) -> None:
        validate_samples(self.samples)
        validate_max_depth(self.max_depth)
        validate_seed(self.seed)
        components = validated_components(
            self.components, error_message="components must be a subset of {valid}"
        )
        validate_bounce_depth(
            self.max_depth,
            components,
            error_message="MC basic scattering requires max_depth >= 1",
        )
        if self.ad_mode not in _VALID_AD_MODES:
            raise ValueError(
                "montecarlo_basic ad_mode must be one of "
                f"{sorted(_VALID_AD_MODES)}"
            )
        validate_workspace_limit_bytes(self.workspace_limit_bytes)
        object.__setattr__(self, "components", components)
