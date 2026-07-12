from __future__ import annotations

from dataclasses import dataclass


# Public component set. transmission and scattering are accepted plumbing in v1:
# they validate and flow through the result contract but export zero paths until
# the physics lands in later waves. transmission depth is capped like reflection
# (chains count wall penetrations); scattering is single-bounce in v1.
_VALID_COMPONENTS = frozenset(
    {"los", "reflection", "diffraction", "transmission", "scattering"}
)
# Default component set is unchanged: the new components are strictly opt-in.
_DEFAULT_COMPONENTS = frozenset({"los", "reflection", "diffraction"})
# Components whose chain length is bounded by max_depth (public cap of 5).
_DEPTH_CAPPED_COMPONENTS = frozenset({"reflection", "transmission"})
_VALID_SORT_KEYS = frozenset({"receiver_transmitter_depth_component"})
_VALID_AD_MODES = frozenset({"none"})
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
    sort_key: str = "receiver_transmitter_depth_component"
    diagnostics: bool = False
    ad_mode: str = "none"
    coupled_paths: bool = False
    coupled_candidate_limit: int = 1_000_000

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        components = frozenset(self.components)
        if not components or not components.issubset(_VALID_COMPONENTS):
            raise ValueError(
                f"components must be a subset of {sorted(_VALID_COMPONENTS)}"
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
        if self.sort_key not in _VALID_SORT_KEYS:
            raise ValueError(f"sort_key must be one of {sorted(_VALID_SORT_KEYS)}")
        if self.ad_mode not in _VALID_AD_MODES:
            raise ValueError(
                "path supports_ad=False in the first replacement release; "
                "ad_mode must be 'none'"
            )
        object.__setattr__(self, "components", components)
