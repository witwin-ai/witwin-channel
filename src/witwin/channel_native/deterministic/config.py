from __future__ import annotations

from dataclasses import dataclass


_VALID_COMPONENTS = frozenset({"los", "reflection", "diffraction"})
_VALID_SORT_KEYS = frozenset({"receiver_transmitter_depth_component"})
_VALID_AD_MODES = frozenset({"none"})
_VALID_MAX_PATHS_SCOPES = frozenset({"global", "per_pair"})


@dataclass(frozen=True, slots=True)
class Config:
    max_depth: int = 1
    max_diffraction_order: int = 1
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str] = (
        _VALID_COMPONENTS
    )
    coherent: bool = True
    return_field: bool = True
    export_paths: bool = False
    max_paths: int | None = None
    max_paths_scope: str = "global"
    sort_key: str = "receiver_transmitter_depth_component"
    diagnostics: bool = False
    ad_mode: str = "none"

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.max_diffraction_order < 0:
            raise ValueError("max_diffraction_order must be 0 or 1")
        if self.max_diffraction_order > 1:
            raise RuntimeError("max_diffraction_order above 1 is not supported yet")

        components = frozenset(self.components)
        if not components or not components.issubset(_VALID_COMPONENTS):
            raise ValueError(
                f"components must be a non-empty subset of {sorted(_VALID_COMPONENTS)}"
            )
        if self.max_depth > 5 and "reflection" in components:
            raise RuntimeError(
                "deterministic reflection currently supports max_depth <= 5"
            )
        if self.max_paths is not None and self.max_paths <= 0:
            raise ValueError("max_paths must be positive when set")
        if self.max_paths_scope not in _VALID_MAX_PATHS_SCOPES:
            raise ValueError("max_paths_scope must be 'global' or 'per_pair'")
        if self.sort_key not in _VALID_SORT_KEYS:
            raise ValueError(f"sort_key must be one of {sorted(_VALID_SORT_KEYS)}")
        if self.ad_mode not in _VALID_AD_MODES:
            raise RuntimeError("deterministic fixed-topology AD is not enabled")

        object.__setattr__(self, "components", components)
