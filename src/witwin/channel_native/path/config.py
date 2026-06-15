from __future__ import annotations

from dataclasses import dataclass


_VALID_COMPONENTS = frozenset({"los", "reflection", "diffraction"})
_VALID_SORT_KEYS = frozenset({"receiver_transmitter_depth_component"})
_VALID_AD_MODES = frozenset({"none"})


@dataclass(frozen=True, slots=True)
class Config:
    max_depth: int = 1
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str] = _VALID_COMPONENTS
    max_paths: int | None = None
    sort_key: str = "receiver_transmitter_depth_component"
    diagnostics: bool = False
    ad_mode: str = "none"

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        components = frozenset(self.components)
        if not components or not components.issubset(_VALID_COMPONENTS):
            raise ValueError(f"components must be a subset of {sorted(_VALID_COMPONENTS)}")
        if self.max_paths is not None and self.max_paths <= 0:
            raise ValueError("max_paths must be positive when set")
        if self.sort_key not in _VALID_SORT_KEYS:
            raise ValueError(f"sort_key must be one of {sorted(_VALID_SORT_KEYS)}")
        if self.ad_mode not in _VALID_AD_MODES:
            raise ValueError("ad_mode is not supported for path topology export")
        object.__setattr__(self, "components", components)
