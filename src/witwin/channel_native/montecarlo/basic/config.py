from __future__ import annotations

from dataclasses import dataclass

from witwin.channel_native.core.kernels.metadata import ACCUMULATION_STRATEGIES


_VALID_COMPONENTS = frozenset({"los", "reflection", "diffraction"})
_VALID_AD_MODES = frozenset({"none", "vjp", "jvp"})
_VALID_REFLECTION_ACCUMULATION_STRATEGIES = frozenset({"auto", "atomic", "staged", "compact"})


@dataclass(frozen=True, slots=True)
class Config:
    samples: int = 4096
    max_depth: int = 1
    seed: int = 0
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str] = _VALID_COMPONENTS
    accumulation_strategy: str = "atomic_add"
    diagnostics: bool = False
    require_reflection: bool = False
    require_diffraction: bool = False
    reflection_accumulation_strategy: str = "auto"
    reflection_compact_min_samples: int = 262_144
    reflection_staged_min_samples_per_cell: int = 64
    ad_mode: str = "none"
    fixed_topology: bool = True
    requires_fixed_seed: bool = True

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        components = frozenset(self.components)
        if not components or not components.issubset(_VALID_COMPONENTS):
            raise ValueError(f"components must be a subset of {sorted(_VALID_COMPONENTS)}")
        if self.accumulation_strategy not in ACCUMULATION_STRATEGIES - {"none"}:
            raise ValueError("accumulation_strategy is not supported for MC basic")
        if self.reflection_accumulation_strategy not in _VALID_REFLECTION_ACCUMULATION_STRATEGIES:
            raise ValueError("reflection_accumulation_strategy is not supported")
        if self.reflection_compact_min_samples < 0:
            raise ValueError("reflection_compact_min_samples must be non-negative")
        if self.reflection_staged_min_samples_per_cell < 0:
            raise ValueError("reflection_staged_min_samples_per_cell must be non-negative")
        if self.ad_mode not in _VALID_AD_MODES:
            raise ValueError(f"ad_mode must be one of {sorted(_VALID_AD_MODES)}")
        object.__setattr__(self, "components", components)
