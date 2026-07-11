from __future__ import annotations

from dataclasses import dataclass


_VALID_COMPONENTS = frozenset({"los", "reflection", "diffraction"})
_VALID_MIS = frozenset({"balance", "power_heuristic", "none"})
_VALID_RECEIVER_STRATEGIES = frozenset({"grid_area", "point_sphere"})
_VALID_ACCUMULATION_STRATEGIES = frozenset({"auto", "atomic", "staged", "compact"})
_VALID_AD_MODES = frozenset({"none"})


@dataclass(frozen=True, slots=True)
class Config:
    samples: int = 4096
    seed: int = 0
    max_depth: int = 3
    max_light_depth: int | None = None
    max_sensor_depth: int | None = None
    max_diffraction_order: int = 1
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str] = _VALID_COMPONENTS
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
        max_light_depth = self.max_depth if self.max_light_depth is None else self.max_light_depth
        max_sensor_depth = self.max_depth if self.max_sensor_depth is None else self.max_sensor_depth
        if max_light_depth < 0:
            raise ValueError("max_light_depth must be non-negative")
        if max_sensor_depth < 0:
            raise ValueError("max_sensor_depth must be non-negative")
        if self.max_diffraction_order not in {0, 1}:
            raise ValueError("max_diffraction_order must be 0 or 1")
        components = frozenset(self.components)
        if not components or not components.issubset(_VALID_COMPONENTS):
            raise ValueError(f"components must be a non-empty subset of {sorted(_VALID_COMPONENTS)}")
        if self.max_depth < 1 and components.intersection({"reflection", "diffraction"}):
            raise RuntimeError("BDPT scattering requires max_depth >= 1")
        if "diffraction" in components and self.max_diffraction_order == 0:
            raise RuntimeError("diffraction requires max_diffraction_order > 0")
        if (
            "diffraction" in components
            and self.mis == "none"
            and self.samples * self.sample_streams >= 2
        ):
            raise RuntimeError(
                "mis='none' double counts the direct+keller diffraction strategies; "
                "use mis='balance' or 'power_heuristic'"
            )
        if self.mis not in _VALID_MIS:
            raise ValueError(f"mis must be one of {sorted(_VALID_MIS)}")
        if self.power_heuristic_beta <= 0.0:
            raise ValueError("power_heuristic_beta must be positive")
        if self.receiver_strategy not in _VALID_RECEIVER_STRATEGIES:
            raise ValueError(f"receiver_strategy must be one of {sorted(_VALID_RECEIVER_STRATEGIES)}")
        if self.accumulation_strategy not in _VALID_ACCUMULATION_STRATEGIES:
            raise ValueError(f"accumulation_strategy must be one of {sorted(_VALID_ACCUMULATION_STRATEGIES)}")
        if self.sample_streams <= 0:
            raise ValueError("sample_streams must be positive")
        if self.max_exported_paths is not None and self.max_exported_paths < 0:
            raise ValueError("max_exported_paths must be non-negative")
        if self.ad_mode not in _VALID_AD_MODES:
            raise ValueError("BDPT ad_mode must be 'none'")
        if self.workspace_limit_bytes is not None and self.workspace_limit_bytes < 0:
            raise ValueError("workspace_limit_bytes must be non-negative")

        object.__setattr__(self, "max_light_depth", max_light_depth)
        object.__setattr__(self, "max_sensor_depth", max_sensor_depth)
        object.__setattr__(self, "components", components)
