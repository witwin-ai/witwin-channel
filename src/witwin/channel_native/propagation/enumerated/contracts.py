from __future__ import annotations

from typing import Protocol


class TopologyConfig(Protocol):
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str]
    max_depth: int
    scattering_samples_per_m2: float
    scattering_power_threshold: float
    scattering_max_paths_per_pair: int
