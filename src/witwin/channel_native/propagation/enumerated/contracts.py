from __future__ import annotations

from typing import Protocol


class TopologyConfig(Protocol):
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str]
    max_depth: int
    scattering_samples_per_m2: float
    scattering_power_threshold: float
    scattering_max_paths_per_pair: int
    # ADR-021 D1 enumerated scatter-chain path class. DEFAULT-OFF: 0 disables
    # chain discovery entirely (the pipeline stays byte-identical). When >= 1 it
    # is the cap on d1 + d2, the combined reflection depth of the two specular
    # legs around the single diffuse vertex; each leg is independently bounded by
    # the native kMaxAdDepth = 8, so the public cap is 2 * 8 = 16.
    scattering_chain_max_depth: int
    # Chain-sample vertex density (samples / m^2). Documented lower density than
    # the single-bounce scattering sampler (scattering_samples_per_m2) because a
    # chain vertex is joined against two specular legs (ADR-021 D1).
    scattering_chain_samples_per_m2: float
    # Per-(tx, rx) keep-strongest cap on joined chain rows (ADR-021 D1 budget).
    scattering_chain_max_rows: int
