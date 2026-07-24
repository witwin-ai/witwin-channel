from __future__ import annotations

from dataclasses import dataclass

from witwin.channel.components import (
    AD_MODES as _VALID_AD_MODES,
    BOUNCE_COMPONENTS as _BOUNCE_COMPONENTS,
    DEFAULT_COMPONENTS as _DEFAULT_COMPONENTS,
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


def _validate_coherent_combine(coherent: bool, components: frozenset[str]) -> None:
    """ADR-019: coherent combine is only defined for the enumerable delta/UTD
    family that carries a complex field. Refuse it loudly for the stochastic
    transmission/scattering samplers rather than silently combining Monte
    Carlo power samples as phasors.

    ADR-022 SUPERSEDES the former coherent+AD refusal: the coherent accumulate
    now carries native backward/jvp companions
    (``bdpt_accumulate_connection_samples_{backward,jvp}``, spec 6.4), so
    coherent solves are differentiable exactly like the power-domain solves."""
    if not coherent:
        return
    refused = components & {"transmission", "scattering"}
    if refused:
        raise RuntimeError(
            "coherent combine supports only {los, reflection, diffraction} "
            f"components; refused for {sorted(refused)}"
        )


def _validate_ad_readiness(ad_mode: str, components: frozenset[str]) -> None:
    """ADR-022 per-feature AD readiness gate.

    ``ad_mode='none'`` is the bitwise default and never builds a tape. Under
    ``jvp``/``vjp`` every BDPT estimator block is differentiable: the frozen
    material/EM/table/frequency/tx_power parameters ride the plan-07 field and
    ADR-015 scattering companions plus the ADR-022 subpath / endpoint /
    accumulate / finalize companions. ``max_scattering_order > 1`` is allowed:
    its extra diffuse factors ride ``scattering_table_eval_ad`` and the subpath
    ``_ad`` wrappers exactly as order 1 does. Any combination whose native
    companions are not registered fails loudly where they are dispatched (a
    missing-symbol error from ``runtime.required_symbol``); it is never silently
    detached. No component combination is refused here in v1 because every
    differentiable-parameter path has a registered companion; geometry
    gradients through the stochastic sampler are refused at the autograd
    boundary (``ad_geometry='enumerated_blocks_only'``), not here."""

    if ad_mode == "none":
        return
    # Reserved for future features whose companions are not yet registered.
    # None exist in v1, so every accepted ``ad_mode`` reaches its native
    # companions; the readiness contract is enforced loudly at dispatch.


def _validate_scattering_order(max_scattering_order: int) -> None:
    """ADR-021 D4: the diffuse multi-order cap must be a positive bounce count.

    Extracted to a module-level validator mirroring ``_validate_coherent_combine``
    so ``__post_init__`` stays within its maintenance-complexity budget.
    """
    if max_scattering_order < 1:
        raise ValueError("max_scattering_order must be >= 1")


@dataclass(frozen=True, slots=True)
class Config:
    samples: int = 4096
    seed: int = 0
    max_depth: int = 3
    max_light_depth: int | None = None
    max_diffraction_order: int = 1
    # ADR-021 D4: maximum number of diffuse-scatter events a single BDPT light
    # subpath may undergo. DEFAULT 1 is today's behavior BIT-IDENTICALLY: a
    # scattered subpath emits its NEE connection and terminates (single-bounce
    # terminal rule). >1 lifts the terminal rule so a scattered subpath
    # continues in its sampled direction and may reflect/transmit/scatter again
    # up to this cap, emitting an NEE row at every scatter vertex (power domain;
    # scattering stays excluded from the ADR-019 coherent combine).
    max_scattering_order: int = 1
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
        _validate_scattering_order(self.max_scattering_order)
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
                "ad_mode must be one of "
                f"{sorted(_VALID_AD_MODES)} (ADR-022 lifted the BDPT AD "
                "refusal to fixed-topology jvp/vjp)"
            )
        _validate_coherent_combine(self.coherent, components)
        _validate_ad_readiness(self.ad_mode, components)
        if self.workspace_limit_bytes is not None and self.workspace_limit_bytes < 0:
            raise ValueError("workspace_limit_bytes must be non-negative")

        object.__setattr__(self, "max_light_depth", max_light_depth)
        object.__setattr__(self, "components", components)
