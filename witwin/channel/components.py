# Copyright Xingyu Chen.
# Implements components.

from __future__ import annotations

from collections.abc import Iterable


VALID_COMPONENTS = frozenset(
    {"los", "reflection", "diffraction", "transmission", "scattering"}
)
DEFAULT_COMPONENTS = frozenset({"los", "reflection", "diffraction"})
BOUNCE_COMPONENTS = frozenset(
    {"reflection", "diffraction", "transmission", "scattering"}
)
NO_AD_MODES = frozenset({"none"})
# Fixed-topology material/frequency AD for the deterministic and path solvers
# (plan 07 AD-1). Monte Carlo solvers keep NO_AD_MODES until their AD phases.
AD_MODES = frozenset({"none", "jvp", "vjp"})
# Components whose chain length is bounded by the solver's max_depth (public
# cap of 5): a transmission chain counts wall penetrations exactly like a
# reflection chain counts bounces.
DEPTH_CAPPED_COMPONENTS = frozenset({"reflection", "transmission"})
# ADR-011 hard work/safety ceiling on a coupled reflection-diffraction
# candidate budget.
MAX_COUPLED_CANDIDATES = 1_000_000
# ADR-021 D1 chain-depth cap: each specular leg is bounded by the native
# kMaxAdDepth = 8, so the public cap on d1 + d2 is 2 * 8 = 16.
MAX_SCATTER_CHAIN_DEPTH = 16


def validated_components(
    components: Iterable[str], *, error_message: str
) -> frozenset[str]:
    normalized = frozenset(components)
    if not normalized or not normalized.issubset(VALID_COMPONENTS):
        raise ValueError(error_message.format(valid=sorted(VALID_COMPONENTS)))
    return normalized


def component_availability_status(
    components: Iterable[str],
    *,
    reflection_available: bool,
    diffraction_available: bool,
    reflection_error: str,
    diffraction_error: str,
    depth_available: bool = True,
    reflection_depth_error: str = "",
    diffraction_depth_error: str = "",
) -> dict[str, str]:
    """Report the status of every component against what this solve can run.

    Every solver states its own error strings because the same missing native
    capability means something different per solver. ``depth_available`` is the
    solver's own bounce-budget check: the path solver is the only caller that
    refuses to call reflection or diffraction enabled without a depth budget,
    and it uses one condition with two messages. Reflection is decided
    completely before diffraction, and availability is checked before depth
    within each component, so a caller that fails two requirements at once
    always sees the first one in that fixed order.
    """

    requested = frozenset(components)
    status = {
        "los": "enabled" if "los" in requested else "not_requested",
        "reflection": "not_requested",
        "diffraction": "not_requested",
        "transmission": (
            "enabled" if "transmission" in requested else "not_requested"
        ),
        "scattering": "enabled" if "scattering" in requested else "not_requested",
    }
    if "reflection" in requested:
        if not reflection_available:
            raise RuntimeError(reflection_error)
        if not depth_available:
            raise RuntimeError(reflection_depth_error)
        status["reflection"] = "enabled"
    if "diffraction" in requested:
        if not diffraction_available:
            raise RuntimeError(diffraction_error)
        if not depth_available:
            raise RuntimeError(diffraction_depth_error)
        status["diffraction"] = "enabled"
    return status


def apply_exported_path_counts(
    status: dict[str, str],
    components: Iterable[str],
    *,
    transmission_path_count: int,
    scattering_path_count: int,
) -> dict[str, str]:
    """Refine transmission/scattering status with what the solve exported.

    Both components export real paths, so a requested component that produced
    no path (every wall too thick to penetrate, every surface smooth) keeps the
    truthful ``enabled_no_paths`` status rather than claiming paths exist.
    Mutates ``status`` in place and returns it.
    """

    requested = frozenset(components)
    counts = {
        "transmission": transmission_path_count,
        "scattering": scattering_path_count,
    }
    for name, count in counts.items():
        if name not in requested:
            status[name] = "not_requested"
        else:
            status[name] = "enabled" if count > 0 else "enabled_no_paths"
    return status


def component_max_depth(
    components: Iterable[str],
    *,
    chain_depth: int,
    single_bounce_depth: int,
) -> dict[str, int]:
    """Per-component interaction depth, ``-1`` for a component not requested.

    LoS is depth 0 by definition. Reflection and transmission are chains and
    carry the solver's depth budget. Diffraction and scattering are
    single-bounce contracts, so their cap is a separate argument rather than
    the same one: BDPT passes ``min(1, effective_max_depth)`` because its
    budget can be smaller than a single bounce, while the enumerated solvers
    pass a literal 1. A solver whose component reaches further than this rule
    (path's coupled 1R1D/1D1R family) states that override at its own call
    site, where the reason for it lives.
    """

    requested = frozenset(components)
    return {
        "los": 0 if "los" in requested else -1,
        "reflection": chain_depth if "reflection" in requested else -1,
        "diffraction": single_bounce_depth if "diffraction" in requested else -1,
        "transmission": chain_depth if "transmission" in requested else -1,
        "scattering": single_bounce_depth if "scattering" in requested else -1,
    }


# --- Shared solver configuration rules ------------------------------------
#
# The four public solver ``Config`` dataclasses declare their own fields, but
# several of those fields carry the identical rule and the identical error
# message in more than one solver. The rule lives here once. Each ``Config``
# calls it at exactly the point in its own ``__post_init__`` where the inline
# check used to run, so a config that violates two rules at once still reports
# the same one first.
#
# Only the rules move. The field declarations stay in each ``Config`` body
# because ``ci/public-api-snapshot.json`` freezes the class body verbatim -
# name, annotation text, default expression and declaration order - so a field
# hoisted into a shared base would be a public-contract change.


def validate_max_depth(max_depth: int) -> None:
    """Validate the interaction-depth budget shared by all four solvers."""

    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")


def validate_samples(samples: int) -> None:
    """Validate the Monte Carlo sample count."""

    if samples <= 0:
        raise ValueError("samples must be positive")


def validate_seed(seed: int) -> None:
    """Validate the Monte Carlo RNG seed."""

    if seed < 0:
        raise ValueError("seed must be non-negative")


def validate_workspace_limit_bytes(workspace_limit_bytes: int | None) -> None:
    """Validate the Monte Carlo workspace budget (``None`` means unbounded)."""

    if workspace_limit_bytes is not None and workspace_limit_bytes < 0:
        raise ValueError("workspace_limit_bytes must be non-negative")


def validate_bounce_depth(
    max_depth: int, components: frozenset[str], *, error_message: str
) -> None:
    """Refuse a bounce-requiring component with no bounce budget.

    Each Monte Carlo solver names itself in the message, so the message is the
    caller's and the rule is shared.
    """

    if max_depth < 1 and components & BOUNCE_COMPONENTS:
        raise RuntimeError(error_message)


def validate_isb_boundary_taper(width: float) -> None:
    """Validate the ADR-017 ISB boundary taper width bound."""

    if not (0.0 < width <= 4.0):
        raise ValueError("isb_boundary_taper_width must be in (0, 4]")


def validate_scatter_chain(
    *,
    max_depth: int,
    samples_per_m2: float,
    max_rows: int,
    components: frozenset[str],
) -> None:
    """Validate the ADR-021 D1 enumerated scatter-chain config (shared)."""

    if max_depth < 0:
        raise ValueError("scattering_chain_max_depth must be non-negative")
    if max_depth > MAX_SCATTER_CHAIN_DEPTH:
        raise ValueError(
            "scattering_chain_max_depth cannot exceed 16 (2 * kMaxAdDepth); each "
            "specular leg is bounded by the native kMaxAdDepth = 8"
        )
    if samples_per_m2 <= 0.0:
        raise ValueError("scattering_chain_samples_per_m2 must be positive")
    if max_rows <= 0:
        raise ValueError("scattering_chain_max_rows must be positive")
    if max_depth >= 1 and "scattering" not in components:
        raise RuntimeError(
            "scattering_chain_max_depth >= 1 requires the 'scattering' component "
            "(ADR-021 D1 appends component_id=6 scatter-chain rows)"
        )


def validate_coupled_gate(
    *, coupled_paths: bool, max_depth: int, components: frozenset[str]
) -> None:
    """Validate the coupled reflection-diffraction opt-in gate (ADR-011).

    Split from the candidate-limit rule below because the path solver checks
    the two at different points in its ``__post_init__``.
    """

    if coupled_paths:
        if max_depth < 2:
            raise RuntimeError(
                "coupled reflection-diffraction paths require max_depth >= 2"
            )
        if not {"reflection", "diffraction"}.issubset(components):
            raise RuntimeError(
                "coupled paths require both reflection and diffraction components"
            )


def validate_coupled_candidate_limit(coupled_candidate_limit: int) -> None:
    """Validate the ADR-011 coupled candidate work/safety budget."""

    if coupled_candidate_limit <= 0:
        raise ValueError("coupled_candidate_limit must be positive")
    if coupled_candidate_limit > MAX_COUPLED_CANDIDATES:
        raise ValueError(
            "coupled_candidate_limit cannot exceed the hard limit of 1000000"
        )