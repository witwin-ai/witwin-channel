# Copyright Xingyu Chen.
# Implements components.

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import math
import numbers
import operator


VALID_COMPONENTS = frozenset(
    {"los", "reflection", "diffraction", "transmission", "scattering"}
)
DEFAULT_COMPONENTS = frozenset({"los", "reflection", "diffraction"})
BOUNCE_COMPONENTS = frozenset(
    {"reflection", "diffraction", "transmission", "scattering"}
)
NO_AD_MODES = frozenset({"none"})
# Fixed-topology material/frequency AD for the deterministic and path solvers
# (material and frequency derivatives). Monte Carlo solvers keep NO_AD_MODES until their AD phases.
AD_MODES = frozenset({"none", "jvp", "vjp"})
# Components whose chain length is bounded by the solver's max_depth (public
# cap of 5): a transmission chain counts wall penetrations exactly like a
# reflection chain counts bounces.
DEPTH_CAPPED_COMPONENTS = frozenset({"reflection", "transmission"})
# coupled reflection and diffraction hard work/safety ceiling on a coupled reflection-diffraction
# candidate budget.
MAX_COUPLED_CANDIDATES = 1_000_000
# coherent scattering chain-depth cap: each specular leg is bounded by the native
# kMaxAdDepth = 8, so the public cap on d1 + d2 is 2 * 8 = 16.
MAX_SCATTER_CHAIN_DEPTH = 16


def validated_components(components: Iterable[str], *, error_message: str) -> frozenset[str]:
    if isinstance(components, (str, bytes)) or not isinstance(
        components, (set, frozenset, tuple, list)
    ):
        raise TypeError(
            "components must be a set, frozenset, tuple or list of component names "
            f"from {sorted(VALID_COMPONENTS)}, got {type(components).__name__}"
        )
    normalized = frozenset(components)
    if not normalized or not normalized.issubset(VALID_COMPONENTS):
        raise ValueError(
            error_message.format(valid=sorted(VALID_COMPONENTS))
            + f", got {sorted(map(str, normalized))}"
        )
    return normalized


def component_availability_status(
    components: Iterable[str], *, reflection_available: bool, diffraction_available: bool,
    reflection_error: str, diffraction_error: str, depth_available: bool = True,
    reflection_depth_error: str = "", diffraction_depth_error: str = "",
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
    status: dict[str, str], components: Iterable[str], *, transmission_path_count: int,
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
    components: Iterable[str], *, chain_depth: int, single_bounce_depth: int,
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
# Each ``Config`` calls it at the matching point in ``__post_init__`` so
# validation error ordering stays consistent across solvers.
#
# Only the rules move. The field declarations stay in each ``Config`` body
# because ``ci/public-api-snapshot.json`` freezes the class body verbatim -
# name, annotation text, default expression and declaration order - so a field
# hoisted into a shared base would be a public-contract change.


# Error classes are the same in all four solvers: a field of the wrong type
# raises TypeError; a value outside its legal range or vocabulary, and a field
# combination the solver does not accept, raise ValueError. Every message names
# the field and its legal range and repeats the rejected value.

# Largest chain depth the enumerated reflection/transmission engines accept.
MAX_ENUMERATED_CHAIN_DEPTH = 5


def _require_bool(where: str, value: object) -> object:
    if type(value) is not bool:
        raise TypeError(f"{where} must be True or False, got {value!r}")
    return value


def _require_int(where: str, value: object, optional: bool) -> object:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(
            f"{where} must be an int{' or None' if optional else ''}, "
            f"got {type(value).__name__} {value!r}"
        )
    return operator.index(value)


def _require_real(where: str, value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{where} must be a real number, got {type(value).__name__} {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{where} must be finite, got {value!r}")
    return value


def _require_str(where: str, value: object) -> object:
    if not isinstance(value, str):
        raise TypeError(f"{where} must be a str, got {type(value).__name__} {value!r}")
    return value


def validate_config_fields(config: object) -> None:
    """Check every scalar ``Config`` field against its declared type before any range check.

    ``int`` fields reject ``bool``, ``float`` and strings; ``float`` fields accept
    real numbers (not ``bool``) and must be finite; ``bool`` and ``str`` fields must
    have exactly that type. Integer-like values (for example a NumPy integer) are
    stored as Python ``int``.
    """

    solver_config = type(config).__module__.removeprefix("witwin.channel.")
    for item in dataclasses.fields(config):  # type: ignore[arg-type]
        kinds = {part.strip() for part in str(item.type).split("|")}
        value = getattr(config, item.name)
        if item.name.startswith("_") or (value is None and "None" in kinds):
            continue
        where = f"{solver_config} Config.{item.name}"
        checked = value
        if kinds <= {"bool"}:
            checked = _require_bool(where, value)
        elif kinds <= {"int", "None"}:
            checked = _require_int(where, value, "None" in kinds)
        elif kinds <= {"float", "None"}:
            checked = _require_real(where, value)
        elif kinds <= {"str"}:
            checked = _require_str(where, value)
        if checked is not value:
            object.__setattr__(config, item.name, checked)


def require_config_type(config: object, expected: type, solver: str) -> None:
    """Refuse a ``Config`` of another solver (or any other object) before any work."""

    if not isinstance(config, expected):
        raise TypeError(
            f"{solver}.solve() requires config to be "
            f"{expected.__module__}.{expected.__qualname__}, got "
            f"{type(config).__module__}.{type(config).__qualname__}"
        )


def validate_vocabulary(name: str, value: str, allowed: Iterable[str]) -> None:
    """Refuse a string field outside its vocabulary, naming the legal values."""

    legal = sorted(allowed)
    if value not in legal:
        raise ValueError(f"{name} must be one of {legal}, got {value!r}")


# Reflection candidate sources of the enumerated solvers. "auto" and "exhaustive" are complete
# for every requested depth; "rays" (ray discovery) is never complete.
DISCOVERY_MODES = ("auto", "exhaustive", "rays")


def discovery_metadata(requested: str, record: dict[str, object] | None) -> dict[str, object]:
    """The ``discovery`` result block: requested source, source per depth, and completeness."""

    if record is None:
        return {"requested": requested, "sources": {}, "complete": True}
    return dict(record)


def validate_max_depth(max_depth: int) -> None:
    """Validate the interaction-depth budget shared by all four solvers."""

    if max_depth < 0:
        raise ValueError(f"max_depth must be a non-negative int, got {max_depth}")


def validate_chain_depth_cap(solver: str, max_depth: int, components: frozenset[str]) -> None:
    """Refuse a reflection/transmission chain deeper than the enumerated engines accept."""

    if max_depth > MAX_ENUMERATED_CHAIN_DEPTH and components & DEPTH_CAPPED_COMPONENTS:
        raise ValueError(
            f"{solver} max_depth must be in [0, {MAX_ENUMERATED_CHAIN_DEPTH}] when "
            f"components include {sorted(components & DEPTH_CAPPED_COMPONENTS)}, "
            f"got {max_depth}"
        )


def validate_samples(samples: int) -> None:
    """Validate the Monte Carlo sample count."""

    if samples <= 0:
        raise ValueError(f"samples must be a positive int, got {samples}")


def validate_seed(seed: int) -> None:
    """Validate the Monte Carlo RNG seed."""

    if seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed}")


def validate_workspace_limit_bytes(workspace_limit_bytes: int | None) -> None:
    """Validate the Monte Carlo workspace budget (``None`` means unbounded)."""

    if workspace_limit_bytes is not None and workspace_limit_bytes < 0:
        raise ValueError(
            "workspace_limit_bytes must be None or a non-negative int, got "
            f"{workspace_limit_bytes}"
        )


def validate_bounce_depth(
    max_depth: int, components: frozenset[str], *, error_message: str,
) -> None:
    """Refuse a bounce-requiring component with no bounce budget.

    Each Monte Carlo solver names itself in the message, so the message is the
    caller's and the rule is shared.
    """

    if max_depth < 1 and components & BOUNCE_COMPONENTS:
        raise ValueError(error_message)


def validate_isb_boundary_taper(width: float) -> None:
    """Validate the ISB boundary taper width bound."""

    if not (0.0 < width <= 4.0):
        raise ValueError(f"isb_boundary_taper_width must be in (0, 4], got {width!r}")


def validate_scatter_chain(
    *, max_depth: int, samples_per_m2: float, max_rows: int, components: frozenset[str],
) -> None:
    """Validate the enumerated scatter-chain config (shared)."""

    if not 0 <= max_depth <= MAX_SCATTER_CHAIN_DEPTH:
        raise ValueError(
            f"scattering_chain_max_depth must be in [0, {MAX_SCATTER_CHAIN_DEPTH}] "
            f"(each specular leg holds at most 8 bounces), got {max_depth}"
        )
    if not samples_per_m2 > 0.0:
        raise ValueError(
            f"scattering_chain_samples_per_m2 must be positive, got {samples_per_m2!r}"
        )
    if max_rows <= 0:
        raise ValueError(f"scattering_chain_max_rows must be a positive int, got {max_rows}")
    if max_depth >= 1 and "scattering" not in components:
        raise ValueError(
            f"scattering_chain_max_depth={max_depth} requires the 'scattering' "
            "component; set scattering_chain_max_depth=0 or add 'scattering'"
        )


def validate_coupled_gate(
    *, coupled_paths: bool, max_depth: int, components: frozenset[str],
) -> None:
    """Validate the coupled reflection-diffraction opt-in gate.

    Split from the candidate-limit rule below because the path solver checks
    the two at different points in its ``__post_init__``.
    """

    if coupled_paths:
        if max_depth < 2:
            raise ValueError(
                f"coupled_paths=True requires max_depth >= 2, got {max_depth}"
            )
        if not {"reflection", "diffraction"}.issubset(components):
            raise ValueError(
                "coupled_paths=True requires both 'reflection' and 'diffraction' in "
                f"components, got {sorted(components)}"
            )


def validate_coupled_candidate_limit(coupled_candidate_limit: int) -> None:
    """Validate the coupled candidate work/safety budget."""

    if not 1 <= coupled_candidate_limit <= MAX_COUPLED_CANDIDATES:
        raise ValueError(
            f"coupled_candidate_limit must be in [1, {MAX_COUPLED_CANDIDATES}], "
            f"got {coupled_candidate_limit}"
        )


def validate_scattering_quadrature(
    *, samples_per_m2: float, max_paths_per_pair: int, power_threshold: float,
) -> None:
    """Validate the rough-surface scattering quadrature fields of the enumerated solvers."""

    if not samples_per_m2 > 0.0:
        raise ValueError(
            f"scattering_samples_per_m2 must be positive, got {samples_per_m2!r}"
        )
    if max_paths_per_pair <= 0:
        raise ValueError(
            f"scattering_max_paths_per_pair must be a positive int, got {max_paths_per_pair}"
        )
    if not power_threshold >= 0.0:
        raise ValueError(
            f"scattering_power_threshold must be non-negative, got {power_threshold!r}"
        )


def validate_max_paths(max_paths: int | None, name: str = "max_paths") -> None:
    if max_paths is not None and max_paths <= 0:
        raise ValueError(f"{name} must be None or a positive int, got {max_paths}")
