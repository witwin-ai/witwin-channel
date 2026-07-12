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
) -> dict[str, str]:
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
        status["reflection"] = "enabled"
    if "diffraction" in requested:
        if not diffraction_available:
            raise RuntimeError(diffraction_error)
        status["diffraction"] = "enabled"
    return status
