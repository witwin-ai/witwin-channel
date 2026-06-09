from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Result:
    """Result placeholder for the reserved path solver."""
