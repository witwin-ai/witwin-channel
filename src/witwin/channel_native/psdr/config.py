from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration placeholder for the reserved PSDR solver."""
