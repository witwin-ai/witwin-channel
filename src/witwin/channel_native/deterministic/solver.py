"""Compatibility facade for the deterministic solver pipeline."""

from witwin.channel_native.propagation.enumerated.engine import (
    evaluate_enumerated_paths,
)
from witwin.channel_native.propagation.enumerated.scattering import (
    append_scattering_evaluated_paths,
)

from .pipeline import _metadata, solve

__all__ = [
    "_metadata",
    "append_scattering_evaluated_paths",
    "evaluate_enumerated_paths",
    "solve",
]
