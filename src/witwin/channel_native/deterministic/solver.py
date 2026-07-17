"""Compatibility facade for the deterministic solver pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from witwin.channel_native.propagation.enumerated.engine import (
    evaluate_enumerated_paths,
)
from witwin.channel_native.propagation.enumerated.scattering import (
    append_scattering_evaluated_paths,
)
from witwin.channel_native.propagation.geometry.endpoints import (
    apply_receiver_layout,
    receiver_positions_and_layout,
)
from witwin.channel_native.scene.tensors import _frequency_scalar

from .config import Config
from .pipeline import _metadata, solve as _solve_pipeline
from .result import Result

if TYPE_CHECKING:
    from witwin.channel_native.scene.models import Scene


def solve(scene: Scene, config: Config) -> Result:
    """Run the deterministic propagation pipeline."""

    return _solve_pipeline(scene, config)

__all__ = [
    "_frequency_scalar",
    "_metadata",
    "apply_receiver_layout",
    "append_scattering_evaluated_paths",
    "evaluate_enumerated_paths",
    "receiver_positions_and_layout",
    "solve",
]
