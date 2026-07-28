"""Public deterministic solver entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING
from witwin.channel.scene.compiler import compile as compile_scene
from witwin.channel.scene.endpoints import bind_solver_scene

from witwin.channel.propagation.enumerated.engine import (
    evaluate_enumerated_paths,
)
from witwin.channel.propagation.enumerated.scattering import (
    append_scattering_evaluated_paths,
)
from witwin.channel.propagation.geometry.endpoints import (
    apply_receiver_layout,
    receiver_positions_and_layout,
)
from witwin.channel.scene.compiler import _frequency_scalar

from .config import Config
from .pipeline import _metadata, solve as _solve_pipeline
from .result import Result

if TYPE_CHECKING:
    from witwin.core import Scene, SceneSnapshot


def solve(
    scene: Scene | SceneSnapshot,
    config: Config,
    *,
    reference_frequency_hz,
) -> Result:
    """Run the deterministic propagation pipeline."""

    compiled = compile_scene(
        scene, reference_frequency_hz=reference_frequency_hz
    )
    return _solve_pipeline(bind_solver_scene(compiled), config)

__all__ = [
    "_frequency_scalar",
    "_metadata",
    "apply_receiver_layout",
    "append_scattering_evaluated_paths",
    "evaluate_enumerated_paths",
    "receiver_positions_and_layout",
    "solve",
]
