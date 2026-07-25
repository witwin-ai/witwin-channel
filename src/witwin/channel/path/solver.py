from __future__ import annotations

from witwin.core import Scene, SceneSnapshot
from witwin.channel.propagation.enumerated.engine import (
    evaluate_enumerated_paths,
)
from witwin.channel.propagation.enumerated.scattering import (
    append_scattering_evaluated_paths,
)

from .config import Config
from .metadata import _metadata
from .pipeline import (
    _DeferredPathResult,
    _receiver_positions,
    _solve_base as _pipeline_solve_base,
    _transmitter_tensors,
    _validate_runtime,
    solve as _pipeline_solve,
)
from .result import PathResult, from_evaluated_paths
from witwin.channel.scene.compiler import compile as compile_scene
from witwin.channel.scene.endpoints import SolverScene, bind_solver_scene


def _solve_base(scene: SolverScene, config: Config) -> _DeferredPathResult:
    """Delegate one centre-endpoint solve through the shared pipeline."""

    return _pipeline_solve_base(
        scene,
        config,
        validate_runtime=_validate_runtime,
        evaluate_enumerated_paths=evaluate_enumerated_paths,
        append_scattering_evaluated_paths=append_scattering_evaluated_paths,
        metadata=_metadata,
        transmitter_tensors=_transmitter_tensors,
        receiver_positions=_receiver_positions,
        pack_evaluated_paths=from_evaluated_paths,
    )


def solve(
    scene: Scene | SceneSnapshot,
    config: Config,
    *,
    reference_frequency_hz,
) -> PathResult:
    """Solve canonical paths and pack synthetic or explicit antenna arrays."""

    compiled = compile_scene(
        scene, reference_frequency_hz=reference_frequency_hz
    )
    return _pipeline_solve(
        bind_solver_scene(compiled), config, solve_base=_solve_base
    )
