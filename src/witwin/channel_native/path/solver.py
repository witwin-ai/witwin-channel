from __future__ import annotations

from witwin.channel_native import Scene
from witwin.channel_native.propagation.enumerated.engine import (
    evaluate_enumerated_paths,
)
from witwin.channel_native.propagation.enumerated.scattering import (
    append_scattering_evaluated_paths,
)

from .config import Config
from .metadata import _metadata
from .pipeline import (
    _receiver_positions,
    _solve_base as _pipeline_solve_base,
    _transmitter_tensors,
    _validate_runtime,
    solve as _pipeline_solve,
)
from .result import PathResult, from_evaluated_paths


def _solve_base(scene: Scene, config: Config) -> PathResult:
    """Delegate one centre-endpoint solve while preserving legacy test seams."""

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


def solve(scene: Scene, config: Config) -> PathResult:
    """Solve canonical paths and pack synthetic or explicit antenna arrays."""

    return _pipeline_solve(scene, config, solve_base=_solve_base)
