from __future__ import annotations

import torch as torch
from witwin.core import Scene, SceneSnapshot

from witwin.channel import build_info
from witwin.channel.montecarlo.bdpt.kernels.maps import (
    bdpt_component_map_buffer as bdpt_component_map_buffer,
    bdpt_store_component_map as bdpt_store_component_map,
)
from witwin.channel.montecarlo.bdpt.kernels.sampling import (
    bdpt_reflection_launch_inputs as bdpt_reflection_launch_inputs,
    bdpt_sample_directions as bdpt_sample_directions,
)
from witwin.channel.scene.compiler import compile as compile_scene
from witwin.channel.scene.antenna import validate_scalar_endpoint_features
from witwin.channel.scene.endpoints import (
    ReceiverGrid,
    _endpoint_views,
    _validate_scalar_endpoint_boundary,
    bind_solver_scene,
)

from .config import Config
from .endpoints import transmitter_tensors as transmitter_tensors
from .pipeline import (
    _BDPTTopologyOptions as _BDPTTopologyOptions,
    _estimate_workspace_bytes as _estimate_workspace_bytes,
    solve as _solve_pipeline,
)
from .result import Result


def solve(
    scene: Scene | SceneSnapshot,
    config: Config,
    *,
    reference_frequency_hz,
) -> Result:
    """Run the native CUDA/OptiX BDPT pipeline."""

    endpoint_views = _endpoint_views(scene)
    if (
        any(isinstance(view, ReceiverGrid) for view in endpoint_views)
        and config.receiver_strategy != "grid_area"
    ):
        raise RuntimeError(
            "receiver_strategy='point_sphere' requires point receivers"
        )
    _validate_scalar_endpoint_boundary(endpoint_views)
    validate_scalar_endpoint_features(
        tuple(view for view in endpoint_views if view.source.role == "tx"),
        tuple(view for view in endpoint_views if view.source.role == "rx"),
        solver="BDPT",
    )
    compiled = compile_scene(
        scene, reference_frequency_hz=reference_frequency_hz
    )
    return _solve_pipeline(
        bind_solver_scene(compiled),
        config,
        build_info_fn=build_info,
        transmitter_tensors_fn=transmitter_tensors,
    )


__all__ = [
    "_BDPTTopologyOptions",
    "_estimate_workspace_bytes",
    "bdpt_component_map_buffer",
    "bdpt_reflection_launch_inputs",
    "bdpt_sample_directions",
    "bdpt_store_component_map",
    "build_info",
    "solve",
    "torch",
    "transmitter_tensors",
]
