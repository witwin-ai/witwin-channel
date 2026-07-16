from __future__ import annotations

import torch as torch

from witwin.channel_native import build_info
from witwin.channel_native.montecarlo.bdpt.kernels.maps import (
    bdpt_component_map_buffer as bdpt_component_map_buffer,
    bdpt_store_component_map as bdpt_store_component_map,
)
from witwin.channel_native.montecarlo.bdpt.kernels.sampling import (
    bdpt_diffraction_state_pack as bdpt_diffraction_state_pack,
    bdpt_diffraction_state_wi as bdpt_diffraction_state_wi,
    bdpt_reflection_launch_inputs as bdpt_reflection_launch_inputs,
    bdpt_sample_directions as bdpt_sample_directions,
    bdpt_selected_edge_indices as bdpt_selected_edge_indices,
)
from witwin.channel_native.scene.models import Scene

from .config import Config
from .connections import transmitter_tensors as transmitter_tensors
from .pipeline import (
    _BDPTTopologyOptions as _BDPTTopologyOptions,
    _diffraction_sample_split as _diffraction_sample_split,
    _estimate_workspace_bytes as _estimate_workspace_bytes,
    solve as _solve_pipeline,
)
from .result import Result


def solve(scene: Scene, config: Config) -> Result:
    """Run the native CUDA/OptiX BDPT pipeline."""

    return _solve_pipeline(
        scene,
        config,
        build_info_fn=build_info,
        transmitter_tensors_fn=transmitter_tensors,
    )


__all__ = [
    "_BDPTTopologyOptions",
    "_diffraction_sample_split",
    "_estimate_workspace_bytes",
    "bdpt_component_map_buffer",
    "bdpt_diffraction_state_pack",
    "bdpt_diffraction_state_wi",
    "bdpt_reflection_launch_inputs",
    "bdpt_sample_directions",
    "bdpt_selected_edge_indices",
    "bdpt_store_component_map",
    "build_info",
    "solve",
    "torch",
    "transmitter_tensors",
]
