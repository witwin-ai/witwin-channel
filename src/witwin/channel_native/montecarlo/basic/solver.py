from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from witwin.channel_native import build_info
from witwin.channel_native.core.antenna import validate_scalar_endpoint_features
from witwin.channel_native.materials.evaluation import (
    _require_frequency_ad_constant_materials,
)

from .config import Config
from .pipeline import (
    _enforce_workspace_budget,
    _face_material_tensors,
    _receiver_count,
    _validate_ad_config,
    solve_pipeline,
)
from .result import Result
from .sampling import make_cuda_generator

if TYPE_CHECKING:
    from witwin.channel_native.scene.models import Scene

__all__ = [
    "_enforce_workspace_budget",
    "_face_material_tensors",
    "_receiver_count",
    "_require_frequency_ad_constant_materials",
    "_validate_ad_config",
    "build_info",
    "make_cuda_generator",
    "solve",
    "torch",
    "validate_scalar_endpoint_features",
]


def solve(scene: Scene, config: Config) -> Result:
    """Run the Monte Carlo Basic solver pipeline."""

    return solve_pipeline(
        scene,
        config,
        build_info_fn=build_info,
        make_cuda_generator_fn=make_cuda_generator,
        validate_scalar_endpoint_features_fn=validate_scalar_endpoint_features,
        require_frequency_ad_constant_materials_fn=(
            _require_frequency_ad_constant_materials
        ),
    )
