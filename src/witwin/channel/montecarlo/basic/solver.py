from __future__ import annotations

import torch
from witwin.core import Scene, SceneSnapshot

from witwin.channel import build_info
from witwin.channel.scene.antenna import validate_scalar_endpoint_features
from witwin.channel.materials.evaluation import (
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
from witwin.channel.scene.compiler import compile as compile_scene
from witwin.channel.scene.endpoints import (
    _endpoint_views,
    _validate_scalar_endpoint_boundary,
    bind_solver_scene,
)

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


def solve(
    scene: Scene | SceneSnapshot,
    config: Config,
    *,
    reference_frequency_hz,
) -> Result:
    """Run the Monte Carlo Basic solver pipeline."""

    endpoint_views = _endpoint_views(scene)
    _validate_scalar_endpoint_boundary(endpoint_views)
    validate_scalar_endpoint_features(
        tuple(view for view in endpoint_views if view.source.role == "tx"),
        tuple(view for view in endpoint_views if view.source.role == "rx"),
        solver="Monte Carlo basic",
    )
    compiled = compile_scene(
        scene, reference_frequency_hz=reference_frequency_hz
    )
    return solve_pipeline(
        bind_solver_scene(compiled),
        config,
        build_info_fn=build_info,
        make_cuda_generator_fn=make_cuda_generator,
        validate_scalar_endpoint_features_fn=validate_scalar_endpoint_features,
        require_frequency_ad_constant_materials_fn=(
            _require_frequency_ad_constant_materials
        ),
    )
