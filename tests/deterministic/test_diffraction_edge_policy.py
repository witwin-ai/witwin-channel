# Copyright Xingyu Chen.
# Tests diffraction edge policy.

import pytest
import torch

from tests.support.scenes import wedge_diffraction_scene
from witwin.channel.deployment import build_info
from witwin.channel.deterministic import Config
from witwin.channel.propagation.enumerated import evaluate_enumerated_paths
from witwin.channel.scene import compile as compile_scene
from witwin.channel.scene.endpoints import bind_solver_scene


def test_diffraction_topology_uses_selected_edge_ids():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diffraction")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    compiled = compile_scene(
        wedge_diffraction_scene(), reference_frequency_hz=3.0e9
    )
    scene = bind_solver_scene(compiled)
    paths, _ = evaluate_enumerated_paths(scene, Config(components={"diffraction"}))
    topology = paths.topology

    assert topology.valid.numel() >= 1
    assert torch.all(topology.component_id == 2)
    assert torch.all(topology.edge_id >= 0)
    assert torch.any(paths.geometry.interaction_position.abs() > 0.0)
    assert int(torch.unique(topology.edge_id).numel()) <= int(
        compiled.geometry.edges.shape[0]
    )