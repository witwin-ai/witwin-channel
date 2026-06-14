import pytest
import torch

from tests.support.scenes import wedge_diffraction_scene
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.deterministic import Config
from witwin.channel_native.deterministic.topology import export_topology


def test_diffraction_topology_uses_selected_edge_ids():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    scene = wedge_diffraction_scene()
    topology = export_topology(scene, Config(components={"diffraction"}))

    assert topology.valid.numel() >= 1
    assert torch.all(topology.component_id == 2)
    assert torch.all(topology.edge_id >= 0)
    assert torch.any(topology.interaction_position.abs() > 0.0)
    assert int(torch.unique(topology.edge_id).numel()) <= scene.diffraction_edge_count()
