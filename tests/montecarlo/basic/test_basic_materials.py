from __future__ import annotations

import importlib

import pytest
import torch

from tests.support.scenes import single_wall_reflection_scene
from witwin.channel_native import ITUMaterial, Scene
from witwin.channel_native.core.scene_loader import itu_material_parameters


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_store_material_tensors_evaluate_itu_material_at_scene_frequency() -> None:
    # Plan 07 AD-3: the solver reads per-face materials from the compiled
    # material store (one source for the primal and the AD modes), so the
    # store must carry the ITU law evaluated at the scene frequency.
    solver = importlib.import_module("witwin.channel_native.montecarlo.basic.solver")
    original = single_wall_reflection_scene()
    scene = Scene(
        structures=[
            original.structures[0].with_material(ITUMaterial(name="concrete"))
        ],
        transmitters=original.transmitters,
        receivers=original.receivers,
        frequency=28.0e9,
    )

    tensors = solver._face_material_tensors(
        scene, device=torch.device("cuda"), ad=False
    )
    eps_r, sigma_e = tensors[0], tensors[1]

    expected_eps_r, expected_sigma_e = itu_material_parameters("concrete", 28.0e9)
    face_count = int(scene.structures[0].faces.shape[0])
    assert eps_r.shape == (face_count,)
    torch.testing.assert_close(
        eps_r,
        torch.full((face_count,), float(expected_eps_r), device=eps_r.device),
    )
    torch.testing.assert_close(
        sigma_e,
        torch.full((face_count,), float(expected_sigma_e), device=sigma_e.device),
    )
    assert not eps_r.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_store_material_tensors_keep_the_store_graph_in_ad_mode() -> None:
    solver = importlib.import_module("witwin.channel_native.montecarlo.basic.solver")
    scene = single_wall_reflection_scene()
    leaf = scene.compile().materials.eps_r
    leaf.requires_grad_(True)
    try:
        tensors = solver._face_material_tensors(
            scene, device=torch.device("cuda"), ad=True
        )
        assert tensors[0].requires_grad
        primal = solver._face_material_tensors(
            scene, device=torch.device("cuda"), ad=False
        )
        assert not primal[0].requires_grad
        torch.testing.assert_close(tensors[0].detach(), primal[0], rtol=0.0, atol=0.0)
    finally:
        leaf.requires_grad_(False)
