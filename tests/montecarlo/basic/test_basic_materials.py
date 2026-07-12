from __future__ import annotations

import importlib

import torch

from tests.support.scenes import single_wall_reflection_scene
from witwin.channel_native import ITUMaterial, Scene
from witwin.channel_native.core.scene_loader import itu_material_parameters


def test_host_material_tensors_evaluate_itu_material_at_scene_frequency(
    monkeypatch,
) -> None:
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
    captured: dict[str, tuple[float, ...]] = {}

    def export(eps_r, sigma_e, mu_r, face_material_id):
        captured["eps_r"] = eps_r
        captured["sigma_e"] = sigma_e
        one = torch.ones((1,), dtype=torch.float32)
        return {
            "eps_r": one,
            "sigma_e": one,
            "mu_r": one,
            "gain": one,
            "valid": torch.ones((1,), dtype=torch.bool),
        }

    monkeypatch.setattr(solver, "bdpt_face_material_tensors_from_host", export)
    solver._host_material_tensors(scene)

    expected_eps_r, expected_sigma_e = itu_material_parameters("concrete", 28.0e9)
    assert captured["eps_r"] == (expected_eps_r,)
    assert captured["sigma_e"] == (expected_sigma_e,)
