from __future__ import annotations

import pytest
import torch

from tests.support.scenes import single_wall_reflection_scene
from witwin.core import PhysicalMaterial
from witwin.channel.scene import compile as compile_scene


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compile_evaluates_core_material_at_reference_frequency() -> None:
    original = single_wall_reflection_scene()
    material_id = int(original.structures[0].material_id)
    material = PhysicalMaterial(
        eps_r=5.3,
        sigma_e=0.12,
        material_id=material_id,
        name="concrete-at-28ghz",
    )
    scene = original.with_material(material_id, material)

    compiled = compile_scene(scene, reference_frequency_hz=28.0e9)

    assert compiled.materials.frequency_hz == 28.0e9
    torch.testing.assert_close(
        compiled.materials.eps_r,
        torch.full_like(compiled.materials.eps_r, 5.3),
    )
    torch.testing.assert_close(
        compiled.materials.sigma_e,
        torch.full_like(compiled.materials.sigma_e, 0.12),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compile_preserves_core_material_tensor_graph() -> None:
    original = single_wall_reflection_scene()
    material_id = int(original.structures[0].material_id)
    eps_r = torch.tensor(4.0, device="cuda", requires_grad=True)
    scene = original.with_material(
        material_id,
        PhysicalMaterial(
            eps_r=eps_r,
            sigma_e=0.02,
            material_id=material_id,
        ),
    )

    compiled = compile_scene(scene, reference_frequency_hz=3.0e9)

    assert compiled.materials.eps_r.requires_grad
    compiled.materials.eps_r.sum().backward()
    torch.testing.assert_close(eps_r.grad, torch.ones_like(eps_r))
