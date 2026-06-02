from __future__ import annotations

import torch
import witwin.channel as wc
from witwin.channel.core.numerics.tensors import to_torch_view


def _wall_scene(material: wc.Material, *, frequency: float) -> wc.Scene:
    return wc.Scene(
        structures=[
            wc.Structure(
                name="wall",
                geometry=wc.Box(
                    position=(0.0, 0.0, 1.5),
                    size=(0.25, 4.0, 3.0),
                    device="cuda",
                ),
                material=material,
            )
        ],
        transmitters=[wc.Transmitter(name="tx", position=(-2.0, -1.0, 1.5))],
        receivers=[wc.Receiver(name="rx", position=(-2.0, 1.0, 1.5))],
        frequency=frequency,
        device="cuda",
    )


def _reflection_amplitude(scene: wc.Scene) -> torch.Tensor:
    result = wc.path.solve(
        scene=scene,
        transmitter="tx",
        receiver="rx",
        config=wc.path.Config(
            num_samples=64,
            max_bounces=1,
            max_diffraction_order=0,
            max_num_paths=4,
        ),
    )
    reflection = result.filter_by_type(wc.path.InteractionType.REFLECTION)
    coeff = reflection.coeff_tensor()
    valid = to_torch_view(reflection.valid, dtype=torch.bool).unsqueeze(-1)
    return torch.abs(torch.where(valid, coeff, torch.zeros_like(coeff))).sum()


def test_path_reflection_response_changes_with_material_mu_r():
    mu1 = _reflection_amplitude(_wall_scene(wc.Material(eps_r=4.0, mu_r=1.0), frequency=3.5e9))
    mu2 = _reflection_amplitude(_wall_scene(wc.Material(eps_r=4.0, mu_r=2.0), frequency=3.5e9))

    assert float(mu1.item()) > 0.0
    assert float(mu2.item()) > 0.0
    assert not torch.isclose(mu1, mu2, rtol=1.0e-3, atol=1.0e-10)


def test_late_bound_itu_material_reevaluates_for_solver_frequency():
    late_bound = _wall_scene(wc.Material.from_itu("concrete"), frequency=2.4e9)
    _reflection_amplitude(late_bound)
    late_bound.frequency = 28.0e9

    late_high = _reflection_amplitude(late_bound)
    fixed_low_material = _reflection_amplitude(
        _wall_scene(wc.Material.from_itu("concrete", frequency=2.4e9), frequency=28.0e9)
    )

    assert not torch.isclose(late_high, fixed_low_material, rtol=1.0e-3, atol=1.0e-10)
