# Copyright Xingyu Chen.
# Aggregated radiomap outputs and frequency gradients repeat bit for bit across solves.

from __future__ import annotations

import pytest
import torch

from benchmarks import synthetic_city
from tests.support.core_world import make_transmitter
from tests.support.scenes import rough_wall_structure
from witwin.channel import radiomap
from witwin.core import Scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

_REPEATS = 3
_LRD = frozenset({"los", "reflection", "diffraction"})


def _city_scene() -> Scene:
    city = synthetic_city.city(4)
    return Scene(
        structures=city.structures,
        endpoints=[
            synthetic_city.transmitter(city.extent_m),
            synthetic_city.receiver_grid(city.extent_m, 32),
        ],
    )


def _rough_scene() -> Scene:
    wall = rough_wall_structure(2.5, rms_height_m=0.015, corr_length_m=0.15, half_size=2.0)
    return Scene(
        structures=[wall],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, -1.0, 0.0])),
            synthetic_city.receiver_grid(4.0, 16),
        ],
    )


def _maps(result, *, detach: bool = True) -> dict[str, torch.Tensor]:
    maps = {"path_gain": result.path_gain}
    maps.update({f"power.{k}": v for k, v in result.component_power.items()})
    if getattr(result, "field", None) is not None:
        maps["field"] = result.field
    for name, value in (getattr(result, "component_fields", None) or {}).items():
        maps[f"field.{name}"] = value
    if not detach:
        return maps
    return {name: value.detach().clone() for name, value in maps.items()}


def _assert_repeats_identical(solve) -> dict[str, torch.Tensor]:
    reference = solve()
    for _ in range(_REPEATS - 1):
        again = solve()
        assert again.keys() == reference.keys()
        for name, value in reference.items():
            assert torch.equal(again[name], value), name
    return reference


def test_coherent_diffraction_and_coupled_maps_repeat_bitwise():
    config = radiomap.Coherent(max_depth=2, components=_LRD, coupled_paths=True)
    maps = _assert_repeats_identical(
        lambda: _maps(radiomap.solve(_city_scene(), config, reference_frequency_hz=3.5e9))
    )
    assert maps["power.diffraction"].abs().sum() > 0
    assert maps["power.coupled"].abs().sum() > 0


def test_coherent_rough_scattering_maps_repeat_bitwise():
    config = radiomap.Coherent(max_depth=1, components={"reflection", "scattering"})
    maps = _assert_repeats_identical(
        lambda: _maps(radiomap.solve(_rough_scene(), config, reference_frequency_hz=3.0e9))
    )
    assert maps["power.scattering"].abs().sum() > 0


def test_stochastic_reflection_and_diffraction_maps_repeat_bitwise():
    config = radiomap.Stochastic(samples=16384, max_depth=1, components=_LRD)
    maps = _assert_repeats_identical(
        lambda: _maps(radiomap.solve(_city_scene(), config, reference_frequency_hz=3.5e9))
    )
    assert maps["power.diffraction"].abs().sum() > 0
    assert maps["power.reflection"].abs().sum() > 0


def test_stochastic_frequency_gradient_repeats_bitwise():
    # The Coherent and Path frequency gradients of diffraction and rough-scattering rows are
    # still summed with float atomics inside RayD's VJP kernels, so only the stochastic
    # estimator's gradient is pinned here.
    config = radiomap.Stochastic(samples=16384, max_depth=1, components=_LRD, ad_mode="vjp")

    def gradient() -> dict[str, torch.Tensor]:
        frequency = torch.tensor(3.5e9, dtype=torch.float64, device="cuda", requires_grad=True)
        result = radiomap.solve(_city_scene(), config, reference_frequency_hz=frequency)
        result.path_gain.sum().backward()
        return {"grad_frequency": frequency.grad.detach().clone()}

    grads = _assert_repeats_identical(gradient)
    assert float(grads["grad_frequency"].abs()) > 0.0


@pytest.mark.parametrize("estimator", ["coherent", "stochastic"])
def test_frequency_tangent_maps_repeat_bitwise(estimator):
    if estimator == "coherent":
        config = radiomap.Coherent(max_depth=1, components=_LRD, ad_mode="jvp")
    else:
        config = radiomap.Stochastic(samples=16384, max_depth=1, components=_LRD, ad_mode="jvp")

    def tangents() -> dict[str, torch.Tensor]:
        primal = torch.tensor(3.5e9, dtype=torch.float64, device="cuda")
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(primal.clone(), torch.ones_like(primal))
            result = radiomap.solve(_city_scene(), config, reference_frequency_hz=dual)
            out = {}
            for name, value in _maps(result, detach=False).items():
                tangent = torch.autograd.forward_ad.unpack_dual(value).tangent
                if tangent is not None:
                    out[name] = tangent.detach().clone()
        return out

    _assert_repeats_identical(tangents)
