"""MC basic straight-penetration transmission radiomap (contract section 4).

Acceptance: a single eps_r=1 vacuum wall reproduces the unobstructed LoS map
exactly (within float tolerance), a lossy wall attenuates it by the stack
power transmittance, and a PEC wall transmits nothing.
"""

import pytest
import torch

from witwin.channel_native import ReceiverGrid, Scene, Structure, Transmitter
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.core.materials import (
    Layer,
    PerfectConductor,
    PhysicalSurface,
)
from witwin.channel_native.montecarlo.basic import Config, solve
from witwin.channel_native.physics.oracle import layer_stack_rt

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA torch is required"
)

_FREQUENCY = 3.0e9


def _require_native() -> None:
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native transmission is not built")


def _wall(material, *, x: float = 2.5) -> Structure:
    return Structure(
        vertices=torch.tensor(
            [
                [x, -4.0, -4.0],
                [x, 4.0, -4.0],
                [x, -4.0, 4.0],
                [x, 4.0, 4.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=material,
        name=f"wall-{x}",
        surface_id=int(x * 10),
    )


def _grid(shape: tuple[int, int] = (4, 4)) -> ReceiverGrid:
    if shape == (1, 1):
        return ReceiverGrid(
            origin=torch.tensor([5.0, 0.0, 0.0]),
            x_axis=torch.tensor([0.0, 1.0, 0.0]),
            y_axis=torch.tensor([0.0, 0.0, 1.0]),
            shape=(1, 1),
            spacing=(1.0, 1.0),
        )
    return ReceiverGrid(
        origin=torch.tensor([5.0, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=shape,
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def _scene(structures, *, grid_shape: tuple[int, int] = (4, 4)) -> Scene:
    return Scene(
        structures=structures,
        transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 0.0]))],
        receivers=[_grid(grid_shape)],
        frequency=_FREQUENCY,
    )


def _solve(scene: Scene, components, *, max_depth: int = 2):
    return solve(
        scene,
        Config(samples=64, seed=3, max_depth=max_depth, components=components),
    )


def test_vacuum_wall_transmission_map_equals_unobstructed_los_map():
    _require_native()
    vacuum = PhysicalSurface(
        layers=(Layer(thickness_m=0.2, eps_r=1.0),), name="vacuum-wall"
    )
    walled = _solve(_scene([_wall(vacuum)]), {"los", "transmission"})
    empty = _solve(_scene([]), {"los"})

    # The wall blocks every tx->cell segment: the exclusive los class is zero.
    assert torch.count_nonzero(walled.component_maps["los"]) == 0
    # A vacuum layer has unit power transmittance, so the transmission map
    # reproduces the unobstructed analytic LoS map (acceptance test).
    torch.testing.assert_close(
        walled.component_maps["transmission"],
        empty.component_maps["los"],
        rtol=1.0e-4,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        walled.component_power["transmission"],
        walled.component_maps["transmission"].sum(),
        rtol=1.0e-5,
        atol=1.0e-10,
    )
    assert walled.metadata["components"]["transmission"] == "enabled"


def test_lossy_wall_attenuates_by_stack_power_transmittance():
    _require_native()
    thickness, eps_r, sigma_e = 0.1, 4.0, 0.05
    lossy = PhysicalSurface(
        layers=(Layer(thickness_m=thickness, eps_r=eps_r, sigma_e=sigma_e),),
        name="lossy-wall",
    )
    # 1x1 grid straight behind the wall: exact normal incidence.
    walled = _solve(
        _scene([_wall(lossy)], grid_shape=(1, 1)), {"transmission"}
    )
    empty = _solve(_scene([], grid_shape=(1, 1)), {"los"})

    oracle = layer_stack_rt(
        [(thickness, eps_r, sigma_e, 1.0)], 1.0, _FREQUENCY
    )
    expected_t = 0.5 * (float(oracle.T_te) + float(oracle.T_tm))
    assert 0.0 < expected_t < 1.0
    torch.testing.assert_close(
        walled.component_maps["transmission"],
        empty.component_maps["los"] * expected_t,
        rtol=5.0e-4,
        atol=1.0e-14,
    )


def test_pec_wall_transmits_nothing():
    _require_native()
    walled = _solve(_scene([_wall(PerfectConductor())]), {"transmission"})
    assert float(walled.component_maps["transmission"].abs().max()) < 1.0e-20
    assert float(walled.component_power["transmission"]) < 1.0e-20


def test_transmission_respects_max_depth_budget():
    _require_native()
    vacuum = PhysicalSurface(
        layers=(Layer(thickness_m=0.2, eps_r=1.0),), name="vacuum-wall"
    )
    scene = _scene([_wall(vacuum, x=2.0), _wall(vacuum, x=3.0)])
    empty = _solve(_scene([]), {"los"})

    # Two walls need two penetrations: max_depth=1 truthfully blocks.
    capped = _solve(scene, {"transmission"}, max_depth=1)
    assert torch.count_nonzero(capped.component_maps["transmission"]) == 0
    # max_depth=2 penetrates both vacuum walls and recovers the LoS map.
    full = _solve(scene, {"transmission"}, max_depth=2)
    torch.testing.assert_close(
        full.component_maps["transmission"],
        empty.component_maps["los"],
        rtol=1.0e-4,
        atol=1.0e-12,
    )
