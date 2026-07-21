"""Deterministic endpoint-connection specular transmission (plan 05 wave 2).

The decisive invariant: a thin_sheet wall made of a single vacuum layer must
reproduce the empty-scene LoS complex field (amplitude AND phase) at normal
and oblique incidence.
"""

import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch

from tests.support.scenes import transmission_wall_structure
from witwin.channel_native import ReceiverPoint, Scene, Transmitter
from witwin.channel_native.materials.kernels import functional as ops
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.core.materials import (
    Layer,
    PerfectConductor,
    PhysicalSurface,
)
from witwin.channel_native.deterministic import Config, solve

_FREQUENCY_HZ = 3.0e9
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA torch is required"
)


def _require_rayd() -> None:
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native scene capability is not built")


def _scene(structures: list, rx_position: list[float]) -> Scene:
    return Scene(
        structures=structures,
        transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 0.0]))],
        receivers=[ReceiverPoint(position=torch.tensor(rx_position))],
        frequency=_FREQUENCY_HZ,
    )


def _vacuum_wall(thickness_m: float = 0.3) -> PhysicalSurface:
    return PhysicalSurface(
        layers=(Layer(thickness_m=thickness_m, eps_r=1.0),), name="vacuum-wall"
    )


def _lossy_wall(name: str = "lossy-wall") -> PhysicalSurface:
    return PhysicalSurface(
        layers=(Layer(thickness_m=0.1, eps_r=4.0, sigma_e=0.05),), name=name
    )


_TRANSMISSION = Config(components={"transmission"}, max_depth=1)
_LOS = Config(components={"los"})


def _stack_t_te(material: PhysicalSurface, cos_theta: float) -> complex:
    layer = material.layers[0]
    stack = ops.em_layer_stack_eval(
        torch.tensor([cos_theta], device="cuda", dtype=torch.float32),
        torch.tensor([0], device="cuda", dtype=torch.int32),
        layer_offset=torch.tensor([0], device="cuda", dtype=torch.int32),
        layer_count=torch.tensor([1], device="cuda", dtype=torch.int32),
        layer_thickness_m=torch.tensor(
            [layer.thickness_m], device="cuda", dtype=torch.float32
        ),
        layer_eps_r=torch.tensor([layer.eps_r], device="cuda", dtype=torch.float32),
        layer_sigma_e=torch.tensor([layer.sigma_e], device="cuda", dtype=torch.float32),
        layer_mu_r=torch.tensor([layer.mu_r], device="cuda", dtype=torch.float32),
        frequency_hz=_FREQUENCY_HZ,
    )
    return complex(stack["t_te_real"][0].item(), stack["t_te_imag"][0].item())


@pytest.mark.parametrize(
    "rx_position",
    (
        [5.0, 0.0, 0.0],  # normal incidence on the x = 2.5 wall
        [5.0, 5.0, 0.0],  # 45 degree oblique incidence (default pol is pure TE)
    ),
)
def test_vacuum_wall_transmission_equals_empty_scene_los(rx_position):
    _require_rayd()

    wall = solve(
        _scene([transmission_wall_structure(2.5, _vacuum_wall())], rx_position),
        _TRANSMISSION,
    )
    empty = solve(_scene([], rx_position), _LOS)

    ratio = wall.component_fields["transmission"] / empty.component_fields["los"]
    assert torch.abs(ratio - 1.0).max().item() <= 1.0e-4
    torch.testing.assert_close(
        wall.component_power["transmission"],
        empty.component_power["los"],
        rtol=1.0e-4,
        atol=0.0,
    )
    assert wall.metadata["components"]["transmission"] == "enabled"
    assert wall.metadata["counts"]["components"]["transmission"] == 1
    assert wall.metadata["transmission"] == {
        "thin_sheet_straight_path_approximation": True,
        "group_delay": "geometric",
    }


@pytest.mark.parametrize(
    ("rx_position", "cos_theta"),
    (
        ([5.0, 0.0, 0.0], 1.0),
        ([5.0, 5.0, 0.0], math.cos(math.pi / 4.0)),
    ),
)
def test_lossy_wall_matches_layer_stack_transmission(rx_position, cos_theta):
    _require_rayd()

    material = _lossy_wall()
    wall = solve(
        _scene([transmission_wall_structure(2.5, material)], rx_position),
        _TRANSMISSION,
    )
    empty = solve(_scene([], rx_position), _LOS)
    expected = abs(_stack_t_te(material, cos_theta))
    assert expected < 0.999  # the wall really attenuates

    observed = (
        wall.component_fields["transmission"].abs()
        / empty.component_fields["los"].abs()
    ).item()
    assert observed == pytest.approx(expected, rel=1.0e-4)
    power_ratio = (
        wall.component_power["transmission"] / empty.component_power["los"]
    ).item()
    assert power_ratio == pytest.approx(expected**2, rel=2.0e-4)


def test_two_walls_transmission_is_the_product_of_single_wall_stacks():
    _require_rayd()

    rx_position = [5.0, 0.0, 0.0]
    wall_a = transmission_wall_structure(
        2.0, _lossy_wall("wall-a"), name="wall-a", surface_id=1
    )
    wall_b = transmission_wall_structure(
        3.0, _vacuum_wall(), name="wall-b", surface_id=2
    )
    empty = solve(_scene([], rx_position), _LOS)
    los_field = empty.component_fields["los"]

    single_a = solve(_scene([wall_a], rx_position), _TRANSMISSION)
    single_b = solve(_scene([wall_b], rx_position), _TRANSMISSION)
    both = solve(
        _scene([wall_a, wall_b], rx_position),
        Config(components={"transmission"}, max_depth=2),
    )

    # Each wall multiplies the LoS field by its stack ratio; a depth-2 chain
    # must carry the complex product of the two single-wall ratios.
    ratio_a = single_a.component_fields["transmission"] / los_field
    ratio_b = single_b.component_fields["transmission"] / los_field
    ratio_both = both.component_fields["transmission"] / los_field
    assert torch.abs(ratio_both - ratio_a * ratio_b).max().item() <= 1.0e-4
    assert both.metadata["counts"]["components"]["transmission"] == 1


def test_max_depth_overflow_is_fail_loud_for_two_walls():
    _require_rayd()
    code = textwrap.dedent(
        """
        import torch

        from tests.deterministic.test_transmission import (
            _TRANSMISSION,
            _scene,
            _vacuum_wall,
        )
        from tests.support.scenes import transmission_wall_structure
        from witwin.channel_native.deterministic import solve

        structures = [
            transmission_wall_structure(
                2.0, _vacuum_wall(), name="wall-a", surface_id=1
            ),
            transmission_wall_structure(
                3.0, _vacuum_wall(), name="wall-b", surface_id=2
            ),
        ]
        solve(_scene(structures, [5.0, 0.0, 0.0]), _TRANSMISSION)
        print("TERMINAL_ENQUEUED", flush=True)
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            print("TERMINAL_SYNC_ERROR", flush=True)
        else:
            raise AssertionError("D + 1 penetration overflow did not fail loudly")
        """
    )
    environment = os.environ.copy()
    source_root = str(_REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert "TERMINAL_ENQUEUED" in completed.stdout
    assert "TERMINAL_SYNC_ERROR" in completed.stdout


def test_los_transmission_exclusivity():
    _require_rayd()

    rx_position = [5.0, 0.0, 0.0]
    config = Config(components={"los", "transmission"}, max_depth=1, coherent=False)

    blocked = solve(
        _scene([transmission_wall_structure(2.5, _lossy_wall())], rx_position), config
    )
    assert torch.count_nonzero(blocked.component_power["los"]) == 0
    assert bool((blocked.component_power["transmission"] > 0).all())
    torch.testing.assert_close(
        blocked.path_gain, blocked.component_power["transmission"]
    )

    empty = solve(_scene([], rx_position), config)
    assert torch.count_nonzero(empty.component_power["transmission"]) == 0
    assert bool((empty.component_power["los"] > 0).all())


def test_pec_wall_transmission_is_negligible():
    _require_rayd()

    rx_position = [5.0, 0.0, 0.0]
    wall = solve(
        _scene([transmission_wall_structure(2.5, PerfectConductor())], rx_position),
        _TRANSMISSION,
    )
    empty = solve(_scene([], rx_position), _LOS)

    ratio = (wall.component_power["transmission"] / empty.component_power["los"]).max()
    assert ratio.item() <= 1.0e-10  # below -100 dB
