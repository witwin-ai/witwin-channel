"""ADR-018 acceptance: BDPT standalone diffraction parity with deterministic.

Ported from artifacts/ws1-alignment/spot_check2.py. Before ADR-018 the BDPT
standalone diffraction component used a crude native power heuristic and was
~434x (grid) / ~2175x (point) above the deterministic UTD reference on the WS1
wedge fixture. Routing it through the shared enumerated engine as a unit-mass
discrete connection (like reflection) brings it into the [0.5x, 2x] gate; in
practice it reproduces the deterministic reference because both solvers consume
the same first-order UTD evaluation.
"""

import pytest
import torch

from tests.support.scenes import wedge_diffraction_scene
from witwin.core import ReceiverGrid, Scene
from tests.support.core_world import make_receiver_grid
from witwin.channel.core.kernels.extension import build_info
from witwin.channel.deterministic import Config as DeterministicConfig
from witwin.channel.deterministic import solve as deterministic_solve
from witwin.channel.montecarlo.bdpt import Config as BDPTConfig
from witwin.channel.montecarlo.bdpt import solve as bdpt_solve


_GATE_LOW = 0.5
_GATE_HIGH = 2.0


def _grid() -> ReceiverGrid:
    return make_receiver_grid(
        origin=torch.tensor([3.0, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def _with_grid(base: Scene, grid: ReceiverGrid) -> Scene:
    return base.with_endpoints(
        (*tuple(endpoint for endpoint in base.endpoints if endpoint.role == "tx"), grid)
    )


def _diffraction_power(result) -> float:
    return float(result.component_power["diffraction"].sum())


def _skip_unless_native() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction parity")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")


def test_bdpt_grid_diffraction_within_2x_of_deterministic():
    _skip_unless_native()

    scene = _with_grid(wedge_diffraction_scene(), _grid())
    reference = _diffraction_power(
        deterministic_solve(
            scene,
            DeterministicConfig(
                components={"diffraction"}, max_depth=1, coherent=False
            ),
            reference_frequency_hz=3.0e9,
        )
    )
    observed = _diffraction_power(
        bdpt_solve(
            scene,
            BDPTConfig(components={"diffraction"}, samples=512, seed=7),
            reference_frequency_hz=3.0e9,
        )
    )

    assert reference > 0.0
    assert observed > 0.0
    ratio = observed / reference
    assert _GATE_LOW <= ratio <= _GATE_HIGH, (
        f"grid diffraction ratio {ratio} outside [0.5, 2]"
    )


def test_bdpt_point_diffraction_within_2x_of_deterministic():
    _skip_unless_native()

    scene = wedge_diffraction_scene()
    reference = _diffraction_power(
        deterministic_solve(
            scene,
            DeterministicConfig(
                components={"diffraction"}, max_depth=1, coherent=False
            ),
            reference_frequency_hz=3.0e9,
        )
    )
    observed = _diffraction_power(
        bdpt_solve(
            scene,
            BDPTConfig(
                components={"diffraction"},
                samples=512,
                seed=7,
                receiver_strategy="point_sphere",
            ),
            reference_frequency_hz=3.0e9,
        )
    )

    assert reference > 0.0
    assert observed > 0.0
    ratio = observed / reference
    assert _GATE_LOW <= ratio <= _GATE_HIGH, (
        f"point diffraction ratio {ratio} outside [0.5, 2]"
    )
