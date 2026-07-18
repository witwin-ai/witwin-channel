"""BDPT specular transmission: straight endpoint chains plus the
event-selected shooting sampler for mixed reflection+transmission chains."""

import pytest
import torch

from witwin.channel_native import (
    ReceiverGrid,
    ReceiverPoint,
    Scene,
    Structure,
    Transmitter,
)
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.core.materials import (
    Layer,
    PerfectConductor,
    PhysicalSurface,
)
from witwin.channel_native.montecarlo.bdpt import Config, solve
from witwin.channel_native.physics.oracle import layer_stack_rt

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA torch is required"
)

_FREQUENCY = 3.0e9


def _require_native() -> None:
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native transmission is not built")


def _wall(material, *, x: float = 2.5, surface_id: int = 1) -> Structure:
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
        name=f"wall-{surface_id}",
        surface_id=surface_id,
    )


def _point_scene(structures) -> Scene:
    return Scene(
        structures=structures,
        transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 0.0]))],
        receivers=[ReceiverPoint(position=torch.tensor([5.0, 0.0, 0.0]))],
        frequency=_FREQUENCY,
    )


def _grid() -> ReceiverGrid:
    return ReceiverGrid(
        origin=torch.tensor([5.0, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def _vacuum() -> PhysicalSurface:
    return PhysicalSurface(
        layers=(Layer(thickness_m=0.2, eps_r=1.0),), name="vacuum-wall"
    )


def _lossy() -> PhysicalSurface:
    return PhysicalSurface(
        layers=(Layer(thickness_m=0.1, eps_r=4.0, sigma_e=0.05),),
        name="lossy-wall",
    )


def test_bdpt_vacuum_wall_transmission_recovers_empty_scene_los():
    _require_native()
    config = Config(samples=1024, seed=5, components={"los", "transmission"})
    walled = solve(_point_scene([_wall(_vacuum())]), config)
    empty = solve(_point_scene([]), Config(samples=1024, seed=5, components={"los"}))

    # The wall blocks the direct segment: the exclusive los class is zero.
    assert float(walled.component_power["los"]) == pytest.approx(0.0, abs=1.0e-20)
    # A vacuum wall has unit power transmittance: the transmission component
    # reproduces the empty-scene LoS value.
    torch.testing.assert_close(
        walled.component_power["transmission"],
        empty.component_power["los"],
        rtol=1.0e-4,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        walled.path_gain, empty.path_gain, rtol=1.0e-4, atol=1.0e-12
    )
    assert walled.metadata["components"]["transmission"] == "enabled"
    assert walled.metadata["transmission"]["component_mask_bit"] == 8
    assert walled.metadata["transmission"]["straight_chain_paths"] == 1


def test_bdpt_lossy_wall_transmission_power_ratio_matches_stack():
    _require_native()
    config = Config(samples=1024, seed=5, components={"transmission"})
    walled = solve(_point_scene([_wall(_lossy())]), config)
    empty = solve(_point_scene([]), Config(samples=1024, seed=5, components={"los"}))

    oracle = layer_stack_rt([(0.1, 4.0, 0.05, 1.0)], 1.0, _FREQUENCY)
    # ADR-020: the transmission component is the full-Jones layer-stack field
    # projected on the polarization, not the unpolarized TE/TM mean. At this
    # exact normal incidence the plane of incidence is degenerate and
    # T_te == T_tm, so the polarized value is simply T_te.
    expected_t = float(oracle.T_te)
    assert float(oracle.T_te) == pytest.approx(float(oracle.T_tm), rel=1.0e-6)
    assert 0.0 < expected_t < 1.0
    observed_ratio = float(walled.component_power["transmission"]) / float(
        empty.component_power["los"]
    )
    assert observed_ratio == pytest.approx(expected_t, rel=5.0e-4)


def test_bdpt_pec_wall_transmits_nothing():
    _require_native()
    config = Config(samples=512, seed=5, components={"transmission"})
    result = solve(_point_scene([_wall(PerfectConductor())]), config)
    assert float(result.component_power["transmission"]) < 1.0e-20


def test_bdpt_transmission_is_seed_reproducible_with_mixed_chains():
    """A lossy front wall plus a PEC back wall creates transmit->reflect
    chains, exercising the event-selected shooting sampler end to end."""

    _require_native()
    scene = _point_scene(
        [
            _wall(_lossy(), x=2.5, surface_id=1),
            _wall(PerfectConductor(), x=6.0, surface_id=2),
        ]
    )
    config = Config(samples=2048, seed=11, max_depth=3, components={"transmission"})
    first = solve(scene, config)
    second = solve(scene, config)

    torch.testing.assert_close(
        first.component_power["transmission"],
        second.component_power["transmission"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(first.path_gain, second.path_gain, rtol=0.0, atol=0.0)
    events = first.metadata["transmission"]["event_counts"]
    assert events == second.metadata["transmission"]["event_counts"]
    assert events["transmit"] > 0
    assert events["reflect"] > 0
    assert float(first.component_power["transmission"]) > 0.0


def test_bdpt_results_unchanged_when_transmission_not_requested():
    """Regression guard: enabling transmission must not perturb the los and
    reflection components (exclusive path classes never overlap)."""

    _require_native()
    scene = _point_scene([_wall(_lossy())])
    base = solve(scene, Config(samples=512, seed=7, components={"los", "reflection"}))
    with_t = solve(
        scene,
        Config(samples=512, seed=7, components={"los", "reflection", "transmission"}),
    )

    for component in ("los", "reflection"):
        torch.testing.assert_close(
            base.component_power[component],
            with_t.component_power[component],
            rtol=1.0e-6,
            atol=1.0e-20,
        )
    assert "transmission" not in base.component_power
    # Determinism of the untouched configuration (bit-identical rerun).
    rerun = solve(scene, Config(samples=512, seed=7, components={"los", "reflection"}))
    torch.testing.assert_close(base.path_gain, rerun.path_gain, rtol=0.0, atol=0.0)


def test_bdpt_grid_transmission_component_map_matches_power():
    _require_native()
    scene = Scene(
        structures=[_wall(_vacuum())],
        transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 0.0]))],
        receivers=[_grid()],
        frequency=_FREQUENCY,
    )
    result = solve(scene, Config(samples=512, seed=5, components={"transmission"}))

    assert result.component_maps is not None
    transmission = result.component_maps["transmission"]
    assert transmission.shape == (1, 4, 4)
    assert torch.count_nonzero(transmission) == 16
    torch.testing.assert_close(
        result.component_power["transmission"],
        transmission.sum(),
        rtol=1.0e-5,
        atol=1.0e-10,
    )
