"""Path solver specular transmission exports (plan 05 wave 2)."""

import math

import pytest
import torch

from tests.support.scenes import transmission_wall_structure
from witwin.channel import ReceiverPoint, Scene, Transmitter
from witwin.channel.core.kernels.extension import build_info
from witwin.channel.core.materials import Layer, PhysicalSurface
from witwin.channel.path import Config, InteractionType, solve

_FREQUENCY_HZ = 3.0e9
_LIGHT_SPEED_M_PER_S = 299792458.0

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


def _vacuum_wall() -> PhysicalSurface:
    return PhysicalSurface(
        layers=(Layer(thickness_m=0.3, eps_r=1.0),), name="vacuum-wall"
    )


def _lossy_wall() -> PhysicalSurface:
    return PhysicalSurface(
        layers=(Layer(thickness_m=0.1, eps_r=4.0, sigma_e=0.05),), name="lossy-wall"
    )


def test_transmission_path_result_events_and_delay():
    _require_rayd()

    rx_position = [5.0, 0.0, 0.0]
    result = solve(
        _scene([transmission_wall_structure(2.5, _lossy_wall())], rx_position),
        Config(components={"transmission"}, max_depth=1),
    )

    assert result.max_num_paths == 1
    assert bool(result.valid[0, 0, 0, 0, 0])
    assert result.interaction_type[0, 0, 0, 0, 0].tolist() == [
        int(InteractionType.TRANSMISSION)
    ]
    assert int(result.primitive_id[0, 0, 0, 0, 0, 0]) >= 0
    assert int(result.material_id[0, 0, 0, 0, 0, 0]) >= 0
    # tau is the geometric straight length over c (group_delay: "geometric").
    expected_tau = math.dist([0.0, 0.0, 0.0], rx_position) / _LIGHT_SPEED_M_PER_S
    assert result.tau[0, 0, 0, 0, 0].item() == pytest.approx(expected_tau, rel=1.0e-6)
    assert torch.abs(result.a[0, 0, 0, 0, 0, 0]).item() > 0.0
    # The penetration event sits on the wall plane x = 2.5.
    assert result.position[0, 0, 0, 0, 0, 0, 0].item() == pytest.approx(
        2.5, abs=1.0e-4
    )
    assert result.metadata["components"]["transmission"] == "enabled"
    assert result.metadata["transmission"] == {
        "thin_sheet_straight_path_approximation": True,
        "group_delay": "geometric",
    }


def test_transmission_complex_a_matches_empty_scene_los_for_vacuum_wall():
    _require_rayd()

    rx_position = [5.0, 5.0, 0.0]  # 45 degree oblique incidence
    wall = solve(
        _scene([transmission_wall_structure(2.5, _vacuum_wall())], rx_position),
        Config(components={"transmission"}, max_depth=1),
    )
    empty = solve(_scene([], rx_position), Config(components={"los"}))

    ratio = wall.a[0, 0, 0, 0, 0, 0] / empty.a[0, 0, 0, 0, 0, 0]
    assert torch.abs(ratio - 1.0).item() <= 1.0e-4
    assert wall.tau[0, 0, 0, 0, 0].item() == pytest.approx(
        empty.tau[0, 0, 0, 0, 0].item(), rel=1.0e-6
    )


def test_transmission_depth2_sequence_and_type_filter():
    _require_rayd()

    rx_position = [5.0, 0.0, 0.0]
    structures = [
        transmission_wall_structure(2.0, _lossy_wall(), name="wall-a", surface_id=1),
        transmission_wall_structure(3.0, _vacuum_wall(), name="wall-b", surface_id=2),
    ]
    result = solve(
        _scene(structures, rx_position),
        Config(components={"transmission"}, max_depth=2),
    )

    assert result.max_num_paths == 1
    assert result.interaction_type[0, 0, 0, 0, 0].tolist() == [
        int(InteractionType.TRANSMISSION),
        int(InteractionType.TRANSMISSION),
    ]
    positions = result.position[0, 0, 0, 0, 0, :, 0]
    assert positions[0].item() == pytest.approx(2.0, abs=1.0e-4)
    assert positions[1].item() == pytest.approx(3.0, abs=1.0e-4)

    kept = result.filter_by_type(int(InteractionType.TRANSMISSION))
    assert int(kept.num_paths.sum()) == 1
    dropped = result.filter_by_type(int(InteractionType.REFLECTION))
    assert int(dropped.num_paths.sum()) == 0


def test_transmission_solver_exports_complex_path():
    _require_rayd()

    rx_position = [5.0, 0.0, 0.0]
    result = solve(
        _scene([transmission_wall_structure(2.5, _lossy_wall())], rx_position),
        Config(components={"los", "transmission"}, max_depth=1),
    )

    assert int(result.num_paths.sum()) == 1
    assert result.interaction_type[result.valid].tolist() == [[int(InteractionType.TRANSMISSION)]]
    assert bool((result.a[result.valid].abs() > 0).all())
    assert result.metadata["components"]["transmission"] == "enabled"
