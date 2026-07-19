"""Acceptance tests for coplanar-group reflection semantics (audit D-1/D-2)."""

import pytest
import torch

from witwin.channel_native import ReceiverPoint, Scene, Structure, Transmitter
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.core.materials import Dielectric
from witwin.channel_native.deterministic import Config, solve


def _require_native() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")


def _subdivided_wall_vertices_faces(splits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A wall at x=2.5 spanning y in [-2, 2], z in [-1, 2] split into a grid."""

    ys = torch.linspace(-2.0, 2.0, splits + 1)
    zs = torch.linspace(-1.0, 2.0, splits + 1)
    vertices = torch.stack(
        [
            torch.full(((splits + 1) ** 2,), 2.5),
            ys.repeat_interleave(splits + 1),
            zs.repeat(splits + 1),
        ],
        dim=1,
    )
    faces = []
    for iy in range(splits):
        for iz in range(splits):
            a = iy * (splits + 1) + iz
            b = (iy + 1) * (splits + 1) + iz
            faces.append([a, b, a + 1])
            faces.append([b, b + 1, a + 1])
    return vertices, torch.tensor(faces)


def _wall_scene(splits: int) -> Scene:
    vertices, faces = _subdivided_wall_vertices_faces(splits)
    wall = Structure(
        vertices=vertices,
        faces=faces,
        material=Dielectric(eps_r=4.0, sigma_e=0.01),
        name=f"wall-{splits}x{splits}",
        surface_id=1,
    )
    return Scene(
        structures=[wall],
        transmitters=[Transmitter(position=torch.tensor([0.0, -1.0, 0.5]))],
        receivers=[ReceiverPoint(position=torch.tensor([0.0, 1.0, 0.5]))],
        frequency=3.0e9,
    )


def test_coplanar_subdivision_invariance():
    """Refining a wall's triangulation must not change the reflected field (D-1)."""

    _require_native()
    config = Config(components={"reflection"}, coherent=True, export_paths=True)
    coarse = solve(_wall_scene(1), config)
    fine = solve(_wall_scene(8), config)

    assert coarse.paths is not None and fine.paths is not None
    assert int(coarse.paths.valid.numel()) == 1
    assert int(fine.paths.valid.numel()) == 1
    torch.testing.assert_close(
        fine.paths.path_gain, coarse.paths.path_gain, rtol=1.0e-4, atol=1.0e-12
    )
    torch.testing.assert_close(
        fine.path_gain, coarse.path_gain, rtol=1.0e-4, atol=1.0e-12
    )


def test_inner_corner_double_reflection_within_one_structure():
    """Consecutive bounces on two walls of the same structure must survive (D-2d)."""

    _require_native()
    corner = Structure(
        vertices=torch.tensor(
            [
                # wall A at x=2 spanning y in [-1, 3]
                [2.0, -1.0, 0.0],
                [2.0, 3.0, 0.0],
                [2.0, -1.0, 2.0],
                [2.0, 3.0, 2.0],
                # wall B at y=2 spanning x in [0, 3]
                [0.0, 2.0, 0.0],
                [3.0, 2.0, 0.0],
                [0.0, 2.0, 2.0],
                [3.0, 2.0, 2.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2], [4, 6, 5], [5, 6, 7]]),
        material=Dielectric(eps_r=3.0, sigma_e=0.005),
        name="l-corner",
        surface_id=7,
    )
    scene = Scene(
        structures=[corner],
        transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 1.0]))],
        receivers=[ReceiverPoint(position=torch.tensor([0.0, 1.0, 1.0]))],
        frequency=3.0e9,
    )

    result = solve(scene, Config(components={"reflection"}, max_depth=2, coherent=True, export_paths=True))

    assert result.paths is not None
    assert bool((result.paths.depth == 2).any())


# F1/R5 (utd-continuity-fix-design): the reflected field carries the
# unnormalized transverse (short-dipole sin(theta)) projection of the default
# z-hat transmit polarization and is projected onto the z-hat receiver
# polarization, so path_gain = |E|^2 for |R| = 1 is the Friis free-space gain
# times sin^2(theta_tx) * sin^2(theta_rx) about the launch/arrival directions.
_REFLECT_TX = torch.tensor([0.0, -1.0, 0.0])
_REFLECT_RX = torch.tensor([0.0, 1.0, 2.0])
_REFLECT_WALL_X = 2.5


def _specular_dipole_factor() -> float:
    zhat = torch.tensor([0.0, 0.0, 1.0])
    image_rx = _REFLECT_RX.clone()
    image_rx[0] = 2.0 * _REFLECT_WALL_X - _REFLECT_RX[0]
    image_tx = _REFLECT_TX.clone()
    image_tx[0] = 2.0 * _REFLECT_WALL_X - _REFLECT_TX[0]
    launch = image_rx - _REFLECT_TX  # TX -> specular point direction
    arrival = _REFLECT_RX - image_tx  # specular point -> RX direction

    def _sin2(direction: torch.Tensor) -> float:
        unit = direction / direction.norm()
        return 1.0 - float(zhat @ unit) ** 2

    return _sin2(launch) * _sin2(arrival)


def test_conductor_skew_reflection_preserves_friis_amplitude():
    """|R| = 1 for a near-perfect conductor at skew incidence (audit D-4).

    The historical scalar TE+TM sum gave |r_te*e_s + r_tm*e_p| which reaches
    sqrt(2) (+3 dB) or 0 depending on the azimuth of the fixed x-hat transmit
    polarization; the vector composition keeps |E| = |E_in| for |r| = 1.
    """

    _require_native()
    wall = Structure(
        vertices=torch.tensor(
            [
                [2.5, -3.0, -2.0],
                [2.5, 3.0, -2.0],
                [2.5, -3.0, 3.0],
                [2.5, 3.0, 3.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=Dielectric(eps_r=1.0, sigma_e=1.0e7),
        name="conductor-wall",
        surface_id=1,
    )
    scene = Scene(
        structures=[wall],
        transmitters=[Transmitter(position=torch.tensor([0.0, -1.0, 0.0]))],
        receivers=[ReceiverPoint(position=torch.tensor([0.0, 1.0, 2.0]))],
        frequency=3.0e9,
    )

    result = solve(scene, Config(components={"reflection"}, coherent=True, export_paths=True))

    assert result.paths is not None
    assert int(result.paths.valid.numel()) == 1
    wavelength = 299_792_458.0 / scene.frequency
    unfolded = torch.tensor([5.0, -1.0, 0.0]) - torch.tensor([0.0, 1.0, 2.0])
    expected_length = float(unfolded.norm())
    # F1/R5: multiply the Friis free-space gain by the z-hat dipole coupling.
    expected_gain = (
        wavelength / (4.0 * torch.pi * expected_length)
    ) ** 2 * _specular_dipole_factor()
    torch.testing.assert_close(
        result.paths.path_length_m.cpu(),
        torch.tensor([expected_length]),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        result.paths.path_gain.cpu(),
        torch.tensor([expected_gain], dtype=torch.float32),
        rtol=5.0e-3,
        atol=1.0e-12,
    )


def test_perfect_conductor_reflects_with_unit_magnitude():
    """PEC materials must reach the |R| = 1 Fresnel limit even though the
    field kernels only receive (eps_r, sigma_e, mu_r)."""

    _require_native()
    from witwin.channel_native.core.materials import PerfectConductor

    wall = Structure(
        vertices=torch.tensor(
            [
                [2.5, -3.0, -2.0],
                [2.5, 3.0, -2.0],
                [2.5, -3.0, 3.0],
                [2.5, 3.0, 3.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=PerfectConductor(),
        name="pec-wall",
        surface_id=1,
    )
    scene = Scene(
        structures=[wall],
        transmitters=[Transmitter(position=torch.tensor([0.0, -1.0, 0.0]))],
        receivers=[ReceiverPoint(position=torch.tensor([0.0, 1.0, 2.0]))],
        frequency=3.0e9,
    )

    result = solve(scene, Config(components={"reflection"}, coherent=True, export_paths=True))

    assert result.paths is not None
    assert int(result.paths.valid.numel()) == 1
    wavelength = 299_792_458.0 / scene.frequency
    unfolded = torch.tensor([5.0, -1.0, 0.0]) - torch.tensor([0.0, 1.0, 2.0])
    # F1/R5: multiply the Friis free-space gain by the z-hat dipole coupling.
    expected_gain = (
        wavelength / (4.0 * torch.pi * float(unfolded.norm()))
    ) ** 2 * _specular_dipole_factor()
    torch.testing.assert_close(
        result.paths.path_gain.cpu(),
        torch.tensor([expected_gain], dtype=torch.float32),
        rtol=5.0e-3,
        atol=1.0e-12,
    )


def test_back_wall_reflection_is_blocked_by_front_wall():
    """A reflection off a hidden back wall must be occluded, even within one structure (D-2b)."""

    _require_native()
    walls = Structure(
        vertices=torch.tensor(
            [
                # front wall at x=2.5
                [2.5, -2.0, -1.0],
                [2.5, 2.0, -1.0],
                [2.5, -2.0, 2.0],
                [2.5, 2.0, 2.0],
                # back wall at x=3.5 (fully shadowed by the front wall)
                [3.5, -2.0, -1.0],
                [3.5, 2.0, -1.0],
                [3.5, -2.0, 2.0],
                [3.5, 2.0, 2.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]]),
        material=Dielectric(eps_r=4.0, sigma_e=0.01),
        name="front-and-back",
        surface_id=3,
    )
    scene = Scene(
        structures=[walls],
        transmitters=[Transmitter(position=torch.tensor([0.0, -1.0, 0.5]))],
        receivers=[ReceiverPoint(position=torch.tensor([0.0, 1.0, 0.5]))],
        frequency=3.0e9,
    )

    result = solve(scene, Config(components={"reflection"}, coherent=True, export_paths=True))

    assert result.paths is not None
    # Only the front-wall specular path may survive; the back-wall path is occluded.
    assert int(result.paths.valid.numel()) == 1
    hit_x = result.paths.interaction_position[:, 0]
    torch.testing.assert_close(hit_x, torch.full_like(hit_x, 2.5))
