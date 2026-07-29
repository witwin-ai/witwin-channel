# Copyright Xingyu Chen.
# Tests multiorder scattering.

"""Tests multiorder scattering."""

import pytest
import torch

from witwin.core import Scene, Structure
from tests.support.core_world import (
    make_mesh_structure,
    make_receiver,
    make_transmitter,
)
from witwin.channel.deployment import build_info
from witwin.core import MaterialLayer, PhysicalMaterial, SurfaceRoughness
from witwin.channel.montecarlo.bdpt import Config, solve


_FREQUENCY = 60.0e9
_SIGMA_H = 1.0e-3
_CORR = 0.01
_EPS_R = 4.0
_SIGMA_E = 0.05
_THICKNESS = 0.1

_TX = torch.tensor([0.0, 0.0, 0.0])
_RX = torch.tensor([0.5, 1.0, 0.3])


def _roughness() -> SurfaceRoughness:
    return SurfaceRoughness(
        rms_height_m=_SIGMA_H,
        correlation_length_x_m=_CORR,
        correlation_length_y_m=_CORR,
    )


def _material() -> PhysicalMaterial:
    return PhysicalMaterial(
        layers=(MaterialLayer(thickness_m=_THICKNESS, eps_r=_EPS_R, sigma_e=_SIGMA_E),),
        roughness_front=_roughness(),
        name="wall-material",
    )


def _wall_x(material: PhysicalMaterial, *, x: float = 2.5) -> Structure:
    return make_mesh_structure(
        vertices=torch.tensor(
            [[x, -4.0, -4.0], [x, 4.0, -4.0], [x, -4.0, 4.0], [x, 4.0, 4.0]]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=material,
        name="wall-x",
        surface_id=1,
    )


def _wall_y(material: PhysicalMaterial, *, y: float = 2.5) -> Structure:
    return make_mesh_structure(
        vertices=torch.tensor(
            [[-4.0, y, -4.0], [4.0, y, -4.0], [-4.0, y, 4.0], [4.0, y, 4.0]]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=material,
        name="wall-y",
        surface_id=2,
    )


def _single_wall_scene() -> Scene:
    return Scene(
        structures=[_wall_x(_material())],
        endpoints=[
            make_transmitter(position=_TX),
            make_receiver(position=_RX),
        ],
    )


def _corner_scene() -> Scene:
    """Two mutually visible rough walls so a tx->wall->wall->rx double-scatter
 path exists."""

    return Scene(
        structures=[_wall_x(_material()), _wall_y(_material())],
        endpoints=[
            make_transmitter(position=_TX),
            make_receiver(position=_RX),
        ],
    )


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT multi-order scattering")


def _require_native() -> None:
    _require_cuda()
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native scattering is not built")


# --------------------------------------------------------------------------- #
# Config contract (pure Python, no CUDA).
# --------------------------------------------------------------------------- #


def test_max_scattering_order_defaults_to_one():
    assert Config().max_scattering_order == 1


def test_max_scattering_order_accepts_values_at_or_above_one():
    assert Config(max_scattering_order=1).max_scattering_order == 1
    assert Config(max_scattering_order=3).max_scattering_order == 3


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_max_scattering_order_rejects_below_one(bad):
    with pytest.raises(ValueError, match="max_scattering_order"):
        Config(max_scattering_order=bad)


def test_coherent_refuses_scattering_regardless_of_order():
    # The coherent combination (coherent combine supports only los/reflection/
    # diffraction) stays intact and applies whatever the scattering order is.
    for order in (1, 2, 3):
        with pytest.raises(RuntimeError, match="coherent"):
            Config(
                coherent=True,
                components={"reflection", "scattering"},
                max_scattering_order=order,
            )


def test_coherent_allowed_without_scattering_even_at_high_order():
    # max_scattering_order > 1 is not itself a coherent-combine conflict: only
    # a scattering component in the coherent solve is refused.
    config = Config(
        coherent=True,
        components={"los", "reflection"},
        max_scattering_order=2,
        ad_mode="none",
    )
    assert config.coherent is True
    assert config.max_scattering_order == 2


# --------------------------------------------------------------------------- #
# Solver behaviour (CUDA + native scattering).
# --------------------------------------------------------------------------- #


def test_default_order_matches_explicit_order_one_bitwise():
    """The new config field is a pure default: an unspecified order and an
 explicit order 1 solve are bit-identical (single-bounce terminal rule)."""

    _require_native()
    scene = _single_wall_scene()
    default = solve(
        scene,
        Config(samples=8192, seed=7, max_depth=2, components={"scattering"}),
        reference_frequency_hz=_FREQUENCY,
    )
    explicit = solve(
        scene,
        Config(
            samples=8192,
            seed=7,
            max_depth=2,
            components={"scattering"},
            max_scattering_order=1,
        ),
        reference_frequency_hz=_FREQUENCY,
    )
    torch.testing.assert_close(
        default.path_gain, explicit.path_gain, rtol=0.0, atol=0.0
    )
    assert (
        default.metadata["scattering"]["event_counts"]
        == explicit.metadata["scattering"]["event_counts"]
    )


def test_order_two_does_not_perturb_an_order_one_rerun():
    """Seed-stream isolation: the multi-order continuation draws from the same
 per-bounce salted direction stream, so interleaving an order-2 solve leaves
 an order-1 solve bit-identical."""

    _require_native()
    scene = _single_wall_scene()
    base = Config(samples=8192, seed=11, max_depth=2, components={"scattering"})
    order_one_first = solve(scene, base, reference_frequency_hz=_FREQUENCY)
    solve(
        scene,
        Config(
            samples=8192,
            seed=11,
            max_depth=2,
            components={"scattering"},
            max_scattering_order=2,
        ),
        reference_frequency_hz=_FREQUENCY,
    )
    order_one_again = solve(scene, base, reference_frequency_hz=_FREQUENCY)
    torch.testing.assert_close(
        order_one_first.path_gain, order_one_again.path_gain, rtol=0.0, atol=0.0
    )


def test_order_two_scattered_power_at_least_order_one():
    """On a two-wall corner (a tx->wall->wall->rx double-scatter path exists),
 order 2 adds nonnegative multi-bounce NEE contributions on top of every
 order-1 row, so its scattered power never drops below order 1."""

    _require_native()
    scene = _corner_scene()
    common = dict(samples=65_536, seed=7, max_depth=3, components={"scattering"})
    order_one = solve(
        scene,
        Config(**common, max_scattering_order=1),
        reference_frequency_hz=_FREQUENCY,
    )
    order_two = solve(
        scene,
        Config(**common, max_scattering_order=2),
        reference_frequency_hz=_FREQUENCY,
    )
    power_one = float(order_one.component_power["scattering"].sum())
    power_two = float(order_two.component_power["scattering"].sum())
    assert power_one > 0.0
    assert power_two >= power_one * (1.0 - 1.0e-6)


def test_metadata_reports_scattering_order_and_depth_rule():
    _require_native()
    scene = _single_wall_scene()
    order_one = solve(
        scene,
        Config(samples=1024, seed=7, max_depth=2, components={"scattering"}),
        reference_frequency_hz=_FREQUENCY,
    )
    assert order_one.metadata["max_scattering_order"] == 1
    assert order_one.metadata["scattering_depth_rule"] == "single_bounce_terminal"

    order_two = solve(
        scene,
        Config(
            samples=1024,
            seed=7,
            max_depth=3,
            components={"scattering"},
            max_scattering_order=2,
        ),
        reference_frequency_hz=_FREQUENCY,
    )
    assert order_two.metadata["max_scattering_order"] == 2
    assert order_two.metadata["scattering_depth_rule"] == "multi_order_continuation"