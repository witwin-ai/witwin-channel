# Copyright Xingyu Chen.
# Tests that an endpoint on or near a reflecting plane keeps its reflection and image-method field.

from __future__ import annotations

import math

import pytest
import torch

from witwin.channel.propagation.consumer import (
    EndpointBatch,
    PropagationRequest,
    evaluate,
)
from witwin.channel.scene import compile
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

FREQUENCY_HZ = 77.0e9
WAVE_NUMBER = 2.0 * math.pi * FREQUENCY_HZ / 299_792_458.0
# The wall x = 4, |y| <= 1.2, |z| <= 3: diagonal 6.46 m, so the minimum segment is 64.6 um.
WALL_X_M = 4.0
SOURCE = (0.0, 0.0, 0.0)
NEAR_M = 5.0e-5


def _structure(vertices, faces, material, structure_id: int) -> Structure:
    return Structure(
        geometry=Mesh(
            vertices=torch.tensor(vertices, dtype=torch.float32),
            faces=torch.tensor(faces),
            recenter=False,
            fill_mode="surface",
            topology_diagnostics=False,
        ),
        material=material,
        structure_id=structure_id,
        material_id=structure_id,
        assignment_id=structure_id,
        surface_id=structure_id,
    )


def _wall_scene(material):
    wall = _structure(
        [[WALL_X_M, -1.2, -3.0], [WALL_X_M, 1.2, -3.0], [WALL_X_M, 1.2, 3.0], [WALL_X_M, -1.2, 3.0]],
        [[0, 1, 2], [0, 2, 3]],
        material,
        1,
    )
    return compile(Scene(structures=(wall,)), reference_frequency_hz=FREQUENCY_HZ)


def _batch(stable_id: int, position, power: float | None = None) -> EndpointBatch:
    return EndpointBatch(
        stable_ids=torch.tensor([stable_id], device="cuda"),
        positions_m=torch.tensor([position], device="cuda", dtype=torch.float32),
        polarizations=torch.tensor([[0.0, 0.0, 1.0]], device="cuda"),
        powers_w=None if power is None else torch.tensor([power], device="cuda"),
    )


def _discover(compiled, source, sink, max_depth: int = 1):
    request = PropagationRequest(
        sources=_batch(1, source, 1.0), sinks=_batch(2, sink), reference_frequency_hz=FREQUENCY_HZ,
        components=frozenset({"los", "reflection"}), max_depth=max_depth, response="scalar_transport",
        topology_mode="discover", ad_mode="none",
    )
    return evaluate(compiled, request).paths


def _image_ratio(source, sink, reflection: complex) -> float:
    """|direct + reflection * image| / |direct| for the wall x = WALL_X_M, in float64."""

    image = (2.0 * WALL_X_M - source[0], source[1], source[2])
    direct = math.dist(source, sink)
    mirrored = math.dist(image, sink)
    total = (
        complex(math.cos(-WAVE_NUMBER * direct), math.sin(-WAVE_NUMBER * direct)) / direct
        + reflection
        * complex(math.cos(-WAVE_NUMBER * mirrored), math.sin(-WAVE_NUMBER * mirrored))
        / mirrored
    )
    return abs(total) * direct


def _total_ratio(paths) -> tuple[list[int], float]:
    depth = paths.topology.depth.tolist()
    field = paths.transport.coefficient.to(torch.complex128)
    los = field[[i for i, d in enumerate(depth) if d == 0]].sum()
    return depth, float(field.sum().abs() / los.abs())


@pytest.mark.parametrize("offset_m", [0.0, NEAR_M])
@pytest.mark.parametrize("site_is_source", [False, True])
def test_endpoint_on_a_pec_plane_keeps_its_reflection_and_matches_the_image_method(
    offset_m: float, site_is_source: bool,
) -> None:
    """A receiver (or a source: one leg of a scatter site's two-leg chain) on a PEC wall or 0.05 mm
    in front of it keeps its reflection, and LoS plus reflection is the image-method field: zero on
    the plane for the tangential (TE) polarization, ``|1 - exp(-2jk delta cos)|`` in front of it."""

    compiled = _wall_scene(PhysicalMaterial.perfect_conductor())
    site = (WALL_X_M - offset_m, 0.6, 0.0)
    source, sink = (site, SOURCE) if site_is_source else (SOURCE, site)
    depth, ratio = _total_ratio(_discover(compiled, source, sink))
    assert sorted(depth) == [0, 1]
    if offset_m == 0.0:
        assert ratio < 1.0e-3
    else:
        expected = _image_ratio(source, sink, -1.0)
        assert 0.1 < expected < 0.2
        assert ratio == pytest.approx(expected, rel=2.0e-2)


def test_two_leg_site_on_a_pec_plane_is_the_product_of_its_legs() -> None:
    """A monostatic site 0.05 mm in front of a PEC wall: each leg keeps its LoS and reflection rows, and
    the round trip is the square of the one-way image factor ((2 sin k delta)^2 at normal incidence)."""

    compiled = _wall_scene(PhysicalMaterial.perfect_conductor())
    radar = (0.0, 0.6, 0.0)
    site = (WALL_X_M - NEAR_M, 0.6, 0.0)
    out_depth, out_ratio = _total_ratio(_discover(compiled, radar, site))
    back_depth, back_ratio = _total_ratio(_discover(compiled, site, radar))
    assert sorted(out_depth) == [0, 1] and sorted(back_depth) == [0, 1]
    one_way = _image_ratio(radar, site, -1.0)
    assert one_way == pytest.approx(2.0 * math.sin(WAVE_NUMBER * NEAR_M), rel=1.0e-3)
    assert out_ratio * back_ratio == pytest.approx(one_way**2, rel=4.0e-2)
