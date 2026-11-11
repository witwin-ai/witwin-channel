# Copyright Xingyu Chen.
# Tests the reflection support, chain and boundary-sample rules the full-wave references pinned.

"""Tests the reflection support, chain and boundary-sample rules the full-wave references pinned."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from witwin.channel.deployment import build_info
from witwin.channel.deterministic import Config, solve
from witwin.core import AntennaState, Mesh, PhysicalMaterial, Scene, Structure
from witwin.core.identity import new_antenna_id


def _require_native() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic solves")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native capability is not built")


def _plate_scene(
    half_width: float, material: PhysicalMaterial, tx: tuple[float, float, float],
    receivers: list[tuple[float, float, float]],
) -> Scene:
    """A plate in ``x = 0`` with ``|y| <= half_width`` and ``z <= 0``: one edge along y at z = 0."""

    w = float(half_width)
    vertices = torch.tensor([[0.0, -w, -2.0 * w], [0.0, w, -2.0 * w], [0.0, w, 0.0], [0.0, -w, 0.0]])
    mesh = Mesh(vertices, torch.tensor([[0, 2, 1], [0, 3, 2]], dtype=torch.int32), recenter=False,
                fill_mode="surface", topology_diagnostics=False)
    polarization = torch.tensor([0.0, 0.0, 1.0])
    endpoints = [AntennaState(new_antenna_id(), "tx", torch.tensor(tx), polarization=polarization, power_w=1.0)]
    endpoints += [AntennaState(new_antenna_id(), "rx", torch.tensor(point), polarization=polarization)
                  for point in receivers]
    return Scene(structures=[Structure(mesh, material, name="plate", surface_id=1)], endpoints=endpoints)


def test_reflection_support_ends_at_the_edge_of_a_large_face():
    """The in-face test is a distance, so a 40 m face's reflection ends at its edge, not millimetres past it."""

    _require_native()
    # Transmitter (-1, 0, 0.5): the reflection shadow boundary of the edge crosses x = -1 at z = -0.5,
    # and a receiver offset dz there moves the reflection point by dz / 2 from the edge.
    offsets = np.arange(-3.0e-3, 3.25e-3, 5.0e-4)
    offsets = offsets[np.abs(offsets) > 1.0e-6]
    receivers = [(-1.0, 0.0, -0.5 + float(dz)) for dz in offsets]
    scene = _plate_scene(20.0, PhysicalMaterial.perfect_conductor(name="pec"), (-1.0, 0.0, 0.5), receivers)
    result = solve(scene, Config(components=frozenset({"reflection"}), max_depth=1, return_field=True),
                   reference_frequency_hz=5.0e9)
    reflected = result.component_fields["reflection"].reshape(-1).abs().cpu().numpy() > 0
    np.testing.assert_array_equal(reflected, offsets < 0.0)
