# Copyright Xingyu Chen.
# Tests the reflection support, chain and boundary-sample rules the full-wave references pinned.

"""Tests the reflection support, chain and boundary-sample rules the full-wave references pinned."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.fullwave_validation import reference_set as rs
from witwin.channel.deployment import build_info
from witwin.channel.radiomap import Coherent, solve
from witwin.core import AntennaState, Mesh, PhysicalMaterial, Scene, Structure
from witwin.core.identity import new_antenna_id

_CHAIN_COMPONENTS = (3, 4)
_CHAIN_INTERIOR_BASE = 8


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
    result = solve(scene, Coherent(components=frozenset({"reflection"}), max_depth=1, return_field=True),
                   reference_frequency_hz=5.0e9)
    reflected = result.component_fields["reflection"].reshape(-1).abs().cpu().numpy() > 0
    np.testing.assert_array_equal(reflected, offsets < 0.0)


def _chain_rows(case_id: str, max_depth: int) -> tuple[object, np.ndarray]:
    case = rs.load_reference_case(case_id)
    planes, _ = rs.load_reference(case)
    plane = dict(planes[0])
    # Every fourth sample of the stored plane keeps the solve small.
    plane["x"] = plane["x"][::4]
    plane["y"] = plane["y"][::4]
    plane["field"] = plane["field"][::4, ::4]
    plane["regions"] = plane["regions"][::4, ::4]
    result = solve(rs.build_channel_scene(case, plane), Coherent(
        components=frozenset({"los", "reflection", "diffraction"}), max_depth=max_depth,
        coupled_paths=True, export_paths=True, return_field=True), reference_frequency_hz=case.frequency_hz)
    paths = result.paths
    assert paths is not None
    component = paths.component_id.cpu().numpy()
    chain = paths.valid.cpu().numpy() & (np.isin(component, _CHAIN_COMPONENTS) |
                                         (component >= _CHAIN_INTERIOR_BASE))
    return paths, chain


def test_chain_legs_leave_the_diffraction_edge_into_open_space():
    """No chain leg crosses a closed cube behind its edge to reach a face from inside."""

    _require_native()
    paths, chain = _chain_rows("three_cube_z", 2)
    assert chain.any()
    case = rs.load_reference_case("three_cube_z")
    boxes = [(np.asarray(b.center), 0.5 * np.asarray(b.size) - 1.0e-3) for b in case.blocks]
    depth = paths.depth.cpu().numpy()
    component = paths.component_id.cpu().numpy()
    positions = paths.interaction_positions.cpu().numpy()
    crossing = 0
    for row in np.flatnonzero(chain):
        d = int(depth[row])
        slot = d - 1 if component[row] == 3 else (0 if component[row] == 4 else component[row] - 8)
        for neighbour in (slot - 1, slot + 1):
            if 0 <= neighbour < d:
                midpoint = 0.5 * (positions[row, slot] + positions[row, neighbour])
                crossing += any(bool(np.all(np.abs(midpoint - c) < h)) for c, h in boxes)
    assert crossing == 0


def test_chains_carry_as_many_reflections_as_the_reflection_paths():
    """The chain that keeps a ``max_depth``-reflection path continuous where it is blocked is enumerated."""

    _require_native()
    paths, chain = _chain_rows("wall_column_z", 2)
    depth = paths.depth.cpu().numpy()
    assert int(depth[chain].max()) == 3


@pytest.mark.parametrize("material", ["pec", "concrete"])
def test_receiver_on_a_reflection_boundary_keeps_the_lit_total(material):
    """On the boundary the reflection is kept and the diffraction takes its lit side: no half-R error."""

    from witwin.core import MaterialLayer

    _require_native()
    face = (PhysicalMaterial.perfect_conductor(name="pec") if material == "pec" else
            PhysicalMaterial(layers=(MaterialLayer(thickness_m=0.1, eps_r=5.24, sigma_e=0.1627),), name="c"))
    # Transmitter (-1, 0, 0.5): the face's reflection boundary crosses x = -1 at z = -0.5 exactly.
    offsets = (-1.0e-4, -1.0e-6, 0.0, 1.0e-6)
    receivers = [(-1.0, 0.0, -0.5 + dz) for dz in offsets]
    scene = _plate_scene(5.0, face, (-1.0, 0.0, 0.5), receivers)
    result = solve(scene, Coherent(components=frozenset({"los", "reflection", "diffraction"}), max_depth=1,
                                 return_field=True), reference_frequency_hz=5.0e9)
    total = result.field.reshape(-1).cpu().numpy()
    reflected = result.component_fields["reflection"].reshape(-1).cpu().numpy()
    assert np.all(np.abs(reflected) > 0)
    scale = np.abs(reflected).max()
    np.testing.assert_array_less(np.abs(total[1:] - total[0]), 0.05 * scale)
