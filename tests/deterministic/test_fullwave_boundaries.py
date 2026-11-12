# Copyright Xingyu Chen.
# Tests the reflection support, chain and boundary-sample rules the full-wave references pinned.

"""Tests the reflection support, chain and boundary-sample rules the full-wave references pinned."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.fullwave_validation import reference_set as rs
from witwin.channel.deployment import build_info
from witwin.channel.deterministic import Config, solve
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
    result = solve(scene, Config(components=frozenset({"reflection"}), max_depth=1, return_field=True),
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
    result = solve(rs.build_channel_scene(case, plane), Config(
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
    result = solve(scene, Config(components=frozenset({"los", "reflection", "diffraction"}), max_depth=1,
                                 return_field=True), reference_frequency_hz=5.0e9)
    total = result.field.reshape(-1).cpu().numpy()
    reflected = result.component_fields["reflection"].reshape(-1).cpu().numpy()
    assert np.all(np.abs(reflected) > 0)
    scale = np.abs(reflected).max()
    np.testing.assert_array_less(np.abs(total[1:] - total[0]), 0.05 * scale)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1.0e-30)


def _in_plane(v: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    return _unit(v - (v * axis).sum(-1, keepdim=True) * axis)


def _rotate(v: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle)[:, None], torch.sin(angle)[:, None]
    return v * c + torch.cross(axis, v, dim=-1) * s + axis * (axis * v).sum(-1, keepdim=True) * (1.0 - c)


def _tilted_toward(n0: torch.Tensor, n1: torch.Tensor, axis: torch.Tensor, leg: torch.Tensor,
                   side: torch.Tensor, angle: float) -> tuple[torch.Tensor, torch.Tensor]:
    """A sheet's faces turned about its edge so the leg runs just off it on the side ``side`` points to."""

    turn = torch.full((int(n0.shape[0]),), angle, device=n0.device)
    # Turning the sheet by +angle moves the leg to the side of -(axis x leg); pick the sign that matches.
    sign = torch.where((torch.cross(axis, leg, dim=-1) * side).sum(-1) < 0.0, 1.0, -1.0)
    return _rotate(n0, axis, sign * turn), _rotate(n1, axis, sign * turn)


def test_a_double_diffraction_leg_along_a_sheet_runs_on_the_junction_side():
    """A D-D leg from a sheet's free edge along the sheet to a junction wedge is the field on the junction's side.

    In the trihedral corner the junction of two plates is a 270 degree wedge whose faces are the plates' outer
    sides, so a leg along a plate reaches it on the outer side; the free edge must evaluate the leg there too.
    The grazing rows equal the same rows with the sheet turned by 1e-4 rad so that the leg runs geometrically
    on that side (to 1e-3 of the largest), and differ from the rows turned the other way.
    """

    _require_native()
    from witwin.channel.kernels import fields as field_kernels

    case = rs.load_reference_case("trihedral_z")
    planes, _ = rs.load_reference(case)
    plane = dict(planes[0], x=np.linspace(0.05, 0.4, 6), y=np.linspace(0.05, 0.4, 6))
    calls: list[tuple[tuple, dict]] = []
    original = field_kernels.field_coupled_dd

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    field_kernels.field_coupled_dd = capture
    try:
        solve(rs.build_channel_scene(case, plane), Config(
            components=frozenset({"los", "reflection", "diffraction"}), max_depth=3, coupled_paths=True),
            reference_frequency_hz=case.frequency_hz)
    finally:
        field_kernels.field_coupled_dd = original
    assert calls, "the trihedral corner must have double-diffraction rows"
    args, kwargs = calls[0]
    q1, a1, n10, n11, q2, a2, n20, n21 = (args[i] for i in (2, 3, 4, 5, 7, 8, 9, 10))
    leg = q2 - q1
    a1u, a2u = _unit(a1), _unit(a2)
    d1, d2 = _in_plane(leg, a1u), _in_plane(-leg, a2u)
    tol = 1.0e-5
    on10, on11 = (d1 * n10).sum(-1).abs() < tol, (d1 * n11).sum(-1).abs() < tol
    on20, on21 = (d2 * n20).sum(-1).abs() < tol, (d2 * n21).sum(-1).abs() < tol
    # A free sheet edge (both faces hold the leg) leading to a junction face (exactly one face holds it).
    rows = on10 & on11 & (on20 ^ on21)
    assert int(rows.sum()) > 0, "the corner must have legs from a free plate edge along the plate to a junction"
    side = torch.where(on20[:, None], n20, n21)
    base = original(*args, **kwargs)["coefficient"][rows]

    def turned(direction: float) -> torch.Tensor:
        t0, t1 = _tilted_toward(n10, n11, a1u, leg, direction * side, 1.0e-4)
        changed = list(args)
        changed[4] = torch.where(rows[:, None], t0, n10).contiguous()
        changed[5] = torch.where(rows[:, None], t1, n11).contiguous()
        return original(*changed, **kwargs)["coefficient"][rows]

    outer, inner = turned(1.0), turned(-1.0)
    peak = float(outer.abs().max())
    relative = (base - outer).abs() / outer.abs().clamp_min(1.0e-12)
    assert float((base - outer).abs().max()) < 1.0e-3 * peak
    assert float(relative.median()) < 1.0e-3
    assert float(((inner - outer).abs() / outer.abs().clamp_min(1.0e-12)).median()) > 0.1
