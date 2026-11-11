# Copyright Xingyu Chen.
# The diffraction edge set is the set of geometric wedges, independent of how the mesh is triangulated.

from __future__ import annotations

import math

import pytest
import torch
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure

from benchmarks import synthetic_city as sc
from tests.accuracy.scenes import antenna

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="edge geometry needs CUDA")


def _edges(structures: list[Structure]) -> dict[str, torch.Tensor]:
    from witwin.channel import scene as channel_scene
    from witwin.channel.propagation.geometry import cached_diffraction_edge_geometry

    scene = Scene(structures=structures, endpoints=[antenna("tx", (0.0, 0.0, 50.0))])
    compiled = channel_scene.compile(scene, reference_frequency_hz=3.5e9)
    geometry = cached_diffraction_edge_geometry(compiled.rayd)
    names = ("selected", "edge_pos", "edge_dir", "lengths", "line_min", "line_max", "n0", "n1",
             "face0", "face1", "exterior_angle")
    selected = geometry[0]
    rows = {name: value[selected].cpu() for name, value in zip(names[1:], geometry[1:])}
    rows["faces"] = compiled.rayd.edge_records().faces.cpu()
    rows["vertices"] = compiled.rayd.edge_records().vertices.cpu()
    return rows


def _structure(vertices: list, faces: list, material=None) -> Structure:
    mesh = Mesh(torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int32),
                recenter=False, fill_mode="surface", topology_diagnostics=False)
    return Structure(mesh, material or PhysicalMaterial.perfect_conductor())


def _segments(rows: dict[str, torch.Tensor]) -> list[tuple]:
    """Each wedge as (sorted end points rounded to 1 mm, exterior angle rounded to 1e-4)."""
    out = []
    for pos, direction, low, high, angle in zip(rows["edge_pos"], rows["edge_dir"], rows["line_min"],
                                                rows["line_max"], rows["exterior_angle"]):
        a = tuple(round(float(v), 3) for v in pos + direction * low)
        b = tuple(round(float(v), 3) for v in pos + direction * high)
        out.append((min(a, b), max(a, b), round(float(angle), 4)))
    return sorted(out)


@pytest.mark.parametrize("subdiv,welded", [(1, True), (2, True), (1, False)])
def test_subdivided_city_has_the_same_wedges(subdiv, welded):
    base = _segments(_edges(sc.city(2).structures))
    other = _segments(_edges(sc.city(2, subdiv=subdiv, welded=welded).structures))
    assert other == base


def test_edge_points_into_face_zero_for_both_windings():
    """``n0 x dir`` is the in-face direction of face 0, whichever way the screen is wound."""
    e = 50.0
    corners = [[0.0, 0.0, -e], [e, 0.0, -e], [e, 0.0, e], [0.0, 0.0, e]]
    for faces in ([[0, 2, 1], [0, 3, 2]], [[0, 1, 2], [0, 2, 3]]):
        rows = _edges([_structure(corners, faces)])
        on_axis = rows["edge_pos"][:, 0].abs() < 1e-3
        assert int(on_axis.sum()) == 1
        into = torch.linalg.cross(rows["n0"][on_axis], rows["edge_dir"][on_axis])[0]
        assert float(into[0]) > 0.99  # the screen occupies x >= 0
        assert int(rows["face1"][on_axis][0]) == int(rows["face0"][on_axis][0])
        assert math.isclose(float(rows["exterior_angle"][on_axis][0]), 2.0 * math.pi, rel_tol=1e-6)


def test_concave_edge_is_not_a_wedge():
    """An L-shaped closed prism has one reflex (concave) vertical edge; it is not selected."""
    # Floor plan: (0,0) (2,0) (2,1) (1,1) (1,2) (0,2); the concave corner is (1,1).
    plan = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    h = 3.0
    vertices = [[x, y, 0.0] for x, y in plan] + [[x, y, h] for x, y in plan]
    faces = []
    n = len(plan)
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i]]
    # Caps (fan triangulation of the L, which is star-shaped from vertex 0 and from (1,1)).
    bottom = [[0, 3, 1], [1, 3, 2], [0, 4, 3], [0, 5, 4]]
    faces += bottom + [[n + a, n + c, n + b] for a, b, c in bottom]
    rows = _edges([_structure(vertices, faces)])
    vertical = rows["edge_dir"][:, 2].abs() > 0.99
    xy = rows["edge_pos"][vertical][:, :2]
    corners = {(round(float(x), 3), round(float(y), 3)) for x, y in xy}
    assert (1.0, 1.0) not in corners
    assert corners == {(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 2.0), (0.0, 2.0)}
