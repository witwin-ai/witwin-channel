# Copyright Xingyu Chen.
# The diffraction edge set is the set of geometric wedges, whatever the triangulation.

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


def test_sheet_edge_resting_on_a_surface_is_not_a_half_plane():
    """A wall sheet standing on a ground plate keeps its free top edge and loses its foot."""
    wall = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 0.0, 3.0], [0.0, 0.0, 3.0]]
    ground = [[-5.0, -5.0, 0.0], [9.0, -5.0, 0.0], [9.0, 5.0, 0.0], [-5.0, 5.0, 0.0]]
    rows = _edges([_structure(wall, [[0, 1, 2], [0, 2, 3]]), _structure(ground, [[0, 1, 2], [0, 2, 3]])])
    along_x = (rows["edge_dir"][:, 0].abs() > 0.99) & (rows["edge_pos"][:, 1].abs() < 1e-3)
    heights = sorted(round(float(z), 3) for z in rows["edge_pos"][along_x][:, 2])
    assert heights == [3.0]


def _box(lo, hi) -> tuple[list, list]:
    """A closed box with outward-wound triangles."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    vertices = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
             [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return vertices, faces


def test_block_base_on_the_ground_is_not_a_wedge():
    """A block standing on a ground block meets it in concave corners: its base edges do not diffract."""
    block = _structure(*_box((0.0, 0.0, 0.0), (2.0, 2.0, 3.0)))
    ground = _structure(*_box((-10.0, -10.0, -5.0), (10.0, 10.0, 0.0)))
    rows = _edges([block, ground])
    on_block = (rows["edge_pos"][:, :2] >= -1e-3).all(dim=1) & (rows["edge_pos"][:, :2] <= 2.001).all(dim=1)
    heights = sorted({round(float(z), 3) for z in rows["edge_pos"][on_block][:, 2]})
    # Four vertical corners (z = 1.5) and four roof edges (z = 3); no base edge at z = 0.
    assert heights == [1.5, 3.0]
    assert int(on_block.sum()) == 8


def test_sheets_meeting_at_their_edges_form_one_wedge():
    """A ground plate and a wall plate joined along their edges are one 270 degree wedge, faces turned out."""
    ground = [[-1.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
    wall = [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0]]
    for winding in ([[0, 1, 2], [0, 2, 3]], [[0, 2, 1], [0, 3, 2]]):
        rows = _edges([_structure(ground, winding), _structure(wall, winding)])
        junction = (rows["edge_pos"][:, 0].abs() < 1e-3) & (rows["edge_pos"][:, 2].abs() < 1e-3)
        assert int(junction.sum()) == 1
        angle = float(rows["exterior_angle"][junction][0])
        assert math.isclose(angle, 1.5 * math.pi, rel_tol=1e-5)
        normals = {tuple(round(float(v), 3) for v in rows[name][junction][0]) for name in ("n0", "n1")}
        # The free sector is below the ground plate and in front of (+x) the wall.
        assert normals == {(0.0, 0.0, -1.0), (1.0, 0.0, 0.0)}
        face_a, face_b = int(rows["face0"][junction][0]), int(rows["face1"][junction][0])
        # One face of each structure (ground triangles 0-1, wall triangles 2-3).
        assert {face_a // 2, face_b // 2} == {0, 1}
        # n0 x dir runs along face 0, away from the edge.
        into = torch.linalg.cross(rows["n0"][junction], rows["edge_dir"][junction])[0]
        centroid = rows["vertices"][rows["faces"][face_a].long()].mean(dim=0)
        assert float((into * (centroid - rows["edge_pos"][junction][0])).sum()) > 0.0
