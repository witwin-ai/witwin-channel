# Copyright Xingyu Chen.
# Tests the Mitsuba XML, PLY, OBJ and glTF scene importers on authored fixtures.

from __future__ import annotations

import json
from pathlib import Path
import struct
import zlib

import numpy as np
import pytest

from witwin.core import Mesh, PhysicalMaterial, Scene, Structure
from witwin.channel.importers import (
    MeshData,
    build_structures,
    load_mesh,
    load_mitsuba,
    read_gltf,
    read_obj,
    read_ply,
    weld_vertices,
)
from witwin.channel.presets import ItuBandDispersion, itu_properties

DATA = Path(__file__).resolve().parents[1] / "data" / "scenes"
QUAD_VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])


def _binary_ply(path: Path, *, endian: str, quad: bool) -> None:
    fmt = "binary_little_endian" if endian == "<" else "binary_big_endian"
    header = (
        f"ply\nformat {fmt} 1.0\nelement vertex 4\nproperty float x\nproperty float y\n"
        "property float z\nproperty uchar red\n"
        f"element face {1 if quad else 2}\nproperty list uchar int vertex_indices\n"
        "property uchar flags\nend_header\n"
    )
    body = b"".join(struct.pack(f"{endian}fffB", *row, 7) for row in QUAD_VERTICES)
    faces = [(0, 1, 2, 3)] if quad else [(0, 1, 2), (0, 2, 3)]
    for face in faces:
        body += struct.pack(f"{endian}B{len(face)}iB", len(face), *face, 1)
    path.write_bytes(header.encode("ascii") + body)


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("quad", [False, True])
def test_binary_ply_matches_reference_quad(tmp_path: Path, endian: str, quad: bool) -> None:
    path = tmp_path / "quad.ply"
    _binary_ply(path, endian=endian, quad=quad)

    mesh = read_ply(path)

    np.testing.assert_allclose(mesh.vertices, QUAD_VERTICES)
    np.testing.assert_array_equal(mesh.faces, [[0, 1, 2], [0, 2, 3]])


def test_ascii_ply_fixture_reads_two_unshared_triangles() -> None:
    mesh = read_ply(DATA / "meshes" / "wall.ply")

    assert mesh.vertices.shape == (6, 3)
    np.testing.assert_array_equal(mesh.faces, [[0, 1, 2], [3, 4, 5]])


def test_weld_merges_duplicates_and_drops_degenerate_faces() -> None:
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 1e-9], [2, 2, 2]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5], [0, 3, 1]])

    welded, new_faces, merged, removed = weld_vertices(vertices, faces, tolerance=1e-6)

    assert merged == 2  # vertices 3 and 4 duplicate vertices 0 and 2
    assert removed == 1  # [0, 3, 1] collapses to a repeated index
    assert welded.shape == (4, 3)  # the unreferenced (2, 2, 2) vertex is dropped
    assert new_faces.shape == (2, 3)
    edges = {tuple(sorted(edge)) for face in new_faces for edge in ((face[0], face[1]),
             (face[1], face[2]), (face[0], face[2]))}
    assert len(edges) == 5  # the diagonal is now one shared edge


# OBJ text lives here because the repository forbids committed *.obj files.
KIOSK_OBJ = """# Closed 2 m x 1 m x 3 m box with quad faces and two materials.
o kiosk
v -1 -0.5 0
v 1 -0.5 0
v 1 0.5 0
v -1 0.5 0
v -1 -0.5 3
v 1 -0.5 3
v 1 0.5 3
v -1 0.5 3
usemtl walls
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f -5 -8 -4 -1
usemtl roof
f 5/1/1 6/2/1 7/3/1 8/4/1
f 4 3 2 1
"""


def _kiosk_obj(tmp_path: Path) -> Path:
    path = tmp_path / "kiosk.obj"
    path.write_text(KIOSK_OBJ)
    return path


def test_ascii_ply_quads_are_fan_triangulated() -> None:
    mesh = read_ply(DATA / "meshes" / "kiosk.ply")

    assert mesh.faces.shape == (12, 3)
    np.testing.assert_array_equal(mesh.faces[:2], [[0, 1, 5], [0, 5, 4]])


def test_obj_splits_by_usemtl_and_triangulates_quads(tmp_path: Path) -> None:
    meshes = read_obj(_kiosk_obj(tmp_path))

    assert [mesh.material for mesh in meshes] == ["walls", "roof"]
    assert [mesh.faces.shape[0] for mesh in meshes] == [8, 4]


def test_obj_groups_keep_only_their_own_vertices(tmp_path: Path) -> None:
    path = tmp_path / "two.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 5 0 0\nv 6 0 0\nv 5 1 0\n"
        "usemtl a\nf 1 2 3\nusemtl b\nf 4 5 6\n"
    )

    meshes = read_obj(path)
    imported = load_mesh(path, PhysicalMaterial())

    assert [mesh.vertices.shape[0] for mesh in meshes] == [3, 3]
    np.testing.assert_allclose(meshes[1].vertices[0], [5, 0, 0])
    assert imported.welded_vertex_count == 0


def test_load_mesh_maps_obj_materials_and_builds_core_structures(tmp_path: Path) -> None:
    walls = PhysicalMaterial(eps_r=5.0, name="walls")
    roof = PhysicalMaterial(eps_r=3.0, name="roof")

    path = _kiosk_obj(tmp_path)
    imported = load_mesh(path, {"walls": walls, "roof": roof})

    assert len(imported.structures) == 2
    assert all(isinstance(structure, Structure) for structure in imported.structures)
    assert all(isinstance(structure.geometry, Mesh) for structure in imported.structures)
    assert {structure.material.name for structure in imported.structures} == {"walls", "roof"}
    assert imported.face_count == 12
    scene = imported.to_scene()
    assert isinstance(scene, Scene)
    with pytest.raises(ValueError, match="no material for mesh material 'roof'"):
        load_mesh(path, {"walls": walls})


def _glb(path: Path) -> None:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.uint16)
    binary = positions.tobytes() + indices.tobytes()
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "parent", "translation": [10.0, 0.0, 0.0], "children": [1]},
            {"name": "child", "mesh": 0, "scale": [2.0, 2.0, 2.0],
             "rotation": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]},
        ],
        "meshes": [{"primitives": [
            {"attributes": {"POSITION": 0}, "indices": 1, "mode": 5, "material": 0},
            {"attributes": {"POSITION": 0}, "mode": 1},
        ]}],
        "materials": [{"name": "facade"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 4, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48},
            {"buffer": 0, "byteOffset": 48, "byteLength": 8},
        ],
        "buffers": [{"byteLength": len(binary)}],
    }
    text = json.dumps(document).encode("utf-8")
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    chunks = (
        struct.pack("<II", len(text), 0x4E4F534A) + text
        + struct.pack("<II", len(binary), 0x004E4942) + binary
    )
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)


def test_glb_applies_node_hierarchy_strips_and_up_axis(tmp_path: Path) -> None:
    path = tmp_path / "scene.glb"
    _glb(path)

    raw, warnings = read_gltf(path, up_axis="y")
    converted, _ = read_gltf(path)

    assert len(raw) == 1 and raw[0].material == "facade"
    assert any("non-triangle mode 1" in warning for warning in warnings)
    np.testing.assert_array_equal(raw[0].faces, [[0, 1, 2], [2, 1, 3]])
    # child: rotate +90 deg about z, scale 2; parent: translate x by 10.
    np.testing.assert_allclose(
        raw[0].vertices, [[10, 0, 0], [10, 2, 0], [8, 0, 0], [8, 2, 0]], atol=1e-6
    )
    # +Y-up to +Z-up maps (x, y, z) to (x, -z, y).
    np.testing.assert_allclose(
        converted[0].vertices, raw[0].vertices[:, [0, 2, 1]] * [1, -1, 1], atol=1e-6
    )


def test_gltf_rejects_required_extensions(tmp_path: Path) -> None:
    path = tmp_path / "draco.gltf"
    path.write_text(json.dumps({"asset": {"version": "2.0"},
                                "extensionsRequired": ["KHR_draco_mesh_compression"]}))

    with pytest.raises(ValueError, match="KHR_draco_mesh_compression"):
        read_gltf(path)


def _mesh(imported, name):
    return next(structure for structure in imported.structures if structure.name == name)


def test_mitsuba_fixture_shapes_transforms_and_materials() -> None:
    imported = load_mitsuba(DATA / "canyon.xml", frequency_hz=3.5e9, merge_shapes=False)

    assert [structure.name for structure in imported.structures] == [
        "ground", "building", "wall", "kiosk"
    ]
    ground = _mesh(imported, "ground").geometry.vertices.numpy()
    np.testing.assert_allclose(ground.min(0), [-50, -50, 0])
    np.testing.assert_allclose(ground.max(0), [50, 50, 0])
    building = _mesh(imported, "building").geometry.vertices.numpy()
    np.testing.assert_allclose(building.min(0), [15, -10, 0])
    np.testing.assert_allclose(building.max(0), [25, 10, 20])
    kiosk = _mesh(imported, "kiosk").geometry.vertices.numpy()
    np.testing.assert_allclose(kiosk.min(0), [-20.5, -1, 0], atol=1e-6)
    np.testing.assert_allclose(kiosk.max(0), [-19.5, 1, 3], atol=1e-6)

    wall_report = next(entry for entry in imported.report if entry.name == "wall")
    assert (wall_report.input_vertices, wall_report.welded_vertices) == (6, 2)
    assert imported.welded_vertex_count >= 2

    concrete = imported.materials["concrete"]
    assert isinstance(concrete.dispersion, ItuBandDispersion)
    assert concrete.thickness_m == pytest.approx(0.2)
    sample = concrete.evaluate_at_frequency(3.5e9)
    eps_r, sigma = itu_properties("concrete", 3.5e9)
    assert sample.eps_r.real == pytest.approx(eps_r)
    assert sample.sigma_e == pytest.approx(sigma)
    custom = imported.materials["custom"]
    assert (custom.eps_r, custom.sigma_e, custom.thickness_m) == (4.5, 0.02, 0.05)
    assert custom.scattering_coefficient == pytest.approx(0.3)
    assert imported.materials["glass"].name == "glass"
    assert imported.materials["mat-itu_wood"].dispersion.material == "wood"


def test_mitsuba_merges_by_material_and_honours_overrides() -> None:
    metal = PhysicalMaterial.perfect_conductor()

    imported = load_mitsuba(DATA / "canyon.xml", frequency_hz=3.5e9, materials={"glass": metal})

    assert len(imported.structures) == 4
    assert imported.materials["glass"] is metal
    assert imported.face_count == 2 + 12 + 2 + 12


def test_mitsuba_rejects_out_of_range_frequency_and_unknown_bsdf(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="brick|glass|concrete|wood"):
        load_mitsuba(DATA / "canyon.xml", frequency_hz=150e9)
    xml = tmp_path / "scene.xml"
    xml.write_text(
        '<scene version="3.0.0"><bsdf type="diffuse" id="paint"/>'
        '<shape type="rectangle"><ref id="paint"/></shape></scene>'
    )
    with pytest.raises(ValueError, match="'paint'.*not a radio material"):
        load_mitsuba(xml, frequency_hz=3.5e9)
    imported = load_mitsuba(xml, frequency_hz=3.5e9, materials={"paint": PhysicalMaterial()})
    assert imported.face_count == 2


def _serialized(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    stream = struct.pack("<I", 0x1000) + b"part\0"
    stream += struct.pack("<QQ", vertices.shape[0], faces.shape[0])
    stream += vertices.astype("<f4").tobytes() + faces.astype("<u4").tobytes()
    blob = struct.pack("<HH", 0x041C, 4) + zlib.compress(stream)
    path.write_bytes(blob + struct.pack("<QI", 0, 1))


def test_mitsuba_serialized_shape(tmp_path: Path) -> None:
    _serialized(tmp_path / "quad.serialized", QUAD_VERTICES, np.array([[0, 1, 2], [0, 2, 3]]))
    xml = tmp_path / "scene.xml"
    xml.write_text(
        '<scene version="3.0.0"><bsdf type="radio-material" id="m"/>'
        '<shape type="serialized"><string name="filename" value="quad.serialized"/>'
        '<transform name="to_world"><translate x="1"/></transform><ref id="m"/></shape>'
        "</scene>"
    )

    imported = load_mitsuba(xml, frequency_hz=3.5e9)

    vertices = imported.structures[0].geometry.vertices.numpy()
    np.testing.assert_allclose(vertices.min(0), [1, 0, 0])
    np.testing.assert_allclose(vertices.max(0), [2, 1, 0])


def test_mitsuba_obj_shape(tmp_path: Path) -> None:
    _kiosk_obj(tmp_path)
    xml = tmp_path / "scene.xml"
    xml.write_text(
        '<scene version="3.0.0"><bsdf type="radio-material" id="m"/>'
        '<shape type="obj"><string name="filename" value="kiosk.obj"/>'
        '<transform name="to_world"><scale z="2"/></transform><ref id="m"/></shape>'
        "</scene>"
    )

    imported = load_mitsuba(xml, frequency_hz=3.5e9)

    assert imported.face_count == 12
    assert float(imported.structures[0].geometry.vertices[:, 2].max()) == 6.0


def test_mirroring_transform_keeps_outward_winding() -> None:
    mesh = MeshData("tri", QUAD_VERTICES[:3], np.array([[0, 1, 2]]))
    mirrored = mesh.transformed(np.diag([-1.0, 1.0, 1.0, 1.0]))

    corners = mirrored.vertices[mirrored.faces[0]]
    normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    assert normal[2] > 0.0


def test_build_structures_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no triangles"):
        build_structures([], PhysicalMaterial())
