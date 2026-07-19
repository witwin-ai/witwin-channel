import struct

import pytest
import torch

from witwin.channel_native import ReceiverPoint, Scene, Transmitter
from witwin.channel_native.path import Config as PathConfig
from witwin.channel_native.path import solve as solve_paths


def _write_ply(path, vertices, faces, *, uv_alias=None):
    properties = ["property float x", "property float y", "property float z"]
    if uv_alias is not None:
        properties.extend(
            [f"property float {uv_alias[0]}", f"property float {uv_alias[1]}"]
        )
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(vertices)}",
            *properties,
            f"element face {len(faces)}",
            "property list uchar int vertex_indices",
            "end_header",
            "",
        ]
    ).encode("ascii")
    payload = bytearray(header)
    for vertex in vertices:
        payload.extend(struct.pack("<" + "f" * len(vertex), *vertex))
    for face in faces:
        payload.extend(struct.pack("<Biii", 3, *face))
    path.write_bytes(payload)


def _scene_xml(shapes: str) -> str:
    return f"""<scene version="3.0.0">
  <bsdf type="diffuse" id="mat-concrete">
    <string name="type" value="concrete"/>
  </bsdf>
  {shapes}
</scene>
"""


def _ply_shape(filename="mesh.ply", *, shape_id="mesh-wall", transform=""):
    return f"""<shape type="ply" id="{shape_id}">
  <string name="filename" value="{filename}"/>
  <ref name="bsdf" id="mat-concrete"/>
  {transform}
</shape>"""


@pytest.mark.parametrize("uv_alias", [("u", "v"), ("s", "t"), ("texture_u", "texture_v")])
def test_binary_ply_vertex_uv_aliases_load_as_structure_uv(tmp_path, uv_alias):
    _write_ply(
        tmp_path / "mesh.ply",
        [(0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 0.0, 1.0)],
        [(0, 1, 2)],
        uv_alias=uv_alias,
    )
    xml = tmp_path / "scene.xml"
    xml.write_text(_scene_xml(_ply_shape()), encoding="utf-8")

    structure = Scene.load_mitsuba(str(xml), merge_shapes=False).structures[0]
    torch.testing.assert_close(
        structure.uv, torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )
    torch.testing.assert_close(
        structure.face_uv, torch.tensor([[0, 1, 2]], dtype=torch.int32)
    )


def test_to_world_operations_are_left_multiplied_in_xml_order(tmp_path):
    _write_ply(
        tmp_path / "mesh.ply",
        [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0)],
        [(0, 1, 2)],
    )
    transform = """<transform name="to_world">
  <translate x="1"/>
  <rotate z="1" angle="90"/>
  <scale value="2, 3, 4"/>
  <matrix value="1 0 0 0 0 1 0 0 0 0 1 5 0 0 0 1"/>
</transform>"""
    xml = tmp_path / "scene.xml"
    xml.write_text(_scene_xml(_ply_shape(transform=transform)), encoding="utf-8")

    vertices = Scene.load_mitsuba(str(xml), merge_shapes=False).structures[0].vertices
    torch.testing.assert_close(vertices[0], torch.tensor([0.0, 6.0, 5.0]), atol=1.0e-6, rtol=0.0)


def test_negative_determinant_reverses_face_winding(tmp_path):
    _write_ply(
        tmp_path / "mesh.ply",
        [(0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 0.0, 1.0)],
        [(0, 1, 2)],
        uv_alias=("u", "v"),
    )
    transform = '<transform name="to_world"><scale x="-1"/></transform>'
    xml = tmp_path / "scene.xml"
    xml.write_text(_scene_xml(_ply_shape(transform=transform)), encoding="utf-8")

    structure = Scene.load_mitsuba(str(xml), merge_shapes=False).structures[0]
    assert structure.faces.tolist() == [[0, 2, 1]]
    assert structure.face_uv.tolist() == [[0, 2, 1]]


def test_shapegroup_instances_expand_with_independent_transforms(tmp_path):
    _write_ply(
        tmp_path / "wall.ply",
        [(2.5, -3.0, -1.0), (2.5, 3.0, -1.0), (2.5, -3.0, 2.0), (2.5, 3.0, 2.0)],
        [(0, 1, 2), (1, 3, 2)],
    )
    shapes = f"""
<shape type="shapegroup" id="wall-group">
  {_ply_shape('wall.ply', shape_id='mesh-panel')}
</shape>
<shape type="instance" id="near"><ref id="wall-group"/></shape>
<shape type="instance" id="far">
  <ref id="wall-group"/>
  <transform name="to_world"><translate y="10"/></transform>
</shape>
"""
    xml = tmp_path / "scene.xml"
    xml.write_text(_scene_xml(shapes), encoding="utf-8")

    scene = Scene.load_mitsuba(str(xml), merge_shapes=False)
    assert len(scene.structures) == 2
    assert len({structure.name for structure in scene.structures}) == 2
    torch.testing.assert_close(
        scene.structures[1].vertices[:, 1], scene.structures[0].vertices[:, 1] + 10.0
    )


def test_instance_geometry_builds_edges_and_real_reflection_paths(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native edge/path acceptance")
    from witwin.channel_native.core.kernels.extension import build_info

    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native scene capability is not built")
    _write_ply(
        tmp_path / "wall.ply",
        [(2.5, -3.0, -1.0), (2.5, 3.0, -1.0), (2.5, -3.0, 2.0), (2.5, 3.0, 2.0)],
        [(0, 1, 2), (1, 3, 2)],
    )
    shapes = f"""<shape type="shapegroup" id="wall-group">
  {_ply_shape('wall.ply')}
</shape>
<shape type="instance" id="wall-instance"><ref id="wall-group"/></shape>"""
    xml = tmp_path / "scene.xml"
    xml.write_text(_scene_xml(shapes), encoding="utf-8")
    scene = Scene.load_mitsuba(str(xml), merge_shapes=False)
    scene.add(Transmitter(position=torch.tensor([0.0, -1.0, 0.5])))
    scene.add(ReceiverPoint(position=torch.tensor([0.0, 1.0, 0.5])))

    assert scene.diffraction_edge_count() > 0
    paths = solve_paths(scene, PathConfig(components={"reflection"}, max_depth=1))
    assert int(paths.valid.sum()) > 0
    assert torch.isfinite(paths.a[paths.valid]).all()


def test_merge_offsets_uv_indices_and_rejects_mixed_uv(tmp_path):
    vertices_uv = [(0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 0.0, 1.0)]
    _write_ply(tmp_path / "a.ply", vertices_uv, [(0, 1, 2)], uv_alias=("u", "v"))
    _write_ply(tmp_path / "b.ply", vertices_uv, [(0, 1, 2)], uv_alias=("u", "v"))
    shapes = "\n".join(
        _ply_shape(name, shape_id=f"mesh-{name[0]}").replace(
            "</shape>", '<boolean name="face_normals" value="true"/></shape>'
        )
        for name in ("a.ply", "b.ply")
    )
    xml = tmp_path / "scene.xml"
    xml.write_text(_scene_xml(shapes), encoding="utf-8")
    merged = Scene.load_mitsuba(str(xml), merge_shapes=True).structures[0]
    assert merged.uv.shape == (6, 2)
    assert merged.face_uv.tolist() == [[0, 1, 2], [3, 4, 5]]

    _write_ply(
        tmp_path / "b.ply",
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [(0, 1, 2)],
    )
    with pytest.raises(ValueError, match="mixed UV"):
        Scene.load_mitsuba(str(xml), merge_shapes=True)


@pytest.mark.parametrize(
    ("shapes", "message"),
    [
        ('<shape type="instance"><ref id="missing"/></shape>', "missing shapegroup"),
        ('<shape type="cube" id="unknown"/>', "does not support shape type"),
    ],
)
def test_invalid_instance_or_shape_fails_fast(tmp_path, shapes, message):
    xml = tmp_path / "scene.xml"
    xml.write_text(_scene_xml(shapes), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        Scene.load_mitsuba(str(xml), merge_shapes=False)


def test_shapegroup_cycle_fails_fast(tmp_path):
    shapes = """
<shape type="shapegroup" id="a"><shape type="instance"><ref id="b"/></shape></shape>
<shape type="shapegroup" id="b"><shape type="instance"><ref id="a"/></shape></shape>
<shape type="instance"><ref id="a"/></shape>
"""
    xml = tmp_path / "scene.xml"
    xml.write_text(_scene_xml(shapes), encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        Scene.load_mitsuba(str(xml), merge_shapes=False)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ('<scale x="0"/>', "singular"),
        ('<translate x="nan"/>', "finite"),
        ('<skew x="1"/>', "not supported"),
    ],
)
def test_invalid_transform_fails_fast(tmp_path, operation, message):
    _write_ply(
        tmp_path / "mesh.ply",
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [(0, 1, 2)],
    )
    transform = f'<transform name="to_world">{operation}</transform>'
    xml = tmp_path / "scene.xml"
    xml.write_text(_scene_xml(_ply_shape(transform=transform)), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        Scene.load_mitsuba(str(xml), merge_shapes=False)
