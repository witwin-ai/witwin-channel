"""Regression coverage for loading Sionna/Mitsuba XML scenes into channel_scene."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from witwin.channel.core.scene import Scene


SIONNA_SOURCE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "reference"
    / "sionna-rt-reference-2.0.1"
    / "src"
)


def _write_tiny_mitsuba_scene(tmp_path: Path) -> Path:
    mesh_path = tmp_path / "wall.ply"
    mesh_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "element face 2",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "1 0 1",
                "0 0 1",
                "3 0 1 2",
                "3 0 2 3",
            ]
        ),
        encoding="utf-8",
    )
    xml_path = tmp_path / "tiny_scene.xml"
    xml_path.write_text(
        "\n".join(
            [
                '<scene version="2.1.0">',
                '  <bsdf type="itu-radio-material" id="itu_concrete">',
                '    <string name="type" value="concrete"/>',
                '    <float name="thickness" value="0.1"/>',
                "  </bsdf>",
                '  <shape type="ply" id="mesh-test-wall">',
                '    <string name="filename" value="wall.ply"/>',
                '    <ref id="itu_concrete" name="bsdf"/>',
                "  </shape>",
                "</scene>",
            ]
        ),
        encoding="utf-8",
    )
    return xml_path


def test_channel_scene_load_mitsuba_imports_xml_mesh_and_material(tmp_path):
    xml_path = _write_tiny_mitsuba_scene(tmp_path)

    scene = Scene.load_mitsuba(
        xml_path,
        device="cpu",
        merge_shapes=False,
        source_root=SIONNA_SOURCE_ROOT,
    )

    assert scene.device == "cpu"
    assert scene.metadata["mitsuba"]["source_path"] == str(xml_path.resolve())
    assert scene.metadata["mitsuba"]["loader"] == "sionna.rt.load_scene"
    assert scene.metadata["mitsuba"]["merge_shapes"] is False
    assert "edge_diffraction" not in scene.metadata["mitsuba"]
    assert "boundary_edge_policy" not in scene.metadata["mitsuba"]
    assert len(scene.structures) == 1
    assert scene.structures[0].name == "test-wall"
    assert scene.structures[0].metadata["sionna"]["radio_material_name"] == "itu_concrete"
    assert scene.tri_data["n_triangles"] == 2

    material = scene.structures[0].material.evaluate_static()
    assert float(material.eps_r) > 1.0
    assert float(material.sigma_e) >= 0.0

    vertices = np.asarray(scene.vertices.x, dtype=np.float32)
    assert vertices.shape == (4,)


def test_channel_scene_load_mitsuba_rejects_conflicting_import_edge_policy(tmp_path):
    xml_path = _write_tiny_mitsuba_scene(tmp_path)

    with pytest.raises(ValueError, match="edge_diffraction=True.*boundary_edge_policy='exclude'"):
        Scene.load_mitsuba(
            xml_path,
            device="cpu",
            merge_shapes=False,
            source_root=SIONNA_SOURCE_ROOT,
            edge_diffraction=True,
            boundary_edge_policy="exclude",
        )
