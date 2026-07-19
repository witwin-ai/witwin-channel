from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from witwin.channel_native import Scene
from witwin.channel_native.core.edge_policy import EdgePolicy


_MUNICH_XML = (
    Path("E:/Code/witwin-platform/channel/reference/sionna-rt-reference-2.0.1/src")
    / "sionna"
    / "rt"
    / "scenes"
    / "munich"
    / "munich.xml"
)
_SIONNA_ROOT = Path("E:/Code/witwin-platform/channel/reference/sionna-rt-reference-2.0.1/src")


def test_edge_diffraction_is_explicitly_enabled_by_default():
    policy = EdgePolicy()

    assert policy.edge_diffraction is True
    assert policy.boundary_edge_policy == "half_plane"


def test_munich_load_mitsuba_matches_original_scene_counts():
    if not _MUNICH_XML.exists():
        pytest.skip("Munich reference scene is not available")

    scene = Scene.load_mitsuba(
        _MUNICH_XML,
        source_root=_SIONNA_ROOT,
        merge_shapes=True,
        frequency=2.4e9,
        edge_selection_mode="all_edges",
        boundary_edge_policy="half_plane",
    )

    assert len(scene.structures) == 11
    assert sum(int(structure.faces.shape[0]) for structure in scene.structures) == 38936
    assert scene.frequency == 2.4e9
    assert scene.metadata["mitsuba"]["source_path"] == str(_MUNICH_XML.resolve())
    assert scene.metadata["sionna_import_edge_policy"].edge_diffraction is True
    assert scene.diffraction_edge_count(
        EdgePolicy(edge_selection_mode="all_edges", boundary_edge_policy="half_plane")
    ) == 51650


def test_munich_load_mitsuba_does_not_import_drjit_or_python_raytracers():
    if not _MUNICH_XML.exists():
        pytest.skip("Munich reference scene is not available")

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root / "src"), str(repo_root), env.get("PYTHONPATH", "")]
    )
    script = f"""
import json
import sys
from pathlib import Path

from tests.support.native_ext import inject_native_paths

inject_native_paths()

from witwin.channel_native import Scene
from witwin.channel_native.core.edge_policy import EdgePolicy

scene = Scene.load_mitsuba(
    Path({str(_MUNICH_XML)!r}),
    source_root=Path({str(_SIONNA_ROOT)!r}),
    merge_shapes=True,
    frequency=2.4e9,
    edge_selection_mode="all_edges",
    boundary_edge_policy="half_plane",
)
print(json.dumps({{
    "bad_modules": {{name: name in sys.modules for name in ("drjit", "mitsuba", "sionna", "rayd")}},
    "structures": len(scene.structures),
    "faces": sum(int(structure.faces.shape[0]) for structure in scene.structures),
    "edges": scene.diffraction_edge_count(
        EdgePolicy(edge_selection_mode="all_edges", boundary_edge_policy="half_plane")
    ),
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["bad_modules"] == {
        "drjit": False,
        "mitsuba": False,
        "sionna": False,
        "rayd": False,
    }
    assert payload["structures"] == 11
    assert payload["faces"] == 38936
    assert payload["edges"] == 51650
