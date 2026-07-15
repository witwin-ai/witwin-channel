from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from witwin.channel_native.core import path_topology as legacy
from witwin.channel_native.propagation.geometry import reevaluate


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

_REEVALUATE_NAMES = (
    "_reflect_points",
    "_coplanar_face_groups",
    "_cached_coplanar_face_groups",
    "_participates_in_ad",
    "_geometry_participates_in_ad",
    "_vertices_participate_in_ad",
    "_opposite_vertex_ids",
    "_reflection_geometry_ad",
)


def test_reevaluate_helpers_are_same_object_compatibility_exports():
    for name in _REEVALUATE_NAMES:
        owner = getattr(reevaluate, name)

        assert owner.__module__ == reevaluate.__name__
        assert getattr(legacy, name) is owner

    assert legacy._PLANE_GROUP_QUANTIZATION == reevaluate._PLANE_GROUP_QUANTIZATION


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.core import path_topology as legacy; "
            "from witwin.channel_native.propagation.geometry import reevaluate"
        ),
        (
            "from witwin.channel_native.propagation.geometry import reevaluate; "
            "from witwin.channel_native.core import path_topology as legacy"
        ),
    ),
)
def test_reevaluate_import_order_preserves_facade_identity(imports: str):
    names = repr(_REEVALUATE_NAMES)
    code = (
        f"{imports}; "
        f"names={names}; "
        "assert all(getattr(legacy, name) is getattr(reevaluate, name) "
        "for name in names)"
    )
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
