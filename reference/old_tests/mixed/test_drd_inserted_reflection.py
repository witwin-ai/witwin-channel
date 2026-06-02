"""Regression test for inserted-reflection mixed diffraction states."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import witwin as wt

from tests._scene_helpers import box_geometry, build_scene as build_test_scene
from witwin.channel import FieldMonitor, Tracer
def build_scene():
    cube1 = box_geometry(center=(-1.8, -1.2, 1.5), size=2.0)
    cube2 = box_geometry(center=(1.8, 1.2, 1.5), size=2.0)
    return build_test_scene(cube1, cube2)


def to_numpy(array):
    return array.numpy() if hasattr(array, "numpy") else np.asarray(array)


def test_inserted_reflection_generates_direct_drd_states():
    scene = build_scene()
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        enable_rd_diffraction=True,
        max_diffractions=2,
    )
    monitor = FieldMonitor(
        "drd_plane",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-4.0, 4.0)),
        grid_size=20,
    )

    result = tracer.trace(
        wt.Point3f(0.0, -4.0, 1.5),
        monitor=monitor,
        verbose=False,
        return_diffraction_audit=True,
    )

    assert result.primary.metadata["path_families"]["D -> R -> D"]["status"] == "approximate"

    audit = result.primary.diffraction_detail["state_audit"]
    order = to_numpy(audit["order"])
    prefix_depth = to_numpy(audit["prefix_reflection_depth"])
    intermediate_depth = to_numpy(audit["intermediate_reflection_depth"])
    suffix_depth = to_numpy(audit["suffix_reflection_depth"])

    direct_drd_mask = (
        (order == 2)
        & (prefix_depth == 0)
        & (intermediate_depth == 1)
        & (suffix_depth == 0)
    )
    direct_drd_idx = np.flatnonzero(direct_drd_mask)
    assert direct_drd_idx.size > 0, "Expected at least one direct S -> D -> R -> D state"

    path_sequences = [audit["path_sequence"][idx] for idx in direct_drd_idx]
    source_types = [audit["source_type"][idx] for idx in direct_drd_idx]
    approx_modes = [audit["approximation_mode"][idx] for idx in direct_drd_idx]

    assert "S -> D -> R -> D" in path_sequences
    assert all(source_type == "direct_tx" for source_type in source_types)
    assert all(mode == "approx_sampled_inserted_reflection" for mode in approx_modes)


