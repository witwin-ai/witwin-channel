# Copyright Xingyu Chen.
# Full-wave reference set: stored maps, region labels, slab and refined-grid case construction.

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from benchmarks.fullwave_validation import reference_set as rs

_STORED = tuple(case_id for case_id in rs.case_ids() if rs.reference_path(case_id).is_file())


def test_every_gated_case_has_a_stored_reference():
    gated = rs.case_ids("fit") + rs.case_ids("holdout") + rs.case_ids("coverage")
    assert set(gated) <= set(_STORED)


@pytest.mark.parametrize("case_id", _STORED)
def test_stored_region_labels_match_the_classifier(case_id):
    case = rs.load_reference_case(case_id)
    planes, metadata = rs.load_reference(case)
    assert metadata["case_fingerprint"] == case.fingerprint
    for plane in planes:
        assert np.isfinite(plane["field"]).all()
        labels = rs.classify_regions(case, plane["x"], plane["y"], plane["z"])
        np.testing.assert_array_equal(labels, plane["regions"])


def test_case_frequency_defaults_to_the_manifest_and_can_be_overridden():
    manifest = rs.load_manifest()
    assert rs.load_reference_case("cube_z").frequency_hz == manifest["frequency_hz"]
    assert rs.load_reference_case("canyon_z_3g5").frequency_hz == 3.5e9
    assert rs.load_reference_case("canyon_z_3g5").calibration == "empty_z_3g5"


def test_slab_block_splits_into_stacked_layer_volumes():
    case = rs.load_reference_case("wall_layered_z")
    volumes = rs._volumes(case.blocks[0])
    assert [round(size[0], 6) for _, size, _ in volumes] == [0.02, 0.15, 0.02]
    assert [round(center[0], 6) for center, _, _ in volumes] == [0.11, 0.195, 0.28]
    assert [material["eps_r"] for _, _, material in volumes] == [3.0, 6.5, 3.0]
    thin = dataclasses.replace(case.blocks[0], size=(0.18, 40.0, 40.0))
    with pytest.raises(ValueError, match="fill the block thickness"):
        rs._volumes(thin)


def test_refined_axis_keeps_every_uniform_lattice_node():
    case = rs.load_reference_case("wall_layered_z")
    steps = rs._refined_steps(case)[0]
    lo, hi = case.domain_bounds_xyz[0]
    nodes = lo + np.concatenate(([0.0], np.cumsum(steps)))
    assert nodes[-1] == pytest.approx(hi, abs=1e-9)
    assert min(steps) == pytest.approx(0.0005)
    lattice = lo + case.grid_m * np.arange(round((hi - lo) / case.grid_m) + 1)
    assert np.min(np.abs(nodes[:, None] - lattice[None, :]), axis=0).max() < 1e-9
    picked = rs._lattice_nodes(nodes[(nodes > -0.43) & (nodes < 0.47)], case.grid_m)
    assert np.allclose(np.diff(nodes[(nodes > -0.43) & (nodes < 0.47)][picked]), 4 * case.grid_m)


def test_lattice_picking_rejects_a_grid_without_the_lattice():
    with pytest.raises(ValueError, match="uniform decimated lattice"):
        rs._lattice_nodes(np.arange(0.0, 0.1, 0.0015), 0.002)


def test_edges_outside_the_domain_label_no_transition_or_corner():
    # The wall crosses the PML, so its edges are not physical and its plane has only lit and shadow.
    case = rs.load_reference_case("wall_concrete_z")
    planes, _ = rs.load_reference(case)
    labels = rs.classify_regions(case, planes[0]["x"], planes[0]["y"], planes[0]["z"])
    assert set(np.unique(labels)) == {rs.INVALID, rs.LIT, rs.SHADOW}


def test_slab_wall_becomes_a_layered_plate_in_its_face_towards_the_transmitter():
    case = rs.load_reference_case("wall_layered_z")
    plane = {"x": np.array([-0.4, -0.392]), "y": np.array([0.0, 0.008]), "z": -0.06}
    vertices, faces, fill_mode = rs._block_mesh(case.blocks[0], np.asarray(case.tx_position))
    assert fill_mode == "surface" and len(faces) == 2
    assert {vertex[0] for vertex in vertices} == {0.1}
    scene = rs.build_channel_scene(case, plane)
    layers = scene.structures[0].material.layers
    assert [layer.thickness_m for layer in layers] == pytest.approx([0.02, 0.15, 0.02])


@pytest.mark.parametrize("case_id", rs.case_ids())
def test_tidy3d_grid_contains_the_lattice_plates_and_planes(case_id):
    pytest.importorskip("tidy3d")
    rs.build_tidy3d_simulation(rs.load_reference_case(case_id))


def test_tidy3d_grid_guard_rejects_a_stretched_grid():
    pytest.importorskip("tidy3d")
    case = rs.load_reference_case("trihedral_z")
    stretched = dataclasses.replace(case, domain_bounds_xyz=((-0.62, 0.42), (-0.62, 0.42), (-0.14, 0.46)))
    with pytest.raises(ValueError, match="not a grid node"):
        rs.build_tidy3d_simulation(stretched)
