"""Regression coverage for RayD edge-BVH higher-order diffraction candidates."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import witwin as wt
import pytest
import torch

import drjit as dr

from tests._scene_helpers import box_geometry, build_scene
from witwin.channel.trace.diffraction import (
    _build_global_to_local_index,
    _build_higher_order_state_arrays,
    _build_tx_first_order_state_arrays,
    _state_lineage_first_edge_idx,
)
from witwin.channel.utils import drjit_to_torch_view
def _build_open_wall_mesh():
    vertices = wt.Point3f(
        wt.Float(-1.0, 1.0, -1.0, 1.0),
        wt.Float(0.0, 0.0, 0.0, 0.0),
        wt.Float(0.0, 0.0, 3.0, 3.0),
    )
    faces = wt.Vector3u(
        wt.UInt32(0, 0),
        wt.UInt32(1, 3),
        wt.UInt32(3, 2),
    )
    return vertices, faces


def _state_pairs(state_arrays):
    first_edge = drjit_to_torch_view(
        _state_lineage_first_edge_idx(state_arrays),
        detach=True,
        dtype=torch.int64,
    ).cpu()
    last_edge = drjit_to_torch_view(state_arrays["edge_idx"], detach=True, dtype=torch.int64).cpu()
    return {
        (int(first_edge[idx]), int(last_edge[idx]))
        for idx in range(int(first_edge.numel()))
    }


def _wavelength_and_k():
    wavelength = 299792458.0 / 1e9
    return wavelength, 2.0 * dr.pi / wavelength


@pytest.mark.gpu
def test_scene_rayd_edge_queries_wrap_mask_and_nearest_edge():
    scene = build_scene(
        _build_open_wall_mesh(),
        boundary_edge_policy="half_plane",
    )
    edge_data = scene.get_edge_data(1.5)["edge_data"]
    initial_mask = scene.edge_mask()

    assert edge_data is not None
    assert int(initial_mask.numel()) > 0

    selected_globals = drjit_to_torch_view(
        edge_data["global_idx"],
        detach=True,
        dtype=torch.int64,
        device=initial_mask.device,
    )
    masked = torch.zeros_like(initial_mask, dtype=torch.bool)
    masked[selected_globals[:1]] = True
    scene.set_edge_mask(masked)
    scene._rayd_scene.sync()

    nearest = scene.nearest_edge(
        wt.Ray(
            wt.Point3f(0.0, -4.0, 1.5),
            wt.Vector3f(0.0, 1.0, 0.0),
        )
    )

    assert torch.equal(scene.edge_mask(), masked)
    assert bool(dr.any(nearest.is_valid()))

    scene.set_edge_mask(initial_mask)
    scene._rayd_scene.sync()
    assert torch.equal(scene.edge_mask(), initial_mask)


@pytest.mark.gpu
def test_higher_order_rayd_edge_bvh_matches_bruteforce_on_half_plane_open_wall():
    scene = build_scene(
        _build_open_wall_mesh(),
        boundary_edge_policy="half_plane",
    )
    edge_data = scene.get_edge_data(1.5)["edge_data"]
    wavelength, k = _wavelength_and_k()
    tx = wt.Point3f(0.0, -4.0, 1.5)
    first_order = _build_tx_first_order_state_arrays(tx, edge_data, wavelength, k, scene=scene)
    global_to_local_idx = _build_global_to_local_index(scene, edge_data)
    initial_mask = scene.edge_mask()

    brute_force = _build_higher_order_state_arrays(
        first_order,
        edge_data,
        k,
        scene=scene,
        wavelength=wavelength,
        global_to_local_idx=global_to_local_idx,
        candidate_backend="bruteforce",
    )
    assert torch.equal(scene.edge_mask(), initial_mask)

    rayd_edge_bvh = _build_higher_order_state_arrays(
        first_order,
        edge_data,
        k,
        scene=scene,
        wavelength=wavelength,
        global_to_local_idx=global_to_local_idx,
        candidate_backend="rayd_edge_bvh",
    )

    assert torch.equal(scene.edge_mask(), initial_mask)
    assert brute_force["n_states"] == 2
    assert rayd_edge_bvh["n_states"] == 2
    assert _state_pairs(rayd_edge_bvh) == _state_pairs(brute_force) == {(0, 1), (1, 0)}


@pytest.mark.gpu
def test_higher_order_rayd_edge_bvh_finds_states_on_two_box_scene():
    scene = build_scene(
        box_geometry(center=(-1.8, -1.2, 1.5), size=2.0),
        box_geometry(center=(1.8, 1.2, 1.5), size=2.0),
    )
    edge_data = scene.get_edge_data(1.5)["edge_data"]
    wavelength, k = _wavelength_and_k()
    tx = wt.Point3f(0.0, -4.0, 1.5)
    first_order = _build_tx_first_order_state_arrays(tx, edge_data, wavelength, k, scene=scene)
    global_to_local_idx = _build_global_to_local_index(scene, edge_data)

    second_order = _build_higher_order_state_arrays(
        first_order,
        edge_data,
        k,
        scene=scene,
        wavelength=wavelength,
        global_to_local_idx=global_to_local_idx,
        candidate_backend="auto",
    )

    assert first_order["n_states"] > 0
    assert second_order["n_states"] > 0


@pytest.mark.gpu
def test_higher_order_auto_backend_requires_rayd_runtime():
    scene = build_scene(
        _build_open_wall_mesh(),
        boundary_edge_policy="half_plane",
    )
    edge_data = scene.get_edge_data(1.5)["edge_data"]
    wavelength, k = _wavelength_and_k()
    tx = wt.Point3f(0.0, -4.0, 1.5)
    first_order = _build_tx_first_order_state_arrays(tx, edge_data, wavelength, k, scene=scene)
    global_to_local_idx = _build_global_to_local_index(scene, edge_data)
    scene._rayd_scene = None

    with pytest.raises(RuntimeError, match="requires RayD edge BVH"):
        _build_higher_order_state_arrays(
            first_order,
            edge_data,
            k,
            scene=scene,
            wavelength=wavelength,
            global_to_local_idx=global_to_local_idx,
            candidate_backend="auto",
        )
