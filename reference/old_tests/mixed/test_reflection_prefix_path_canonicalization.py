"""Regression coverage for canonical reflection-prefix path collection."""

from __future__ import annotations

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import drjit as dr
import witwin as wt
import pytest

from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from witwin.channel import Field, Tracer, compute_reflection_field
from witwin.channel.trace.reflection import api as reflection_api_module
from witwin.channel.trace.reflection.api import (
    _select_reflection_ray_directions,
    _trace_reflection_paths_legacy,
    _trace_reflection_paths_rayd,
)
from witwin.channel.trace.reflection.paths import (
    _collect_reflection_prefix_paths_from_rayd_chain,
    _collect_unique_reflection_paths,
)


FREQ = 1e9
REFLECTION_N_RAYS = 10000
REFLECTION_MAX_BOUNCES = 3
REFLECTION_COEF = 0.8
CANONICAL_POS_TOL = 1e-5


def build_scene():
    cube1 = box_drjit_geometry(center=(-2.5, -3.0, 1.5), size=2.0, rotation=None).to_mesh()
    cube2 = box_drjit_geometry(center=(2.5, 1.0, 1.5), size=2.0, rotation=None).to_mesh()
    return build_test_scene(cube1, cube2)


def _prim_center_x(scene, prim_idx: int) -> float:
    tri_v0x = scene.tri_data_gpu["v0"].x
    tri_v1x = scene.tri_data_gpu["v1"].x
    tri_v2x = scene.tri_data_gpu["v2"].x
    prim_idx = int(prim_idx)
    return float((tri_v0x[prim_idx] + tri_v1x[prim_idx] + tri_v2x[prim_idx]) / 3.0)


def _path_key(paths, idx: int) -> tuple[tuple[int, ...], tuple[int, int, int]]:
    chain_depth = int(paths.get("chain_depth", 0))
    chain = tuple(int(paths[f"path_prim_idx_{slot}"][idx]) for slot in range(chain_depth))
    image_source = paths["image_source"]
    quantized_pos = (
        int(round(float(image_source.x[idx]) / CANONICAL_POS_TOL)),
        int(round(float(image_source.y[idx]) / CANONICAL_POS_TOL)),
        int(round(float(image_source.z[idx]) / CANONICAL_POS_TOL)),
    )
    return chain, quantized_pos


def _path_summary(paths) -> dict[tuple[tuple[int, ...], tuple[int, int, int]], int]:
    return {
        _path_key(paths, idx): int(paths["discovery_count"][idx])
        for idx in range(int(paths.get("n_paths", 0)))
    }


def test_reflection_prefix_paths_merge_close_image_sources_within_tolerance():
    active = wt.Bool(True, True)
    image_source = wt.Point3f(
        wt.Float(-2.0335752964019775, -2.0335748195648193),
        wt.Float(5.794851303100586, 5.794851303100586),
        wt.Float(1.5, 1.5),
    )
    chain_prim_history = [wt.Int32(4, 4)]
    chain_plane_point_history = [wt.Point3f(wt.Float(1.0, 1.0), wt.Float(0.0, 0.0), wt.Float(0.0, 0.0))]
    chain_plane_normal_history = [wt.Vector3f(wt.Float(0.0, 0.0), wt.Float(1.0, 1.0), wt.Float(0.0, 0.0))]
    chain_hit_point_history = [wt.Point3f(wt.Float(1.0, 1.0), wt.Float(0.5, 0.5), wt.Float(0.0, 0.0))]

    paths = _collect_unique_reflection_paths(
        active=active,
        image_source=image_source,
        chain_prim_history=chain_prim_history,
        chain_depth=1,
        chain_plane_point_history=chain_plane_point_history,
        chain_plane_normal_history=chain_plane_normal_history,
        chain_hit_point_history=chain_hit_point_history,
    )

    assert int(paths["n_paths"]) == 1
    assert int(paths["discovery_count"][0]) == 2
    assert int(paths["path_prim_idx_0"][0]) == 4
    assert abs(float(paths["image_source"].x[0] - image_source.x[0])) < 1e-8
    assert abs(float(paths["path_plane_point_0"].x[0] - 1.0)) < 1e-8
    assert abs(float(paths["path_plane_normal_0"].y[0] - 1.0)) < 1e-8
    assert abs(float(paths["path_hit_point_0"].y[0] - 0.5)) < 1e-8


def test_rayd_prefix_collector_applies_canonical_remap_before_grouping():
    class FakeChain:
        ray_count = 2
        max_bounces = 1
        bounce_count = wt.Int32(1, 1)
        discovery_count = wt.Int32(2, 3)
        representative_ray_index = wt.Int32(5, 2)
        prim_ids = wt.Int32(4, 5)
        image_sources = wt.Point3f(wt.Float(1.0, 1.0), wt.Float(2.0, 2.0), wt.Float(3.0, 3.0))
        plane_points = wt.Point3f(wt.Float(10.0, 20.0), wt.Float(0.0, 0.0), wt.Float(0.0, 0.0))
        plane_normals = wt.Vector3f(wt.Float(0.0, 0.0), wt.Float(1.0, 2.0), wt.Float(0.0, 0.0))
        hit_points = wt.Point3f(wt.Float(30.0, 40.0), wt.Float(0.5, 0.6), wt.Float(0.0, 0.0))

    paths = _collect_reflection_prefix_paths_from_rayd_chain(
        FakeChain(),
        chain_depth=1,
        surface_canonical_prims=wt.Int32(0, 1, 2, 3, 7, 7),
    )[0]

    assert int(paths["n_paths"]) == 1
    assert int(paths["discovery_count"][0]) == 5
    assert int(paths["path_prim_idx_0"][0]) == 7
    assert abs(float(paths["path_plane_point_0"].x[0] - 20.0)) < 1e-8
    assert abs(float(paths["path_plane_normal_0"].y[0] - 2.0)) < 1e-8
    assert abs(float(paths["path_hit_point_0"].x[0] - 40.0)) < 1e-8


def test_rayd_prefix_collector_accepts_symbolic_trace_keep_mask_payload():
    class FakeBounce:
        def __init__(self, *, prim_ids, image_x, plane_x, hit_x):
            self.prim_ids = wt.Int32(*prim_ids)
            self.image_sources = wt.Point3f(wt.Float(*image_x), wt.Float(0.0, 0.0), wt.Float(1.5, 1.5))
            self.plane_points = wt.Point3f(wt.Float(*plane_x), wt.Float(0.0, 0.0), wt.Float(0.0, 0.0))
            self.plane_normals = wt.Vector3f(wt.Float(0.0, 0.0), wt.Float(1.0, 2.0), wt.Float(0.0, 0.0))
            self.hit_points = wt.Point3f(wt.Float(*hit_x), wt.Float(0.5, 0.6), wt.Float(0.0, 0.0))
            self.geo_normals = self.plane_normals

    class FakeTrace:
        ray_count = 2
        max_bounces = 1
        bounce_count = wt.Int32(1, 1)
        discovery_count = wt.Int32(2, 3)
        representative_ray_index = wt.Int32(5, 2)
        dedup_keep_mask = wt.Bool(False, True)
        bounces = [FakeBounce(prim_ids=(4, 5), image_x=(1.0, 1.0), plane_x=(10.0, 20.0), hit_x=(30.0, 40.0))]

        def bounce(self, index):
            return self.bounces[index]

    paths = _collect_reflection_prefix_paths_from_rayd_chain(
        FakeTrace(),
        chain_depth=1,
        surface_canonical_prims=wt.Int32(0, 1, 2, 3, 7, 8),
    )[0]

    assert int(paths["n_paths"]) == 1
    assert int(paths["discovery_count"][0]) == 3
    assert int(paths["path_prim_idx_0"][0]) == 8
    assert abs(float(paths["path_plane_point_0"].x[0] - 20.0)) < 1e-8
    assert abs(float(paths["path_hit_point_0"].x[0] - 40.0)) < 1e-8


@pytest.mark.gpu
def test_rayd_reflection_prefix_paths_match_legacy_on_same_sampled_rays():
    scene = build_scene()
    tx_pos = wt.Point3f(0.0, -5.0, 1.5)
    ray_dir, ray_sampling_metadata = _select_reflection_ray_directions(
        axis="z",
        bounds=((-6, 6), (-6, 6)),
        tx_pos=tx_pos,
        n_rays=REFLECTION_N_RAYS,
        mode="2d",
        plane_position=1.5,
        ray_sampling="auto",
    )
    common_kwargs = dict(
        tx_pos=tx_pos,
        scene=scene,
        wavelength=299792458.0 / FREQ,
        n_rays=REFLECTION_N_RAYS,
        max_reflections=REFLECTION_MAX_BOUNCES,
        mode="2d",
        reflection_coef=REFLECTION_COEF,
        ray_sampling="auto",
        tx_polarization=(1.0, 0.0, 0.0),
        reflection_relative_permittivity=5.0,
        reflection_conductivity=0.0,
        reflection_material=None,
        use_scene_materials=False,
        sampling_axis="z",
        sampling_bounds=((-6, 6), (-6, 6)),
        sampling_plane_position=1.5,
        tri_data=scene.tri_data_gpu,
        ray_dir=ray_dir,
        ray_sampling_metadata=ray_sampling_metadata,
        on_segment=None,
    )

    legacy = _trace_reflection_paths_legacy(**common_kwargs)
    rayd = _trace_reflection_paths_rayd(**common_kwargs)

    for depth in range(REFLECTION_MAX_BOUNCES):
        assert _path_summary(legacy["source_paths_per_bounce"][depth]) == _path_summary(
            rayd["source_paths_per_bounce"][depth]
        )


@pytest.mark.gpu
def test_rayd_reflection_prefix_paths_preserve_tx_gradients_in_symbolic_mode():
    scene = build_scene()
    rayd_scene = getattr(scene, "_rayd_scene", None)
    if rayd_scene is None or not hasattr(rayd_scene, "trace_reflections"):
        pytest.skip("RayDi reflection tracing is unavailable in the current environment")

    tx_pos = wt.Point3f(0.0, -5.0, 1.5)
    dr.enable_grad(tx_pos.x, tx_pos.y, tx_pos.z)
    ray_dir, ray_sampling_metadata = _select_reflection_ray_directions(
        axis="z",
        bounds=((-6, 6), (-6, 6)),
        tx_pos=tx_pos,
        n_rays=REFLECTION_N_RAYS,
        mode="2d",
        plane_position=1.5,
        ray_sampling="auto",
    )
    result = _trace_reflection_paths_rayd(
        tx_pos=tx_pos,
        scene=scene,
        wavelength=299792458.0 / FREQ,
        n_rays=REFLECTION_N_RAYS,
        max_reflections=1,
        mode="2d",
        reflection_coef=REFLECTION_COEF,
        ray_sampling="auto",
        tx_polarization=(1.0, 0.0, 0.0),
        reflection_relative_permittivity=5.0,
        reflection_conductivity=0.0,
        reflection_material=None,
        use_scene_materials=False,
        sampling_axis="z",
        sampling_bounds=((-6, 6), (-6, 6)),
        sampling_plane_position=1.5,
        tri_data=scene.tri_data_gpu,
        ray_dir=ray_dir,
        ray_sampling_metadata=ray_sampling_metadata,
        on_segment=None,
    )

    first_paths = result["source_paths_per_bounce"][0]
    n_paths = int(first_paths["n_paths"])
    assert n_paths > 0
    assert dr.grad_enabled(first_paths["image_source"].x)

    weights = wt.Float([float(index + 1) for index in range(n_paths)])
    dr.backward(dr.sum(first_paths["image_source"].z * weights))

    assert abs(float(dr.grad(tx_pos.z)[0])) > 1e-8


@pytest.mark.gpu
def test_compute_reflection_field_explicit_replay_skips_legacy_segment_trace(monkeypatch):
    scene = build_scene()
    rayd_scene = getattr(scene, "_rayd_scene", None)
    if rayd_scene is None or not hasattr(rayd_scene, "trace_reflections"):
        pytest.skip("RayDi reflection tracing is unavailable in the current environment")

    field = Field(bounds=((-6, 6), (-6, 6)), size=(16, 16))
    coords = field.get_coordinates()

    _, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=wt.Point3f(0.0, -5.0, 1.5),
        scene=scene,
        wavelength=299792458.0 / FREQ,
        k=2.0 * math.pi / (299792458.0 / FREQ),
        n_rays=512,
        max_reflections=2,
        mode="2d",
        reflection_coef=REFLECTION_COEF,
        grid_data=coords,
    )

    def _unexpected_legacy(*args, **kwargs):
        raise AssertionError("EPC reflection evaluation should not enter the legacy trace path.")

    monkeypatch.setattr(reflection_api_module, "_trace_reflection_paths_legacy", _unexpected_legacy)

    _, _, replay_detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=wt.Point3f(0.0, -5.0, 1.5),
        scene=scene,
        wavelength=299792458.0 / FREQ,
        k=2.0 * math.pi / (299792458.0 / FREQ),
        n_rays=512,
        max_reflections=2,
        mode="2d",
        reflection_coef=REFLECTION_COEF,
        grid_data=coords,
        reflection_detail=reflection_detail,
    )

    assert replay_detail["dda_stats"]["backend"] == "epc"
    assert replay_detail["dda_stats"]["implementation"] == "epc"


@pytest.mark.gpu
def test_reflection_prefix_paths_are_canonicalized_for_mixed_diffraction():
    scene = build_scene()
    tracer = Tracer(
        frequency=FREQ,
        scene=scene,
        reflection_n_rays=REFLECTION_N_RAYS,
        reflection_max_bounces=REFLECTION_MAX_BOUNCES,
        reflection_coef=REFLECTION_COEF,
        enable_rd_diffraction=True,
    )
    field = Field(bounds=((-6, 6), (-6, 6)), size=(16, 16))
    coords = field.get_coordinates()

    _, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=wt.Point3f(0.0, -5.0, 1.5),
        scene=scene,
        wavelength=tracer.wavelength,
        k=tracer.k,
        n_rays=tracer.reflection_n_rays,
        max_reflections=tracer.reflection_max_bounces,
        mode="2d",
        reflection_coef=tracer.reflection_coef,
        return_per_bounce=True,
        grid_data=coords,
    )

    path_counts = []
    for paths in reflection_detail["source_paths_per_bounce"]:
        n_paths = int(paths.get("n_paths", 0))
        path_counts.append(n_paths)
        seen_keys = set()
        for idx in range(n_paths):
            key = _path_key(paths, idx)
            assert key not in seen_keys, f"duplicate canonical reflection-prefix path remained: {key}"
            seen_keys.add(key)
            assert f"path_plane_point_0" in paths
            assert f"path_plane_normal_0" in paths
        if n_paths > 0:
            max_discovery_count = max(int(paths["discovery_count"][idx]) for idx in range(n_paths))
            assert max_discovery_count > 1

    assert path_counts == [4, 2, 0]

    bounce1_paths = reflection_detail["source_paths_per_bounce"][0]
    bounce2_paths = reflection_detail["source_paths_per_bounce"][1]
    bounce1_idx_c1 = [
        idx
        for idx in range(int(bounce1_paths["n_paths"]))
        if _prim_center_x(scene, bounce1_paths["path_prim_idx_0"][idx]) < 0.0
    ]
    bounce2_idx_c2 = [
        idx
        for idx in range(int(bounce2_paths["n_paths"]))
        if _prim_center_x(scene, bounce2_paths["path_prim_idx_1"][idx]) > 0.0
    ]

    assert len(bounce1_idx_c1) == 2
    assert len(bounce2_idx_c2) == 2


@pytest.mark.gpu
def test_rotated_single_cube_reflection_paths_remain_canonical():
    scene = build_test_scene(
        box_drjit_geometry(
            center=(0.0, 0.0, 2.0),
            size=4.0,
            rotation=float(math.radians(15.0)),
        ).to_mesh()
    )
    tracer = Tracer(
        frequency=FREQ,
        scene=scene,
        reflection_n_rays=REFLECTION_N_RAYS,
        reflection_max_bounces=1,
        reflection_coef=1.0,
    )
    field = Field(bounds=((-8, 8), (-8, 8)), size=(16, 16))
    coords = field.get_coordinates()

    _, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=wt.Point3f(-5.0, 5.0, 1.5),
        scene=scene,
        wavelength=tracer.wavelength,
        k=tracer.k,
        n_rays=tracer.reflection_n_rays,
        max_reflections=tracer.reflection_max_bounces,
        mode="2d",
        reflection_coef=tracer.reflection_coef,
        return_per_bounce=True,
        grid_data=coords,
    )

    bounce1_paths = reflection_detail["source_paths_per_bounce"][0]
    n_paths = int(bounce1_paths.get("n_paths", 0))
    assert n_paths == 2

    seen_keys = set()
    for idx in range(n_paths):
        key = _path_key(bounce1_paths, idx)
        assert key not in seen_keys, f"duplicate rotated reflection-prefix path remained: {key}"
        seen_keys.add(key)

    assert min(int(bounce1_paths["discovery_count"][idx]) for idx in range(n_paths)) > 1

