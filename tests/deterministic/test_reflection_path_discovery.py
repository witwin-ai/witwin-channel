from __future__ import annotations

import pytest

from witwin.channel.deterministic import types as wt
from witwin.channel.core.runtime import Material as RuntimeMaterial
from witwin.channel.core.runtime import Tx, Wave
from witwin.channel.core.scene import Scene
from witwin.channel.deterministic.reflection import paths
from witwin.core import Material, Mesh, Structure


def _open_wall_scene(*, device: str = "cuda") -> Scene:
    mesh = Mesh(
        vertices=(
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 3.0),
            (1.0, 0.0, 3.0),
        ),
        faces=((0, 1, 3), (0, 3, 2)),
        device=device,
    )
    return Scene(
        structures=[
            Structure(
                geometry=mesh,
                material=Material(eps_r=4.0, sigma_e=0.0),
                name="open_wall",
            )
        ],
        device=device,
    )


@pytest.mark.gpu
def test_first_order_reflection_discovery_uses_analytic_surface_enumeration(monkeypatch):
    scene = _open_wall_scene(device="cuda")
    tx = Tx(position=(-2.0, -1.0, 1.5))
    wave = Wave.from_frequency(3.5e9)
    tri_data = scene._triangle_runtime()

    class _NoRayDTrace:
        def trace_reflections(self, *args, **kwargs):
            raise AssertionError("first-order analytic discovery must not call RayD trace_reflections")

    monkeypatch.setattr(scene, "_rayd_scene", _NoRayDTrace())

    trace_data = paths.trace_paths(
        tx=tx,
        scene=scene,
        wave=wave,
        n_rays=32768,
        max_reflections=1,
        mode="3d",
        material=RuntimeMaterial(reflection_coef=1.0),
        ray_sampling="full_sphere",
        sampling_axis="y",
        sampling_bounds=((-3.0, 3.0), (0.0, 3.0)),
        sampling_plane_position=0.0,
        tri_data=tri_data,
    )

    first = trace_data["source_paths_per_bounce"][0]
    assert int(first.chain_depth) == 1
    assert int(first.n_paths) >= 1
    assert trace_data["reflection_model"] == "materialized"


@pytest.mark.gpu
def test_multi_bounce_reflection_discovery_uses_native_prefix_compaction_when_available(monkeypatch):
    scene = _open_wall_scene(device="cuda")
    tx = Tx(position=(-2.0, -1.0, 1.5))
    wave = Wave.from_frequency(3.5e9)
    tri_data = scene._triangle_runtime()

    class _FakeChain:
        ray_count = 1
        max_bounces = 2

    fake_chain = _FakeChain()

    class _NativeRayDTrace:
        def trace_reflections(self, *args, **kwargs):
            return fake_chain

    def _point(value: float) -> wt.Point3f:
        return wt.Point3f([value], [value + 1.0], [value + 2.0])

    native_depth_two = paths.SourcePathSet(
        image_source=_point(10.0),
        discovery_count=wt.UInt32([3]),
        chain_depth=2,
        n_paths=1,
        path_prim_idx=(wt.Int32([0]), wt.Int32([1])),
        path_plane_point=(_point(20.0), _point(30.0)),
        path_plane_normal=(_point(40.0), _point(50.0)),
        path_hit_point=(_point(60.0), _point(70.0)),
    )
    calls = []

    def fake_native_compaction(
        chain,
        canonical_prim_table,
        image_source_tolerance,
        max_prefix_depth,
        min_prefix_depth,
    ):
        calls.append(
            (
                chain,
                int(max_prefix_depth),
                int(min_prefix_depth),
                float(image_source_tolerance),
                int(wt.Int32(canonical_prim_table)[0]),
            )
        )
        return (native_depth_two,)

    def fail_python_prefix_compaction(*args, **kwargs):
        raise AssertionError("native prefix compaction must bypass Python collect_prefix_paths")

    monkeypatch.setattr(scene, "_rayd_scene", _NativeRayDTrace())
    monkeypatch.setattr(
        paths.rayd,
        "compact_reflection_prefix_paths",
        fake_native_compaction,
        raising=False,
    )
    monkeypatch.setattr(paths, "collect_prefix_paths", fail_python_prefix_compaction)

    trace_data = paths.trace_paths(
        tx=tx,
        scene=scene,
        wave=wave,
        n_rays=8,
        max_reflections=2,
        mode="3d",
        material=RuntimeMaterial(reflection_coef=1.0),
        ray_sampling="full_sphere",
        sampling_axis="y",
        sampling_bounds=((-3.0, 3.0), (0.0, 3.0)),
        sampling_plane_position=0.0,
        tri_data=tri_data,
    )

    source_paths = trace_data["source_paths_per_bounce"]
    assert len(calls) == 1
    assert calls[0][0] is fake_chain
    assert calls[0][1:3] == (2, 2)
    assert source_paths[0].chain_depth == 1
    assert source_paths[1] is native_depth_two
    assert trace_data["reflection_discovery_backend"] == "rayd_trace_native_prefix_compaction"


@pytest.mark.gpu
def test_channel_native_prefix_compaction_matches_python_prefix_bucketing():
    class _FlatChain:
        ray_count = 4
        max_bounces = 2
        bounce_count = wt.Int32([2, 2, 1, 2])
        discovery_count = wt.Int32([1, 2, 5, 3])
        representative_ray_index = wt.Int32([10, 5, 2, 7])
        global_prim_ids = wt.Int32([
            0, 1,
            0, 1,
            0, -1,
            0, 2,
        ])
        image_sources = wt.Point3f(
            [1.0, 11.0, 1.0, 11.0, 1.0, 0.0, 1.0, 21.0],
            [2.0, 12.0, 2.0, 12.0, 2.0, 0.0, 2.0, 22.0],
            [3.0, 13.0, 3.0, 13.0, 3.0, 0.0, 3.0, 23.0],
        )
        plane_points = image_sources
        plane_normals = image_sources
        hit_points = image_sources

    chain = _FlatChain()
    canonical_prims = wt.Int32([0, 1, 2])

    expected_depth_two = paths.collect_prefix_paths(
        chain,
        chain_depth=2,
        surface_canonical_prims=canonical_prims,
        image_source_tolerance=1e-5,
    )[1]
    native_depth_two = paths._collect_prefix_paths_channel_native(
        chain,
        chain_depth=2,
        surface_canonical_prims=canonical_prims,
        image_source_tolerance=1e-5,
    )[0]

    assert native_depth_two.chain_depth == expected_depth_two.chain_depth
    assert native_depth_two.n_paths == expected_depth_two.n_paths == 2
    assert [int(v) for v in native_depth_two.discovery_count] == [
        int(v) for v in expected_depth_two.discovery_count
    ]
    assert [int(v) for v in native_depth_two.path_prim_idx[0]] == [
        int(v) for v in expected_depth_two.path_prim_idx[0]
    ]
    assert [int(v) for v in native_depth_two.path_prim_idx[1]] == [
        int(v) for v in expected_depth_two.path_prim_idx[1]
    ]
    assert [float(v) for v in native_depth_two.image_source.x] == [
        float(v) for v in expected_depth_two.image_source.x
    ]
