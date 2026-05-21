from __future__ import annotations

import drjit as dr
import pytest
import rayd

from witwin.channel.core.runtime import Tx, Wave
from witwin.channel.core.scene import Scene
from witwin.channel.deterministic import types as wt
from witwin.channel.deterministic.reflection import epc
from witwin.channel.deterministic.reflection.detail import build_trace_detail
from witwin.channel.deterministic.reflection.paths import enumerate_first_bounce_surface_paths
from witwin.core import Box, Material, Mesh, Structure


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


def _cube_scene(*, device: str = "cuda") -> Scene:
    return Scene(
        structures=[
            Structure(
                geometry=Box(position=(0.0, 0.0, 1.5), size=(2.0, 2.0, 2.0), device=device),
                material=Material(eps_r=4.0, sigma_e=0.0),
                name="cube",
            )
        ],
        device=device,
    )


def _centroid_blocked_reflector_scene(*, device: str = "cuda") -> Scene:
    reflector = Mesh(
        vertices=(
            (-4.0, 0.0, -1.0),
            (4.0, 0.0, -1.0),
            (-4.0, 0.0, 1.0),
            (4.0, 0.0, 1.0),
        ),
        faces=((0, 1, 3), (0, 3, 2)),
        device=device,
    )
    blocker = Mesh(
        vertices=(
            (0.5, -1.0, -0.35),
            (0.8, -1.0, -0.35),
            (0.5, -1.0, -0.05),
            (0.8, -1.0, -0.05),
        ),
        faces=((0, 3, 1), (0, 2, 3)),
        recenter=False,
        device=device,
    )
    return Scene(
        structures=[
            Structure(
                geometry=reflector,
                material=Material(eps_r=4.0, sigma_e=0.0),
                name="reflector",
            ),
            Structure(
                geometry=blocker,
                material=Material(eps_r=4.0, sigma_e=0.0),
                name="blocker",
            ),
        ],
        device=device,
    )


class _CountingRayDScene:
    def __init__(self, inner):
        self._inner = inner
        self.epc_calls = 0
        self.epc_field_calls = 0
        self.epc_field_direct_calls = 0
        self.ray_counts = []
        self.field_ray_counts = []
        self.field_direct_ray_counts = []
        self.options = []
        self.field_options = []
        self.field_direct_options = []

    def trace_reflection_epc(self, *args, **kwargs):
        self.epc_calls += 1
        self.ray_counts.append(dr.width(args[0].o.x))
        options = kwargs.get("options")
        if options is None and len(args) >= 4:
            options = args[3]
        self.options.append(options)
        return self._inner.trace_reflection_epc(*args, **kwargs)

    def trace_reflection_epc_field(self, *args, **kwargs):
        self.epc_field_calls += 1
        self.field_ray_counts.append(dr.width(args[0].o.x))
        options = kwargs.get("options")
        if options is None and len(args) >= 4:
            options = args[3]
        self.field_options.append(options)
        return self._inner.trace_reflection_epc_field(*args, **kwargs)

    def trace_reflection_epc_field_direct(self, *args, **kwargs):
        self.epc_field_direct_calls += 1
        self.field_direct_ray_counts.append(dr.width(args[0].x))
        options = kwargs.get("options")
        if options is None and len(args) >= 4:
            options = args[3]
        self.field_direct_options.append(options)
        return self._inner.trace_reflection_epc_field_direct(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.mark.gpu
def test_first_bounce_surface_paths_store_interior_representative_point():
    scene = _open_wall_scene(device="cuda")
    tx = Tx(position=(0.0, -1.0, 0.0))
    paths = enumerate_first_bounce_surface_paths(tx=tx, tri_data=scene._triangle_runtime())

    assert int(paths.n_paths) >= 1
    anchor = paths.hit_point(0)
    plane_point = paths.plane_point(0)
    assert abs(float(anchor.y[0])) < 1e-5
    assert abs(float(anchor.x[0]) - float(plane_point.x[0])) > 1e-3
    assert abs(float(anchor.z[0]) - float(plane_point.z[0])) > 1e-3


@pytest.mark.gpu
def test_hard_non_ad_chain_to_target_uses_direct_rayd_epc_field(monkeypatch):
    scene = _open_wall_scene(device="cuda")
    tx = Tx(position=(0.0, -1.0, 0.0), polarization=(1.0, 0.0, 0.0))
    wave = Wave.from_frequency(3.5e9)
    paths = enumerate_first_bounce_surface_paths(tx=tx, tri_data=scene._triangle_runtime())
    detail = build_trace_detail(
        reflection_model="materialized",
        reflection_model_source="runtime",
        reflection_gain=1.0,
        source_paths_per_bounce=(paths,),
    )
    counter = _CountingRayDScene(scene._rayd_scene)
    monkeypatch.setattr(scene, "_rayd_scene", counter)

    def fail_native_forward(**kwargs):
        raise AssertionError("hard non-AD reflection EPC must use RayD direct field EPC")

    monkeypatch.setattr(epc, "launch_native_forward", fail_native_forward)

    valid, chain_vector, geometry = epc.chain_to_target(
        paths=paths,
        path_idx=wt.UInt32([0]),
        target_pos=wt.Point3f([0.5], [-1.0], [0.0]),
        scene=scene,
        reflection_detail=detail,
        wave=wave,
        tx=tx,
        return_geometry=True,
    )

    dr.eval(valid, geometry["hit_points"][0], chain_vector["x"].real)
    assert counter.epc_calls == 0
    assert counter.epc_field_calls == 0
    assert counter.epc_field_direct_calls == 1
    assert counter.field_direct_ray_counts == [1]
    assert isinstance(counter.field_direct_options[0], rayd.ReflectionEpcFieldOptions)
    assert counter.field_direct_options[0].return_geometry is True
    assert counter.field_direct_options[0].return_hit_points is True
    assert counter.field_direct_options[0].return_normals is True
    assert counter.field_direct_options[0].return_resolved_prim_ids is False
    assert counter.field_direct_options[0].return_surface_group_ids is False
    assert dr.width(counter.field_direct_options[0].slot_plane_point.x) == int(paths.chain_depth)
    assert bool(valid[0])
    assert abs(float(geometry["hit_points"][0].y[0])) < 1e-5
    assert float(dr.abs(chain_vector["x"])[0]) > 0.0


@pytest.mark.gpu
def test_rayd_epc_matches_native_epc_across_surface_group_members():
    scene = _open_wall_scene(device="cuda")
    tx = Tx(position=(0.0, -1.0, 0.0), polarization=(1.0, 0.0, 0.0))
    wave = Wave.from_frequency(3.5e9)
    paths = enumerate_first_bounce_surface_paths(tx=tx, tri_data=scene._triangle_runtime())
    detail = build_trace_detail(
        reflection_model="materialized",
        reflection_model_source="runtime",
        reflection_gain=1.0,
        source_paths_per_bounce=(paths,),
    )
    target_pos = wt.Point3f([-0.5, 0.5], [-1.0, -1.0], [0.0, 0.0])
    path_idx = wt.UInt32([0, 0])

    native_valid, native_vector, native_geometry = epc.chain_to_target(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        reflection_detail=detail,
        wave=wave,
        tx=tx,
        return_geometry=True,
        prefer_rayd_epc=False,
    )
    rayd_valid, rayd_vector, rayd_geometry = epc.chain_to_target(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        reflection_detail=detail,
        wave=wave,
        tx=tx,
        return_geometry=True,
        prefer_rayd_epc=True,
    )
    dr.eval(native_valid, rayd_valid, native_geometry["hit_points"][0], rayd_geometry["hit_points"][0])

    assert [bool(v) for v in rayd_valid] == [bool(v) for v in native_valid] == [True, True]
    native_hit = native_geometry["hit_points"][0]
    rayd_hit = rayd_geometry["hit_points"][0]
    assert float(dr.max(dr.abs(native_hit.x - rayd_hit.x))[0]) < 1e-4
    assert float(dr.max(dr.abs(native_hit.y - rayd_hit.y))[0]) < 1e-4
    assert float(dr.max(dr.abs(native_hit.z - rayd_hit.z))[0]) < 1e-4
    assert float(dr.max(dr.abs(native_vector["x"] - rayd_vector["x"]))[0]) < 1e-4


@pytest.mark.gpu
def test_rayd_epc_direct_plane_ignores_blocked_seed_trace():
    scene = _centroid_blocked_reflector_scene(device="cuda")
    tx = Tx(position=(0.0, -2.0, 0.0), polarization=(1.0, 0.0, 0.0))
    wave = Wave(wavelength=0.1)
    paths = enumerate_first_bounce_surface_paths(tx=tx, tri_data=scene._triangle_runtime())
    detail = build_trace_detail(
        reflection_model="materialized",
        reflection_model_source="runtime",
        reflection_gain=1.0,
        source_paths_per_bounce=(paths,),
    )
    path_idx = wt.UInt32([0])
    target_pos = wt.Point3f([6.0], [-2.0], [0.0])

    native_valid, native_vector, native_geometry = epc.chain_to_target(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        reflection_detail=detail,
        wave=wave,
        tx=tx,
        return_geometry=True,
        prefer_rayd_epc=False,
    )
    rayd_valid, rayd_vector, rayd_geometry = epc.chain_to_target(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        reflection_detail=detail,
        wave=wave,
        tx=tx,
        return_geometry=True,
        prefer_rayd_epc=True,
    )
    dr.eval(native_valid, rayd_valid, native_geometry["hit_points"][0], rayd_geometry["hit_points"][0])

    assert bool(native_valid[0])
    assert bool(rayd_valid[0]) == bool(native_valid[0])
    assert float(rayd_geometry["hit_points"][0].x[0]) == pytest.approx(3.0, abs=1e-4)
    assert float(dr.abs(native_vector["x"] - rayd_vector["x"])[0]) < 1e-4


@pytest.mark.gpu
def test_rayd_epc_direct_plane_handles_triangle_boundary_hit():
    scene = _open_wall_scene(device="cuda")
    tx = Tx(position=(-0.5, -1.0, 1.5), polarization=(1.0, 0.0, 0.0))
    wave = Wave.from_frequency(3.5e9)
    paths = enumerate_first_bounce_surface_paths(tx=tx, tri_data=scene._triangle_runtime())
    detail = build_trace_detail(
        reflection_model="materialized",
        reflection_model_source="runtime",
        reflection_gain=1.0,
        source_paths_per_bounce=(paths,),
    )
    path_idx = wt.UInt32([0])
    target_pos = wt.Point3f([0.5], [-1.0], [1.5])

    native_valid, native_vector, native_geometry = epc.chain_to_target(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        reflection_detail=detail,
        wave=wave,
        tx=tx,
        return_geometry=True,
        prefer_rayd_epc=False,
    )
    rayd_valid, rayd_vector, rayd_geometry = epc.chain_to_target(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        reflection_detail=detail,
        wave=wave,
        tx=tx,
        return_geometry=True,
        prefer_rayd_epc=True,
    )
    dr.eval(native_valid, rayd_valid, native_geometry["hit_points"][0], rayd_geometry["hit_points"][0])

    assert bool(rayd_valid[0]) == bool(native_valid[0]) == True
    assert float(rayd_geometry["hit_points"][0].x[0]) == pytest.approx(0.0, abs=1e-4)
    assert float(rayd_geometry["hit_points"][0].z[0]) == pytest.approx(1.5, abs=1e-4)
    assert float(dr.abs(native_vector["x"] - rayd_vector["x"])[0]) < 1e-4


@pytest.mark.gpu
def test_rayd_epc_uses_group_indexed_surface_members_for_cube_boundary():
    scene = _cube_scene(device="cuda")
    tx = Tx(position=(-2.0, -5.0, 4.0), polarization=(1.0, 0.0, 0.0))
    wave = Wave.from_frequency(1.0e9)
    paths = enumerate_first_bounce_surface_paths(tx=tx, tri_data=scene._triangle_runtime())
    prim_ids = list(map(int, paths.prim_idx(0)))
    front_path_idx = prim_ids.index(8)
    detail = build_trace_detail(
        reflection_model="materialized",
        reflection_model_source="runtime",
        reflection_gain=1.0,
        source_paths_per_bounce=(paths,),
    )
    target_pos = wt.Point3f([-0.254], [-4.0], [1.0])
    path_idx = wt.UInt32([front_path_idx])

    native_valid, _, _ = epc.chain_to_target(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        reflection_detail=detail,
        wave=wave,
        tx=tx,
        return_geometry=True,
        prefer_rayd_epc=False,
    )
    rayd_valid, _, _ = epc.chain_to_target(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        reflection_detail=detail,
        wave=wave,
        tx=tx,
        return_geometry=True,
        prefer_rayd_epc=True,
    )
    dr.eval(native_valid, rayd_valid)

    assert not bool(native_valid[0])
    assert bool(rayd_valid[0]) == bool(native_valid[0])
