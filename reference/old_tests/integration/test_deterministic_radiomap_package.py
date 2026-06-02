"""Integration coverage for the standalone deterministic radiomap package."""

from __future__ import annotations

import drjit as dr
import numpy as np
import pytest
import rayd
import witwin as wt
import witwin.deterministic as drm

from witwin.channel import (
    RadioMapMonitor,
    Scene as LegacyScene,
    Tracer,
)
from witwin.channel_scene import Mesh as ChannelMesh
from witwin.channel_scene import ReceiverGrid, Scene as ChannelScene, Transmitter
from witwin.channel_utils.mesh_buffers import to_point3f, to_vector3u
from witwin.core import Box, Material, Structure
from witwin.deterministic import (
    Config,
    GridSpec,
    native_extension_available,
    solve,
)
from witwin.deterministic.path.diffraction_impl.state import Geo
from witwin.deterministic.path.reflection import discover_paths
from witwin.deterministic.runtime import Material as RuntimeMaterial
from witwin.deterministic.runtime import Tx, Wave
pytestmark = pytest.mark.gpu

BOUNDS = ((-3.0, 3.0), (-3.0, 3.0))
TX_POS = wt.Point3f(-2.5, 0.0, 1.5)
FREQUENCY = 3.5e9


def _scalar(value) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _wall_structure(*, x_offset=0.0, eps_r=4.0) -> Structure:
    return Structure(
        name="wall",
        geometry=Box(
            position=(x_offset, 0.0, 1.5),
            size=(0.25, 4.0, 3.0),
            device="cuda",
        ),
        material=Material(eps_r=eps_r, sigma_e=0.0),
    )


def _cube_structure(*, x_center=0.0, eps_r=4.0) -> Structure:
    return Structure(
        name="cube",
        geometry=Box(
            position=(x_center, 0.0, 1.5),
            size=(2.0, 2.0, 2.0),
            device="cuda",
        ),
        material=Material(eps_r=eps_r, sigma_e=0.0),
    )


def _build_channel_scene(structure: Structure | None = None) -> ChannelScene:
    return ChannelScene(
        structures=[_wall_structure() if structure is None else structure],
        device="cuda",
        edge_selection_mode="all_edges",
    )


def _build_legacy_scene(structure: Structure | None = None) -> LegacyScene:
    return LegacyScene(
        structures=[_wall_structure() if structure is None else structure],
        device="cuda",
        edge_selection_mode="all_edges",
    )


def _triangle_mesh_scene(vertices=None) -> ChannelScene:
    if vertices is None:
        vertices = wt.Point3f(
            wt.Float([0.0, 1.0, 0.0]),
            wt.Float([0.0, 0.0, 1.0]),
            wt.Float([0.0, 0.0, 0.0]),
        )
    faces = wt.Vector3u(wt.UInt32([0]), wt.UInt32([1]), wt.UInt32([2]))
    return ChannelScene(
        structures=[
            Structure(
                name="triangle",
                geometry=ChannelMesh(vertices, faces),
                material=Material(eps_r=4.0, sigma_e=0.0),
            )
        ],
        device="cuda",
        edge_selection_mode="all_edges",
    )


def _two_triangle_mesh_scene() -> ChannelScene:
    faces = wt.Vector3u(wt.UInt32([0]), wt.UInt32([1]), wt.UInt32([2]))
    tri0 = wt.Point3f(
        wt.Float([0.0, 1.0, 0.0]),
        wt.Float([0.0, 0.0, 1.0]),
        wt.Float([0.0, 0.0, 0.0]),
    )
    tri1 = wt.Point3f(
        wt.Float([2.0, 3.0, 2.0]),
        wt.Float([0.0, 0.0, 1.0]),
        wt.Float([0.0, 0.0, 0.0]),
    )
    return ChannelScene(
        structures=[
            Structure(
                name="triangle_0",
                geometry=ChannelMesh(tri0, faces),
                material=Material(eps_r=4.0, sigma_e=0.0),
            ),
            Structure(
                name="triangle_1",
                geometry=ChannelMesh(tri1, faces),
                material=Material(eps_r=4.0, sigma_e=0.0),
            ),
        ],
        device="cuda",
        edge_selection_mode="all_edges",
    )


def _three_cube_scene() -> ChannelScene:
    centers = (
        (-2.5, -3.0, 1.5),
        (2.0, 0.5, 1.5),
        (-0.5, 3.5, 1.5),
    )
    return ChannelScene(
        structures=[
            Structure(
                name=f"cube_{index}",
                geometry=Box(
                    position=center,
                    size=(2.0, 2.0, 2.0),
                    device="cuda",
                ),
                material=Material(eps_r=4.0, sigma_e=0.0),
            )
            for index, center in enumerate(centers)
        ],
        device="cuda",
        edge_selection_mode="all_edges",
    )


def _legacy_result(*, native: bool = False):
    tracer = Tracer(
        frequency=FREQUENCY,
        scene=_build_legacy_scene(),
        reflection_n_rays=128,
        reflection_max_bounces=0,
        max_diffractions=0,
        reflection_field_backend="native" if native else "drjit",
    )
    monitor = RadioMapMonitor(
        "legacy_baseline",
        axis="z",
        position=1.5,
        bounds=BOUNDS,
        grid_shape=(8, 8),
        metric="path_gain",
        combine_mode="coherent",
        receiver_model="matched_isotropic",
        accumulation_backend="baseline",
        ray_mode="3d",
        max_diffractions=0,
    )
    return tracer.trace(TX_POS, monitor=monitor, verbose=False)


def _standalone_result(*, native: bool = False):
    return solve(
        scene=_build_channel_scene(),
        frequency=FREQUENCY,
        tx_pos=TX_POS,
        grid=GridSpec(
            axis="z",
            position=1.5,
            bounds=BOUNDS,
            grid_shape=(8, 8),
        ),
        config=Config(
            metric="path_gain",
            shadow_boundary_correction=False,
            reflection_n_rays=128,
            reflection_max_bounces=0,
            max_diffractions=0,
            reflection_field_backend="native" if native else "drjit",
        ),
    )


def _assert_metric_close(modern, legacy) -> None:
    np.testing.assert_allclose(
        np.asarray(modern.path_gain, dtype=np.float32),
        np.asarray(legacy.path_gain, dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(modern.rss, dtype=np.float32),
        np.asarray(legacy.rss, dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_grid_spec_validates_shape_contract():
    with pytest.raises(ValueError, match="exactly one of grid_shape or cell_size"):
        GridSpec(
            axis="z",
            position=1.5,
            bounds=((-1.0, 1.0), (-1.0, 1.0)),
        )

    with pytest.raises(ValueError, match="exactly one of grid_shape or cell_size"):
        GridSpec(
            axis="z",
            position=1.5,
            bounds=((-1.0, 1.0), (-1.0, 1.0)),
            grid_shape=(8, 8),
            cell_size=0.5,
        )

    grid = GridSpec(
        axis="Z",
        position=1.5,
        bounds=((-2.0, 2.0), (-3.0, 3.0)),
        cell_size=(0.25, 0.5),
    )

    assert grid.axis == "z"
    assert grid.cell_size == (0.25, 0.5)


def test_solve_uses_scene_transmitter_and_receiver_grid_endpoints():
    scene = _build_channel_scene()
    scene.add_transmitter(
        Transmitter(
            name="tx",
            position=TX_POS,
            polarization=(0.0, 1.0, 0.0),
            power=2.5,
        )
    )
    scene.add_receiver(
        ReceiverGrid(
            name="map",
            axis="z",
            position=1.5,
            bounds=BOUNDS,
            grid_shape=(4, 4),
            polarization=None,
        )
    )

    result = solve(
        scene=scene,
        frequency=FREQUENCY,
        transmitter="tx",
        receiver="map",
        config=Config(
            metric="path_gain",
            shadow_boundary_correction=False,
            reflection_n_rays=128,
            reflection_max_bounces=0,
            max_diffractions=0,
        ),
    )

    assert result.grid_shape == (4, 4)
    assert result.tx_power == pytest.approx(2.5)


def test_sionna_first_order_diffraction_point_selection_matches_reference():
    edge_origin = wt.Point3f(1.0, 0.0, 0.5)
    edge_dir = wt.Vector3f(0.0, 0.0, 1.0)

    coplanar = Geo.first_order_diffraction_parameter(
        wt.Point3f(0.0, -5.0, 1.5),
        wt.Point3f(0.0, 5.0, 1.5),
        edge_origin,
        edge_dir,
    )
    elevated = Geo.first_order_diffraction_parameter(
        wt.Point3f(0.0, -5.0, 3.0),
        wt.Point3f(0.0, 5.0, 1.0),
        edge_origin,
        edge_dir,
    )

    assert float(coplanar[0]) == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert float(elevated[0]) == pytest.approx(1.5, rel=1e-6, abs=1e-6)


def test_deterministic_radiomap_config_rejects_removed_options_and_invalid_modes():
    for removed in (
        "tx_power",
        "noise_power",
        "reflection_coef",
        "reflection_relative_permittivity",
        "reflection_conductivity",
        "tx_polarization",
        "rx_polarization",
    ):
        with pytest.raises(TypeError, match=removed):
            Config(**{removed: 1.0})

    with pytest.raises(TypeError, match="accumulation_backend"):
        Config(accumulation_backend="native_monte_carlo")

    with pytest.raises(TypeError, match="combine_mode"):
        Config(combine_mode="invalid")

    with pytest.raises(TypeError, match="combine_mode"):
        Config(combine_mode="incoherent")

    with pytest.raises(TypeError, match="receiver_model"):
        Config(receiver_model="projected_polarized")

    with pytest.raises(TypeError, match="accumulation_backend"):
        Config(accumulation_backend="native_coherent")

    with pytest.raises(TypeError, match="shadow_boundary_mode"):
        Config(shadow_boundary_mode="matched_isb_completion")

    with pytest.raises(TypeError, match="completion"):
        Config(completion=False)

    with pytest.raises(TypeError, match="completion_backend"):
        Config(completion_backend="auto")

    with pytest.raises(ValueError, match="metric"):
        Config(metric="invalid")

    cfg = Config(
        reflection_n_rays=64,
        reflection_max_bounces=1,
        max_diffractions=0,
    )

    assert not hasattr(cfg, "combine_mode")
    assert not hasattr(cfg, "receiver_model")
    assert not hasattr(cfg, "accumulation_backend")
    assert not hasattr(cfg, "shadow_boundary_mode")
    assert cfg.shadow_boundary_correction is True
    assert cfg.max_diffractions == 0


def test_package_root_exports_only_short_public_names():
    assert drm.__all__ == [
        "Config",
        "FieldResult",
        "FieldSpec",
        "GridSpec",
        "Result",
        "native_extension_available",
        "solve",
        "solve_field",
    ]
    assert hasattr(drm, "Config")
    assert hasattr(drm, "FieldResult")
    assert hasattr(drm, "FieldSpec")
    assert hasattr(drm, "GridSpec")
    assert hasattr(drm, "Result")
    assert hasattr(drm, "solve")
    assert hasattr(drm, "solve_field")
    assert hasattr(drm, "native_extension_available")
    assert not hasattr(drm, "RadioMapConfig")
    assert not hasattr(drm, "AxisAlignedGridSpec")
    assert not hasattr(drm, "RadioMapResult")
    assert not hasattr(drm, "RadioMapSolver")
    assert not hasattr(drm, "Solver")


def test_channel_scene_adapter_exposes_runtime_views():
    adapted = _build_channel_scene(_cube_structure())

    ray = rayd.Ray(wt.Point3f(-4.0, 0.0, 1.5), wt.Vector3f(1.0, 0.0, 0.0))
    si = adapted.ray_intersect(ray)
    candidates = adapted.get_triangle_surface_edge_candidates(wt.UInt32(0))
    edge_data = adapted.get_edge_data(1.5)

    assert bool(np.asarray(adapted.ray_test(ray), dtype=np.bool_).reshape(-1)[0])
    assert bool(np.asarray(si.is_valid(), dtype=np.bool_).reshape(-1)[0])
    assert int(np.asarray(si.global_prim_id, dtype=np.int32).reshape(-1)[0]) >= 0
    assert adapted._triangle_runtime() is not None
    assert int(np.asarray(adapted.gather_structure_indices(wt.UInt32(0)), dtype=np.int32).reshape(-1)[0]) == 0
    assert int(np.asarray(candidates["count"], dtype=np.uint32).reshape(-1)[0]) >= 0
    assert edge_data["edge_data"] is not None
    assert len(edge_data["diffraction_points"]) > 0


def test_channel_scene_adapter_globalizes_multi_mesh_primitive_indices():
    adapted = _two_triangle_mesh_scene()
    ray = rayd.Ray(wt.Point3f(2.25, 0.25, 1.0), wt.Vector3f(0.0, 0.0, -1.0))

    si = adapted.ray_intersect(ray)

    prim_index = int(np.asarray(si.global_prim_id, dtype=np.int32).reshape(-1)[0])
    owner_index = int(
        np.asarray(adapted.gather_structure_indices(wt.UInt32(prim_index)), dtype=np.int32).reshape(-1)[0]
    )

    assert bool(np.asarray(si.is_valid(), dtype=np.bool_).reshape(-1)[0])
    assert prim_index == 1
    assert owner_index == 1


def test_channel_scene_adapter_preserves_geometry_and_material_gradients():
    mesh_vertices = wt.Point3f(
        wt.Float([0.0, 1.0, 0.0]),
        wt.Float([0.0, 0.0, 1.0]),
        wt.Float([0.0, 0.0, 0.0]),
    )
    dr.enable_grad(mesh_vertices.x, mesh_vertices.y, mesh_vertices.z)
    scene = _triangle_mesh_scene(mesh_vertices)
    adapted = scene
    ray = rayd.Ray(wt.Point3f(0.25, 0.25, 1.0), wt.Vector3f(0.0, 0.0, -1.0))

    tri_data = adapted._triangle_runtime()
    dr.enable_grad(tri_data["material_eps_r"])

    si = adapted.ray_intersect(ray)
    material = adapted.triangle_material(wt.UInt32(0))

    assert bool(np.asarray(si.is_valid(), dtype=np.bool_).reshape(-1)[0])
    assert dr.grad_enabled(si.t)
    assert dr.grad_enabled(material["eps_r"])

    dr.backward(dr.sum(si.t) + dr.sum(material["eps_r"]))

    assert float(np.sum(np.abs(np.asarray(dr.grad(mesh_vertices.z), dtype=np.float64)))) > 0.0
    assert float(np.sum(np.abs(np.asarray(dr.grad(tri_data["material_eps_r"]), dtype=np.float64)))) > 0.0


def test_channel_scene_adapter_clone_and_update_vertices_syncs_runtime():
    adapted = _build_channel_scene(_cube_structure())
    clone = adapted.clone()
    vertices_before = clone._merged_vertices()
    before_sum_x = _scalar(dr.sum(vertices_before.x))
    vertex_count = int(dr.width(vertices_before.x))
    translated = wt.Point3f(
        vertices_before.x + 0.25,
        vertices_before.y,
        vertices_before.z,
    )

    clone.update_vertices(translated)

    after_sum_x = _scalar(dr.sum(clone._merged_vertices().x))
    assert after_sum_x == pytest.approx(before_sum_x + 0.25 * vertex_count, abs=1.0e-5)
    assert _scalar(dr.sum(adapted._merged_vertices().x)) == pytest.approx(before_sum_x, abs=1.0e-5)


def test_standalone_solver_matches_legacy_deterministic_on_simple_scene():
    legacy = _legacy_result()
    modern = _standalone_result()

    _assert_metric_close(modern, legacy)
    assert set(modern.components) >= {"los", "reflection", "diffraction", "path_gain"}
    np.testing.assert_allclose(
        np.asarray(modern.components["reflection"], dtype=np.float32),
        np.asarray(legacy.incoherent["reflection"], dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    for removed_field in (
        "coherent",
        "coherent_power",
        "power_diagnostics",
        "timing",
    ):
        assert not hasattr(modern, removed_field)


def test_standalone_result_exposes_runtime_metadata():
    result = _standalone_result()

    metadata = result.metadata
    assert metadata["scene_summary"]["n_structures"] == 1
    assert metadata["scene_summary"]["n_triangles"] > 0
    assert metadata["solver_controls"]["selected"] == "accuracy"
    assert metadata["shadow_boundary_correction"]["enabled"] is False
    for key in (
        "grid_resolution_seconds",
        "los_trace_seconds",
        "reflection_trace_seconds",
        "diffraction_state_preparation_seconds",
        "diffraction_accumulation_seconds",
        "shadow_boundary_correction_seconds",
        "result_shaping_seconds",
        "total_solve_seconds",
    ):
        assert key in metadata["performance_timing"]
        assert metadata["performance_timing"][key] >= 0.0


def test_standalone_shadow_boundary_correction_metadata_reports_statistics_backend():
    result = solve(
        scene=_build_channel_scene(),
        frequency=FREQUENCY,
        tx_pos=TX_POS,
        grid=GridSpec(
            axis="z",
            position=1.5,
            bounds=BOUNDS,
            grid_shape=(8, 8),
        ),
        config=Config(
            metric="path_gain",
            shadow_boundary_correction=True,
            shadow_boundary_backend="auto",
            reflection_n_rays=32,
            reflection_max_bounces=0,
            enable_rd_diffraction=False,
            max_diffractions=0,
        ),
    )

    correction_metadata = result.metadata["shadow_boundary_correction"]
    stats = correction_metadata["statistics"]
    assert correction_metadata["enabled"] is True
    assert correction_metadata["applied"] is True
    assert correction_metadata["backend"] == "dense_native"
    assert stats["resolved_backend"] == "dense_native"
    assert stats["full_pair_count"] <= stats["dense_pair_limit"]


def test_standalone_result_exposes_diffraction_builder_reports():
    result = solve(
        scene=_build_channel_scene(),
        frequency=FREQUENCY,
        tx_pos=TX_POS,
        grid=GridSpec(
            axis="z",
            position=1.5,
            bounds=BOUNDS,
            grid_shape=(8, 8),
        ),
        config=Config(
            metric="path_gain",
            shadow_boundary_correction=False,
            reflection_n_rays=32,
            reflection_max_bounces=0,
            enable_rd_diffraction=True,
            max_diffractions=1,
        ),
    )

    builder_reports = result.metadata["diffraction"]["builder_reports"]
    assert len(builder_reports) == 1
    report = builder_reports[0]
    assert report["edge_count"] > 0
    assert report["candidate_backend"] == "not_used"
    assert report["stages"]["tx_first"] >= 0
    assert report["stages"]["first_order"] >= report["stages"]["tx_first"]
    assert report["final_state_count"] == result.metadata["diffraction"]["state_count_total"]
    assert result.metadata["diffraction"]["receiver_tile_count"] == 1
    assert result.metadata["diffraction"]["receiver_tiling_enabled"] is False


def test_standalone_solver_rejects_native_coherent_backend():
    with pytest.raises(TypeError, match="accumulation_backend"):
        Config(accumulation_backend="native_coherent")


def test_standalone_native_module_exports_expected_symbols_when_available():
    if not native_extension_available():
        pytest.skip("Standalone deterministic native extension is unavailable in this environment.")

    from witwin.deterministic._native import _extension
    ext = _extension()
    assert hasattr(ext, "utd_accumulate_tiled_vectors")
    assert hasattr(ext, "utd_pair_vectors")
    assert not hasattr(ext, "utd_accumulate_tiled_arrays_v2")
    assert not hasattr(ext, "utd_accumulate_scalar_power_arrays")
    assert hasattr(ext, "reflection_accumulate")
    assert hasattr(ext, "reflection_epc_targets_forward_arrays")
    assert hasattr(ext, "reflection_prefix_filter_arrays")
    assert hasattr(ext, "radiomap_accumulate_vector_power_pairs")
    assert hasattr(ext, "suffix_grid_forward_raw")
    assert hasattr(ext, "pack_state_arrays_raw")
    assert hasattr(ext, "cartesian_filter_bruteforce_arrays")
    assert hasattr(ext, "prune_state_arrays_by_budget_arrays")


def test_standalone_solver_keeps_tx_gradient_flow():
    tx_x = wt.Float(-2.5)
    dr.enable_grad(tx_x)
    config = Config(
        reflection_n_rays=32,
        reflection_max_bounces=0,
        max_diffractions=0,
    )
    grid = GridSpec(
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
        grid_shape=(4, 4),
    )

    result = solve(
        scene=_build_channel_scene(),
        frequency=FREQUENCY,
        tx_pos=wt.Point3f(tx_x, 0.0, 1.5),
        grid=grid,
        config=config,
    )
    loss = dr.sum(result.path_gain)
    dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
    grad = _scalar(dr.grad(tx_x))

    plus = solve(
        scene=_build_channel_scene(),
        frequency=FREQUENCY,
        tx_pos=wt.Point3f(-2.5 + 1.0e-3, 0.0, 1.5),
        grid=grid,
        config=config,
    )
    minus = solve(
        scene=_build_channel_scene(),
        frequency=FREQUENCY,
        tx_pos=wt.Point3f(-2.5 - 1.0e-3, 0.0, 1.5),
        grid=grid,
        config=config,
    )
    fd = float(
        (
            np.asarray(plus.path_gain, dtype=np.float64)
            - np.asarray(minus.path_gain, dtype=np.float64)
        ).sum()
        / (2.0e-3)
    )

    assert np.isfinite(grad)
    assert abs(grad) > 0.0
    assert grad == pytest.approx(fd, rel=2.0e-1, abs=5.0e-2)


def test_standalone_matched_isb_profile_keeps_geometry_gradient_flow():
    wall_x = wt.Float(0.0)
    dr.enable_grad(wall_x)
    config = Config(
        reflection_n_rays=32,
        reflection_max_bounces=1,
        enable_rd_diffraction=True,
        max_diffractions=1,
    )
    grid = GridSpec(
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
        grid_shape=(8, 8),
    )
    base_vertices, base_faces = Box(
        position=(0.0, 0.0, 0.0),
        size=(0.25, 6.0, 3.0),
        device="cuda",
    ).to_mesh()
    base_vertices = to_point3f(base_vertices)
    structure = Structure(
        name="wall",
        geometry=ChannelMesh(
            vertices=wt.Point3f(
                base_vertices.x + wall_x,
                base_vertices.y,
                base_vertices.z + 1.5,
            ),
            faces=to_vector3u(base_faces),
        ),
        material=Material(eps_r=4.0),
    )

    result = solve(
        scene=_build_channel_scene(structure),
        frequency=FREQUENCY,
        tx_pos=TX_POS,
        grid=grid,
        config=config,
    )
    dr.set_grad(wall_x, 1.0)
    jvp = np.asarray(
        dr.forward_to(
            result.path_gain,
            flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad,
        ),
        dtype=np.float64,
    )

    assert jvp.shape == (8, 8)
    assert np.all(np.isfinite(jvp))


def test_standalone_matched_isb_profile_runs_without_native_extension():
    if native_extension_available():
        pytest.skip("This regression covers the no-native matched-ISB fallback path.")

    tx_x = wt.Float(-2.5)
    dr.enable_grad(tx_x)
    config = Config(
        reflection_n_rays=32,
        reflection_max_bounces=1,
        enable_rd_diffraction=True,
        max_diffractions=1,
    )
    grid = GridSpec(
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
        grid_shape=(8, 8),
    )

    result = solve(
        scene=_build_channel_scene(),
        frequency=FREQUENCY,
        tx_pos=wt.Point3f(tx_x, 0.0, 1.5),
        grid=grid,
        config=config,
    )

    path_gain = np.asarray(result.path_gain, dtype=np.float64)
    assert path_gain.shape == (8, 8)
    assert np.all(np.isfinite(path_gain))
    assert set(result.components) >= {"los", "reflection", "diffraction", "path_gain"}
    for values in result.components.values():
        component = np.asarray(values, dtype=np.float64)
        assert component.shape == (8, 8)
        assert np.all(np.isfinite(component))
    assert not hasattr(result, "combine_mode")
    assert not hasattr(result, "receiver_model")
    for removed_field in (
        "coherent",
        "coherent_power",
        "power_diagnostics",
        "timing",
    ):
        assert not hasattr(result, removed_field)

    dr.set_grad(tx_x, 1.0)
    jvp = np.asarray(
        dr.forward_to(
            result.path_gain,
            flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad,
        ),
        dtype=np.float64,
    )
    assert jvp.shape == (8, 8)
    assert np.all(np.isfinite(jvp))
    assert float(np.sum(np.abs(jvp))) > 0.0


def test_reflection_path_discovery_globalizes_multi_mesh_primitives():
    scene = _three_cube_scene()
    tx_pos = wt.Point3f(0.0, -5.0, 4.0)
    wavelength = 299792458.0 / FREQUENCY
    k = 2.0 * np.pi / wavelength

    detail = discover_paths(
        tx=Tx(tx_pos),
        scene=scene,
        wave=Wave(wavelength=wavelength, k=k),
        n_rays=4096,
        max_reflections=3,
        mode="3d",
        ray_sampling="full_sphere",
        sampling_axis="z",
        sampling_plane_position=1.0,
        sampling_bounds=((-10.0, 10.0), (-10.0, 10.0)),
        material=RuntimeMaterial(reflection_coef=0.7),
    )

    first_bounce = detail.source_paths_per_bounce[0]
    owners = np.asarray(scene.gather_structure_indices(first_bounce.path_prim_idx[0]), dtype=np.int32)
    unique_owners = {int(value) for value in owners.reshape(-1).tolist() if int(value) >= 0}

    assert int(first_bounce.n_paths) > 0
    assert len(unique_owners) >= 2
