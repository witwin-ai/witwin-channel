"""Integration coverage for the standalone Monte Carlo radiomap package."""

from __future__ import annotations

import math

import drjit as dr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import rayd
import witwin as wt
import witwin.montecarlo as mc_rm

from examples.monte_carlo_radiomap_three_cubes import (
    DEFAULT_FD_STEP,
    DEFAULT_FREQUENCY_HZ,
    SolveSnapshot,
    ThreeCubeExperiment,
    plot_forward_components,
    plot_shadow_boundary,
)
from witwin.channel import RadioMapMonitor, Scene as LegacyScene, Tracer
from witwin.channel_scene import ReceiverGrid, Scene as ChannelScene, Transmitter
import witwin.channel.monitors.radio_map.backend as legacy_rm_backend
from witwin.core import Box, Material, Structure
from witwin.montecarlo.config import ResolvedTraceConfig, resolve_solver_controls
from witwin.montecarlo import (
    GridSpec,
    Config,
    NativeExtension,
    solve,
)
import witwin.montecarlo.integrators.basic as package_rm_integrator
from witwin.montecarlo.integrators.basic import Basic
from witwin.montecarlo.integrators.basic_ad import BasicIntegratorAD
from witwin.montecarlo.integrators.bdpt_diffraction import BDPTDiffractionMIS
from witwin.montecarlo.kernels.sparse_coeff import SparseCoeffKernel
from witwin.montecarlo.kernels.transport_grid import TransportGridKernel
from witwin.montecarlo.grid import Grid
from witwin.montecarlo import grid
from witwin.montecarlo import materials
from witwin.montecarlo.path.ad_support import SceneQuery
from witwin.montecarlo.path.diffraction_ad import DiffractionAD
from witwin.montecarlo.path.diffraction_support import (
    DiffractionEdgeSampler,
    DiffractionScene,
)
from witwin.montecarlo.path.reflection_ad import ReflectionAD
from witwin.montecarlo.sampler import Sampler
from witwin.channel_utils import arrays
from witwin.channel_utils.arrays import scalar
pytestmark = pytest.mark.gpu


def _wall_structure() -> Structure:
    return Structure(
        name="wall",
        geometry=Box(
            position=(0.0, 0.0, 1.5),
            size=(0.25, 4.0, 3.0),
            device="cuda",
        ),
        material=Material(eps_r=4.0, sigma_e=0.0),
    )


def _build_channel_scene() -> ChannelScene:
    return ChannelScene(
        structures=[_wall_structure()],
        device="cuda",
    )


def _build_three_cube_scene() -> ChannelScene:
    material = Material(eps_r=4.0, sigma_e=0.0)
    centers = (
        (-2.5, -3.0, 1.5),
        (2.0, 0.5, 1.5),
        (-0.5, 3.5, 1.5),
    )
    return ChannelScene(
        structures=[
            Structure(
                name=f"cube_{index}",
                geometry=Box(position=center, size=(2.0, 2.0, 2.0), device="cuda"),
                material=material,
            )
            for index, center in enumerate(centers)
        ],
        device="cuda",
        edge_selection_mode="all_edges",
    )


def _build_legacy_scene() -> LegacyScene:
    return LegacyScene(
        structures=[_wall_structure()],
        device="cuda",
    )


def test_monte_carlo_metadata_reports_channel_scene_edge_diffraction_default():
    result = solve(
        scene=_build_channel_scene(),
        frequency=DEFAULT_FREQUENCY_HZ,
        tx_pos=wt.Point3f(-2.0, -2.0, 1.5),
        grid=GridSpec(
            axis="z",
            position=1.0,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(4, 4),
        ),
        config=Config(
            integrator="basic",
            reflection_n_rays=4,
            reflection_max_bounces=0,
            samples_per_tx=4,
            enable_rd_diffraction=False,
            max_diffractions=0,
            shadow_boundary_mode="none",
            seed=3,
        ),
    )

    scene_metadata = result.metadata["scene"]
    assert scene_metadata["edge_diffraction"] is True
    assert scene_metadata["boundary_edge_policy"] == "half_plane"
    assert scene_metadata["n_diffraction_edges"] == _build_channel_scene().n_diffraction_edges


def test_bdpt_reflection_budget_uses_reflection_n_rays_not_diffraction_samples():
    if not NativeExtension.native_extension_available():
        pytest.skip("Monte Carlo native extension is unavailable in this environment.")

    result = solve(
        scene=_build_channel_scene(),
        frequency=DEFAULT_FREQUENCY_HZ,
        tx_pos=wt.Point3f(-2.0, -2.0, 1.5),
        grid=GridSpec(
            axis="z",
            position=1.0,
            bounds=((-3.0, 3.0), (-3.0, 3.0)),
            grid_shape=(8, 8),
        ),
        config=Config(
            integrator="bdpt",
            reflection_n_rays=16,
            reflection_max_bounces=1,
            samples_per_tx=3,
            enable_rd_diffraction=False,
            max_diffractions=0,
            shadow_boundary_mode="none",
            seed=5,
        ),
    )

    reflection_runtime = result.metadata["runtime_backends"]["reflection"]
    assert reflection_runtime["ray_budget"] == 16
    assert result.metadata["ray_sampling"]["samples_per_tx"] == 16
    assert result.metadata["ray_sampling"]["diffraction_samples_per_tx"] == 3
    assert result.metadata["monte_carlo"]["reflection_n_rays"] == 16


def test_three_cube_example_plots_shadow_boundary_correction_component():
    values = np.ones((2, 2), dtype=np.float64) * 1.0e-5
    correction = np.array(
        [[-1.0e-6, 0.0], [5.0e-7, -2.0e-6]],
        dtype=np.float64,
    )
    snapshot = SolveSnapshot(
        path_gain=values + correction,
        coords_x=np.array([-0.5, 0.5], dtype=np.float64),
        coords_y=np.array([-0.5, 0.5], dtype=np.float64),
        metadata={},
        components={
            "los": values,
            "reflection": values * 0.1,
            "diffraction": values * 0.01,
            "shadow_boundary_correction": correction,
        },
    )

    try:
        axes = plot_forward_components(snapshot, bounds=((-1.0, 1.0), (-1.0, 1.0)))
        axes_flat = np.asarray(axes, dtype=object).reshape(-1)
        assert len(axes_flat) == 4
        assert "Shadow Boundary Correction" in axes_flat[-1].get_title()

        shadow_ax = plot_shadow_boundary(snapshot, bounds=((-1.0, 1.0), (-1.0, 1.0)))
        assert "Shadow Boundary Correction" in shadow_ax.get_title()
    finally:
        plt.close("all")


def _tent_splat_reference(*, grid: Grid, coord_0, coord_1, power, active):
    n_coord_0 = int(grid.grid_shape[0])
    n_coord_1 = int(grid.grid_shape[1])
    out = dr.zeros(wt.Float, int(grid.n_cells))
    scaled_0 = (coord_0 - wt.Float(grid.bounds[0][0])) / wt.Float(grid.cell_size[0]) - wt.Float(0.5)
    scaled_1 = (coord_1 - wt.Float(grid.bounds[1][0])) / wt.Float(grid.cell_size[1]) - wt.Float(0.5)
    base_0_float = dr.floor(scaled_0)
    base_1_float = dr.floor(scaled_1)
    frac_0 = scaled_0 - base_0_float
    frac_1 = scaled_1 - base_1_float
    base_0 = wt.Int32(base_0_float)
    base_1 = wt.Int32(base_1_float)
    next_0 = base_0 + wt.Int32(1)
    next_1 = base_1 + wt.Int32(1)
    weight_0 = (wt.Float(1.0) - frac_0, frac_0)
    weight_1 = (wt.Float(1.0) - frac_1, frac_1)
    index_0 = (base_0, next_0)
    index_1 = (base_1, next_1)
    valid_0 = (
        (base_0 >= 0) & (base_0 < wt.Int32(n_coord_0)),
        (next_0 >= 0) & (next_0 < wt.Int32(n_coord_0)),
    )
    valid_1 = (
        (base_1 >= 0) & (base_1 < wt.Int32(n_coord_1)),
        (next_1 >= 0) & (next_1 < wt.Int32(n_coord_1)),
    )
    for yi in range(2):
        for xi in range(2):
            cell_active = active & valid_0[xi] & valid_1[yi]
            safe_cell_idx = wt.UInt32(
                dr.select(
                    cell_active,
                    index_1[yi] * wt.Int32(n_coord_0) + index_0[xi],
                    wt.Int32(0),
                )
            )
            weighted_power = power * weight_0[xi] * weight_1[yi]
            dr.scatter_reduce(
                dr.ReduceOp.Add,
                out,
                dr.select(cell_active, weighted_power, wt.Float(0.0)),
                safe_cell_idx,
                cell_active,
            )
    return out


def _reference_best_edge_indices(*, tx_pos, ray_directions, prim_index, hit_p, hit_n, hit_geo_n, hit, scene):
    n_rays = int(dr.width(ray_directions.x))
    if n_rays <= 0:
        return dr.zeros(wt.Int32, 0)
    triangle_edges = scene.get_triangle_surface_edge_candidates(prim_index)
    candidate_slots = triangle_edges.get("slots", ())
    if len(candidate_slots) <= 0:
        return dr.full(wt.Int32, -1, n_rays)
    viewpoint = DiffractionEdgeSampler.silhouette_viewpoint(hit_p, hit_n, hit_geo_n, ray_directions)
    tx_lanes = arrays.broadcast_point(tx_pos, n_rays)
    best_edge_idx = dr.full(wt.Int32, -1, n_rays)
    best_distance = dr.full(wt.Float, 1.0e30, n_rays)
    for slot_index, edge_idx in enumerate(candidate_slots):
        candidate_active = hit & (triangle_edges["count"] > wt.UInt32(slot_index)) & (edge_idx >= 0)
        edge_data = DiffractionEdgeSampler.gather_edge_subset(scene, edge_idx, valid_mask=candidate_active)
        edge_point, _ = DiffractionEdgeSampler.closest_point_on_edge(
            query_point=viewpoint,
            edge_pos=edge_data["pos"],
            edge_dir=edge_data["edge_dir"],
            line_min=edge_data["line_min"],
            line_max=edge_data["line_max"],
        )
        flip = dr.dot(ray_directions, edge_data["n0"]) > 0.0
        oriented_n0 = dr.select(flip, edge_data["n_face_n"], edge_data["n0"])
        oriented_nn = dr.select(flip, edge_data["n0"], edge_data["n_face_n"])
        exterior = DiffractionScene.wedge_exterior(
            tx_lanes - edge_point,
            edge_data["edge_dir"],
            oriented_n0,
            oriented_nn,
        )
        view_vec = viewpoint - edge_point
        face0_front = dr.dot(view_vec, edge_data["n0"]) > wt.Float(1.0e-6)
        face1_front = dr.dot(view_vec, edge_data["n_face_n"]) > wt.Float(1.0e-6)
        silhouette = (edge_data["adjacent_face1"] < 0) | (face0_front != face1_front)
        candidate_distance = dr.dot(view_vec, view_vec)
        better = candidate_active & exterior & silhouette & (candidate_distance < best_distance)
        best_distance = dr.select(better, candidate_distance, best_distance)
        best_edge_idx = dr.select(better, wt.Int32(edge_idx), best_edge_idx)
    return best_edge_idx


def _prepare_three_cube_fixed_tape_context(*, grid_size: int = 16, samples_per_tx: int = 1024):
    experiment = ThreeCubeExperiment(
        grid_shape=(grid_size, grid_size),
        samples_per_tx=samples_per_tx,
        forward_reflection_n_rays=32,
        gradient_reflection_n_rays=32,
        seed=7,
    )
    mc_config = experiment.gradient_config
    trace_config = mc_config.to_trace_config()
    resolved = ResolvedTraceConfig.from_config(
        frequency=DEFAULT_FREQUENCY_HZ,
        config=trace_config,
    )
    solver_controls = resolve_solver_controls(
        trace_config,
        execution_intent="radio_map_incoherent",
        max_diffractions_override=int(mc_config.max_diffractions),
    )
    tx_pos = wt.Point3f(*experiment.tx_pos)
    base_scene = experiment._build_channel_scene(cube1_x=wt.Float(experiment.base_centers[0][0]))
    detached = BasicIntegratorAD.detached_workload(base_scene, tx_pos, resolved)
    primal_state = Basic.primal(
        detached["tx_pos"],
        experiment.grid,
        mc_config,
        detached["scene"],
        resolved,
        solver_controls,
        accumulation_backend=str(mc_config.accumulation_backend),
        return_timing=False,
        resolved_ad_mode=True,
        ad_backend="test_fixed_tape_fd",
        loop_mode="symbolic",
        collect_ad_tapes=True,
    )
    grid = primal_state.grid
    diff_gain_scale = wt.Float(
        (float(resolved.wavelength) / (4.0 * math.pi)) ** 2
        / float(grid.cell_size[0] * grid.cell_size[1])
    )
    solid_angle_per_ray = Sampler.solid_angle(
        Sampler.metadata(
            axis=str(grid.axis),
            plane_position=float(grid.position),
            tx_pos=detached["tx_pos"],
        ),
        int(mc_config.samples_per_tx),
    )
    return {
        "experiment": experiment,
        "mc_config": mc_config,
        "resolved": resolved,
        "tx_pos": tx_pos,
        "base_scene": base_scene,
        "detached": detached,
        "primal_state": primal_state,
        "grid": grid,
        "diff_gain_scale": diff_gain_scale,
        "solid_angle_per_ray": float(solid_angle_per_ray),
        "material_omega": materials.material_angular_frequency(resolved.wavelength),
        "total_length_weight": float(primal_state.diff_length_weight),
    }


def test_axis_aligned_grid_spec_validates_shape_contract():
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


def test_monte_carlo_radiomap_config_rejects_unsupported_contracts():
    for removed in (
        "reflection_coef",
        "tx_power",
        "noise_power",
        "reflection_material",
        "diffraction_material",
        "tx_polarization",
        "rx_polarization",
    ):
        with pytest.raises(TypeError, match=removed):
            Config(**{removed: 1.0})

    with pytest.raises(ValueError, match="max_diffractions"):
        Config(max_diffractions=2)

    with pytest.raises(ValueError, match="accumulation_backend"):
        Config(accumulation_backend="baseline")

    with pytest.raises(ValueError, match="shadow_boundary_mode"):
        Config(shadow_boundary_mode="utd_cross_term_surrogate")

    cfg = Config(
        reflection_n_rays=64,
        reflection_max_bounces=1,
        samples_per_tx=128,
        max_diffractions=1,
        accumulation_backend="auto",
    )

    assert cfg.reflection_n_rays == 64
    assert cfg.samples_per_tx == 128
    assert cfg.max_diffractions == 1
    assert cfg.shadow_boundary_mode == "utd_power_smoothing"
    assert Config(shadow_boundary_mode="none").shadow_boundary_mode == "none"

    bdpt_cfg = Config(integrator="bdpt", max_diffractions=3)
    assert bdpt_cfg.max_diffractions == 3


def test_monte_carlo_solve_uses_scene_transmitter_and_receiver_grid_endpoints():
    scene = ChannelScene(
        structures=[
            Structure(
                name="distant_block",
                geometry=Box(position=(5.0, 5.0, 5.0), size=(0.25, 0.25, 0.25), device="cuda"),
                material=Material(eps_r=4.0, sigma_e=0.0),
            )
        ],
        transmitters=[Transmitter(name="tx", position=(0.0, 0.0, 1.0), power=2.5)],
        receivers=[
            ReceiverGrid(
                name="map",
                axis="z",
                position=1.0,
                bounds=((-1.0, 1.0), (-1.0, 1.0)),
                grid_shape=(4, 4),
            )
        ],
        device="cuda",
    )
    result = solve(
        scene=scene,
        frequency=1.0e9,
        transmitter="tx",
        receiver="map",
        config=Config(
            reflection_n_rays=8,
            reflection_max_bounces=0,
            samples_per_tx=8,
            max_diffractions=0,
            shadow_boundary_mode="none",
            ad=False,
        ),
    )

    assert result.tx_power == 2.5
    assert result.grid_shape == (4, 4)
    assert result.metadata["tx_power"] == 2.5


def test_power_domain_shadow_boundary_finalization_smooths_incident_step():
    smoothed = package_rm_integrator._empty_radio_map(4)
    smoothed["incoherent"]["los"] = wt.Float([1.0, 1.0, 0.0, 0.0])
    smoothed["incoherent"]["diffraction"] = wt.Float([0.25, 0.25, 0.25, 0.25])
    smoothed["incoherent"]["continued_incident_power"] = wt.Float(
        [1.0, 1.0, 1.0, 1.0]
    )
    smoothed["incoherent"]["incident_shadow_boundary_weight"] = wt.Float(
        [1.0, 1.0, 1.0, 1.0]
    )
    smoothed["incoherent"]["diffraction_incident_transition_power"] = wt.Float(
        [0.25, 0.25, 0.25, 0.25]
    )
    package_rm_integrator._finalize_component_totals(
        smoothed,
        shadow_boundary_mode="utd_power_smoothing",
    )

    raw = np.asarray(smoothed["incoherent"]["raw_total"], dtype=np.float32)
    total = np.asarray(smoothed["incoherent"]["total"], dtype=np.float32)
    correction = np.asarray(
        smoothed["incoherent"]["shadow_boundary_correction"],
        dtype=np.float32,
    )
    assert np.max(np.abs(np.diff(total))) < np.max(np.abs(np.diff(raw)))
    assert correction[0] < 0.0
    assert correction[-1] >= -1.0e-12
    assert np.all(total >= 0.0)


def test_power_domain_shadow_boundary_requires_sampled_incident_transition_support():
    smoothed = package_rm_integrator._empty_radio_map(2)
    smoothed["incoherent"]["continued_incident_power"] = wt.Float([1.0, 1.0])
    smoothed["incoherent"]["incident_shadow_boundary_weight"] = wt.Float([1.0, 1.0])
    package_rm_integrator._finalize_component_totals(
        smoothed,
        shadow_boundary_mode="utd_power_smoothing",
    )

    np.testing.assert_array_equal(
        np.asarray(smoothed["incoherent"]["shadow_boundary_correction"], dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(smoothed["incoherent"]["total"], dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )


def test_power_domain_shadow_boundary_finalization_smooths_reflection_step():
    smoothed = package_rm_integrator._empty_radio_map(4)
    smoothed["incoherent"]["reflection"] = wt.Float([0.64, 0.64, 0.0, 0.0])
    smoothed["incoherent"]["diffraction"] = wt.Float([0.16, 0.16, 0.16, 0.16])
    smoothed["incoherent"]["reflection_shadow_boundary_weight"] = wt.Float(
        [1.0, 1.0, 1.0, 1.0]
    )
    smoothed["incoherent"]["diffraction_reflection_transition_power"] = wt.Float(
        [0.16, 0.16, 0.16, 0.16]
    )
    package_rm_integrator._finalize_component_totals(
        smoothed,
        shadow_boundary_mode="utd_power_smoothing",
    )

    none = package_rm_integrator._empty_radio_map(4)
    none["incoherent"]["reflection"] = smoothed["incoherent"]["reflection"]
    none["incoherent"]["diffraction"] = smoothed["incoherent"]["diffraction"]
    package_rm_integrator._finalize_component_totals(
        none,
        shadow_boundary_mode="none",
    )

    default_total = np.asarray(smoothed["incoherent"]["total"], dtype=np.float32)
    none_total = np.asarray(none["incoherent"]["total"], dtype=np.float32)
    reflection_proxy = np.asarray(
        smoothed["incoherent"]["diffraction_reflection_transition_power"],
        dtype=np.float32,
    )
    correction = np.asarray(
        smoothed["incoherent"]["shadow_boundary_correction"],
        dtype=np.float32,
    )
    assert np.max(np.abs(np.diff(default_total))) < np.max(np.abs(np.diff(none_total)))
    assert np.any(reflection_proxy > 0.0)
    assert np.all(correction <= 0.0)
    np.testing.assert_allclose(
        np.asarray(none["incoherent"]["raw_total"], dtype=np.float32),
        none_total,
    )


def test_package_root_exports_only_short_public_names():
    assert mc_rm.__all__ == [
        "Config",
        "ComponentFilterConfig",
        "FilterConfig",
        "GridSpec",
        "NativeExtension",
        "Result",
        "solve",
    ]
    assert hasattr(mc_rm, "Config")
    assert hasattr(mc_rm, "ComponentFilterConfig")
    assert hasattr(mc_rm, "FilterConfig")
    assert hasattr(mc_rm, "GridSpec")
    assert hasattr(mc_rm, "NativeExtension")
    assert hasattr(mc_rm, "Result")
    assert hasattr(mc_rm, "solve")
    assert not hasattr(mc_rm, "Solver")
    assert not hasattr(mc_rm, "RadioMapConfig")
    assert not hasattr(mc_rm, "AxisAlignedGridSpec")
    assert not hasattr(mc_rm, "RadioMapResult")
    assert not hasattr(mc_rm, "RadioMapSolver")


def test_channel_scene_adapter_exposes_scene_runtime_views():
    scene = _build_channel_scene()
    adapted = scene

    assert adapted._triangle_runtime() is not None
    assert adapted._selected_edge_runtime() is not None
    assert len(adapted._selected_edge_views()) > 0

    material = adapted.triangle_material(wt.UInt32(0))
    assert scalar(material["eps_r"]) == pytest.approx(4.0)
    assert bool(np.asarray(material["specified"]).reshape(-1)[0])
    assert int(scalar(material["structure_idx"])) == 0
    assert int(scalar(adapted.gather_structure_indices(wt.UInt32(0)))) == 0

    candidates = adapted.get_triangle_surface_edge_candidates(wt.UInt32(0))
    assert int(scalar(candidates["count"])) >= 0


def test_standalone_solver_matches_legacy_monte_carlo_on_simple_scene(monkeypatch):
    monkeypatch.setattr(legacy_rm_backend, "native_extension_available", lambda: True)
    monkeypatch.setattr(
        package_rm_integrator.NativeExtension,
        "native_extension_available",
        staticmethod(lambda: True),
    )

    legacy_scene = _build_legacy_scene()
    channel_scene = _build_channel_scene()
    tx_pos = wt.Point3f(-2.0, 0.0, 1.5)

    legacy_tracer = Tracer(
        frequency=3.5e9,
        scene=legacy_scene,
        reflection_n_rays=128,
        reflection_max_bounces=1,
        max_diffractions=0,
    )
    legacy_monitor = RadioMapMonitor(
        "legacy_monte_carlo",
        axis="z",
        position=1.5,
        bounds=((-3.0, 3.0), (-3.0, 3.0)),
        grid_shape=(8, 8),
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        accumulation_backend="auto",
        sampling_mode="monte_carlo",
        samples_per_tx=128,
        max_diffractions=0,
        seed=0,
    )

    legacy = legacy_tracer.trace(tx_pos, monitor=legacy_monitor, verbose=False)
    modern = solve(
        scene=channel_scene,
        frequency=3.5e9,
        tx_pos=tx_pos,
        grid=GridSpec(
            axis="z",
            position=1.5,
            bounds=((-3.0, 3.0), (-3.0, 3.0)),
            grid_shape=(8, 8),
        ),
        config=Config(
            reflection_n_rays=128,
            reflection_max_bounces=1,
            samples_per_tx=128,
            max_diffractions=0,
            accumulation_backend="auto",
            seed=0,
        ),
    )

    np.testing.assert_allclose(
        np.asarray(modern.incoherent["reflection"], dtype=np.float32),
        np.asarray(legacy.incoherent["reflection"], dtype=np.float32),
        rtol=5.0e-4,
        atol=1.0e-6,
    )
    assert modern.metadata["sampling_mode"] == "monte_carlo"
    assert modern.metadata["accumulation_backend"]["resolved"] == "native_monte_carlo"
    assert modern.metadata["accumulation_backend"]["cell_accumulation_mode"] == "native_monte_carlo_scatter_delta"
    assert modern.metadata["monte_carlo"]["samples_per_tx"] == 128
    assert modern.metadata["monte_carlo"]["shadow_boundary_mode"] == "utd_power_smoothing"
    assert modern.metadata["monte_carlo"]["los_strategy"] == "cell_center_visibility"
    assert "raw_total" in modern.incoherent
    assert "shadow_boundary_correction" in modern.incoherent
    assert "diffraction_incident_transition_power" in modern.incoherent
    assert "diffraction_reflection_transition_power" in modern.incoherent
    assert "incident_shadow_boundary_weight" in modern.incoherent
    assert "reflection_shadow_boundary_weight" in modern.incoherent
    np.testing.assert_allclose(
        np.asarray(modern.path_gain, dtype=np.float32),
        np.asarray(modern.incoherent["total"], dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        np.asarray(modern.incoherent["raw_total"], dtype=np.float32),
        np.asarray(modern.incoherent["total"], dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-9,
    )
    assert float(np.sum(np.asarray(modern.incoherent["los"], dtype=np.float32))) > 0.0
    assert modern.metadata["path_counts"]["diffraction"] == 0
    assert modern.metadata["runtime_backends"]["reflection"]["cell_scatter_backend"] == "native_monte_carlo_scatter_delta"
    assert modern.metadata["runtime_backends"]["reflection"]["implementation"] == "cell_center_los_plus_tx_emitted_specular_reflection_symbolic_loop_plus_native_scatter_delta"


def test_cell_center_los_is_independent_of_sample_budget():
    if not NativeExtension.native_extension_available():
        pytest.skip("Standalone Monte Carlo native extension is unavailable in this environment.")

    scene = _build_channel_scene()
    grid_spec = GridSpec(
        axis="z",
        position=1.5,
        bounds=((-3.0, 3.0), (-3.0, 3.0)),
        grid_shape=(8, 8),
    )
    common = dict(
        reflection_n_rays=16,
        reflection_max_bounces=0,
        max_diffractions=0,
        accumulation_backend="auto",
        seed=5,
    )
    low = solve(
        scene=scene,
        frequency=3.5e9,
        tx_pos=wt.Point3f(-2.0, 0.0, 1.5),
        grid=grid_spec,
        config=Config(samples_per_tx=16, **common),
    )
    high = solve(
        scene=scene,
        frequency=3.5e9,
        tx_pos=wt.Point3f(-2.0, 0.0, 1.5),
        grid=grid_spec,
        config=Config(samples_per_tx=256, **common),
    )

    np.testing.assert_allclose(
        np.asarray(low.incoherent["los"], dtype=np.float32),
        np.asarray(high.incoherent["los"], dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-9,
    )
    assert low.metadata["path_counts"]["los"] == high.metadata["path_counts"]["los"]
    assert low.metadata["monte_carlo"]["los_strategy"] == "cell_center_visibility"


def test_bdpt_integrator_runs_primal_and_reports_mis_metadata():
    if not NativeExtension.native_extension_available():
        pytest.skip("Standalone Monte Carlo native extension is unavailable in this environment.")

    result = solve(
        scene=_build_channel_scene(),
        frequency=3.5e9,
        tx_pos=wt.Point3f(-2.0, 0.0, 1.5),
        grid=GridSpec(
            axis="z",
            position=1.5,
            bounds=((-3.0, 3.0), (-3.0, 3.0)),
            grid_shape=(8, 8),
        ),
        config=Config(
            integrator="bdpt",
            reflection_n_rays=32,
            reflection_max_bounces=1,
            samples_per_tx=64,
            max_diffractions=1,
            accumulation_backend="auto",
            seed=3,
            ad=False,
        ),
    )

    path_gain = np.asarray(result.path_gain, dtype=np.float32)
    total = np.asarray(result.incoherent["total"], dtype=np.float32)
    raw_total = np.asarray(result.incoherent["raw_total"], dtype=np.float32)
    correction = np.asarray(
        result.incoherent["shadow_boundary_correction"],
        dtype=np.float32,
    )
    assert path_gain.shape == (8, 8)
    assert np.all(np.isfinite(path_gain))
    np.testing.assert_allclose(path_gain, total, rtol=1.0e-6, atol=1.0e-9)
    np.testing.assert_allclose(raw_total + correction, total, rtol=1.0e-6, atol=1.0e-9)
    assert np.all(np.isfinite(correction))
    assert np.all(total >= -1.0e-12)
    assert "diffraction_incident_transition_power" in result.incoherent
    assert "diffraction_reflection_transition_power" in result.incoherent
    assert "incident_shadow_boundary_weight" in result.incoherent
    assert "reflection_shadow_boundary_weight" in result.incoherent
    assert result.metadata["integrator"] == "bdpt"
    assert result.metadata["monte_carlo"]["shadow_boundary_mode"] == "utd_power_smoothing"
    assert result.metadata["bdpt"]["mis_policy"]["heuristic"] == "balance"
    assert result.metadata["bdpt"]["mis_policy"]["sample_sequence"] == "sobol"
    assert result.metadata["bdpt"]["mis_policy"]["active_strategy_count"] == 3
    assert result.metadata["runtime_backends"]["diffraction"]["sample_sequence"] == "sobol"
    assert (
        result.metadata["runtime_backends"]["diffraction"]["first_order_edge_sampler"]
        == "length_receiver_solid_angle_source_power_mixture"
    )
    first_order_policy = result.metadata["bdpt"]["mis_policy"]["first_order_edge_sampler"]
    assert first_order_policy["proposal"] == "length_receiver_solid_angle_source_power_mixture"
    assert first_order_policy["pdf_correction"] == "edge_measure_weight"
    assert first_order_policy["applies_to_orders"] == [1]
    assert result.metadata["bdpt"]["reflection_policy"] == "forward_sampled_specular_only"
    strategies = result.metadata["bdpt"]["strategies"]
    assert strategies["direct_wedge_connection"]["samples"] == 22
    assert strategies["keller_cone_plane_hit"]["samples"] == 21
    assert strategies["specular_suffix_connection"]["samples"] == 21
    assert (
        strategies["direct_wedge_connection"]["samples"]
        + strategies["keller_cone_plane_hit"]["samples"]
        + strategies["specular_suffix_connection"]["samples"]
    ) == 64
    assert strategies["direct_wedge_connection"]["gain_scale"] == "lambda_over_4pi_squared"


def test_bdpt_integrator_accepts_depth_three_diffraction():
    if not NativeExtension.native_extension_available():
        pytest.skip("Standalone Monte Carlo native extension is unavailable in this environment.")

    result = solve(
        scene=_build_channel_scene(),
        frequency=3.5e9,
        tx_pos=wt.Point3f(-2.0, 0.0, 1.5),
        grid=GridSpec(
            axis="z",
            position=1.5,
            bounds=((-3.0, 3.0), (-3.0, 3.0)),
            grid_shape=(4, 4),
        ),
        config=Config(
            integrator="bdpt",
            reflection_n_rays=32,
            reflection_max_bounces=1,
            samples_per_tx=96,
            max_diffractions=3,
            accumulation_backend="auto",
            seed=7,
            ad=False,
        ),
    )

    path_gain = np.asarray(result.path_gain, dtype=np.float32)
    assert path_gain.shape == (4, 4)
    assert np.all(np.isfinite(path_gain))
    bdpt = result.metadata["bdpt"]
    assert bdpt["max_diffraction_depth_supported"] == 3
    assert bdpt["max_diffraction_depth_active"] == 3
    assert bdpt["path_families"]["S -> D -> ... -> D"] == "active"
    assert set(bdpt["order_breakdown"]) == {1, 2, 3}
    total_samples = sum(
        order_data["samples"]["direct_wedge_connection"]
        + order_data["samples"]["keller_cone_plane_hit"]
        + order_data["samples"]["specular_suffix_connection"]
        for order_data in bdpt["order_breakdown"].values()
    )
    assert total_samples == 96


def test_bdpt_can_run_pure_two_order_diffraction_without_reflection_coupling():
    if not NativeExtension.native_extension_available():
        pytest.skip("Standalone Monte Carlo native extension is unavailable in this environment.")

    result = solve(
        scene=_build_channel_scene(),
        frequency=3.5e9,
        tx_pos=wt.Point3f(-2.0, 0.0, 1.5),
        grid=GridSpec(
            axis="z",
            position=1.5,
            bounds=((-3.0, 3.0), (-3.0, 3.0)),
            grid_shape=(4, 4),
        ),
        config=Config(
            integrator="bdpt",
            reflection_n_rays=32,
            reflection_max_bounces=1,
            samples_per_tx=96,
            max_diffractions=2,
            enable_bdpt_reflection_coupled_diffraction=False,
            accumulation_backend="auto",
            seed=7,
            ad=False,
        ),
    )

    bdpt = result.metadata["bdpt"]
    assert bdpt["max_diffraction_depth_active"] == 2
    assert bdpt["mis_policy"]["active_order_range"] == [1, 2]
    assert bdpt["mis_policy"]["active_strategy_count"] == 2
    assert bdpt["mis_policy"]["reflection_coupled_diffraction"] is False
    assert bdpt["path_families"]["S -> D"] == "active"
    assert bdpt["path_families"]["S -> D -> ... -> D"] == "active"
    assert bdpt["path_families"]["R^n -> D"] == "disabled"
    assert bdpt["path_families"]["D -> R"] == "disabled"
    assert result.metadata["monte_carlo"]["state_pool"]["reflection_prefix"] == 0
    assert set(bdpt["order_breakdown"]) == {1, 2}
    for order_data in bdpt["order_breakdown"].values():
        assert order_data["samples"]["direct_wedge_connection"] > 0
        assert order_data["samples"]["keller_cone_plane_hit"] > 0
        assert order_data["samples"]["specular_suffix_connection"] == 0
        assert order_data["accepted"]["specular_suffix_connection"] == 0


def test_bdpt_first_order_direct_connection_is_balance_weighted():
    if not NativeExtension.native_extension_available():
        pytest.skip("Standalone Monte Carlo native extension is unavailable in this environment.")

    direct_strategy = BDPTDiffractionMIS.DIRECT_STRATEGY
    keller_strategy = BDPTDiffractionMIS.KELLER_STRATEGY
    suffix_strategy = BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY

    def run_with_allocation(allocation_fn=None) -> float:
        original_allocate_samples = BDPTDiffractionMIS.allocate_samples
        if allocation_fn is not None:
            def allocated_samples(
                samples_per_tx,
                max_depth=1,
                include_suffix_reflection=True,
            ):
                del max_depth, include_suffix_reflection
                return allocation_fn(int(samples_per_tx))

            BDPTDiffractionMIS.allocate_samples = staticmethod(allocated_samples)
        try:
            result = solve(
                scene=_build_three_cube_scene(),
                frequency=1.0e9,
                tx_pos=wt.Point3f(0.0, -5.0, 4.0),
                grid=GridSpec(
                    axis="z",
                    position=1.0,
                    bounds=((-10.0, 10.0), (-10.0, 10.0)),
                    grid_shape=(16, 16),
                ),
                config=Config(
                    integrator="bdpt",
                    reflection_n_rays=1024,
                    reflection_max_bounces=1,
                    samples_per_tx=2048,
                    enable_rd_diffraction=True,
                    max_diffractions=1,
                    enable_bdpt_reflection_coupled_diffraction=False,
                    accumulation_backend="auto",
                    shadow_boundary_mode="none",
                    seed=7,
                    ad=False,
                ),
            )
        finally:
            BDPTDiffractionMIS.allocate_samples = original_allocate_samples
        return float(
            np.sum(np.asarray(result.incoherent["diffraction"], dtype=np.float64))
        )

    keller_only_sum = run_with_allocation(
        lambda samples: {
            1: {
                direct_strategy: 0,
                keller_strategy: samples,
                suffix_strategy: 0,
            }
        }
    )
    balanced_sum = run_with_allocation()

    assert keller_only_sum > 0.0
    assert balanced_sum < keller_only_sum * 3.0


def test_bdpt_diffraction_scale_stays_below_los_on_wall_scene():
    if not NativeExtension.native_extension_available():
        pytest.skip("Standalone Monte Carlo native extension is unavailable in this environment.")

    result = solve(
        scene=_build_channel_scene(),
        frequency=3.5e9,
        tx_pos=wt.Point3f(-2.0, 0.0, 1.5),
        grid=GridSpec(
            axis="z",
            position=1.5,
            bounds=((-3.0, 3.0), (-3.0, 3.0)),
            grid_shape=(16, 16),
        ),
        config=Config(
            integrator="bdpt",
            reflection_n_rays=512,
            reflection_max_bounces=1,
            samples_per_tx=2048,
            max_diffractions=1,
            accumulation_backend="auto",
            seed=3,
            ad=False,
        ),
    )

    los_sum = float(np.sum(np.asarray(result.incoherent["los"], dtype=np.float64)))
    diffraction_sum = float(np.sum(np.asarray(result.incoherent["diffraction"], dtype=np.float64)))
    assert los_sum > 0.0
    assert diffraction_sum < los_sum * 1.0e-2


def test_bdpt_multi_diffraction_increment_stays_below_first_order_on_wall_scene():
    if not NativeExtension.native_extension_available():
        pytest.skip("Standalone Monte Carlo native extension is unavailable in this environment.")

    scene = _build_channel_scene()
    grid = GridSpec(
        axis="z",
        position=1.5,
        bounds=((-3.0, 3.0), (-3.0, 3.0)),
        grid_shape=(8, 8),
    )
    common = dict(
        integrator="bdpt",
        reflection_n_rays=128,
        reflection_max_bounces=1,
        accumulation_backend="auto",
        seed=13,
        ad=False,
    )
    first_order = solve(
        scene=scene,
        frequency=3.5e9,
        tx_pos=wt.Point3f(-2.0, 0.0, 1.5),
        grid=grid,
        config=Config(samples_per_tx=512, max_diffractions=1, **common),
    )
    third_order = solve(
        scene=scene,
        frequency=3.5e9,
        tx_pos=wt.Point3f(-2.0, 0.0, 1.5),
        grid=grid,
        config=Config(samples_per_tx=1536, max_diffractions=3, **common),
    )

    first_sum = float(np.sum(np.asarray(first_order.incoherent["diffraction"], dtype=np.float64)))
    third_sum = float(np.sum(np.asarray(third_order.incoherent["diffraction"], dtype=np.float64)))
    higher_order_increment = max(0.0, third_sum - first_sum)
    assert first_sum > 0.0
    assert higher_order_increment < max(first_sum * 1.0e-2, 1.0e-12)


def test_bdpt_reflection_prefix_diffraction_is_repeatable_on_three_cube_scene():
    if not NativeExtension.native_extension_available():
        pytest.skip("Standalone Monte Carlo native extension is unavailable in this environment.")

    def run_once():
        return solve(
            scene=_build_three_cube_scene(),
            frequency=1.0e9,
            tx_pos=wt.Point3f(0.0, -5.0, 4.0),
            grid=GridSpec(
                axis="z",
                position=1.0,
                bounds=((-10.0, 10.0), (-10.0, 10.0)),
                grid_shape=(24, 24),
            ),
            config=Config(
                integrator="bdpt",
                reflection_n_rays=4096,
                reflection_max_bounces=1,
                samples_per_tx=4096,
                enable_rd_diffraction=True,
                max_diffractions=3,
                accumulation_backend="auto",
                seed=7,
                ad=False,
            ),
        )

    first = run_once()
    second = run_once()
    np.testing.assert_allclose(
        np.asarray(first.incoherent["diffraction"], dtype=np.float64),
        np.asarray(second.incoherent["diffraction"], dtype=np.float64),
        rtol=1.0e-6,
        atol=1.0e-10,
    )
    assert first.metadata["bdpt"]["order_breakdown"] == second.metadata["bdpt"]["order_breakdown"]


def test_bdpt_diffraction_scene_material_changes_direct_utd_term():
    if not NativeExtension.native_extension_available():
        pytest.skip("Standalone Monte Carlo native extension is unavailable in this environment.")

    def scene_with_material(eps_r: float) -> ChannelScene:
        material = Material(eps_r=eps_r, sigma_e=0.0)
        centers = (
            (-2.5, -3.0, 1.5),
            (2.0, 0.5, 1.5),
            (-0.5, 3.5, 1.5),
        )
        return ChannelScene(
            structures=[
                Structure(
                    name=f"cube_{index}",
                    geometry=Box(position=center, size=(2.0, 2.0, 2.0), device="cuda"),
                    material=material,
                )
                for index, center in enumerate(centers)
            ],
            device="cuda",
            edge_selection_mode="all_edges",
        )

    def diffraction_sum(eps_r: float) -> float:
        result = solve(
            scene=scene_with_material(eps_r),
            frequency=1.0e9,
            tx_pos=wt.Point3f(0.0, -5.0, 4.0),
            grid=GridSpec(
                axis="z",
                position=1.0,
                bounds=((-10.0, 10.0), (-10.0, 10.0)),
                grid_shape=(24, 24),
            ),
            config=Config(
                integrator="bdpt",
                reflection_n_rays=4096,
                reflection_max_bounces=0,
                samples_per_tx=4096,
                enable_rd_diffraction=True,
                max_diffractions=1,
                accumulation_backend="auto",
                seed=7,
                ad=False,
            ),
        )
        return float(np.sum(np.asarray(result.incoherent["diffraction"], dtype=np.float64)))

    low_permittivity = diffraction_sum(2.0)
    high_permittivity = diffraction_sum(8.0)
    assert low_permittivity > 0.0
    assert high_permittivity > 0.0
    assert high_permittivity != pytest.approx(low_permittivity)


def test_bdpt_ad_true_tx_scalar_loss_gradient_is_finite():
    if not SparseCoeffKernel.available():
        pytest.skip("Standalone Monte Carlo native AD kernels are unavailable in this environment.")

    tx_x = wt.Float(-2.0)
    dr.enable_grad(tx_x)
    result = solve(
        scene=_build_channel_scene(),
        frequency=3.5e9,
        tx_pos=wt.Point3f(tx_x, 0.0, 1.5),
        grid=GridSpec(
            axis="z",
            position=1.5,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(4, 4),
        ),
        config=Config(
            integrator="bdpt",
            reflection_n_rays=16,
            reflection_max_bounces=1,
            samples_per_tx=32,
            max_diffractions=1,
            accumulation_backend="auto",
            shadow_boundary_mode="none",
            seed=3,
            ad=True,
        ),
    )

    loss = dr.sum(result.path_gain)
    dr.backward(loss)
    grad = dr.grad(tx_x)
    dr.eval(loss, grad)

    assert result.metadata["bdpt"]["ad_mode"] is True
    assert result.metadata["monte_carlo"]["ad_mode"] is True
    assert result.metadata["bdpt"]["tape_layout_version"] == "bdpt_diffraction_fixed_width_v1"
    assert np.isfinite(float(np.asarray(loss).reshape(-1)[0]))
    assert np.isfinite(float(np.asarray(grad).reshape(-1)[0]))
    assert abs(float(np.asarray(grad).reshape(-1)[0])) > 0.0


def test_bdpt_ad_scene_vertex_and_material_gradients_are_finite():
    if not SparseCoeffKernel.available():
        pytest.skip("Standalone Monte Carlo native AD kernels are unavailable in this environment.")

    experiment = ThreeCubeExperiment(
        grid_shape=(4, 4),
        samples_per_tx=64,
        forward_reflection_n_rays=32,
        gradient_reflection_n_rays=32,
        seed=7,
    )
    cube1_x = wt.Float(experiment.base_centers[0][0])
    cube1_eps = wt.Float(4.0)
    dr.enable_grad(cube1_x, cube1_eps)
    scene = experiment._build_channel_scene(cube1_x=cube1_x)
    scene = scene.set_structure_material_parameters("cube_0", eps_r=cube1_eps)
    result = solve(
        scene=scene,
        frequency=3.5e9,
        tx_pos=wt.Point3f(*experiment.tx_pos),
        grid=experiment.grid,
        config=Config(
            integrator="bdpt",
            reflection_n_rays=32,
            reflection_max_bounces=1,
            samples_per_tx=64,
            max_diffractions=1,
            accumulation_backend="auto",
            shadow_boundary_mode="none",
            seed=7,
            ad=True,
        ),
    )

    loss = dr.sum(result.path_gain)
    dr.backward(loss)
    cube_grad = dr.grad(cube1_x)
    eps_grad = dr.grad(cube1_eps)
    dr.eval(loss, cube_grad, eps_grad)

    assert result.metadata["bdpt"]["ad_mode"] is True
    assert np.isfinite(float(np.asarray(cube_grad).reshape(-1)[0]))
    assert np.isfinite(float(np.asarray(eps_grad).reshape(-1)[0]))
    assert abs(float(np.asarray(cube_grad).reshape(-1)[0])) > 0.0
    assert abs(float(np.asarray(eps_grad).reshape(-1)[0])) > 0.0


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_bdpt_ad_supports_depth_limited_path_families(depth: int):
    if not SparseCoeffKernel.available():
        pytest.skip("Standalone Monte Carlo native AD kernels are unavailable in this environment.")

    tx_x = wt.Float(-2.0)
    dr.enable_grad(tx_x)
    result = solve(
        scene=_build_channel_scene(),
        frequency=3.5e9,
        tx_pos=wt.Point3f(tx_x, 0.0, 1.5),
        grid=GridSpec(
            axis="z",
            position=1.5,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(2, 2),
        ),
        config=Config(
            integrator="bdpt",
            reflection_n_rays=8,
            reflection_max_bounces=1,
            samples_per_tx=12,
            max_diffractions=depth,
            accumulation_backend="auto",
            shadow_boundary_mode="none",
            seed=3,
            ad=True,
        ),
    )

    loss = dr.sum(result.path_gain)
    dr.backward(loss)
    grad = dr.grad(tx_x)
    dr.eval(loss, grad)

    assert result.metadata["bdpt"]["max_diffraction_depth_active"] == depth
    assert result.metadata["bdpt"]["ad_mode"] is True
    assert np.isfinite(float(np.asarray(loss).reshape(-1)[0]))
    assert np.isfinite(float(np.asarray(grad).reshape(-1)[0]))


def test_bdpt_ad_reflection_coupled_diffraction_can_be_disabled():
    if not SparseCoeffKernel.available():
        pytest.skip("Standalone Monte Carlo native AD kernels are unavailable in this environment.")

    tx_x = wt.Float(-2.0)
    dr.enable_grad(tx_x)
    result = solve(
        scene=_build_channel_scene(),
        frequency=3.5e9,
        tx_pos=wt.Point3f(tx_x, 0.0, 1.5),
        grid=GridSpec(
            axis="z",
            position=1.5,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(2, 2),
        ),
        config=Config(
            integrator="bdpt",
            reflection_n_rays=8,
            reflection_max_bounces=1,
            samples_per_tx=12,
            max_diffractions=2,
            enable_bdpt_reflection_coupled_diffraction=False,
            accumulation_backend="auto",
            shadow_boundary_mode="none",
            seed=3,
            ad=True,
        ),
    )

    loss = dr.sum(result.path_gain)
    dr.backward(loss)
    grad = dr.grad(tx_x)
    dr.eval(loss, grad)

    assert result.metadata["bdpt"]["ad_mode"] is True
    assert result.metadata["bdpt"]["mis_policy"]["reflection_coupled_diffraction"] is False
    assert np.isfinite(float(np.asarray(loss).reshape(-1)[0]))
    assert np.isfinite(float(np.asarray(grad).reshape(-1)[0]))


def test_bdpt_ad_none_auto_detects_and_ad_false_detaches():
    if not SparseCoeffKernel.available():
        pytest.skip("Standalone Monte Carlo native AD kernels are unavailable in this environment.")

    tx_x = wt.Float(-2.0)
    dr.enable_grad(tx_x)
    common = dict(
        integrator="bdpt",
        reflection_n_rays=8,
        reflection_max_bounces=1,
        samples_per_tx=12,
        max_diffractions=1,
        accumulation_backend="auto",
        shadow_boundary_mode="none",
        seed=3,
    )
    result_auto = solve(
        scene=_build_channel_scene(),
        frequency=3.5e9,
        tx_pos=wt.Point3f(tx_x, 0.0, 1.5),
        grid=GridSpec(
            axis="z",
            position=1.5,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(2, 2),
        ),
        config=Config(ad=None, **common),
    )
    assert result_auto.metadata["bdpt"]["ad_mode"] is True

    result_disabled = solve(
        scene=_build_channel_scene(),
        frequency=3.5e9,
        tx_pos=wt.Point3f(tx_x, 0.0, 1.5),
        grid=GridSpec(
            axis="z",
            position=1.5,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(2, 2),
        ),
        config=Config(ad=False, **common),
    )
    disabled_loss = dr.sum(result_disabled.path_gain)
    assert result_disabled.metadata["bdpt"]["ad_mode"] is False
    assert not dr.grad_enabled(disabled_loss)


def test_standalone_native_module_exports_expected_symbols_when_available():
    if not NativeExtension.native_extension_available():
        pytest.skip("Standalone Monte Carlo native extension is unavailable in this environment.")

    ext = NativeExtension.load()
    assert hasattr(ext, "radiomap_monte_carlo_scatter_axis_aligned")
    assert hasattr(ext, "monte_carlo_sparse_coeff_jvp_into")
    assert hasattr(ext, "monte_carlo_sparse_coeff_vjp_into")
    assert hasattr(ext, "monte_carlo_transport_grid_forward_raw")
    assert hasattr(ext, "monte_carlo_transport_grid_jvp_raw")
    assert hasattr(ext, "monte_carlo_transport_grid_backward_raw")
    assert hasattr(ext, "monte_carlo_transport_vertex_jvp_into")
    assert hasattr(ext, "monte_carlo_transport_vertex_vjp_into")
    assert hasattr(ext, "monte_carlo_diffraction_sample_slots")
    assert hasattr(ext, "monte_carlo_diffraction_best_edge_indices")
    assert hasattr(ext, "monte_carlo_diffraction_discover_edges")
    assert hasattr(ext, "monte_carlo_diffraction_build_state_arrays")


def test_transport_grid_native_matches_reference_jvp_and_vjp():
    if not NativeExtension.native_extension_available():
        pytest.skip("Standalone Monte Carlo native extension is unavailable in this environment.")

    grid = Grid.from_spec(
        GridSpec(
            axis="z",
            position=1.0,
            bounds=((0.0, 3.0), (0.0, 2.0)),
            grid_shape=(3, 2),
        )
    )
    coord_0_native = wt.Float([0.30, 1.20, 2.40, 1.85])
    coord_1_native = wt.Float([0.40, 0.80, 1.25, 1.60])
    power_native = wt.Float([1.0, 2.0, 0.5, 3.0])
    coord_0_ref = wt.Float(coord_0_native)
    coord_1_ref = wt.Float(coord_1_native)
    power_ref = wt.Float(power_native)
    active = wt.Bool([True, True, True, False])
    upstream = wt.Float([1.0, 0.25, -0.5, 0.75, 0.5, -1.0])

    dr.enable_grad(coord_0_native, coord_1_native, power_native)
    dr.enable_grad(coord_0_ref, coord_1_ref, power_ref)

    native_map = TransportGridKernel.tent_splat(
        coord_0=coord_0_native,
        coord_1=coord_1_native,
        power=power_native,
        active=active,
        bounds=grid.bounds,
        cell_size=grid.cell_size,
        grid_shape=grid.grid_shape,
    )
    reference_map = _tent_splat_reference(
        grid=grid,
        coord_0=coord_0_ref,
        coord_1=coord_1_ref,
        power=power_ref,
        active=active,
    )

    tangent_0 = wt.Float([0.1, -0.2, 0.0, 0.3])
    tangent_1 = wt.Float([-0.05, 0.15, 0.2, -0.1])
    tangent_power = wt.Float([0.4, 0.0, -0.3, 0.2])
    dr.set_grad(coord_0_native, tangent_0)
    dr.set_grad(coord_1_native, tangent_1)
    dr.set_grad(power_native, tangent_power)
    dr.set_grad(coord_0_ref, tangent_0)
    dr.set_grad(coord_1_ref, tangent_1)
    dr.set_grad(power_ref, tangent_power)

    native_jvp = dr.forward_to(native_map)
    reference_jvp = dr.forward_to(reference_map)
    np.testing.assert_allclose(
        np.asarray(native_map, dtype=np.float32),
        np.asarray(reference_map, dtype=np.float32),
        rtol=5.0e-5,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(native_jvp, dtype=np.float32),
        np.asarray(reference_jvp, dtype=np.float32),
        rtol=5.0e-5,
        atol=1.0e-6,
    )

    dr.backward(dr.sum(native_map * upstream))
    dr.backward(dr.sum(reference_map * upstream))
    np.testing.assert_allclose(
        np.asarray(dr.grad(coord_0_native), dtype=np.float32),
        np.asarray(dr.grad(coord_0_ref), dtype=np.float32),
        rtol=5.0e-5,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(dr.grad(coord_1_native), dtype=np.float32),
        np.asarray(dr.grad(coord_1_ref), dtype=np.float32),
        rtol=5.0e-5,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(dr.grad(power_native), dtype=np.float32),
        np.asarray(dr.grad(power_ref), dtype=np.float32),
        rtol=5.0e-5,
        atol=1.0e-6,
    )


def test_diffraction_builder_sample_slots_matches_reference():
    scene = ThreeCubeExperiment(
        grid_shape=(16, 16),
        samples_per_tx=256,
        forward_reflection_n_rays=16,
        gradient_reflection_n_rays=16,
        seed=7,
    )._build_channel_scene()
    line_length = scene._selected_edge_runtime()["length"]
    sampler = DiffractionEdgeSampler.from_line_length(line_length)
    if sampler is None:
        pytest.skip("Scene does not expose diffraction states.")

    sample_index = dr.arange(wt.UInt32, 64)
    native_slots = sampler.sample_slots(sample_index, seed=11)
    sample_u = np.asarray(
        Sampler.hash_uniform(sample_index, stream=601, seed=11) * wt.Float(sampler.total_length_scalar),
        dtype=np.float32,
    )
    reference_slots = np.searchsorted(
        np.asarray(sampler.cdf, dtype=np.float32),
        sample_u,
        side="left",
    ).astype(np.uint32)

    np.testing.assert_array_equal(
        np.asarray(native_slots, dtype=np.uint32),
        reference_slots,
    )


def test_diffraction_builder_best_edge_matches_reference():
    experiment = ThreeCubeExperiment(
        grid_shape=(16, 16),
        samples_per_tx=256,
        forward_reflection_n_rays=16,
        gradient_reflection_n_rays=16,
        seed=7,
    )
    scene = experiment._build_channel_scene()
    tx_pos = wt.Point3f(*experiment.tx_pos)
    ray_index = dr.arange(wt.UInt32, 128)
    ray_dir = Sampler.directions(256, ray_index=ray_index)
    si = scene.ray_intersect(
        rayd.Ray(arrays.broadcast_point(tx_pos, 128), ray_dir),
        active=dr.full(wt.Bool, 128, True),
    )
    hit = si.is_valid()
    if not dr.any(hit):
        pytest.skip("Generated rays did not hit the scene.")

    native_best = DiffractionEdgeSampler.best_edge_indices_from_hit_data(
        tx_pos=tx_pos,
        ray_directions=ray_dir,
        prim_index=si.global_prim_id,
        hit_p=si.p,
        hit_n=si.n,
        hit_geo_n=si.geo_n,
        hit=hit,
        scene=scene,
    )
    reference_best = _reference_best_edge_indices(
        tx_pos=tx_pos,
        ray_directions=ray_dir,
        prim_index=si.global_prim_id,
        hit_p=si.p,
        hit_n=si.n,
        hit_geo_n=si.geo_n,
        hit=hit,
        scene=scene,
    )

    np.testing.assert_array_equal(
        np.asarray(native_best, dtype=np.int32),
        np.asarray(reference_best, dtype=np.int32),
    )


def test_reflection_transport_matches_fixed_tape_cube1_x_fd():
    ctx = _prepare_three_cube_fixed_tape_context(grid_size=16, samples_per_tx=1024)
    tape = ctx["primal_state"].reflection_tape
    if int(dr.width(tape.transport_blocker_prim_idx)) <= 0:
        pytest.skip("Reflection transport tape is empty for this workload.")

    n_vertices = int(dr.width(ctx["base_scene"]._merged_vertices().x))
    vertex_tangent_x = dr.zeros(wt.Float, n_vertices)
    dr.scatter(vertex_tangent_x, wt.Float(1.0), dr.arange(wt.UInt32, 8))
    zero_vertex_tangent = dr.zeros(wt.Float, n_vertices)
    upstream = wt.Float(
        np.linspace(
            -0.75,
            0.85,
            int(ctx["grid"].n_cells),
            dtype=np.float32,
        )
    )

    transport_jvp = ReflectionAD.vertex_jvp(
        tape=tape,
        scene=ctx["detached"]["scene"],
        tx_pos=ctx["detached"]["tx_pos"],
        grid=ctx["grid"],
        config=ctx["resolved"],
        solid_angle_per_ray=ctx["solid_angle_per_ray"],
        cell_area=float(ctx["grid"].cell_size[0] * ctx["grid"].cell_size[1]),
        material_omega=ctx["material_omega"],
        vertex_tangent=wt.Point3f(
            vertex_tangent_x,
            zero_vertex_tangent,
            zero_vertex_tangent,
        ),
    )
    transport_vjp = ReflectionAD.vertex_vjp(
        tape=tape,
        scene=ctx["detached"]["scene"],
        tx_pos=ctx["detached"]["tx_pos"],
        grid=ctx["grid"],
        config=ctx["resolved"],
        solid_angle_per_ray=ctx["solid_angle_per_ray"],
        cell_area=float(ctx["grid"].cell_size[0] * ctx["grid"].cell_size[1]),
        material_omega=ctx["material_omega"],
        upstream_component=upstream,
        n_vertices=n_vertices,
    )

    def replay_sum(scene):
        state = ReflectionAD.vertex_state(
            tape=tape,
            scene=scene,
            tx_pos=ctx["detached"]["tx_pos"],
            grid=ctx["grid"],
            config=ctx["resolved"],
            solid_angle_per_ray=ctx["solid_angle_per_ray"],
            cell_area=float(ctx["grid"].cell_size[0] * ctx["grid"].cell_size[1]),
            material_omega=ctx["material_omega"],
        )
        assert state is not None
        return float(
            np.dot(
                np.asarray(state["transport_map"], dtype=np.float64),
                np.asarray(upstream, dtype=np.float64),
            )
        )

    scene_plus = BasicIntegratorAD.detached_workload(
        ctx["experiment"]._build_channel_scene(cube1_x=wt.Float(ctx["experiment"].base_centers[0][0] + DEFAULT_FD_STEP)),
        ctx["tx_pos"],
        ctx["resolved"],
    )["scene"]
    scene_minus = BasicIntegratorAD.detached_workload(
        ctx["experiment"]._build_channel_scene(cube1_x=wt.Float(ctx["experiment"].base_centers[0][0] - DEFAULT_FD_STEP)),
        ctx["tx_pos"],
        ctx["resolved"],
    )["scene"]
    fixed_tape_fd = (replay_sum(scene_plus) - replay_sum(scene_minus)) / (2.0 * DEFAULT_FD_STEP)

    jvp_sum = float(
        np.dot(
            np.asarray(transport_jvp, dtype=np.float64),
            np.asarray(upstream, dtype=np.float64),
        )
    )
    vjp_dot = float(np.dot(np.asarray(transport_vjp.x, dtype=np.float64), np.asarray(vertex_tangent_x, dtype=np.float64)))

    assert np.isfinite(jvp_sum)
    assert np.isfinite(vjp_dot)
    assert np.isfinite(fixed_tape_fd)
    assert abs(fixed_tape_fd) > 0.0
    assert jvp_sum == pytest.approx(fixed_tape_fd, rel=0.2, abs=1.0e-7)
    assert vjp_dot == pytest.approx(fixed_tape_fd, rel=0.2, abs=1.0e-7)


def test_diffraction_transport_matches_fixed_tape_vertex_fd():
    ctx = _prepare_three_cube_fixed_tape_context(grid_size=16, samples_per_tx=1024)
    tape = ctx["primal_state"].diffraction_tape
    if int(dr.width(tape.cell_idx)) <= 0:
        pytest.skip("Diffraction tape is empty for this workload.")

    state = DiffractionAD.vertex_transport_state(
        tape=tape,
        scene=ctx["detached"]["scene"],
        tx_pos=ctx["detached"]["tx_pos"],
        grid=ctx["grid"],
        config=ctx["resolved"],
        diff_gain_scale=ctx["diff_gain_scale"],
        total_length_weight=ctx["total_length_weight"],
    )
    assert state is not None

    active_indices = np.unique(
        np.concatenate(
            [
                np.asarray(state["edge_v0_idx"], dtype=np.int32),
                np.asarray(state["edge_v1_idx"], dtype=np.int32),
                np.asarray(state["face0_third_idx"], dtype=np.int32),
                np.asarray(state["face1_third_idx"], dtype=np.int32),
            ]
        )
    )
    active_indices = active_indices[active_indices >= 0]
    if active_indices.size == 0:
        pytest.skip("Diffraction transport state did not reference any vertices.")

    n_vertices = int(dr.width(ctx["base_scene"]._merged_vertices().x))
    vertex_tangent_x = dr.zeros(wt.Float, n_vertices)
    dr.scatter(vertex_tangent_x, wt.Float(1.0), wt.UInt32(active_indices.tolist()))
    zero_vertex_tangent = dr.zeros(wt.Float, n_vertices)
    upstream = wt.Float(
        np.linspace(
            -0.75,
            0.85,
            int(ctx["grid"].n_cells),
            dtype=np.float32,
        )
    )

    transport_jvp = DiffractionAD.vertex_transport_jvp(
        tape=tape,
        scene=ctx["detached"]["scene"],
        tx_pos=ctx["detached"]["tx_pos"],
        grid=ctx["grid"],
        config=ctx["resolved"],
        diff_gain_scale=ctx["diff_gain_scale"],
        total_length_weight=ctx["total_length_weight"],
        vertex_tangent=wt.Point3f(
            vertex_tangent_x,
            zero_vertex_tangent,
            zero_vertex_tangent,
        ),
    )
    transport_vjp = DiffractionAD.vertex_transport_vjp(
        tape=tape,
        scene=ctx["detached"]["scene"],
        tx_pos=ctx["detached"]["tx_pos"],
        grid=ctx["grid"],
        config=ctx["resolved"],
        diff_gain_scale=ctx["diff_gain_scale"],
        total_length_weight=ctx["total_length_weight"],
        upstream_component=upstream,
        n_vertices=n_vertices,
    )

    jvp_sum = float(
        np.dot(
            np.asarray(transport_jvp, dtype=np.float64),
            np.asarray(upstream, dtype=np.float64),
        )
    )
    vjp_dot = float(np.dot(np.asarray(transport_vjp.x, dtype=np.float64), np.asarray(vertex_tangent_x, dtype=np.float64)))

    assert np.isfinite(jvp_sum)
    assert np.isfinite(vjp_dot)
    assert abs(jvp_sum) > 0.0
    assert abs(vjp_dot) > 0.0
    assert jvp_sum == pytest.approx(vjp_dot, rel=5.0e-4, abs=1.0e-7)


def test_native_scatter_delta_matches_drjit_reference():
    if not NativeExtension.native_extension_available():
        pytest.skip("Standalone Monte Carlo native extension is unavailable in this environment.")

    grid_obj = Grid.from_spec(
        GridSpec(
            axis="z",
            position=1.0,
            bounds=((0.0, 2.0), (0.0, 2.0)),
            grid_shape=(2, 2),
        )
    )
    coord_0 = wt.Float([0.25, 0.75, 1.25, 1.75, 1.50, -0.10])
    coord_1 = wt.Float([0.25, 1.25, 0.75, 1.75, 1.50, 0.50])
    los_power = wt.Float([1.0, 2.0, 0.0, 4.0, 0.5, 9.0])
    reflection_power = wt.Float([0.0, 3.0, 5.0, 0.0, 0.5, 8.0])
    diffraction_power = wt.Float([7.0, 0.0, 6.0, 0.0, 0.25, 10.0])
    diffraction_incident_transition_power = wt.Float([0.7, 0.0, 0.6, 0.0, 0.025, 1.0])
    diffraction_reflection_transition_power = wt.Float([0.2, 0.0, 0.4, 0.0, 0.010, 2.0])

    deltas = grid.scatter_component_deltas(
        grid=grid_obj,
        coord_0=coord_0,
        coord_1=coord_1,
        component_power={
            "los": los_power,
            "reflection": reflection_power,
            "diffraction": diffraction_power,
            "diffraction_incident_transition_power": diffraction_incident_transition_power,
            "diffraction_reflection_transition_power": diffraction_reflection_transition_power,
        },
    )

    active = (
        (coord_0 >= wt.Float(grid_obj.bounds[0][0]))
        & (coord_0 < wt.Float(grid_obj.bounds[0][1]))
        & (coord_1 >= wt.Float(grid_obj.bounds[1][0]))
        & (coord_1 < wt.Float(grid_obj.bounds[1][1]))
    )
    cell_idx = grid.cell_index(grid=grid_obj, coord_0=coord_0, coord_1=coord_1)

    def _reference(power):
        out = dr.zeros(wt.Float, int(grid_obj.n_cells))
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            out,
            dr.select(active, power, wt.Float(0.0)),
            cell_idx,
            active,
        )
        return np.asarray(out, dtype=np.float32)

    np.testing.assert_allclose(
        np.asarray(deltas["los"], dtype=np.float32),
        _reference(los_power),
        rtol=5.0e-4,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(deltas["reflection"], dtype=np.float32),
        _reference(reflection_power),
        rtol=5.0e-4,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(deltas["diffraction"], dtype=np.float32),
        _reference(diffraction_power),
        rtol=5.0e-4,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(deltas["diffraction_incident_transition_power"], dtype=np.float32),
        _reference(diffraction_incident_transition_power),
        rtol=5.0e-4,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(deltas["diffraction_reflection_transition_power"], dtype=np.float32),
        _reference(diffraction_reflection_transition_power),
        rtol=5.0e-4,
        atol=1.0e-6,
    )


def test_three_cube_channel_scene_rebuild_preserves_cube1_x_ad():
    experiment = ThreeCubeExperiment(
        grid_shape=(16, 16),
        samples_per_tx=2000,
        forward_reflection_n_rays=32,
        gradient_reflection_n_rays=32,
        seed=7,
    )
    parameter = wt.Float(experiment.base_centers[0][0])
    dr.enable_grad(parameter)
    scene = experiment._build_channel_scene(cube1_x=parameter)
    objective = dr.sum(scene.vertices.x)
    dr.backward(objective)

    assert float(dr.grad(parameter)[0]) == pytest.approx(8.0, abs=1.0e-6)


def test_standalone_three_cube_cube1_diffraction_jvp_is_nonzero():
    experiment = ThreeCubeExperiment(
        grid_shape=(32, 32),
        samples_per_tx=4000,
        forward_reflection_n_rays=64,
        gradient_reflection_n_rays=64,
        seed=7,
    )
    parameter = wt.Float(experiment.base_centers[0][0])
    dr.enable_grad(parameter)
    result = experiment._solve(cube1_x=parameter, config=experiment.gradient_config)
    dr.set_grad(parameter, 1.0)
    diffraction_jvp = np.asarray(
        dr.forward_to(
            result.incoherent["diffraction"],
            flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad,
        ),
        dtype=np.float64,
    )

    plus = experiment._solve(
        cube1_x=wt.Float(experiment.base_centers[0][0] + DEFAULT_FD_STEP),
        config=experiment.gradient_config,
    )
    minus = experiment._solve(
        cube1_x=wt.Float(experiment.base_centers[0][0] - DEFAULT_FD_STEP),
        config=experiment.gradient_config,
    )
    diffraction_fd = (
        np.asarray(plus.incoherent["diffraction"], dtype=np.float64)
        - np.asarray(minus.incoherent["diffraction"], dtype=np.float64)
    ) / (2.0 * DEFAULT_FD_STEP)

    assert float(np.sum(np.abs(diffraction_fd))) > 0.0
    assert float(np.sum(np.abs(diffraction_jvp))) > 0.0


def test_three_cube_surface_edge_candidates_use_selected_edge_slots():
    experiment = ThreeCubeExperiment(
        grid_shape=(16, 16),
        samples_per_tx=256,
        forward_reflection_n_rays=16,
        gradient_reflection_n_rays=16,
        seed=7,
    )
    scene = experiment.base_channel_scene
    tri_data = scene._triangle_runtime()
    edge_runtime = scene._selected_edge_runtime()

    edge_indices = np.asarray(tri_data["surface_edge_indices"], dtype=np.int64).reshape(-1)
    valid_edge_indices = edge_indices[edge_indices >= 0]

    assert valid_edge_indices.size > 0
    assert int(valid_edge_indices.min()) >= 0
    assert int(valid_edge_indices.max()) < int(edge_runtime["n_edges"])


def test_three_cube_diffraction_strength_regression_from_surface_edge_slots():
    experiment = ThreeCubeExperiment(
        grid_shape=(128, 128),
        samples_per_tx=200000,
        forward_reflection_n_rays=128,
        gradient_reflection_n_rays=128,
        seed=7,
    )

    snapshot = experiment.forward()

    assert snapshot.metadata["monte_carlo"]["state_pool"]["total"] >= 20
    assert snapshot.metadata["path_counts"]["diffraction"] >= 7000
    assert float(np.sum(snapshot.components["diffraction"])) > 5.0e-3


def test_three_cube_example_component_gradients_are_not_empty():
    experiment = ThreeCubeExperiment(
        grid_shape=(16, 16),
        samples_per_tx=256,
        forward_reflection_n_rays=16,
        gradient_reflection_n_rays=16,
        seed=7,
    )

    tx_gradient = experiment.gradient("tx_x", fd_step=DEFAULT_FD_STEP)
    cube_gradient = experiment.gradient("cube1_x", fd_step=DEFAULT_FD_STEP)

    assert float(np.sum(np.abs(tx_gradient.component_jvp["los"]))) > 0.0
    assert float(np.sum(np.abs(tx_gradient.component_jvp["diffraction"]))) > 0.0
    assert float(np.sum(np.abs(cube_gradient.component_jvp["reflection"]))) > 0.0
    assert float(np.sum(np.abs(cube_gradient.component_jvp["diffraction"]))) > 0.0


def test_diffraction_sparse_coeff_matches_fixed_tape_cube1_x_fd():
    if not SparseCoeffKernel.available():
        pytest.skip("Standalone Monte Carlo native AD kernels are unavailable in this environment.")

    experiment = ThreeCubeExperiment(
        grid_shape=(16, 16),
        samples_per_tx=1024,
        forward_reflection_n_rays=32,
        gradient_reflection_n_rays=32,
        seed=7,
    )
    mc_config = experiment.gradient_config
    trace_config = mc_config.to_trace_config()
    resolved = ResolvedTraceConfig.from_config(
        frequency=DEFAULT_FREQUENCY_HZ,
        config=trace_config,
    )
    solver_controls = resolve_solver_controls(
        trace_config,
        execution_intent="radio_map_incoherent",
        max_diffractions_override=int(mc_config.max_diffractions),
    )
    tx_pos = wt.Point3f(*experiment.tx_pos)
    base_scene = experiment._build_channel_scene(cube1_x=wt.Float(experiment.base_centers[0][0]))
    detached = BasicIntegratorAD.detached_workload(base_scene, tx_pos, resolved)
    primal_state = Basic.primal(
        detached["tx_pos"],
        experiment.grid,
        mc_config,
        detached["scene"],
        resolved,
        solver_controls,
        accumulation_backend=str(mc_config.accumulation_backend),
        return_timing=False,
        resolved_ad_mode=True,
        ad_backend="test_fixed_tape_fd",
        loop_mode="symbolic",
        collect_ad_tapes=True,
    )
    tape = primal_state.diffraction_tape
    grid = primal_state.grid
    diff_gain_scale = wt.Float(
        (float(resolved.wavelength) / (4.0 * math.pi)) ** 2
        / float(grid.cell_size[0] * grid.cell_size[1])
    )
    total_length_weight = float(primal_state.diff_length_weight)
    buffers = DiffractionAD.sparse_coeffs(
        tape=tape,
        scene=detached["scene"],
        tx_pos=detached["tx_pos"],
        grid=grid,
        config=resolved,
        diff_gain_scale=diff_gain_scale,
        total_length_weight=total_length_weight,
    )

    n_vertices = int(dr.width(base_scene._merged_vertices().x))
    vertex_tangent_x = dr.zeros(wt.Float, n_vertices)
    dr.scatter(vertex_tangent_x, wt.Float(1.0), dr.arange(wt.UInt32, 8))
    zero_vertex_tangent = dr.zeros(wt.Float, n_vertices)
    sparse_jvp = SparseCoeffKernel.launch_jvp_into(
        buffers=buffers,
        tx_tangent=wt.Point3f(0.0, 0.0, 0.0),
        vertex_tangent=wt.Point3f(
            vertex_tangent_x,
            zero_vertex_tangent,
            zero_vertex_tangent,
        ),
        material_tangent={
            "eps_r": dr.zeros(wt.Float, 0),
            "sigma_e": dr.zeros(wt.Float, 0),
        },
        out_size=int(grid.n_cells),
    )
    sparse_sum = float(np.asarray(sparse_jvp, dtype=np.float64).sum())

    def replay_sum(scene):
        width = int(dr.width(tape.cell_idx))
        local_tx = SceneQuery.tx_lanes(detached["tx_pos"], width)
        geo = DiffractionAD.edge_geometry(
            tape=tape,
            scene=scene,
            local_tx=local_tx,
            grid=grid,
            config=resolved,
            width=width,
        )
        contribution = (
            geo["field_power"]
            * diff_gain_scale
            * geo["integration_weight"]
            * wt.Float(total_length_weight)
            * geo["exterior_angle"]
        )
        return float(np.asarray(dr.sum(contribution), dtype=np.float64).reshape(-1)[0])

    scene_plus = BasicIntegratorAD.detached_workload(
        experiment._build_channel_scene(cube1_x=wt.Float(experiment.base_centers[0][0] + DEFAULT_FD_STEP)),
        tx_pos,
        resolved,
    )["scene"]
    scene_minus = BasicIntegratorAD.detached_workload(
        experiment._build_channel_scene(cube1_x=wt.Float(experiment.base_centers[0][0] - DEFAULT_FD_STEP)),
        tx_pos,
        resolved,
    )["scene"]
    fixed_tape_fd = (replay_sum(scene_plus) - replay_sum(scene_minus)) / (2.0 * DEFAULT_FD_STEP)

    assert buffers.vertex_slot_count == 4
    assert np.isfinite(sparse_sum)
    assert np.isfinite(fixed_tape_fd)
    assert abs(fixed_tape_fd) > 0.0
    assert sparse_sum == pytest.approx(fixed_tape_fd, rel=0.15, abs=1.0e-7)
