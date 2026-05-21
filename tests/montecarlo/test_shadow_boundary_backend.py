from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import drjit as dr
import witwin.channel as wt
from witwin.channel.core.scene import EdgePolicy, ReceiverGrid, Scene, Transmitter
from witwin.channel.core.grid import Grid, GridSpec
from witwin.core import Box, Material, Mesh, Structure
from witwin.channel.montecarlo import Config, IntegratorOptions, NativeExtension, Tuning, solve
from witwin.channel.montecarlo.config import ResolvedTraceConfig
from witwin.channel.montecarlo.trace.diffraction import DiffractionStates
from witwin.channel.montecarlo.trace.postprocessing import ShadowBoundary
from witwin.channel.montecarlo import types as mc_types


FREQUENCY = 3.5e9


def _three_cube_scene() -> Scene:
    centers = ((-2.5, 0.0, 1.0), (2.0, 0.0, 1.2), (0.0, 3.0, 1.0))
    structures = [
        Structure(
            name=f"cube_{index}",
            geometry=Box(position=center, size=(1.5, 1.5, 2.0), device="cuda"),
            material=Material(eps_r=5.0, sigma_e=0.01),
        )
        for index, center in enumerate(centers)
    ]
    return Scene(structures=structures, frequency=FREQUENCY, device="cuda")


def _open_wall_scene() -> Scene:
    mesh = Mesh(
        vertices=(
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 3.0),
            (1.0, 0.0, 3.0),
        ),
        faces=((0, 1, 3), (0, 3, 2)),
        recenter=False,
        device="cpu",
    )
    return Scene(
        structures=[Structure(geometry=mesh, material=Material(), name="open_wall")],
        device="cpu",
    )


def _two_edge_states_for_axial_visibility() -> DiffractionStates:
    material = {
        "eta_r": mc_types.Float([5.0, 5.0]),
        "sigma": mc_types.Float([0.0, 0.0]),
        "gain": mc_types.Float([1.0, 1.0]),
        "use_fresnel": mc_types.Bool([True, True]),
    }
    return DiffractionStates(
        edge_index=mc_types.Int32([0, 1]),
        edge_pos=mc_types.Point3f(
            mc_types.Float([0.5, 1.5]),
            mc_types.Float([1.0, -2.0]),
            mc_types.Float([1.5, 1.5]),
        ),
        edge_dir=mc_types.Vector3f(
            mc_types.Float([0.0, 0.0]),
            mc_types.Float([0.0, 0.0]),
            mc_types.Float([1.0, 1.0]),
        ),
        n0=mc_types.Vector3f(
            mc_types.Float([1.0, 1.0]),
            mc_types.Float([0.0, 0.0]),
            mc_types.Float([0.0, 0.0]),
        ),
        nn=mc_types.Vector3f(
            mc_types.Float([0.0, 0.0]),
            mc_types.Float([1.0, 1.0]),
            mc_types.Float([0.0, 0.0]),
        ),
        wedge_n=mc_types.Float([1.5, 1.5]),
        edge_line_min=mc_types.Float([-0.25, -0.25]),
        edge_line_max=mc_types.Float([0.25, 0.25]),
        source_pos=mc_types.Point3f(
            mc_types.Float([0.5, 1.5]),
            mc_types.Float([-1.0, -1.0]),
            mc_types.Float([1.5, 1.5]),
        ),
        adjacent_face0=mc_types.Int32([0, 0]),
        adjacent_face1=mc_types.Int32([1, 1]),
        face0_material=material,
        face1_material=material,
    )


def _add_radio_map_endpoints(scene: Scene, *, tx_pos, grid) -> Scene:
    scene.add(Transmitter("tx", tx_pos))
    scene.add(
        ReceiverGrid(
            "rm",
            axis=grid.axis,
            position=grid.position,
            bounds=grid.bounds,
            grid_shape=grid.grid_shape,
            cell_size=grid.cell_size,
        )
    )
    return scene


def _resolved(config: Config) -> ResolvedTraceConfig:
    return ResolvedTraceConfig.from_config(
        frequency=FREQUENCY,
        config=config.to_trace_config(),
    )


def _as_np(values, key: str) -> np.ndarray:
    dr.eval(values[key])
    dr.sync_thread()
    return np.asarray(values[key], dtype=np.float64)


def _assert_response_close(
    *,
    native_values,
    drjit_values,
    response_key: str,
    weight_key: str,
) -> None:
    native_response = _as_np(native_values, response_key)
    drjit_response = _as_np(drjit_values, response_key)
    native_weight = _as_np(native_values, weight_key)
    drjit_weight = _as_np(drjit_values, weight_key)
    # Raw response values in very low-weight cells are numerically unstable and
    # are only used after weight damping. Keep the direct response comparison on
    # cells where the response materially affects the correction, and compare
    # weighted responses across the full grid below.
    active = np.maximum(native_weight, drjit_weight) > 1.0e-3
    if np.any(active):
        np.testing.assert_allclose(
            native_response[active],
            drjit_response[active],
            rtol=3.0e-3,
            atol=1.0e-4,
        )
    np.testing.assert_allclose(
        native_response * native_weight,
        drjit_response * drjit_weight,
        rtol=2.5e-4,
        atol=2.0e-6,
    )


def test_finite_edge_support_cuts_out_stationary_points_outside_segment():
    config = _resolved(Config())
    _, _, support = ShadowBoundary._finite_edge_factor(
        source_pos=mc_types.Point3f(0.0, -1.0, 0.0),
        target_pos=mc_types.Point3f(
            mc_types.Float([0.0, 2.1]),
            mc_types.Float([1.0, 1.0]),
            mc_types.Float([0.0, 0.0]),
        ),
        edge_pos=mc_types.Point3f(0.0, 0.0, 0.0),
        edge_dir=mc_types.Vector3f(1.0, 0.0, 0.0),
        edge_line_min=mc_types.Float(-1.0),
        edge_line_max=mc_types.Float(1.0),
        k=config.k,
    )

    support_values = np.asarray(support, dtype=np.float32)
    assert support_values[0] > 0.0
    assert support_values[1] == 0.0


@pytest.mark.gpu
def test_source_visible_edge_mask_uses_rayd_axial_visibility_without_segment_loop():
    scene = _open_wall_scene()
    states = _two_edge_states_for_axial_visibility()

    def fail_segment_visible(*args, **kwargs):
        raise AssertionError("source edge visibility should use RayD axial visibility")

    scene.segment_visible = fail_segment_visible
    visible = ShadowBoundary._source_visible_edge_mask(states=states, scene=scene)
    dr.eval(visible)

    np.testing.assert_array_equal(np.asarray(visible, dtype=bool), np.array([False, True]))


def test_shadow_boundary_config_defaults_and_validation():
    cfg = Config()
    assert cfg.tuning.shadow_boundary_backend == "auto"
    assert cfg.tuning.shadow_boundary_tile_shape == (8, 8)
    assert cfg.tuning.shadow_boundary_band_width_wavelengths == 3.0
    assert cfg.tuning.shadow_boundary_max_candidate_factor == 64.0

    explicit = Config(
        tuning=Tuning(
            shadow_boundary_backend="native_candidate",
            shadow_boundary_tile_shape=(4, 16),
            shadow_boundary_band_width_wavelengths=5.0,
            shadow_boundary_max_candidate_factor=32.0,
        ),
    )
    assert explicit.to_trace_config().shadow_boundary_backend == "native_candidate"

    with pytest.raises(ValueError, match="shadow_boundary_backend"):
        Config(tuning=Tuning(shadow_boundary_backend="cuda"))
    with pytest.raises(ValueError, match="shadow_boundary_tile_shape"):
        Config(tuning=Tuning(shadow_boundary_tile_shape=(0, 8)))
    with pytest.raises(ValueError, match="shadow_boundary_band_width_wavelengths"):
        Config(tuning=Tuning(shadow_boundary_band_width_wavelengths=0.0))
    with pytest.raises(ValueError, match="shadow_boundary_max_candidate_factor"):
        Config(tuning=Tuning(shadow_boundary_max_candidate_factor=0.0))


def test_native_candidate_matches_drjit_reference_with_full_band():
    if not NativeExtension.native_extension_available():
        pytest.skip("Monte Carlo native extension is unavailable in this environment.")

    scene = _three_cube_scene()
    grid = Grid.from_spec(
        GridSpec(axis="z", position=1.0, bounds=((-8.0, 8.0), (-8.0, 8.0)), grid_shape=(6, 6))
    )
    tx_pos = wt.Point3f(0.0, -5.0, 4.0)
    common_tuning = {
        "enable_rd_diffraction": True,
        "shadow_boundary_mode": "utd_power_smoothing",
        "shadow_boundary_tile_shape": (3, 3),
        "shadow_boundary_band_width_wavelengths": 1.0e9,
        "shadow_boundary_max_candidate_factor": 1.0e9,
    }

    drjit_config = Config(
        max_diffraction_order=1,
        edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
        tuning=Tuning(shadow_boundary_backend="drjit", **common_tuning),
    )
    native_config = Config(
        max_diffraction_order=1,
        edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
        tuning=Tuning(shadow_boundary_backend="native_candidate", **common_tuning),
    )
    scene.diffraction_edge_count(edge_policy=drjit_config.edge_policy)
    drjit_values = ShadowBoundary.transition_weights_for_grid(
        scene=scene,
        tx_pos=tx_pos,
        grid=grid,
        config=_resolved(drjit_config),
    )
    scene.diffraction_edge_count(edge_policy=native_config.edge_policy)
    native_values = ShadowBoundary.transition_weights_for_grid(
        scene=scene,
        tx_pos=tx_pos,
        grid=grid,
        config=_resolved(native_config),
    )

    for key in (
        "incident_shadow_boundary_weight",
        "reflection_shadow_boundary_weight",
    ):
        np.testing.assert_allclose(
            _as_np(native_values, key),
            _as_np(drjit_values, key),
            rtol=2.5e-4,
            atol=2.5e-5,
        )
    _assert_response_close(
        native_values=native_values,
        drjit_values=drjit_values,
        response_key="incident_transition_response_real",
        weight_key="incident_shadow_boundary_weight",
    )
    _assert_response_close(
        native_values=native_values,
        drjit_values=drjit_values,
        response_key="incident_transition_response_imag",
        weight_key="incident_shadow_boundary_weight",
    )
    _assert_response_close(
        native_values=native_values,
        drjit_values=drjit_values,
        response_key="reflection_transition_response_real",
        weight_key="reflection_shadow_boundary_weight",
    )
    _assert_response_close(
        native_values=native_values,
        drjit_values=drjit_values,
        response_key="reflection_transition_response_imag",
        weight_key="reflection_shadow_boundary_weight",
    )

    metadata = native_values["_metadata"]
    assert metadata["backend"] == "native_candidate"
    assert metadata["candidate_tiles"] > 0
    assert metadata["candidate_ratio"] <= 1.0


def test_shadow_boundary_explicit_empty_edge_set_does_not_fallback_to_scene_edges():
    scene = _three_cube_scene()
    grid = Grid.from_spec(
        GridSpec(axis="z", position=1.0, bounds=((-8.0, 8.0), (-8.0, 8.0)), grid_shape=(4, 4))
    )
    values = ShadowBoundary.transition_weights_for_grid(
        scene=scene,
        tx_pos=wt.Point3f(0.0, -5.0, 4.0),
        grid=grid,
        config=_resolved(
            Config(
                max_diffraction_order=1,
                edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
                tuning=Tuning(
                    shadow_boundary_backend="drjit",
                    enable_rd_diffraction=True,
                ),
            )
        ),
        edge_indices=dr.zeros(wt.UInt32, 0),
    )

    metadata = values["_metadata"]
    assert metadata["backend"] == "none"
    assert metadata["source_total_edges"] == 0
    assert metadata["source_visible_edges"] == 0
    np.testing.assert_array_equal(
        _as_np(values, "incident_shadow_boundary_weight"),
        np.zeros(int(grid.n_cells), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        _as_np(values, "reflection_shadow_boundary_weight"),
        np.zeros(int(grid.n_cells), dtype=np.float64),
    )


def test_basic_shadow_boundary_uses_discovered_diffraction_edges():
    grid = ReceiverGrid(
        "rm",
        axis="z",
        position=1.0,
        bounds=((-8.0, 8.0), (-8.0, 8.0)),
        grid_shape=(6, 6),
    )
    result = solve(
        scene=_add_radio_map_endpoints(
            _three_cube_scene(),
            tx_pos=wt.Point3f(0.0, -5.0, 4.0),
            grid=grid,
        ),
        transmitter="tx",
        receiver="rm",
        config=Config(
            num_samples=512,
            max_bounces=1,
            max_diffraction_order=1,
            edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
            tuning=Tuning(
                enable_rd_diffraction=True,
                shadow_boundary_mode="utd_power_smoothing",
                shadow_boundary_backend="drjit",
                shadow_boundary_tile_shape=(3, 3),
                shadow_boundary_band_width_wavelengths=1.0e9,
                shadow_boundary_max_candidate_factor=1.0e9,
            ),
            integrator_options=IntegratorOptions(
                integrator="basic",
                samples_per_tx=512,
                accumulation_backend="auto",
                seed=7,
                ad=False,
            ),
        ),
    )

    state_pool = result.metadata["monte_carlo"]["state_pool"]
    correction = result.metadata["monte_carlo"]["shadow_boundary_correction"]
    assert 0 < state_pool["kept"] < result.metadata["scene"]["n_diffraction_edges"]
    assert correction["source_total_edges"] == state_pool["kept"]


def test_bdpt_shadow_boundary_uses_accepted_first_order_edges():
    grid = ReceiverGrid(
        "rm",
        axis="z",
        position=1.0,
        bounds=((-8.0, 8.0), (-8.0, 8.0)),
        grid_shape=(6, 6),
    )
    result = solve(
        scene=_add_radio_map_endpoints(
            _three_cube_scene(),
            tx_pos=wt.Point3f(0.0, -5.0, 4.0),
            grid=grid,
        ),
        transmitter="tx",
        receiver="rm",
        config=Config(
            num_samples=128,
            max_bounces=1,
            max_diffraction_order=1,
            edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
            tuning=Tuning(
                enable_rd_diffraction=True,
                enable_bdpt_reflection_coupled_diffraction=False,
                shadow_boundary_mode="utd_power_smoothing",
                shadow_boundary_backend="drjit",
                shadow_boundary_tile_shape=(3, 3),
                shadow_boundary_band_width_wavelengths=1.0e9,
                shadow_boundary_max_candidate_factor=1.0e9,
            ),
            integrator_options=IntegratorOptions(
                integrator="bdpt",
                samples_per_tx=128,
                accumulation_backend="auto",
                seed=7,
                ad=False,
            ),
        ),
    )

    correction = result.metadata["monte_carlo"]["shadow_boundary_correction"]
    assert 0 < correction["source_total_edges"] < result.metadata["scene"]["n_diffraction_edges"]
    assert correction["source_total_edges"] <= result.metadata["path_counts"]["diffraction"]


def test_bdpt_uses_native_shadow_boundary_backend_metadata():
    if not NativeExtension.native_extension_available():
        pytest.skip("Monte Carlo native extension is unavailable in this environment.")

    grid = ReceiverGrid(
        "rm",
        axis="z",
        position=1.0,
        bounds=((-8.0, 8.0), (-8.0, 8.0)),
        grid_shape=(8, 8),
    )
    result = solve(
        scene=_add_radio_map_endpoints(
            _three_cube_scene(),
            tx_pos=wt.Point3f(0.0, -5.0, 4.0),
            grid=grid,
        ),
        transmitter="tx",
        receiver="rm",
        config=Config(
            num_samples=256,
            max_bounces=1,
            max_diffraction_order=1,
            edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
            tuning=Tuning(
                enable_rd_diffraction=True,
                enable_bdpt_reflection_coupled_diffraction=False,
                shadow_boundary_mode="utd_power_smoothing",
                shadow_boundary_backend="native_candidate",
                shadow_boundary_tile_shape=(4, 4),
                shadow_boundary_band_width_wavelengths=1.0e9,
                shadow_boundary_max_candidate_factor=1.0e9,
            ),
            integrator_options=IntegratorOptions(
                integrator="bdpt",
                samples_per_tx=256,
                accumulation_backend="auto",
                seed=3,
                ad=False,
            ),
        ),
    )

    correction = result.metadata["monte_carlo"]["shadow_boundary_correction"]
    assert correction["enabled"] is True
    assert correction["backend"] == "native_candidate"
    assert correction["candidate_tiles"] > 0
    assert np.isfinite(np.asarray(result.path_gain, dtype=np.float64)).all()


def test_bdpt_ad_tx_x_jvp_is_finite():
    tx_x = mc_types.Float(0.0)
    dr.enable_grad(tx_x)
    grid = ReceiverGrid(
        "rm",
        axis="z",
        position=1.0,
        bounds=((-8.0, 8.0), (-8.0, 8.0)),
        grid_shape=(4, 4),
    )
    result = solve(
        scene=_add_radio_map_endpoints(
            _three_cube_scene(),
            tx_pos=mc_types.Point3f(tx_x, -5.0, 4.0),
            grid=grid,
        ),
        transmitter="tx",
        receiver="rm",
        config=Config(
            num_samples=64,
            max_bounces=1,
            max_diffraction_order=1,
            edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
            tuning=Tuning(
                enable_rd_diffraction=True,
                enable_bdpt_reflection_coupled_diffraction=False,
                shadow_boundary_mode="none",
            ),
            integrator_options=IntegratorOptions(
                integrator="bdpt",
                samples_per_tx=64,
                accumulation_backend="auto",
                seed=11,
                ad=True,
            ),
        ),
    )

    dr.set_grad(tx_x, 1.0)
    jvp = dr.forward_to(
        result.path_gain,
        flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad,
    )
    jvp_np = np.asarray(jvp, dtype=np.float64)

    assert jvp_np.shape == (1, 4, 4)
    assert np.isfinite(jvp_np).all()


def test_munich_shadow_boundary_culls_tx_hidden_edges_before_candidate_smoothing():
    if not NativeExtension.native_extension_available():
        pytest.skip("Monte Carlo native extension is unavailable in this environment.")

    repo_root = Path(__file__).resolve().parents[2]
    scene_path = (
        repo_root
        / "sionna-rt-reference-2.0.0"
        / "src"
        / "sionna"
        / "rt"
        / "scenes"
        / "munich"
        / "munich.xml"
    )
    if not scene_path.exists():
        pytest.skip("Bundled Munich Sionna RT scene is unavailable.")

    scene = Scene.load_mitsuba(
        scene_path,
        device="cuda",
        merge_shapes=True,
        frequency=FREQUENCY,
    )
    grid = Grid.from_spec(
        GridSpec(
            axis="z",
            position=1.5,
            bounds=((-120.0, 120.0), (-120.0, 140.0)),
            grid_shape=(32, 32),
        )
    )
    config = Config(
        max_diffraction_order=2,
        edge_policy=EdgePolicy(edge_selection_mode="vertical_only"),
        tuning=Tuning(
            enable_rd_diffraction=True,
            shadow_boundary_backend="native_candidate",
            shadow_boundary_tile_shape=(8, 8),
            shadow_boundary_band_width_wavelengths=3.0,
            shadow_boundary_max_candidate_factor=1.0e9,
        ),
        integrator_options=IntegratorOptions(integrator="bdpt"),
    )
    scene.diffraction_edge_count(edge_policy=config.edge_policy)
    values = ShadowBoundary.transition_weights_for_grid(
        scene=scene,
        tx_pos=wt.Point3f(8.5, 21.0, 27.0),
        grid=grid,
        config=_resolved(config),
    )

    metadata = values["_metadata"]
    assert metadata["source_visibility_samples_per_edge"] >= 3
    assert metadata["source_total_edges"] == scene.n_diffraction_edges
    assert 0 < metadata["source_visible_edges"] < 0.25 * metadata["source_total_edges"]
    reflection_weight = _as_np(values, "reflection_shadow_boundary_weight")
    assert float(np.percentile(reflection_weight, 95.0)) < 0.99
    assert float(np.mean(reflection_weight >= 0.999)) < 0.01


def test_munich_shadow_boundary_does_not_continue_incident_power_from_unmatched_blockers():
    if not NativeExtension.native_extension_available():
        pytest.skip("Monte Carlo native extension is unavailable in this environment.")

    repo_root = Path(__file__).resolve().parents[2]
    scene_path = (
        repo_root
        / "sionna-rt-reference-2.0.0"
        / "src"
        / "sionna"
        / "rt"
        / "scenes"
        / "munich"
        / "munich.xml"
    )
    if not scene_path.exists():
        pytest.skip("Bundled Munich Sionna RT scene is unavailable.")

    scene = Scene.load_mitsuba(
        scene_path,
        device="cuda",
        merge_shapes=True,
        frequency=FREQUENCY,
    )
    grid = ReceiverGrid(
        "rm",
        axis="z",
        position=1.5,
        bounds=((-120.0, 120.0), (-120.0, 140.0)),
        grid_shape=(32, 32),
    )
    _add_radio_map_endpoints(scene, tx_pos=wt.Point3f(8.5, 21.0, 27.0), grid=grid)
    common_config = dict(
        num_samples=32768,
        max_bounces=1,
        max_diffraction_order=2,
        edge_policy=EdgePolicy(edge_selection_mode="vertical_only"),
    )
    common_tuning = dict(
        enable_rd_diffraction=True,
        enable_bdpt_reflection_coupled_diffraction=False,
        shadow_boundary_tile_shape=(8, 8),
        shadow_boundary_band_width_wavelengths=3.0,
        shadow_boundary_max_candidate_factor=128.0,
        shadow_boundary_backend="native_candidate",
    )
    common_options = dict(
        integrator="bdpt",
        samples_per_tx=32768,
        seed=11,
    )
    raw_result = solve(
        scene=scene,
        transmitter="tx",
        receiver="rm",
        config=Config(
            **common_config,
            tuning=Tuning(shadow_boundary_mode="none", **common_tuning),
            integrator_options=IntegratorOptions(**common_options),
        ),
    )
    smoothed_result = solve(
        scene=scene,
        transmitter="tx",
        receiver="rm",
        config=Config(
            **common_config,
            tuning=Tuning(shadow_boundary_mode="utd_power_smoothing", **common_tuning),
            integrator_options=IntegratorOptions(**common_options),
        ),
    )

    raw_total = np.asarray(raw_result.incoherent["total"], dtype=np.float64)
    los = np.asarray(smoothed_result.incoherent["los"], dtype=np.float64)
    correction = np.asarray(
        smoothed_result.incoherent["shadow_boundary_correction"],
        dtype=np.float64,
    )
    dark_unlit = (raw_total < 1.0e-14) & (los <= 0.0)
    assert int(np.count_nonzero(dark_unlit)) > 100
    assert float(np.percentile(correction[dark_unlit], 99.0)) < 1.0e-11


def test_munich_all_edges_shadow_boundary_uses_sampled_transition_support():
    if not NativeExtension.native_extension_available():
        pytest.skip("Monte Carlo native extension is unavailable in this environment.")

    repo_root = Path(__file__).resolve().parents[2]
    scene_path = (
        repo_root
        / "sionna-rt-reference-2.0.0"
        / "src"
        / "sionna"
        / "rt"
        / "scenes"
        / "munich"
        / "munich.xml"
    )
    if not scene_path.exists():
        pytest.skip("Bundled Munich Sionna RT scene is unavailable.")

    scene = Scene.load_mitsuba(
        scene_path,
        device="cuda",
        merge_shapes=True,
        frequency=FREQUENCY,
    )
    grid = ReceiverGrid(
        "rm",
        axis="z",
        position=1.5,
        bounds=((-120.0, 120.0), (-120.0, 140.0)),
        grid_shape=(24, 24),
    )
    _add_radio_map_endpoints(scene, tx_pos=wt.Point3f(8.5, 21.0, 27.0), grid=grid)
    common_config = dict(
        num_samples=8192,
        max_bounces=1,
        max_diffraction_order=1,
        edge_policy=EdgePolicy(
            edge_selection_mode="all_edges",
            edge_diffraction=True,
            boundary_edge_policy=None,
        ),
    )
    common_tuning = dict(
        enable_rd_diffraction=True,
        enable_bdpt_reflection_coupled_diffraction=False,
        shadow_boundary_backend="native_candidate",
        shadow_boundary_tile_shape=(8, 8),
        shadow_boundary_band_width_wavelengths=3.0,
        shadow_boundary_max_candidate_factor=128.0,
    )
    common_options = dict(
        integrator="bdpt",
        samples_per_tx=8192,
        seed=11,
        ad=False,
    )
    raw_result = solve(
        scene=scene,
        transmitter="tx",
        receiver="rm",
        config=Config(
            **common_config,
            tuning=Tuning(shadow_boundary_mode="none", **common_tuning),
            integrator_options=IntegratorOptions(**common_options),
        ),
    )
    smoothed_result = solve(
        scene=scene,
        transmitter="tx",
        receiver="rm",
        config=Config(
            **common_config,
            tuning=Tuning(shadow_boundary_mode="utd_power_smoothing", **common_tuning),
            integrator_options=IntegratorOptions(**common_options),
        ),
    )

    correction_metadata = smoothed_result.metadata["monte_carlo"][
        "shadow_boundary_correction"
    ]
    assert 0 < correction_metadata["source_total_edges"] < scene.n_diffraction_edges

    raw_total = np.asarray(raw_result.incoherent["total"], dtype=np.float64)
    los = np.asarray(smoothed_result.incoherent["los"], dtype=np.float64)
    correction = np.asarray(
        smoothed_result.incoherent["shadow_boundary_correction"],
        dtype=np.float64,
    )
    dark_unlit = (raw_total < 1.0e-14) & (los <= 0.0)
    assert int(np.count_nonzero(dark_unlit)) > 100
    assert float(np.percentile(correction[dark_unlit], 99.0)) <= 0.0
