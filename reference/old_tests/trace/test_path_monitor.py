from __future__ import annotations

import drjit as dr
import numpy as np
import torch
import pytest

from tests._scene_helpers import box_geometry, build_scene
from witwin.channel import InteractionType, PathMonitor, FieldMonitor, Tracer
import witwin as wt
from witwin.channel.validation import build_single_wedge_case
import witwin.channel.trace.reflection.api as reflection_field
import witwin.channel.trace.reflection.epc as reflection_epc_module
import witwin.channel.monitors.path.collectors as path_collectors_module
import witwin.channel.monitors.path.trace as trace_path_module
import witwin.channel.monitors.path.result as result_module
from witwin.channel.monitors.path.result import PathResult
from witwin.channel.trace.diffraction.state import PATH_EXPORT_REDUCED_STATE_LAYOUT


C = 299792458.0


def _path_amplitudes(paths: PathResult) -> np.ndarray:
    return np.asarray(paths.a, dtype=np.complex64).reshape(paths.path_shape)


def _dense_raw_paths(
    *,
    rx_index,
    amplitudes,
    delays,
    interaction_type,
):
    count = len(rx_index)
    return {
        "rx_index": wt.UInt32(rx_index),
        "a": wt.Complex2f(
            wt.Float([float(value) for value in amplitudes]),
            wt.Float([0.0] * count),
        ),
        "tau": wt.Float([float(value) for value in delays]),
        "theta_t": wt.Float([0.1] * count),
        "phi_t": wt.Float([0.2] * count),
        "theta_r": wt.Float([0.3] * count),
        "phi_r": wt.Float([0.4] * count),
        "type_slots": (dr.full(wt.Int32, int(interaction_type), count),),
        "vertex_slots": None,
        "normal_slots": None,
        "object_slots": None,
        "metadata": {"n_paths": count},
    }


@pytest.mark.gpu
def test_path_monitor_los_shapes_and_channel_helpers():
    tx = torch.tensor([-1.0, -2.0, 1.5], dtype=torch.float32)
    rx_positions = torch.tensor(
        [
            [0.0, 3.0, 1.5],
            [1.0, 4.0, 1.5],
        ],
        dtype=torch.float32,
    )
    scene = build_scene()
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )

    paths = tracer.trace(
        tx,
        monitor=PathMonitor(
            "rx",
            positions=rx_positions,
            max_diffractions=0,
        ),
        verbose=False,
    )

    assert not isinstance(paths.a, torch.Tensor)
    assert not isinstance(paths.tau, torch.Tensor)
    assert _path_amplitudes(paths).shape == (2, 1)
    assert paths.tau.shape == (2, 1)
    np.testing.assert_array_equal(np.asarray(paths.valid, dtype=np.bool_), np.ones((2, 1), dtype=np.bool_))
    np.testing.assert_array_equal(np.asarray(paths.num_paths, dtype=np.int32), np.asarray([1, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(paths.types, dtype=np.int32)[:, 0, 0],
        np.asarray([InteractionType.NONE, InteractionType.NONE], dtype=np.int32),
    )

    expected_tau = torch.linalg.norm(rx_positions - tx.unsqueeze(0), dim=1) / C
    np.testing.assert_allclose(np.asarray(paths.tau, dtype=np.float32)[:, 0], expected_tau.cpu().numpy(), rtol=1e-5, atol=1e-8)

    cir_a, cir_tau = paths.cir(normalize_delays=True)
    assert cir_a.shape == paths.path_shape
    assert cir_tau.shape == paths.tau.shape
    torch.testing.assert_close(cir_tau[:, 0].cpu(), torch.zeros((2,), dtype=torch.float32), atol=1e-8, rtol=0.0)

    cfr = paths.cfr(torch.linspace(27.5e9, 28.5e9, 16))
    taps = paths.taps(bandwidth=100e6, num_taps=8)
    assert cfr.shape == (2, 16)
    assert taps.shape == (2, 8)


@pytest.mark.gpu
def test_trace_supports_mixed_plane_and_path_monitors():
    plane = FieldMonitor(
        "field_xy",
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
        grid_size=(6, 6),
    )
    path = PathMonitor(
        "rx_array",
        positions=torch.tensor(
            [
                [0.0, 2.5, 1.5],
                [1.0, 2.5, 1.5],
            ],
            dtype=torch.float32,
        ),
        max_diffractions=0,
    )
    scene = build_scene(monitors=[plane, path])
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )

    result = tracer.trace((0.0, -3.0, 1.5), verbose=False)
    field_payload = result["field_xy"]
    paths = result["rx_array"]

    assert set(result.keys()) == {"field_xy", "rx_array"}
    assert field_payload.name == "field_xy"
    assert paths.num_rx == 2


@pytest.mark.gpu
def test_tracer_trace_many_supports_multi_tx_multi_rx_path_monitor():
    tx_positions = (
        torch.tensor([-1.0, -2.0, 1.5], dtype=torch.float32),
        torch.tensor([2.0, -2.0, 1.5], dtype=torch.float32),
    )
    rx_positions = torch.tensor(
        [
            [0.0, 3.0, 1.5],
            [1.0, 4.0, 1.5],
        ],
        dtype=torch.float32,
    )
    scene = build_scene()
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )
    monitor = PathMonitor(
        "rx",
        positions=rx_positions,
        max_diffractions=0,
    )

    expected_results = tuple(
        tracer.trace(tx, monitor=monitor, verbose=False)
        for tx in tx_positions
    )
    results = tracer.trace_many(tx_positions, monitor=monitor, verbose=False)

    assert len(results) == 2
    for result, expected_result, tx in zip(results, expected_results, tx_positions):
        paths = result
        expected_paths = expected_result
        assert paths.num_rx == 2
        assert paths.tx_pos == tuple(float(value) for value in tx.tolist())
        np.testing.assert_allclose(_path_amplitudes(paths), _path_amplitudes(expected_paths))
        np.testing.assert_allclose(np.asarray(paths.tau, dtype=np.float32), np.asarray(expected_paths.tau, dtype=np.float32))
        np.testing.assert_array_equal(np.asarray(paths.valid, dtype=np.bool_), np.asarray(expected_paths.valid, dtype=np.bool_))
        np.testing.assert_array_equal(np.asarray(paths.num_paths, dtype=np.int32), np.asarray(expected_paths.num_paths, dtype=np.int32))


@pytest.mark.gpu
def test_trace_many_supports_per_request_tx_and_rx_overrides():
    base_monitor = PathMonitor(
        "rx",
        positions=torch.tensor([[0.0, 3.0, 1.5]], dtype=torch.float32),
        max_diffractions=0,
    )
    override_rx_a = torch.tensor(
        [
            [0.0, 3.0, 1.5],
            [1.0, 4.0, 1.5],
        ],
        dtype=torch.float32,
    )
    override_rx_b = torch.tensor(
        [
            [-1.0, 2.5, 1.5],
            [1.5, 3.5, 1.5],
            [2.5, 4.5, 1.5],
        ],
        dtype=torch.float32,
    )
    tx_a = torch.tensor([-1.0, -2.0, 1.5], dtype=torch.float32)
    tx_b = torch.tensor([2.0, -2.5, 1.5], dtype=torch.float32)
    scene = build_scene(monitors=[base_monitor])
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )

    expected_results = (
        tracer.trace(
            tx_a,
            monitor_overrides={"rx": {"positions": override_rx_a}},
            verbose=False,
        ),
        tracer.trace(
            tx_b,
            monitor_overrides={"rx": {"positions": override_rx_b}},
            verbose=False,
        ),
    )
    results = tracer.trace_many(
        [
            {
                "tx_pos": tx_a,
                "monitor_overrides": {"rx": {"positions": override_rx_a}},
            },
            {
                "tx_pos": tx_b,
                "monitor_overrides": {"rx": {"positions": override_rx_b}},
            },
        ],
        verbose=False,
    )

    assert base_monitor.num_rx == 1
    np.testing.assert_allclose(
        np.stack(
            [
                np.asarray(base_monitor.positions.x, dtype=np.float32),
                np.asarray(base_monitor.positions.y, dtype=np.float32),
                np.asarray(base_monitor.positions.z, dtype=np.float32),
            ],
            axis=-1,
        ),
        np.asarray([[0.0, 3.0, 1.5]], dtype=np.float32),
    )
    for result, expected_result, tx, rx_positions in zip(
        results,
        expected_results,
        (tx_a, tx_b),
        (override_rx_a, override_rx_b),
    ):
        paths = result
        expected_paths = expected_result
        assert paths.tx_pos == tuple(float(value) for value in tx.tolist())
        assert paths.num_rx == int(rx_positions.shape[0])
        np.testing.assert_allclose(
            np.asarray(paths.rx_positions, dtype=np.float32),
            rx_positions.cpu().numpy(),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(_path_amplitudes(paths), _path_amplitudes(expected_paths))
        np.testing.assert_allclose(np.asarray(paths.tau, dtype=np.float32), np.asarray(expected_paths.tau, dtype=np.float32))
        np.testing.assert_array_equal(np.asarray(paths.valid, dtype=np.bool_), np.asarray(expected_paths.valid, dtype=np.bool_))
        np.testing.assert_array_equal(np.asarray(paths.num_paths, dtype=np.int32), np.asarray(expected_paths.num_paths, dtype=np.int32))


@pytest.mark.gpu
def test_path_monitor_collects_reflection_paths_and_geometry():
    scene = build_scene(
        box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0)),
    )
    monitor = PathMonitor(
        "rx",
        positions=torch.tensor(
            [
                [-3.0, 5.0, 1.5],
                [-2.0, 4.0, 1.5],
            ],
            dtype=torch.float32,
        ),
        max_diffractions=0,
        return_geometry=True,
    )
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=8192,
        reflection_max_bounces=1,
        max_diffractions=0,
    )

    paths = tracer.trace((-3.0, -5.0, 1.5), monitor=monitor, verbose=False)
    reflection_only = paths.filter_by_type(InteractionType.REFLECTION)

    assert np.all(np.asarray(paths.num_paths, dtype=np.int32) >= 2)
    np.testing.assert_array_equal(np.asarray(reflection_only.num_paths, dtype=np.int32), np.asarray([1, 1], dtype=np.int32))
    assert reflection_only.path_shape == (2, 1)
    assert tuple(reflection_only.coeff_tensor().shape) == (2, 1)
    assert paths.vertices is not None
    assert paths.normals is not None
    assert paths.objects is not None
    assert paths.metadata["reflection_sampling"]["selected_ray_sampling"] == "full_sphere"

    reflection_mask = np.asarray(paths.valid, dtype=np.bool_) & (
        np.asarray(paths.types, dtype=np.int32)[:, :, 0] == InteractionType.REFLECTION
    )
    np.testing.assert_array_equal(reflection_mask.sum(axis=1), np.asarray([1, 1], dtype=np.int64))

    reflection_vertices = np.asarray(paths.vertices, dtype=np.float32)[reflection_mask]
    reflection_objects = np.asarray(paths.objects, dtype=np.int32)[reflection_mask]
    assert np.all(np.linalg.norm(reflection_vertices[:, 0, :], axis=-1) > 0.0)
    assert np.all(reflection_objects[:, 0] >= 0)


@pytest.mark.gpu
def test_path_monitor_collects_reflection_paths_without_geometry():
    scene = build_scene(
        box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0)),
    )
    monitor = PathMonitor(
        "rx",
        positions=torch.tensor(
            [
                [-3.0, 5.0, 1.5],
                [-2.0, 4.0, 1.5],
            ],
            dtype=torch.float32,
        ),
        max_diffractions=0,
        return_geometry=False,
    )
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=8192,
        reflection_max_bounces=1,
        max_diffractions=0,
    )

    paths = tracer.trace((-3.0, -5.0, 1.5), monitor=monitor, verbose=False)
    reflection_only = paths.filter_by_type(InteractionType.REFLECTION)

    assert np.all(np.asarray(paths.num_paths, dtype=np.int32) >= 2)
    np.testing.assert_array_equal(np.asarray(reflection_only.num_paths, dtype=np.int32), np.asarray([1, 1], dtype=np.int32))
    assert paths.vertices is None
    assert paths.normals is None
    assert paths.objects is None


@pytest.mark.gpu
def test_path_monitor_reflection_collection_reuses_prepared_family_descriptors(monkeypatch):
    scene = build_scene(
        box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0)),
    )
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=0,
    )
    seen_descriptor_shapes = []
    original_replay = path_collectors_module.epc_reflection_chain_to_target

    def _wrapped_replay(*args, **kwargs):
        descriptor = kwargs.get("epc_descriptor")
        target_pos = kwargs.get("target_pos")
        if descriptor is not None and target_pos is not None:
            seen_descriptor_shapes.append(
                (int(descriptor.n_paths), int(dr.width(target_pos.x)))
            )
        return original_replay(*args, **kwargs)

    monkeypatch.setattr(
        path_collectors_module,
        "epc_reflection_chain_to_target",
        _wrapped_replay,
    )

    paths = tracer.trace(
        (-3.0, -5.0, 1.5),
        monitor=PathMonitor(
            "rx",
            positions=torch.tensor(
                [
                    [-3.0, 5.0, 1.5],
                    [-2.0, 4.0, 1.5],
                ],
                dtype=torch.float32,
            ),
            max_diffractions=0,
        ),
        verbose=False,
    )

    assert seen_descriptor_shapes
    assert any(n_paths < n_pairs for n_paths, n_pairs in seen_descriptor_shapes)
    assert np.all(np.asarray(paths.num_paths, dtype=np.int32) >= 2)


@pytest.mark.gpu
def test_prepared_reflection_epc_descriptor_matches_direct_epc_for_nonzero_path_index():
    scene = build_scene(
        box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0)),
    )
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=8192,
        reflection_max_bounces=1,
        max_diffractions=0,
    )
    config = tracer._resolved_trace_config
    tx_pos = wt.Point3f(-3.0, -5.0, 1.5)
    rx_positions = wt.Point3f(
        wt.Float([-3.0, -2.0]),
        wt.Float([5.0, 4.0]),
        wt.Float([1.5, 1.5]),
    )

    _, detail = path_collectors_module.collect_reflection_paths(
        scene=scene,
        rx_positions=rx_positions,
        tx_pos=tx_pos,
        wavelength=config.wavelength,
        k=config.k,
        n_rays=tracer.reflection_n_rays,
        max_reflections=tracer.reflection_max_bounces,
        mode="3d",
        tx_polarization=config.tx_polarization,
        rx_polarization=config.rx_polarization,
        reflection_coef=config.reflection_coef,
        min_ray_contribution_threshold=config.min_ray_contribution_threshold,
        reflection_relative_permittivity=config.reflection_relative_permittivity,
        reflection_conductivity=config.reflection_conductivity,
        reflection_material=config.reflection_material,
        use_scene_materials=config.use_scene_materials_for_reflection,
        return_geometry=False,
    )
    paths = detail["source_paths_per_bounce"][0]
    assert int(paths.get("n_paths", 0)) >= 2

    absolute_path_idx = wt.UInt32([1])
    local_path_idx = wt.UInt32([0])
    target_pos = wt.Point3f(wt.Float([-3.0]), wt.Float([5.0]), wt.Float([1.5]))
    descriptor = reflection_epc_module.build_reflection_epc_descriptor(
        paths=paths,
        path_idx=absolute_path_idx,
        scene=scene,
        reflection_detail=detail,
    )

    direct_valid, direct_vector, direct_geometry = reflection_epc_module.epc_reflection_chain_to_target(
        paths=paths,
        path_idx=absolute_path_idx,
        target_pos=target_pos,
        scene=scene,
        target_adjacent_faces=(),
        reflection_detail=detail,
        wavelength=config.wavelength,
        tx_polarization=config.tx_polarization,
        return_geometry=False,
        return_endpoints=True,
    )
    prepared_valid, prepared_vector, prepared_geometry = reflection_epc_module.epc_reflection_chain_to_target(
        paths=paths,
        path_idx=local_path_idx,
        target_pos=target_pos,
        scene=scene,
        target_adjacent_faces=(),
        reflection_detail=detail,
        wavelength=config.wavelength,
        tx_polarization=config.tx_polarization,
        return_geometry=False,
        return_endpoints=True,
        epc_descriptor=descriptor,
    )

    np.testing.assert_array_equal(
        np.asarray(prepared_valid, dtype=np.bool_),
        np.asarray(direct_valid, dtype=np.bool_),
    )
    for axis in ("x", "y", "z"):
        np.testing.assert_allclose(
            np.asarray(prepared_vector[axis], dtype=np.complex64),
            np.asarray(direct_vector[axis], dtype=np.complex64),
            rtol=1e-5,
            atol=1e-6,
        )
    for point_name in ("tx_pos", "first_hit", "last_hit"):
        np.testing.assert_allclose(
            np.asarray(prepared_geometry[point_name], dtype=np.float32),
            np.asarray(direct_geometry[point_name], dtype=np.float32),
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.gpu
def test_path_result_reflection_refs_skip_replay_in_no_geometry_assembly(monkeypatch):
    scene = build_scene(
        box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0)),
    )
    rx_positions = torch.tensor(
        [
            [-3.0, 5.0, 1.5],
            [-2.0, 4.0, 1.5],
        ],
        dtype=torch.float32,
    )
    rx_positions_bk = wt.Point3f(
        wt.Float(rx_positions[:, 0].tolist()),
        wt.Float(rx_positions[:, 1].tolist()),
        wt.Float(rx_positions[:, 2].tolist()),
    )
    tx_pos = (-3.0, -5.0, 1.5)
    tx_pos_bk = wt.Point3f(*tx_pos)
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=8192,
        reflection_max_bounces=1,
        max_diffractions=0,
    )
    config = tracer._resolved_trace_config

    reflection_raw, _ = path_collectors_module.collect_reflection_paths(
        scene=scene,
        rx_positions=rx_positions_bk,
        tx_pos=tx_pos_bk,
        wavelength=config.wavelength,
        k=config.k,
        n_rays=tracer.reflection_n_rays,
        max_reflections=tracer.reflection_max_bounces,
        mode="exhaustive",
        tx_polarization=config.tx_polarization,
        rx_polarization=config.rx_polarization,
        reflection_coef=config.reflection_coef,
        min_ray_contribution_threshold=config.min_ray_contribution_threshold,
        reflection_relative_permittivity=config.reflection_relative_permittivity,
        reflection_conductivity=config.reflection_conductivity,
        reflection_material=config.reflection_material,
        use_scene_materials=config.use_scene_materials_for_reflection,
        return_geometry=False,
    )

    assert reflection_raw.get("path_depth") is not None
    assert reflection_raw.get("theta_t") is not None
    reflection_raw["reflection_detail"] = None

    def _fail_replay(*args, **kwargs):
        raise AssertionError("no-geometry reflection assembly should not replay paths")

    monkeypatch.setattr(
        path_collectors_module,
        "epc_reflection_chain_to_target",
        _fail_replay,
    )

    paths = PathResult.from_raw_collections(
        name="rx",
        num_rx=2,
        max_num_paths=None,
        tx_pos=tx_pos,
        rx_positions=rx_positions,
        frequency=float(config.frequency),
        wavelength=float(config.wavelength),
        raw_collections=[reflection_raw],
        return_geometry=False,
        metadata={},
    )

    reflection_mask = np.asarray(paths.valid, dtype=np.bool_) & (
        np.asarray(paths.types, dtype=np.int32)[:, :, 0] == InteractionType.REFLECTION
    )
    assert np.any(reflection_mask)
    assert np.all(np.asarray(paths.num_paths, dtype=np.int32) >= 1)
    assert paths.vertices is None
    assert paths.normals is None
    assert paths.objects is None


@pytest.mark.gpu
def test_path_monitor_collects_first_order_diffraction_paths():
    case = build_single_wedge_case()
    monitor = PathMonitor(
        "rx",
        positions=torch.tensor(
            [
                [0.0, 3.5, case.calculation_height],
                [1.0, 3.5, case.calculation_height],
            ],
            dtype=torch.float32,
        ),
        max_diffractions=1,
    )
    tracer = Tracer(
        frequency=28e9,
        scene=case.scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )

    paths = tracer.trace(case.tx_pos, monitor=monitor, verbose=False)
    diffraction_only = paths.filter_by_type(InteractionType.DIFFRACTION)

    np.testing.assert_array_equal(np.asarray(paths.num_paths, dtype=np.int32), np.asarray([1, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(paths.types, dtype=np.int32)[:, 0, 0],
        np.asarray([InteractionType.DIFFRACTION, InteractionType.DIFFRACTION], dtype=np.int32),
    )
    np.testing.assert_array_equal(np.asarray(diffraction_only.num_paths, dtype=np.int32), np.asarray([1, 1], dtype=np.int32))
    assert paths.metadata["solver_mode"]["requested"]["max_diffractions"] == 1
    assert paths.metadata["solver_mode"]["effective"]["max_diffractions"] == 1
    assert paths.metadata["execution_intent"]["kind"] == "path_export"
    assert paths.metadata["execution_intent"]["path_export_enabled"]
    assert paths.metadata["path_counts"]["diffraction"] >= 2
    assert len(paths.metadata["diffraction_groups"]) == 1
    assert paths.metadata["diffraction_groups"][0]["grouping_axis"] == "z"
    assert paths.metadata["diffraction_groups"][0]["state_layout"] == PATH_EXPORT_REDUCED_STATE_LAYOUT
    assert paths.metadata["diffraction_groups"][0]["path_collection"]["output_paths"] >= 2
    assert (
        paths.metadata["diffraction_groups"][0]["path_collection"]["state_layout"]
        == PATH_EXPORT_REDUCED_STATE_LAYOUT
    )
    assert (
        paths.metadata["runtime_reuse"]["diffraction_state_prep_cache"]["state_layout"]
        == PATH_EXPORT_REDUCED_STATE_LAYOUT
    )
    assert "performance_memory" in paths.metadata


@pytest.mark.gpu
def test_field_monitor_local_max_diffractions_override_enables_diffraction():
    case = build_single_wedge_case()
    monitor = FieldMonitor(
        "field_xy",
        axis="z",
        position=case.calculation_height,
        bounds=((-2.0, 2.0), (2.5, 4.5)),
        grid_size=(8, 8),
        max_diffractions=1,
    )
    tracer = Tracer(
        frequency=28e9,
        scene=case.scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )

    payload = tracer.trace(case.tx_pos, monitor=monitor, verbose=False)
    metadata = payload.metadata

    assert metadata["solver_mode"]["requested"]["max_diffractions"] == 1
    assert metadata["solver_mode"]["effective"]["max_diffractions"] == 1
    assert not metadata.get("diffraction_skipped", False)
    assert np.max(np.abs(np.asarray(payload.field.diffraction, dtype=np.complex64))) > 0.0


@pytest.mark.gpu
def test_path_monitor_default_uses_first_order_and_none_inherits_tracer_limit():
    case = build_single_wedge_case()
    rx_positions = torch.tensor(
        [
            [0.0, 3.5, case.calculation_height],
            [1.0, 3.5, case.calculation_height],
        ],
        dtype=torch.float32,
    )
    default_monitor = PathMonitor("rx_default", positions=rx_positions)
    inherited_monitor = PathMonitor(
        "rx_inherit",
        positions=rx_positions,
        max_diffractions=None,
    )
    tracer = Tracer(
        frequency=28e9,
        scene=case.scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=2,
    )

    result = tracer.trace(
        case.tx_pos,
        monitor=[default_monitor, inherited_monitor],
        verbose=False,
    )

    default_paths = result["rx_default"]
    inherited_paths = result["rx_inherit"]

    assert default_monitor.max_diffractions == 1
    assert default_paths.metadata["solver_mode"]["requested"]["max_diffractions"] == 1
    assert default_paths.metadata["solver_mode"]["effective"]["max_diffractions"] == 1

    assert inherited_monitor.max_diffractions is None
    assert inherited_paths.metadata["solver_mode"]["requested"]["max_diffractions"] == 2
    assert inherited_paths.metadata["solver_mode"]["effective"]["max_diffractions"] == 2


@pytest.mark.gpu
def test_path_monitor_uses_reduced_diffraction_state_layout_for_path_export(monkeypatch):
    case = build_single_wedge_case()
    monitor = PathMonitor(
        "rx",
        positions=torch.tensor(
            [
                [0.0, 3.5, case.calculation_height],
                [1.0, 3.5, case.calculation_height],
            ],
            dtype=torch.float32,
        ),
        max_diffractions=1,
    )
    tracer = Tracer(
        frequency=28e9,
        scene=case.scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )

    original_collect = trace_path_module.collect_diffraction_state_paths
    seen_layouts = []
    eval_fields_present = []

    def _wrapped_collect(*args, **kwargs):
        state_arrays = kwargs["state_arrays"]
        seen_layouts.append(state_arrays.get("__path_export_state_layout__"))
        eval_fields_present.append(
            (
                "incident_field" in state_arrays,
                "incident_vector_x" in state_arrays,
            )
        )
        return original_collect(*args, **kwargs)

    monkeypatch.setattr(
        trace_path_module,
        "collect_diffraction_state_paths",
        _wrapped_collect,
    )

    paths = tracer.trace(case.tx_pos, monitor=monitor, verbose=False)

    assert seen_layouts == [PATH_EXPORT_REDUCED_STATE_LAYOUT]
    assert eval_fields_present == [(True, True)]
    assert paths.metadata["diffraction_groups"][0]["state_layout"] == PATH_EXPORT_REDUCED_STATE_LAYOUT
    assert (
        paths.metadata["runtime_reuse"]["diffraction_state_prep_cache"]["state_layout"]
        == PATH_EXPORT_REDUCED_STATE_LAYOUT
    )


@pytest.mark.gpu
def test_path_monitor_reuses_diffraction_state_prep_cache_across_calls(monkeypatch):
    case = build_single_wedge_case()
    monitor = PathMonitor(
        "rx",
        positions=torch.tensor(
            [
                [0.0, 3.5, case.calculation_height],
                [1.0, 3.5, case.calculation_height],
            ],
            dtype=torch.float32,
        ),
        max_diffractions=1,
    )
    tracer = Tracer(
        frequency=28e9,
        scene=case.scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )

    original_prepare = trace_path_module._prepare_diffraction_state_arrays
    call_count = 0

    def _counted_prepare(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        trace_path_module,
        "_prepare_diffraction_state_arrays",
        _counted_prepare,
    )

    first = tracer.trace(case.tx_pos, monitor=monitor, verbose=False)
    second = tracer.trace(case.tx_pos, monitor=monitor, verbose=False)

    assert call_count == 1
    assert first.metadata["runtime_reuse"]["diffraction_state_prep_cache"]["misses"] == 1
    assert first.metadata["runtime_reuse"]["diffraction_state_prep_cache"]["hits"] == 0
    assert second.metadata["runtime_reuse"]["diffraction_state_prep_cache"]["misses"] == 0
    assert second.metadata["runtime_reuse"]["diffraction_state_prep_cache"]["hits"] == 1
    assert (
        second.metadata["runtime_reuse"]["diffraction_state_prep_cache"]["state_layout"]
        == PATH_EXPORT_REDUCED_STATE_LAYOUT
    )
    assert second.metadata["diffraction_groups"][0]["state_prep_cache_hit"]


@pytest.mark.gpu
def test_path_monitor_reuses_diffraction_state_prep_cache_for_same_z_with_moving_xy(monkeypatch):
    case = build_single_wedge_case()
    first_monitor = PathMonitor(
        "rx_first",
        positions=torch.tensor(
            [
                [0.0, 3.5, case.calculation_height],
                [1.0, 3.5, case.calculation_height],
            ],
            dtype=torch.float32,
        ),
        max_diffractions=1,
    )
    second_monitor = PathMonitor(
        "rx_second",
        positions=torch.tensor(
            [
                [0.3, 3.8, case.calculation_height],
                [1.3, 3.8, case.calculation_height],
            ],
            dtype=torch.float32,
        ),
        max_diffractions=1,
    )
    tracer = Tracer(
        frequency=28e9,
        scene=case.scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )

    original_prepare = trace_path_module._prepare_diffraction_state_arrays
    call_count = 0

    def _counted_prepare(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        trace_path_module,
        "_prepare_diffraction_state_arrays",
        _counted_prepare,
    )

    first = tracer.trace(case.tx_pos, monitor=first_monitor, verbose=False)
    second = tracer.trace(case.tx_pos, monitor=second_monitor, verbose=False)

    assert call_count == 1
    assert first.metadata["runtime_reuse"]["diffraction_state_prep_cache"]["misses"] == 1
    assert second.metadata["runtime_reuse"]["diffraction_state_prep_cache"]["hits"] == 1
    assert second.metadata["diffraction_groups"][0]["state_prep_cache_hit"]


@pytest.mark.gpu
def test_path_monitor_diffraction_state_prep_cache_invalidates_after_scene_update(monkeypatch):
    case = build_single_wedge_case()
    monitor = PathMonitor(
        "rx",
        positions=torch.tensor(
            [
                [0.0, 3.5, case.calculation_height],
                [1.0, 3.5, case.calculation_height],
            ],
            dtype=torch.float32,
        ),
        max_diffractions=1,
    )
    tracer = Tracer(
        frequency=28e9,
        scene=case.scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )

    original_prepare = trace_path_module._prepare_diffraction_state_arrays
    call_count = 0

    def _counted_prepare(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        trace_path_module,
        "_prepare_diffraction_state_arrays",
        _counted_prepare,
    )

    tracer.trace(case.tx_pos, monitor=monitor, verbose=False)
    tracer.update_scene(case.scene.vertices, recompute_edges=True)
    refreshed = tracer.trace(case.tx_pos, monitor=monitor, verbose=False)

    assert call_count == 2
    assert refreshed.metadata["runtime_reuse"]["diffraction_state_prep_cache"]["misses"] == 1
    assert refreshed.metadata["runtime_reuse"]["diffraction_state_prep_cache"]["hits"] == 0


@pytest.mark.gpu
def test_diffraction_state_path_materialization_reduced_layout_avoids_full_gather(monkeypatch):
    def _fail_full_gather(*args, **kwargs):
        raise AssertionError("full gather_inserted_reflection_state_fields should not be used")

    def _fake_slots(*, keep_states, edge_data, edge_object_idx, return_geometry):
        del keep_states, edge_data, edge_object_idx, return_geometry
        return ((dr.full(wt.Int32, InteractionType.DIFFRACTION, 1),), None, None, None, 1)

    monkeypatch.setattr(
        path_collectors_module,
        "gather_inserted_reflection_state_fields",
        _fail_full_gather,
    )
    monkeypatch.setattr(
        path_collectors_module,
        "_build_type_and_geometry_slots",
        _fake_slots,
    )

    raw = {
        "payload_kind": "diffraction_state_refs_v1",
        "rx_index": wt.UInt32([0]),
        "local_rx_index": wt.UInt32([0]),
        "state_idx": wt.UInt32([0]),
        "a": wt.Complex2f(wt.Float([1.0]), wt.Float([0.0])),
        "tau": wt.Float([1.0e-9]),
        "tx_pos": wt.Point3f(0.0, 0.0, 0.0),
        "rx_positions": wt.Point3f(
            wt.Float([0.0]),
            wt.Float([3.0]),
            wt.Float([1.5]),
        ),
        "state_arrays": {
            "__path_export_state_layout__": PATH_EXPORT_REDUCED_STATE_LAYOUT,
            "edge_pos": wt.Point3f(wt.Float([1.0]), wt.Float([2.0]), wt.Float([3.0])),
            "path_length_prefix": wt.Float([2.0]),
            "first_interaction_pos": wt.Point3f(
                wt.Float([0.0]),
                wt.Float([1.0]),
                wt.Float([2.0]),
            ),
            "source_type_code": wt.UInt32([0]),
            "prefix_reflection_depth": wt.UInt32([0]),
            "intermediate_reflection_depth": wt.UInt32([0]),
            "suffix_reflection_depth": wt.UInt32([0]),
            "order": wt.UInt32([1]),
            "n_states": 1,
        },
        "edge_data": None,
        "edge_object_idx": None,
        "metadata": {"n_paths": 1},
    }

    materialized = path_collectors_module._materialize_diffraction_state_path_refs(
        raw,
        return_geometry=False,
    )

    np.testing.assert_array_equal(
        np.asarray(materialized["rx_index"], dtype=np.uint32),
        np.asarray([0], dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        np.asarray(materialized["type_slots"], dtype=np.int32),
        np.asarray([[InteractionType.DIFFRACTION]], dtype=np.int32),
    )


@pytest.mark.gpu
def test_path_monitor_diffraction_geometry_survives_reduced_state_layout():
    case = build_single_wedge_case()
    monitor = PathMonitor(
        "rx",
        positions=torch.tensor(
            [
                [0.0, 3.5, case.calculation_height],
                [1.0, 3.5, case.calculation_height],
            ],
            dtype=torch.float32,
        ),
        max_diffractions=1,
        return_geometry=True,
    )
    tracer = Tracer(
        frequency=28e9,
        scene=case.scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )

    paths = tracer.trace(case.tx_pos, monitor=monitor, verbose=False)

    diffraction_mask = np.asarray(paths.valid, dtype=np.bool_) & (
        np.asarray(paths.types, dtype=np.int32)[:, :, 0] == InteractionType.DIFFRACTION
    )

    assert paths.vertices is not None
    assert paths.normals is not None
    assert paths.objects is not None
    assert np.any(diffraction_mask)
    assert paths.metadata["diffraction_groups"][0]["state_layout"] == PATH_EXPORT_REDUCED_STATE_LAYOUT
    assert (
        paths.metadata["diffraction_groups"][0]["path_collection"]["state_layout"]
        == PATH_EXPORT_REDUCED_STATE_LAYOUT
    )

    diffraction_vertices = np.asarray(paths.vertices, dtype=np.float32)[diffraction_mask]
    diffraction_objects = np.asarray(paths.objects, dtype=np.int32)[diffraction_mask]
    assert np.any(np.linalg.norm(diffraction_vertices[:, 0, :], axis=-1) > 0.0)
    assert np.any(diffraction_objects[:, 0] >= 0)


@pytest.mark.gpu
@pytest.mark.parametrize("return_geometry", [False, True])
def test_path_result_sparse_diffraction_replay_selects_before_materialization(
    monkeypatch,
    return_geometry: bool,
):
    call_log = []

    def _fake_materialize(raw, *, return_geometry, path_indices=None):
        assert path_indices is not None
        selected = np.asarray(path_indices, dtype=np.uint32)
        call_log.append({
            "indices": selected.tolist(),
            "return_geometry": bool(return_geometry),
        })
        count = int(selected.shape[0])
        rx_index = dr.gather(wt.UInt32, raw["rx_index"], path_indices)
        a = dr.gather(wt.Complex2f, raw["a"], path_indices)
        tau = dr.gather(wt.Float, raw["tau"], path_indices)
        payload = {
            "rx_index": rx_index,
            "a": a,
            "tau": tau,
            "theta_t": dr.full(wt.Float, 0.1, count),
            "phi_t": dr.full(wt.Float, 0.2, count),
            "theta_r": dr.full(wt.Float, 0.3, count),
            "phi_r": dr.full(wt.Float, 0.4, count),
            "type_slots": (dr.full(wt.Int32, InteractionType.DIFFRACTION, count),),
            "vertex_slots": None,
            "normal_slots": None,
            "object_slots": None,
            "metadata": dict(raw.get("metadata", {})),
        }
        if return_geometry:
            payload["vertex_slots"] = (
                wt.Point3f(
                    dr.full(wt.Float, 1.0, count),
                    dr.full(wt.Float, 2.0, count),
                    dr.full(wt.Float, 3.0, count),
                ),
            )
            payload["normal_slots"] = (
                wt.Vector3f(
                    dr.full(wt.Float, 0.0, count),
                    dr.full(wt.Float, 1.0, count),
                    dr.full(wt.Float, 0.0, count),
                ),
            )
            payload["object_slots"] = (dr.full(wt.Int32, 7, count),)
        return payload

    monkeypatch.setattr(
        path_collectors_module,
        "_materialize_diffraction_state_path_refs",
        _fake_materialize,
    )

    raw = {
        "payload_kind": "diffraction_state_refs_v1",
        "rx_index": wt.UInt32([0, 0, 0]),
        "local_rx_index": wt.UInt32([0, 0, 0]),
        "state_idx": wt.UInt32([0, 1, 2]),
        "a": wt.Complex2f(
            wt.Float([1.0, 3.0, 2.0]),
            wt.Float([0.0, 0.0, 0.0]),
        ),
        "tau": wt.Float([3.0e-9, 1.0e-9, 2.0e-9]),
        "tx_pos": wt.Point3f(0.0, 0.0, 1.5),
        "rx_positions": wt.Point3f(
            wt.Float([0.0]),
            wt.Float([3.0]),
            wt.Float([1.5]),
        ),
        "state_arrays": {
            "prefix_reflection_depth": wt.UInt32([0, 1, 0]),
            "intermediate_reflection_depth": wt.UInt32([0, 2, 1]),
            "suffix_reflection_depth": wt.UInt32([0, 0, 0]),
            "order": wt.UInt32([1, 2, 1]),
        },
        "edge_data": None,
        "edge_object_idx": None,
        "metadata": {"n_paths": 3},
    }

    paths = PathResult.from_raw_collections(
        name="rx",
        num_rx=1,
        max_num_paths=1,
        tx_pos=(0.0, 0.0, 1.5),
        rx_positions=wt.Point3f(
            wt.Float([0.0]),
            wt.Float([3.0]),
            wt.Float([1.5]),
        ),
        frequency=28e9,
        wavelength=C / 28e9,
        raw_collections=[raw],
        return_geometry=return_geometry,
        metadata={"test_case": "phase7_sparse_replay"},
    )

    assert call_log == [{"indices": [1], "return_geometry": return_geometry}]
    assert paths.max_num_paths == 1
    assert paths.max_depth == 5
    np.testing.assert_array_equal(
        np.asarray(paths.num_paths, dtype=np.int32),
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_allclose(
        _path_amplitudes(paths),
        np.asarray([[3.0 + 0.0j]], dtype=np.complex64),
    )
    if return_geometry:
        assert paths.vertices is not None
        assert paths.normals is not None
        assert paths.objects is not None
        np.testing.assert_array_equal(
            np.asarray(paths.objects, dtype=np.int32)[0, 0, :],
            np.asarray([7, -1, -1, -1, -1], dtype=np.int32),
        )
    else:
        assert paths.vertices is None
        assert paths.normals is None
        assert paths.objects is None


@pytest.mark.gpu
def test_diffraction_state_path_materialization_uses_minimal_native_gather(monkeypatch):
    call_log = []

    def _fail_full_gather(*args, **kwargs):
        raise AssertionError("full gather_state_arrays should not be used for sparse path replay")

    def _fake_minimal_gather(state_arrays, indices):
        del state_arrays
        call_log.append(int(dr.width(indices)))
        count = int(dr.width(indices))
        return {
            "edge_pos": wt.Point3f(
                dr.full(wt.Float, 1.0, count),
                dr.full(wt.Float, 2.0, count),
                dr.full(wt.Float, 3.0, count),
            ),
            "first_interaction_pos": wt.Point3f(
                dr.full(wt.Float, 0.0, count),
                dr.full(wt.Float, 1.0, count),
                dr.full(wt.Float, 2.0, count),
            ),
            "prefix_reflection_depth": dr.zeros(wt.UInt32, count),
            "intermediate_reflection_depth": dr.zeros(wt.UInt32, count),
            "suffix_reflection_depth": dr.zeros(wt.UInt32, count),
            "order": dr.full(wt.UInt32, 1, count),
            "n_states": count,
        }

    def _fake_slots(*, keep_states, edge_data, edge_object_idx, return_geometry):
        del keep_states, edge_data, edge_object_idx, return_geometry
        return ((dr.full(wt.Int32, InteractionType.DIFFRACTION, 2),), None, None, None, 1)

    monkeypatch.setattr(path_collectors_module, "gather_state_arrays", _fail_full_gather)
    monkeypatch.setattr(
        path_collectors_module,
        "gather_inserted_reflection_state_fields",
        _fake_minimal_gather,
    )
    monkeypatch.setattr(path_collectors_module, "_build_type_and_geometry_slots", _fake_slots)

    raw = {
        "payload_kind": "diffraction_state_refs_v1",
        "rx_index": wt.UInt32([0, 1]),
        "local_rx_index": wt.UInt32([0, 1]),
        "state_idx": wt.UInt32([3, 4]),
        "a": wt.Complex2f(
            wt.Float([1.0, 2.0]),
            wt.Float([0.0, 0.0]),
        ),
        "tau": wt.Float([1.0e-9, 2.0e-9]),
        "tx_pos": wt.Point3f(0.0, 0.0, 0.0),
        "rx_positions": wt.Point3f(
            wt.Float([0.0, 0.0]),
            wt.Float([3.0, 4.0]),
            wt.Float([1.5, 1.5]),
        ),
        "state_arrays": {"n_states": 5},
        "edge_data": None,
        "edge_object_idx": None,
        "metadata": {"n_paths": 2},
    }

    materialized = path_collectors_module._materialize_diffraction_state_path_refs(
        raw,
        return_geometry=False,
    )

    assert call_log == [2]
    np.testing.assert_array_equal(
        np.asarray(materialized["rx_index"], dtype=np.uint32),
        np.asarray([0, 1], dtype=np.uint32),
    )
    np.testing.assert_allclose(
        np.asarray(materialized["tau"], dtype=np.float32),
        np.asarray([1.0e-9, 2.0e-9], dtype=np.float32),
    )


@pytest.mark.gpu
def test_path_result_sparse_diffraction_replay_materializes_in_chunks(monkeypatch):
    call_log = []

    def _fake_materialize(raw, *, return_geometry, path_indices=None):
        assert path_indices is not None
        selected = np.asarray(path_indices, dtype=np.uint32)
        call_log.append(selected.tolist())
        count = int(selected.shape[0])
        rx_index = dr.gather(wt.UInt32, raw["rx_index"], path_indices)
        a = dr.gather(wt.Complex2f, raw["a"], path_indices)
        tau = dr.gather(wt.Float, raw["tau"], path_indices)
        return {
            "rx_index": rx_index,
            "a": a,
            "tau": tau,
            "theta_t": dr.full(wt.Float, 0.1, count),
            "phi_t": dr.full(wt.Float, 0.2, count),
            "theta_r": dr.full(wt.Float, 0.3, count),
            "phi_r": dr.full(wt.Float, 0.4, count),
            "type_slots": (dr.full(wt.Int32, InteractionType.DIFFRACTION, count),),
            "vertex_slots": None,
            "normal_slots": None,
            "object_slots": None,
            "metadata": dict(raw.get("metadata", {})),
        }

    monkeypatch.setattr(
        path_collectors_module,
        "_materialize_diffraction_state_path_refs",
        _fake_materialize,
    )
    monkeypatch.setattr(result_module, "_PATH_RESULT_REPLAY_CHUNK_SIZE", 2)

    raw = {
        "payload_kind": "diffraction_state_refs_v1",
        "rx_index": wt.UInt32([0, 0, 0, 0, 0]),
        "local_rx_index": wt.UInt32([0, 0, 0, 0, 0]),
        "state_idx": wt.UInt32([0, 1, 2, 3, 4]),
        "a": wt.Complex2f(
            wt.Float([1.0, 2.0, 3.0, 4.0, 5.0]),
            wt.Float([0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        "tau": wt.Float([1.0e-9, 2.0e-9, 3.0e-9, 4.0e-9, 5.0e-9]),
        "tx_pos": wt.Point3f(0.0, 0.0, 1.5),
        "rx_positions": wt.Point3f(
            wt.Float([0.0]),
            wt.Float([3.0]),
            wt.Float([1.5]),
        ),
        "state_arrays": {
            "prefix_reflection_depth": wt.UInt32([0, 0, 0, 0, 0]),
            "intermediate_reflection_depth": wt.UInt32([0, 0, 0, 0, 0]),
            "suffix_reflection_depth": wt.UInt32([0, 0, 0, 0, 0]),
            "order": wt.UInt32([1, 1, 1, 1, 1]),
        },
        "edge_data": None,
        "edge_object_idx": None,
        "metadata": {"n_paths": 5},
    }

    paths = PathResult.from_raw_collections(
        name="rx",
        num_rx=1,
        max_num_paths=None,
        tx_pos=(0.0, 0.0, 1.5),
        rx_positions=wt.Point3f(
            wt.Float([0.0]),
            wt.Float([3.0]),
            wt.Float([1.5]),
        ),
        frequency=28e9,
        wavelength=C / 28e9,
        raw_collections=[raw],
        return_geometry=False,
        metadata={"test_case": "phase7_sparse_replay_chunking"},
    )

    assert call_log == [[0, 1], [2, 3], [4]]
    assert paths.max_num_paths == 5
    np.testing.assert_array_equal(
        np.asarray(paths.num_paths, dtype=np.int32),
        np.asarray([5], dtype=np.int32),
    )
    np.testing.assert_allclose(
        _path_amplitudes(paths),
        np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j, 5.0 + 0.0j]], dtype=np.complex64),
    )


@pytest.mark.gpu
def test_path_result_sparse_reflection_replay_selects_before_materialization(monkeypatch):
    call_log = []

    def _fake_materialize(raw, *, return_geometry, path_indices=None):
        assert path_indices is not None
        selected = np.asarray(path_indices, dtype=np.uint32)
        call_log.append({
            "indices": selected.tolist(),
            "return_geometry": bool(return_geometry),
        })
        count = int(selected.shape[0])
        rx_index = dr.gather(wt.UInt32, raw["rx_index"], path_indices)
        a = dr.gather(wt.Complex2f, raw["a"], path_indices)
        tau = dr.gather(wt.Float, raw["tau"], path_indices)
        payload = {
            "rx_index": rx_index,
            "a": a,
            "tau": tau,
            "theta_t": dr.full(wt.Float, 0.1, count),
            "phi_t": dr.full(wt.Float, 0.2, count),
            "theta_r": dr.full(wt.Float, 0.3, count),
            "phi_r": dr.full(wt.Float, 0.4, count),
            "type_slots": (
                dr.full(wt.Int32, InteractionType.REFLECTION, count),
                dr.full(wt.Int32, InteractionType.REFLECTION, count),
            ),
            "vertex_slots": None,
            "normal_slots": None,
            "object_slots": None,
            "metadata": dict(raw.get("metadata", {})),
        }
        if return_geometry:
            payload["vertex_slots"] = (
                wt.Point3f(
                    dr.full(wt.Float, 1.0, count),
                    dr.full(wt.Float, 2.0, count),
                    dr.full(wt.Float, 3.0, count),
                ),
                wt.Point3f(
                    dr.full(wt.Float, 4.0, count),
                    dr.full(wt.Float, 5.0, count),
                    dr.full(wt.Float, 6.0, count),
                ),
            )
            payload["normal_slots"] = (
                wt.Vector3f(
                    dr.full(wt.Float, 0.0, count),
                    dr.full(wt.Float, 1.0, count),
                    dr.full(wt.Float, 0.0, count),
                ),
                wt.Vector3f(
                    dr.full(wt.Float, 1.0, count),
                    dr.full(wt.Float, 0.0, count),
                    dr.full(wt.Float, 0.0, count),
                ),
            )
            payload["object_slots"] = (
                dr.full(wt.Int32, 3, count),
                dr.full(wt.Int32, 5, count),
            )
        return payload

    monkeypatch.setattr(
        path_collectors_module,
        "_materialize_reflection_path_refs",
        _fake_materialize,
    )

    raw = {
        "payload_kind": "reflection_path_refs_v1",
        "rx_index": wt.UInt32([0, 0, 0]),
        "path_group_index": wt.UInt32([0, 0, 1]),
        "path_idx": wt.UInt32([0, 1, 0]),
        "a": wt.Complex2f(
            wt.Float([1.0, 2.0, 3.0]),
            wt.Float([0.0, 0.0, 0.0]),
        ),
        "tau": wt.Float([3.0e-9, 2.0e-9, 1.0e-9]),
        "tx_pos": wt.Point3f(0.0, 0.0, 1.5),
        "rx_positions": wt.Point3f(
            wt.Float([0.0]),
            wt.Float([3.0]),
            wt.Float([1.5]),
        ),
        "scene": None,
        "reflection_detail": {
            "detail_kind": "reflection_trace_detail",
            "reflection_model": "test",
            "reflection_model_source": "test",
            "reflection_gain": 1.0,
            "reflection_material": None,
            "use_scene_materials": False,
            "source_paths_per_bounce": (
                {"chain_depth": 1, "n_paths": 2},
                {"chain_depth": 2, "n_paths": 1},
            ),
        },
        "wavelength": C / 1.0e9,
        "tx_polarization": (1.0, 0.0, 0.0),
        "max_depth_hint": 2,
        "metadata": {"n_paths": 3},
    }

    paths = PathResult.from_raw_collections(
        name="rx",
        num_rx=1,
        max_num_paths=1,
        tx_pos=(0.0, 0.0, 1.5),
        rx_positions=wt.Point3f(
            wt.Float([0.0]),
            wt.Float([3.0]),
            wt.Float([1.5]),
        ),
        frequency=1.0e9,
        wavelength=C / 1.0e9,
        raw_collections=[raw],
        return_geometry=False,
        metadata={"test_case": "phase7_sparse_reflection_replay"},
    )

    assert call_log == [{"indices": [2], "return_geometry": False}]
    assert paths.max_num_paths == 1
    assert paths.max_depth == 2
    np.testing.assert_array_equal(
        np.asarray(paths.num_paths, dtype=np.int32),
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_allclose(
        _path_amplitudes(paths),
        np.asarray([[3.0 + 0.0j]], dtype=np.complex64),
    )


@pytest.mark.gpu
def test_path_result_limits_per_rx_by_strength_then_orders_kept_paths_by_tau():
    raw = _dense_raw_paths(
        rx_index=[0, 0, 0, 1, 1, 1],
        amplitudes=[1.0, 4.0, 3.0, 5.0, 1.0, 4.0],
        delays=[4.0e-9, 1.0e-9, 2.0e-9, 3.0e-9, 1.0e-9, 2.0e-9],
        interaction_type=InteractionType.DIFFRACTION,
    )

    paths = PathResult.from_raw_collections(
        name="rx",
        num_rx=2,
        max_num_paths=2,
        tx_pos=(0.0, 0.0, 1.5),
        rx_positions=wt.Point3f(
            wt.Float([0.0, 1.0]),
            wt.Float([3.0, 3.0]),
            wt.Float([1.5, 1.5]),
        ),
        frequency=28e9,
        wavelength=C / 28e9,
        raw_collections=[raw],
        return_geometry=False,
        metadata={"test_case": "phase3_topk_selection"},
    )

    np.testing.assert_array_equal(
        np.asarray(paths.num_paths, dtype=np.int32),
        np.asarray([2, 2], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(paths.tau, dtype=np.float32),
        np.asarray(
            [
                [1.0e-9, 2.0e-9],
                [2.0e-9, 3.0e-9],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        _path_amplitudes(paths),
        np.asarray(
            [
                [4.0 + 0.0j, 3.0 + 0.0j],
                [4.0 + 0.0j, 5.0 + 0.0j],
            ],
            dtype=np.complex64,
        ),
    )


@pytest.mark.gpu
def test_path_result_dense_no_geometry_assembly_skips_normalize(monkeypatch):
    def _fail_normalize(*args, **kwargs):
        raise AssertionError("dense no-geometry assembly should use summary tensor cache")

    monkeypatch.setattr(
        result_module,
        "_normalize_raw_path_collection",
        _fail_normalize,
    )

    raw = _dense_raw_paths(
        rx_index=[0, 0, 1, 1],
        amplitudes=[1.0, 2.0, 3.0, 4.0],
        delays=[4.0e-9, 1.0e-9, 3.0e-9, 2.0e-9],
        interaction_type=InteractionType.NONE,
    )

    paths = PathResult.from_raw_collections(
        name="rx",
        num_rx=2,
        max_num_paths=None,
        tx_pos=(0.0, 0.0, 0.0),
        rx_positions=torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        frequency=1e9,
        wavelength=0.3,
        raw_collections=[raw],
        return_geometry=False,
        metadata={},
    )

    np.testing.assert_allclose(
        np.asarray(paths.tau, dtype=np.float32),
        np.asarray(
            [
                [1.0e-9, 4.0e-9],
                [2.0e-9, 3.0e-9],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(paths.types, dtype=np.int32)[:, :, 0],
        np.asarray(
            [
                [InteractionType.NONE, InteractionType.NONE],
                [InteractionType.NONE, InteractionType.NONE],
            ],
            dtype=np.int32,
        ),
    )


@pytest.mark.gpu
def test_path_result_keeps_cross_summary_tau_order_when_unbounded():
    los_raw = _dense_raw_paths(
        rx_index=[0, 0],
        amplitudes=[1.0, 2.0],
        delays=[3.0e-9, 1.0e-9],
        interaction_type=InteractionType.NONE,
    )
    reflection_raw = _dense_raw_paths(
        rx_index=[0, 0],
        amplitudes=[5.0, 6.0],
        delays=[2.0e-9, 4.0e-9],
        interaction_type=InteractionType.REFLECTION,
    )

    paths = PathResult.from_raw_collections(
        name="rx",
        num_rx=1,
        max_num_paths=None,
        tx_pos=(0.0, 0.0, 1.5),
        rx_positions=wt.Point3f(
            wt.Float([0.0]),
            wt.Float([3.0]),
            wt.Float([1.5]),
        ),
        frequency=1.0e9,
        wavelength=C / 1.0e9,
        raw_collections=[los_raw, reflection_raw],
        return_geometry=False,
        metadata={"test_case": "phase3_cross_summary_tau_order"},
    )

    np.testing.assert_array_equal(
        np.asarray(paths.num_paths, dtype=np.int32),
        np.asarray([4], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(paths.tau, dtype=np.float32),
        np.asarray([[1.0e-9, 2.0e-9, 3.0e-9, 4.0e-9]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(paths.types, dtype=np.int32)[0, :, 0],
        np.asarray(
            [
                InteractionType.NONE,
                InteractionType.REFLECTION,
                InteractionType.NONE,
                InteractionType.REFLECTION,
            ],
            dtype=np.int32,
        ),
    )


@pytest.mark.gpu
def test_trace_shares_reflection_discovery_between_plane_and_path_monitors(monkeypatch):
    scene = build_scene(
        box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0)),
    )
    plane = FieldMonitor(
        "field_xy",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_size=(8, 8),
        ray_mode="3d",
        ray_sampling="full_sphere",
    )
    path = PathMonitor(
        "rx",
        positions=torch.tensor([[-3.0, 5.0, 1.5]], dtype=torch.float32),
        ray_mode="3d",
        max_diffractions=0,
    )
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=0,
    )

    original = reflection_field._trace_reflection_paths
    call_count = 0

    def _counted_trace(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(reflection_field, "_trace_reflection_paths", _counted_trace)

    result = tracer.trace((-3.0, -5.0, 1.5), monitor=[plane, path], verbose=False)

    assert call_count == 1
    assert int(np.asarray(result["rx"].num_paths, dtype=np.int32)[0]) >= 2


@pytest.mark.gpu
def test_path_monitor_reflection_replay_uses_endpoints_without_full_geometry(monkeypatch):
    scene = build_scene(
        box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0)),
    )
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=0,
    )

    original_replay = path_collectors_module.epc_reflection_chain_to_target
    seen_flags = []

    def _wrapped_replay(*args, **kwargs):
        seen_flags.append((kwargs.get("return_geometry"), kwargs.get("return_endpoints")))
        return original_replay(*args, **kwargs)

    monkeypatch.setattr(
        path_collectors_module,
        "epc_reflection_chain_to_target",
        _wrapped_replay,
    )

    paths = tracer.trace(
        (-3.0, -5.0, 1.5),
        monitor=PathMonitor(
            "rx",
            positions=torch.tensor([[-3.0, 5.0, 1.5]], dtype=torch.float32),
            max_diffractions=0,
        ),
        verbose=False,
    )

    assert paths.vertices is None
    assert seen_flags
    assert all(flags == (False, True) for flags in seen_flags)
