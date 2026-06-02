"""Integration coverage for the standalone path solver package."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
import witwin as wt
import witwin.path as path_solver

from witwin.channel import PathMonitor, Scene as LegacyScene, Tracer
from witwin.channel.scene.sionna import load_sionna_rt
from witwin.channel_scene import Receiver, Scene as ChannelScene, Transmitter
from witwin.channel_utils.constants import SPEED_OF_LIGHT
from witwin.core import Box, Material, Structure
from witwin.path import Config, InteractionType, ReceiverSpec, Result, TransmitterSpec, solve


pytestmark = pytest.mark.gpu

FREQUENCY = 3.5e9


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


def _channel_wall_scene() -> ChannelScene:
    return ChannelScene(
        structures=[_wall_structure()],
        device="cuda",
        edge_selection_mode="all_edges",
    )


def _legacy_wall_scene() -> LegacyScene:
    return LegacyScene(
        structures=[_wall_structure()],
        device="cuda",
        edge_selection_mode="all_edges",
    )


def _as_numpy(value, *, dtype=np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def test_package_root_exports_solver_api():
    assert path_solver.__all__ == [
        "Config",
        "InteractionType",
        "ReceiverSpec",
        "Result",
        "TransmitterSpec",
        "solve",
    ]
    assert path_solver.Config is Config
    assert path_solver.ReceiverSpec is ReceiverSpec
    assert path_solver.Result is Result
    assert path_solver.TransmitterSpec is TransmitterSpec
    assert path_solver.InteractionType is InteractionType
    assert path_solver.solve is solve


def test_path_config_rejects_removed_physical_setup_fields():
    for removed in (
        "reflection_coef",
        "reflection_relative_permittivity",
        "reflection_conductivity",
        "reflection_material",
        "diffraction_material",
        "use_scene_materials_for_reflection",
        "use_scene_materials_for_diffraction",
        "tx_polarization",
        "rx_polarization",
    ):
        with pytest.raises(TypeError, match=removed):
            Config(**{removed: 1.0})


def test_path_solve_uses_scene_transmitter_and_receiver_endpoints():
    scene = ChannelScene(
        structures=[],
        transmitters=[
            Transmitter(
                name="tx",
                position=(0.0, 0.0, 1.0),
                polarization=(0.0, 1.0, 0.0),
            )
        ],
        receivers=[
            Receiver(
                name="rx",
                position=(1.0, 0.0, 1.0),
                polarization=(0.0, 0.0, 1.0),
            )
        ],
        device="cuda",
    )
    result = solve(
        scene=scene,
        frequency=FREQUENCY,
        transmitter="tx",
        receiver="rx",
        config=Config(reflection_max_bounces=0, max_diffractions=0),
    )

    assert result.num_tx == 1
    assert result.num_rx == 1
    assert result.metadata["transmitter_sampling"]["name"] == "tx"
    assert result.metadata["receiver_sampling"]["name"] == "rx"
    assert result.metadata["polarization_transport"]["tx_polarization"] == (0.0, 1.0, 0.0)
    assert result.metadata["polarization_transport"]["rx_polarization"] == (0.0, 0.0, 1.0)


def test_path_package_does_not_import_legacy_channel_package():
    package_root = Path(__file__).resolve().parents[2] / "witwin" / "path"
    forbidden: list[str] = []
    for module_path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "witwin.channel" or alias.name.startswith("witwin.channel."):
                        forbidden.append(f"{module_path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level >= 2:
                    forbidden.append(f"{module_path.name}: relative import level {node.level}")
                if module == "witwin.channel" or module.startswith("witwin.channel."):
                    forbidden.append(f"{module_path.name}: from {module}")

    assert forbidden == []


def test_los_solver_shapes_and_signal_helpers():
    scene = ChannelScene(structures=[], device="cuda")
    result = solve(
        scene=scene,
        frequency=FREQUENCY,
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        receivers=ReceiverSpec(
            [
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ]
        ),
        config=Config(reflection_max_bounces=0, max_diffractions=0),
    )

    assert result.num_tx == 1
    assert result.path_shape == (2, 1, 1)
    assert result.depth_shape == (2, 1, 1, 1)
    assert _as_numpy(result.valid, dtype=np.bool_).all()
    np.testing.assert_array_equal(_as_numpy(result.num_paths, dtype=np.int32), [[1], [1]])
    np.testing.assert_allclose(
        _as_numpy(result.tau)[:, 0, 0],
        np.array([1.0, 2.0], dtype=np.float32) / SPEED_OF_LIGHT,
        rtol=1.0e-6,
        atol=1.0e-10,
    )
    np.testing.assert_array_equal(
        _as_numpy(result.types, dtype=np.int32),
        np.zeros((2, 1, 1, 1), dtype=np.int32),
    )

    coeff, delay = result.cir()
    assert coeff.shape == (2, 1, 1)
    assert delay.shape == (2, 1, 1)
    assert result.coeff_tensor().shape == (2, 1, 1)
    assert result.cfr(torch.tensor([FREQUENCY, FREQUENCY + 1.0e6])).shape == (2, 1, 2)
    assert result.taps(20.0e6, 8).shape == (2, 1, 8)


def test_los_solver_accepts_multi_transmitter_gpu_batch():
    scene = ChannelScene(structures=[], device="cuda")
    tx_positions = TransmitterSpec(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        name="tx",
    )
    receivers = ReceiverSpec(
        [
            [0.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
        ],
        name="rx",
    )
    result = solve(
        scene=scene,
        frequency=FREQUENCY,
        tx_pos=tx_positions,
        receivers=receivers,
        config=Config(reflection_max_bounces=0, max_diffractions=0),
    )

    assert result.num_tx == 2
    assert result.num_rx == 2
    assert result.path_shape == (2, 2, 1)
    assert result.depth_shape == (2, 2, 1, 1)
    assert _as_numpy(result.valid, dtype=np.bool_).all()
    np.testing.assert_array_equal(
        _as_numpy(result.num_paths, dtype=np.int32),
        np.ones((2, 2), dtype=np.int32),
    )
    expected_distances = np.array(
        [
            [1.0, np.sqrt(2.0)],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        _as_numpy(result.tau)[..., 0],
        expected_distances / SPEED_OF_LIGHT,
        rtol=1.0e-6,
        atol=1.0e-10,
    )
    coeff, delay = result.cir()
    assert coeff.shape == (2, 2, 1)
    assert delay.shape == (2, 2, 1)
    assert result.coeff_tensor().shape == (2, 2, 1)
    assert result.cfr(torch.tensor([FREQUENCY, FREQUENCY + 1.0e6])).shape == (2, 2, 2)
    assert result.taps(20.0e6, 8).shape == (2, 2, 8)


def test_multi_transmitter_los_matches_sionna_path_solver():
    try:
        import_result = load_sionna_rt(prefer_local=True)
    except Exception as exc:
        pytest.skip(f"Sionna RT is unavailable: {exc}")

    rt = import_result.rt
    scene = ChannelScene(structures=[], device="cuda")
    tx_values = [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ]
    rx_values = [
        [0.0, 1.0, 1.0],
        [2.0, 0.0, 1.0],
    ]
    tx_positions = TransmitterSpec(
        tx_values,
        name="tx",
    )
    receivers = ReceiverSpec(
        rx_values,
        name="rx",
    )
    result = solve(
        scene=scene,
        frequency=FREQUENCY,
        tx_pos=tx_positions,
        receivers=receivers,
        config=Config(reflection_max_bounces=0, max_diffractions=0),
    )

    sionna_scene = rt.Scene()
    sionna_scene.frequency = FREQUENCY
    iso_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    sionna_scene.tx_array = iso_array
    sionna_scene.rx_array = iso_array
    for tx_index, tx_pos in enumerate(tx_values):
        sionna_scene.add(rt.Transmitter(name=f"tx-{tx_index}", position=tx_pos))
    for rx_index, rx_pos in enumerate(rx_values):
        sionna_scene.add(rt.Receiver(name=f"rx-{rx_index}", position=rx_pos))

    sionna_paths = rt.PathSolver()(
        scene=sionna_scene,
        max_depth=0,
        synthetic_array=True,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        samples_per_src=1,
        seed=7,
    )

    np.testing.assert_array_equal(
        _as_numpy(result.valid, dtype=np.bool_),
        np.asarray(sionna_paths.valid, dtype=np.bool_),
    )
    np.testing.assert_allclose(
        _as_numpy(result.tau),
        np.asarray(sionna_paths.tau, dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-10,
    )


def test_reflection_solver_geometry_filtering_and_metadata():
    result = solve(
        scene=_channel_wall_scene(),
        frequency=FREQUENCY,
        tx_pos=wt.Point3f(-2.0, -1.0, 1.5),
        receivers=ReceiverSpec([[-2.0, 1.0, 1.5]], name="wall_paths"),
        config=Config(
            reflection_n_rays=256,
            reflection_max_bounces=1,
            max_diffractions=0,
            max_num_paths=4,
            return_geometry=True,
        ),
        return_timing=True,
    )

    reflection_only = result.filter_by_type(InteractionType.REFLECTION)
    assert int(_as_numpy(result.num_paths, dtype=np.int32)[0, 0]) >= 2
    assert int(_as_numpy(reflection_only.num_paths, dtype=np.int32)[0, 0]) >= 1
    assert result.vertices is not None
    assert result.normals is not None
    assert result.objects is not None
    assert _as_numpy(result.vertices).shape == (1, 1, 4, 1, 3)
    assert (_as_numpy(result.objects, dtype=np.int32) >= -1).all()
    assert result.metadata["receiver_sampling"]["name"] == "wall_paths"
    assert result.metadata["path_counts"]["reflection"] >= 1
    assert "timing" in result.metadata


def test_standalone_path_solver_matches_legacy_path_monitor_smoke():
    tx_pos = wt.Point3f(-2.0, -1.0, 1.5)
    receivers = wt.Point3f(
        wt.Float([-2.0]),
        wt.Float([1.0]),
        wt.Float([1.5]),
    )
    modern = solve(
        scene=_channel_wall_scene(),
        frequency=FREQUENCY,
        tx_pos=tx_pos,
        receivers=ReceiverSpec(receivers, name="path"),
        config=Config(
            reflection_n_rays=256,
            reflection_max_bounces=1,
            max_diffractions=0,
            max_num_paths=4,
        ),
    )
    legacy = Tracer(
        frequency=FREQUENCY,
        scene=_legacy_wall_scene(),
        reflection_n_rays=256,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        max_diffractions=0,
    ).trace(
        tx_pos,
        monitor=PathMonitor(
            "path",
            positions=receivers,
            ray_mode="3d",
            max_num_paths=4,
            max_diffractions=0,
        ),
        verbose=False,
    )

    modern_valid = _as_numpy(modern.valid, dtype=np.bool_)
    legacy_valid = _as_numpy(legacy.valid, dtype=np.bool_)
    np.testing.assert_array_equal(_as_numpy(modern.num_paths, dtype=np.int32)[:, 0], _as_numpy(legacy.num_paths, dtype=np.int32))
    np.testing.assert_array_equal(modern_valid[:, 0], legacy_valid)
    np.testing.assert_array_equal(
        _as_numpy(modern.types, dtype=np.int32)[:, 0],
        _as_numpy(legacy.types, dtype=np.int32),
    )
    np.testing.assert_allclose(
        _as_numpy(modern.tau)[:, 0][modern_valid[:, 0]],
        _as_numpy(legacy.tau)[legacy_valid],
        rtol=1.0e-5,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        modern.coeff_tensor().detach().cpu().numpy()[:, 0][modern_valid[:, 0]],
        legacy.coeff_tensor().detach().cpu().numpy()[legacy_valid],
        rtol=1.0e-4,
        atol=1.0e-7,
    )
