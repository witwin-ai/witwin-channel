from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from witwin.channel import (
    InteractionType,
    Material,
    Mesh,
    PathMonitor,
    Scene,
    Structure,
    Tracer,
    load_sionna_rt,
    scene_to_sionna_scene,
)
from witwin.channel.validation import build_single_wedge_case


pytestmark = [pytest.mark.gpu, pytest.mark.validation]


_FREQUENCY_HZ = 28e9
_WITWIN_TYPE_LABELS = {
    int(InteractionType.NONE): "los",
    int(InteractionType.REFLECTION): "reflection",
    int(InteractionType.DIFFRACTION): "diffraction",
    int(InteractionType.TRANSMISSION): "transmission",
    int(InteractionType.SCATTERING): "scattering",
}
_SIONNA_TYPE_LABELS = {
    0: "los",
    1: "reflection",
    2: "scattering",
    4: "transmission",
    8: "diffraction",
}


@dataclass(frozen=True)
class _NormalizedPath:
    interactions: tuple[str, ...]
    vertices: tuple[tuple[float, float, float], ...]
    tau_seconds: float


@pytest.fixture(scope="module")
def sionna_rt():
    try:
        return load_sionna_rt(prefer_local=True).rt
    except Exception as exc:  # pragma: no cover - exercised only when Sionna is unavailable
        pytest.skip(f"Local Sionna RT reference is unavailable: {exc}")


def _iso_array(rt):
    return rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )


def _interaction_labels(type_codes: np.ndarray, mapping: dict[int, str]) -> tuple[str, ...]:
    labels = tuple(mapping[int(code)] for code in type_codes if int(code) != 0)
    return ("los",) if len(labels) == 0 else labels


def _rounded_vertices(vertices: np.ndarray, depth: int) -> tuple[tuple[float, float, float], ...]:
    if depth <= 0:
        return ()
    rounded = np.round(vertices[:depth], decimals=4)
    return tuple(tuple(float(value) for value in point) for point in rounded.tolist())


def _normalize_witwin_pair(paths, rx_index: int) -> list[_NormalizedPath]:
    valid = np.asarray(paths.valid, dtype=np.bool_)[rx_index]
    tau = np.asarray(paths.tau, dtype=np.float32)[rx_index]
    type_rows = np.asarray(paths.types, dtype=np.int32)[rx_index]
    vertex_rows = None if paths.vertices is None else np.asarray(paths.vertices, dtype=np.float32)[rx_index]

    normalized = []
    for path_index in range(valid.shape[0]):
        if not bool(valid[path_index]):
            continue
        interactions = _interaction_labels(type_rows[path_index], _WITWIN_TYPE_LABELS)
        depth = 0 if interactions == ("los",) else len(interactions)
        vertices = () if vertex_rows is None else _rounded_vertices(vertex_rows[path_index], depth)
        normalized.append(
            _NormalizedPath(
                interactions=interactions,
                vertices=vertices,
                tau_seconds=float(tau[path_index]),
            )
        )
    return sorted(normalized, key=lambda item: (item.interactions, item.vertices, item.tau_seconds))


def _normalize_sionna_pair(paths, *, rx_index: int, tx_index: int) -> list[_NormalizedPath]:
    valid = np.asarray(paths.valid, dtype=np.bool_)[rx_index, tx_index]
    tau = np.asarray(paths.tau, dtype=np.float32)[rx_index, tx_index]
    interaction_rows = np.asarray(paths.interactions, dtype=np.int32)[:, rx_index, tx_index, :]
    vertex_rows = np.asarray(paths.vertices, dtype=np.float32)[:, rx_index, tx_index, :, :]

    normalized = []
    for path_index in range(valid.shape[0]):
        if not bool(valid[path_index]):
            continue
        interactions = _interaction_labels(interaction_rows[:, path_index], _SIONNA_TYPE_LABELS)
        depth = 0 if interactions == ("los",) else len(interactions)
        normalized.append(
            _NormalizedPath(
                interactions=interactions,
                vertices=_rounded_vertices(vertex_rows[:, path_index, :], depth),
                tau_seconds=float(tau[path_index]),
            )
        )
    return sorted(normalized, key=lambda item: (item.interactions, item.vertices, item.tau_seconds))


def _assert_pairwise_path_parity(witwin_results, sionna_paths, rx_positions: torch.Tensor) -> None:
    for tx_index, witwin_paths in enumerate(witwin_results):
        for rx_index in range(int(rx_positions.shape[0])):
            witwin_pair = _normalize_witwin_pair(witwin_paths, rx_index)
            sionna_pair = _normalize_sionna_pair(sionna_paths, rx_index=rx_index, tx_index=tx_index)

            assert [item.interactions for item in witwin_pair] == [item.interactions for item in sionna_pair]
            assert len(witwin_pair) == len(sionna_pair)
            for witwin_item, sionna_item in zip(witwin_pair, sionna_pair):
                assert len(witwin_item.vertices) == len(sionna_item.vertices)
                if len(witwin_item.vertices) > 0:
                    np.testing.assert_allclose(
                        np.asarray(witwin_item.vertices, dtype=np.float32),
                        np.asarray(sionna_item.vertices, dtype=np.float32),
                        rtol=0.0,
                        atol=1e-3,
                    )
            np.testing.assert_allclose(
                [item.tau_seconds for item in witwin_pair],
                [item.tau_seconds for item in sionna_pair],
                rtol=1e-5,
                atol=1e-10,
            )


def _run_witwin_paths(
    *,
    scene: Scene,
    tx_positions: tuple[torch.Tensor, ...],
    rx_positions: torch.Tensor,
    reflection_n_rays: int,
    reflection_max_bounces: int,
    monitor_max_diffractions: int,
    return_geometry: bool,
):
    tracer = Tracer(
        frequency=_FREQUENCY_HZ,
        scene=scene,
        reflection_n_rays=reflection_n_rays,
        reflection_max_bounces=reflection_max_bounces,
        max_diffractions=0,
    )
    monitor = PathMonitor(
        "rx",
        positions=rx_positions,
        max_diffractions=monitor_max_diffractions,
        return_geometry=return_geometry,
    )
    return tracer.trace_many(tx_positions, monitor=monitor, verbose=False)


def _run_sionna_paths(
    *,
    rt,
    scene: Scene,
    tx_positions: tuple[torch.Tensor, ...],
    rx_positions: torch.Tensor,
    los: bool,
    specular_reflection: bool,
    diffraction: bool,
    samples_per_src: int = 10000,
):
    converted = scene_to_sionna_scene(scene, prefer_local=True)
    sionna_scene = converted.scene
    sionna_scene.frequency = _FREQUENCY_HZ
    array = _iso_array(rt)
    sionna_scene.tx_array = array
    sionna_scene.rx_array = array

    for tx_index, tx_pos in enumerate(tx_positions):
        sionna_scene.add(
            rt.Transmitter(
                name=f"tx-{tx_index}",
                position=tx_pos.detach().cpu().tolist(),
            )
        )
    for rx_index, rx_pos in enumerate(rx_positions):
        sionna_scene.add(
            rt.Receiver(
                name=f"rx-{rx_index}",
                position=rx_pos.detach().cpu().tolist(),
            )
        )

    solver = rt.PathSolver()
    return solver(
        scene=sionna_scene,
        max_depth=1,
        synthetic_array=True,
        los=los,
        specular_reflection=specular_reflection,
        diffuse_reflection=False,
        refraction=False,
        diffraction=diffraction,
        edge_diffraction=False,
        samples_per_src=samples_per_src,
        seed=7,
    )


def _los_case():
    scene = Scene(structures=[], device="cuda")
    tx_positions = (
        torch.tensor((0.0, 0.0, 1.5), dtype=torch.float32),
        torch.tensor((1.0, 0.0, 1.5), dtype=torch.float32),
    )
    rx_positions = torch.tensor(
        [
            (0.0, 3.0, 1.5),
            (1.0, 3.0, 1.5),
        ],
        dtype=torch.float32,
    )
    return scene, tx_positions, rx_positions


def _full_3d_los_case():
    scene = Scene(structures=[], device="cuda")
    tx_positions = (
        torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32),
        torch.tensor((1.0, -0.5, 2.2), dtype=torch.float32),
        torch.tensor((2.5, 0.5, 0.7), dtype=torch.float32),
    )
    rx_positions = torch.tensor(
        [
            (0.0, 3.0, 1.8),
            (1.0, 3.5, 0.9),
            (2.0, 2.8, 2.5),
        ],
        dtype=torch.float32,
    )
    return scene, tx_positions, rx_positions


def _reflection_case():
    wall = Mesh(
        vertices=[
            [0.0, -4.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 4.0, 3.0],
            [0.0, -4.0, 3.0],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
        position=(0.0, 0.0, 0.0),
        recenter=False,
        device="cpu",
    )
    scene = Scene(
        structures=[
            Structure(
                geometry=wall,
                material=Material(name="reflector", eps_r=4.0, sigma_e=0.1),
                name="wall",
            )
        ],
        device="cuda",
    )
    tx_positions = (
        torch.tensor((-3.0, -5.0, 1.5), dtype=torch.float32),
        torch.tensor((-2.0, -5.0, 1.5), dtype=torch.float32),
    )
    rx_positions = torch.tensor(
        [
            (-3.0, 5.0, 1.5),
            (-2.0, 4.0, 1.5),
        ],
        dtype=torch.float32,
    )
    return scene, tx_positions, rx_positions


def _full_3d_reflection_case():
    wall = Mesh(
        vertices=[
            [0.0, -4.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 4.0, 4.0],
            [0.0, -4.0, 4.0],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
        position=(0.0, 0.0, 0.0),
        recenter=False,
        device="cpu",
    )
    scene = Scene(
        structures=[
            Structure(
                geometry=wall,
                material=Material(name="reflector-3d", eps_r=4.0, sigma_e=0.1),
                name="wall-3d",
            )
        ],
        device="cuda",
    )
    tx_positions = (
        torch.tensor((-3.0, -5.0, 1.4), dtype=torch.float32),
        torch.tensor((-2.0, -5.0, 2.2), dtype=torch.float32),
        torch.tensor((1.5, -4.5, 0.7), dtype=torch.float32),
    )
    rx_positions = torch.tensor(
        [
            (-3.0, 5.0, 0.9),
            (-2.0, 4.0, 1.8),
            (1.5, 3.0, 2.6),
        ],
        dtype=torch.float32,
    )
    return scene, tx_positions, rx_positions


def _diffraction_case():
    case = build_single_wedge_case()
    tx_positions = (
        torch.tensor(case.tx_pos, dtype=torch.float32),
        torch.tensor((1.0, -6.0, case.calculation_height), dtype=torch.float32),
    )
    rx_positions = torch.tensor(
        [
            (0.0, 3.5, case.calculation_height),
            (1.0, 3.5, case.calculation_height),
        ],
        dtype=torch.float32,
    )
    return case.scene, tx_positions, rx_positions


def _mixed_first_order_case():
    case = build_single_wedge_case()
    wall = Mesh(
        vertices=[
            [2.0, -4.0, 0.0],
            [2.0, 4.0, 0.0],
            [2.0, 4.0, 3.0],
            [2.0, -4.0, 3.0],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
        position=(0.0, 0.0, 0.0),
        recenter=False,
        device="cpu",
    )
    scene = Scene(
        structures=[
            *case.scene.structures,
            Structure(
                geometry=wall,
                material=Material(name="reflector-material", eps_r=4.0, sigma_e=0.1),
                name="reflector-wall",
            ),
        ],
        device="cuda",
    )
    tx_positions = (
        torch.tensor(case.tx_pos, dtype=torch.float32),
        torch.tensor((1.0, -6.0, case.calculation_height), dtype=torch.float32),
        torch.tensor((4.0, -3.5, case.calculation_height), dtype=torch.float32),
    )
    rx_positions = torch.tensor(
        [
            (0.0, 3.5, case.calculation_height),
            (1.0, 3.5, case.calculation_height),
            (3.0, 3.0, case.calculation_height),
        ],
        dtype=torch.float32,
    )
    return scene, tx_positions, rx_positions


def test_sionna_path_solver_matches_multi_tx_multi_rx_los_paths(sionna_rt):
    scene, tx_positions, rx_positions = _los_case()

    witwin_results = _run_witwin_paths(
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        monitor_max_diffractions=0,
        return_geometry=False,
    )
    sionna_paths = _run_sionna_paths(
        rt=sionna_rt,
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        los=True,
        specular_reflection=False,
        diffraction=False,
    )

    _assert_pairwise_path_parity(witwin_results, sionna_paths, rx_positions)


def test_sionna_path_solver_matches_complex_scene_multi_tx_multi_rx_simultaneous_paths(sionna_rt):
    scene, tx_positions, rx_positions = _mixed_first_order_case()

    witwin_results = _run_witwin_paths(
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        reflection_n_rays=8192,
        reflection_max_bounces=1,
        monitor_max_diffractions=1,
        return_geometry=True,
    )
    sionna_paths = _run_sionna_paths(
        rt=sionna_rt,
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        los=True,
        specular_reflection=True,
        diffraction=True,
        samples_per_src=50000,
    )

    _assert_pairwise_path_parity(witwin_results, sionna_paths, rx_positions)


def test_sionna_path_solver_matches_full_3d_multi_tx_multi_rx_los_paths(sionna_rt):
    scene, tx_positions, rx_positions = _full_3d_los_case()

    witwin_results = _run_witwin_paths(
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        monitor_max_diffractions=0,
        return_geometry=False,
    )
    sionna_paths = _run_sionna_paths(
        rt=sionna_rt,
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        los=True,
        specular_reflection=False,
        diffraction=False,
    )

    _assert_pairwise_path_parity(witwin_results, sionna_paths, rx_positions)

    theta_t = np.asarray(witwin_results[0].theta_t, dtype=np.float32)
    assert np.max(np.abs(theta_t - (np.pi * 0.5))) > 0.05


def test_sionna_path_solver_matches_multi_tx_multi_rx_first_order_reflection_paths(sionna_rt):
    scene, tx_positions, rx_positions = _reflection_case()

    witwin_results = _run_witwin_paths(
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        monitor_max_diffractions=0,
        return_geometry=True,
    )
    sionna_paths = _run_sionna_paths(
        rt=sionna_rt,
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        los=True,
        specular_reflection=True,
        diffraction=False,
    )

    _assert_pairwise_path_parity(witwin_results, sionna_paths, rx_positions)


def test_sionna_path_solver_matches_full_3d_multi_tx_multi_rx_reflection_paths(sionna_rt):
    scene, tx_positions, rx_positions = _full_3d_reflection_case()

    witwin_results = _run_witwin_paths(
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        reflection_n_rays=8192,
        reflection_max_bounces=1,
        monitor_max_diffractions=0,
        return_geometry=True,
    )
    sionna_paths = _run_sionna_paths(
        rt=sionna_rt,
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        los=True,
        specular_reflection=True,
        diffraction=False,
        samples_per_src=50000,
    )

    _assert_pairwise_path_parity(witwin_results, sionna_paths, rx_positions)

    reflection_vertices = np.asarray(witwin_results[0].vertices, dtype=np.float32)
    reflection_mask = np.asarray(witwin_results[0].valid, dtype=np.bool_) & (
        np.asarray(witwin_results[0].types, dtype=np.int32)[:, :, 0] == InteractionType.REFLECTION
    )
    assert np.max(np.abs(reflection_vertices[reflection_mask][:, 0, 2] - 1.5)) > 0.1


def test_sionna_path_solver_matches_multi_tx_multi_rx_first_order_diffraction_paths(sionna_rt):
    scene, tx_positions, rx_positions = _diffraction_case()

    witwin_results = _run_witwin_paths(
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        monitor_max_diffractions=1,
        return_geometry=True,
    )
    sionna_paths = _run_sionna_paths(
        rt=sionna_rt,
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        los=True,
        specular_reflection=False,
        diffraction=True,
    )

    _assert_pairwise_path_parity(witwin_results, sionna_paths, rx_positions)
