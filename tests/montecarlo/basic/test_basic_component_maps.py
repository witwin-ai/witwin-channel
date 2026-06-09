import pytest
import torch

from tests.support.scenes import single_wall_reflection_scene, wedge_diffraction_scene
from witwin.channel_native import ReceiverGrid, Transmitter
from witwin.channel_native.core.edge_policy import EdgePolicy
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.core.scene import _RAYD_EDGE_INFO_PLANE_TOL, _selected_diffraction_edges
from witwin.channel_native.montecarlo.basic import Config, solve
import witwin.channel_native.montecarlo.basic.backend as basic_backend
import witwin.channel_native.montecarlo.basic.raydn_components as raydn_components


def _grid_at_x(x: float) -> ReceiverGrid:
    return ReceiverGrid(
        origin=torch.tensor([x, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def _non_square_grid_at_x(x: float) -> ReceiverGrid:
    return ReceiverGrid(
        origin=torch.tensor([x, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(2, 3),
        spacing=(1.0, 0.5),
    )


def _safe_normalize_vectors(vectors: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    return torch.nn.functional.normalize(vectors, dim=1, eps=eps)


def _unsigned_angle(a: torch.Tensor, b: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    cross = torch.cross(a, b, dim=1)
    signed_norm = torch.sign((cross * axis).sum(dim=1)) * torch.linalg.vector_norm(cross, dim=1)
    angle = torch.atan2(signed_norm, (a * b).sum(dim=1))
    return torch.where(angle < 0.0, angle + 2.0 * torch.pi, angle)


def _torch_diffraction_edge_geometry(records) -> tuple[torch.Tensor, ...]:
    selected = _selected_diffraction_edges(
        vertices=records.vertices,
        faces=records.faces,
        face_normals=records.face_normals,
        edge_v0=records.edge_v0,
        edge_v1=records.edge_v1,
        face0=records.face0,
        face1=records.face1,
        edge_policy=EdgePolicy(edge_selection_mode="all_edges", boundary_edge_policy="half_plane"),
        plane_tol=_RAYD_EDGE_INFO_PLANE_TOL,
    )
    edge_v0 = records.edge_v0.to(dtype=torch.long)
    edge_v1 = records.edge_v1.to(dtype=torch.long)
    vertices = records.vertices
    face0 = records.face0.to(dtype=torch.long)
    face1 = records.face1.to(dtype=torch.long)
    start = vertices[edge_v0]
    end = vertices[edge_v1]
    vectors = vertices[edge_v1] - vertices[edge_v0]
    lengths = torch.linalg.vector_norm(vectors, dim=1).clamp_min(1.0e-12)
    edge_dir = vectors / lengths[:, None]
    safe0 = face0.clamp_min(0)
    safe1 = face1.clamp_min(0)
    n0_cand = _safe_normalize_vectors(records.face_normals[safe0])
    n1_cand = _safe_normalize_vectors(records.face_normals[safe1])

    to1 = _safe_normalize_vectors(torch.cross(n0_cand, edge_dir, dim=1))
    tn1 = _safe_normalize_vectors(torch.cross(n1_cand, edge_dir, dim=1))
    to2 = _safe_normalize_vectors(torch.cross(n1_cand, edge_dir, dim=1))
    tn2 = _safe_normalize_vectors(torch.cross(n0_cand, edge_dir, dim=1))
    choose_first = _unsigned_angle(to1, tn1, edge_dir) < _unsigned_angle(to2, tn2, edge_dir)
    ordered_n0 = torch.where(choose_first[:, None], n0_cand, n1_cand)
    ordered_n1 = torch.where(choose_first[:, None], n1_cand, n0_cand)

    interior = (face0 >= 0) & (face1 >= 0)
    boundary = face1 < 0
    n0 = torch.where(interior[:, None], ordered_n0, n0_cand)
    n1 = torch.where(interior[:, None], ordered_n1, n1_cand)
    n1 = torch.where(boundary[:, None], -n0_cand, n1)
    normal_dot = (n0 * n1).sum(dim=1)
    interior_angle = torch.acos(torch.clamp(-normal_dot, -1.0, 1.0))
    exterior_angle = torch.where(
        interior,
        2.0 * torch.pi - interior_angle,
        torch.full_like(interior_angle, 2.0 * torch.pi),
    )
    return (
        selected,
        ((start + end) * 0.5).contiguous(),
        edge_dir.contiguous(),
        lengths.contiguous(),
        (-0.5 * lengths).contiguous(),
        (0.5 * lengths).contiguous(),
        n0.contiguous(),
        n1.contiguous(),
        records.face0.contiguous(),
        records.face1.contiguous(),
        exterior_angle.contiguous(),
    )


def test_basic_solver_returns_los_component_map_for_receiver_grid():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic component maps")

    scene = single_wall_reflection_scene()
    scene = type(scene)(
        structures=[],
        transmitters=scene.transmitters,
        receivers=[_grid_at_x(5.0)],
        frequency=scene.frequency,
    )

    result = solve(scene, Config(samples=128, seed=3, components={"los"}))

    assert result.component_maps is not None
    assert result.component_maps["los"].shape == (1, 4, 4)
    torch.testing.assert_close(
        result.component_maps["los"].reshape(1, -1),
        result.path_gain,
        rtol=0.0,
        atol=0.0,
    )


def test_basic_solver_los_component_map_uses_public_yx_grid_layout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic component maps")

    scene = single_wall_reflection_scene()
    grid = _non_square_grid_at_x(5.0)
    scene = type(scene)(
        structures=[],
        transmitters=scene.transmitters,
        receivers=[grid],
        frequency=scene.frequency,
    )

    result = solve(scene, Config(samples=128, seed=3, components={"los"}))

    assert result.component_maps is not None
    assert result.component_maps["los"].shape == (1, 3, 2)

    tx = scene.transmitters[0].position.to(device=result.path_gain.device)
    points = grid.points().reshape(*grid.shape, 3).transpose(0, 1).reshape(-1, 3).to(device=result.path_gain.device)
    distance = torch.linalg.vector_norm(tx[None, :] - points, dim=1).clamp_min(1.0e-6)
    wavelength = 299_792_458.0 / scene.frequency
    expected = scene.transmitters[0].power_w / ((4.0 * torch.pi * distance / wavelength) ** 2)

    torch.testing.assert_close(
        result.component_maps["los"].reshape(1, -1),
        expected.reshape(1, -1),
        rtol=1e-6,
        atol=1e-12,
    )
    torch.testing.assert_close(
        result.path_gain,
        expected.reshape(1, -1),
        rtol=1e-6,
        atol=1e-12,
    )


def test_basic_solver_reuses_los_export_for_single_receiver_grid(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic component maps")

    scene = single_wall_reflection_scene()
    scene = type(scene)(
        structures=[],
        transmitters=scene.transmitters,
        receivers=[_grid_at_x(5.0)],
        frequency=scene.frequency,
    )
    call_count = 0
    original = basic_backend.path_los_export

    def counted(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(basic_backend, "path_los_export", counted)

    solve(scene, Config(samples=128, seed=3, components={"los"}))

    assert call_count == 1


def test_basic_solver_returns_native_reflection_component_map_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic component maps")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native reflection is not built")

    scene = single_wall_reflection_scene().add(_grid_at_x(5.0))
    result = solve(scene, Config(samples=2048, seed=5, components={"reflection"}))

    assert result.component_maps is not None
    assert result.component_maps["reflection"].is_cuda
    assert result.component_maps["reflection"].shape == (1, 4, 4)
    assert result.metadata["components"]["reflection"] == "enabled"
    torch.testing.assert_close(
        result.component_power["reflection"],
        result.component_maps["reflection"].sum(),
        rtol=1e-5,
        atol=1e-8,
    )
    torch.testing.assert_close(
        result.path_gain,
        result.component_maps["reflection"].reshape(1, -1),
        rtol=1e-5,
        atol=1e-8,
    )


def test_basic_solver_returns_native_diffraction_component_map_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic component maps")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    scene = wedge_diffraction_scene().add(_grid_at_x(3.0))
    result = solve(scene, Config(samples=512, seed=7, components={"diffraction"}))

    assert result.component_maps is not None
    assert result.component_maps["diffraction"].is_cuda
    assert result.component_maps["diffraction"].shape == (1, 4, 4)
    assert result.metadata["components"]["diffraction"] == "enabled"
    torch.testing.assert_close(
        result.component_power["diffraction"],
        result.component_maps["diffraction"].sum(),
        rtol=1e-5,
        atol=1e-8,
    )
    torch.testing.assert_close(
        result.path_gain,
        result.component_maps["diffraction"].reshape(1, -1),
        rtol=1e-5,
        atol=1e-8,
    )


def test_basic_solver_reports_native_fused_schedule_for_raydn_components():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic component maps")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native components are not built")

    scene = wedge_diffraction_scene().add(_grid_at_x(3.0))
    result = solve(scene, Config(samples=512, seed=7, components={"reflection", "diffraction"}))

    kernel = result.metadata["kernel"]
    assert kernel["raydn_native"] is True
    assert kernel["scheduling_strategy"] == "native_fused"
    assert kernel["fused_stages"] >= 1


def test_diffraction_edge_geometry_native_matches_torch_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    scene = wedge_diffraction_scene()
    records = scene.compile().raydn.edge_records()

    reference = _torch_diffraction_edge_geometry(records)
    native = raydn_components._diffraction_edge_geometry(records)

    torch.testing.assert_close(native[0], reference[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(native[8], reference[8], rtol=0.0, atol=0.0)
    torch.testing.assert_close(native[9], reference[9], rtol=0.0, atol=0.0)
    for idx in (1, 2, 3, 4, 5, 6, 7, 10):
        torch.testing.assert_close(native[idx], reference[idx], rtol=1e-6, atol=1e-6)


def test_diffraction_edge_candidates_are_cached_across_transmitters(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    base = wedge_diffraction_scene()
    scene = type(base)(
        structures=base.structures,
        transmitters=[
            base.transmitters[0],
            Transmitter(position=base.transmitters[0].position + torch.tensor([0.0, -0.25, 0.0])),
        ],
        receivers=[_grid_at_x(3.0)],
        frequency=base.frequency,
    )
    call_count = 0
    original = raydn_components._native_surface_group_edge_candidates

    def counted(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(raydn_components, "_native_surface_group_edge_candidates", counted)

    solve(scene, Config(samples=512, seed=7, components={"reflection", "diffraction"}))

    assert call_count == 1


def test_diffraction_edge_candidates_are_cached_across_solves(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    scene = wedge_diffraction_scene().add(_grid_at_x(3.0))
    call_count = 0
    original = raydn_components._native_surface_group_edge_candidates

    def counted(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(raydn_components, "_native_surface_group_edge_candidates", counted)

    solve(scene, Config(samples=512, seed=7, components={"reflection", "diffraction"}))
    solve(scene, Config(samples=512, seed=7, components={"reflection", "diffraction"}))

    assert call_count == 1
