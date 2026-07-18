import pytest
import torch

from tests.support.scenes import wedge_diffraction_scene
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.deterministic import Config, solve
from witwin.channel_native.path import Config as PathConfig
from witwin.channel_native.path import solve as solve_paths


def test_single_wedge_diffraction_matches_path_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    scene = wedge_diffraction_scene()
    result = solve(scene, Config(components={"diffraction"}, coherent=False, export_paths=True, return_field=False))
    reference = solve_paths(scene, PathConfig(components={"diffraction"}))

    assert result.paths is not None
    assert result.paths.valid.numel() == reference.valid.numel()
    torch.testing.assert_close(result.paths.edge_id, reference.primitive_id[reference.valid, 0])
    # Real UTD paths (audit DF-1): one merged record for the shared wedge
    # edge (audit D-6), Keller stationary-point delays, and K-P amplitudes.
    # F2 admitted the weak horizontal edge 4 after the 5 cm gate removal, but
    # the finite-edge truncation refinements (F5e monotone even part + F5f
    # boundary-distance odd blend) legitimately push its field back below the
    # 1e-30 existence floor, so the record set returns to edge ids 0, 1, 2, 5;
    # the deterministic and path solvers still agree edge-for-edge.
    torch.testing.assert_close(
        result.paths.edge_id,
        torch.tensor([0, 1, 2, 5], device=result.paths.edge_id.device, dtype=torch.int32),
    )
    expected_length = torch.tensor(
        [
            3.650281668,
            4.744879723,
            3.768759727,
            4.046976566,
        ],
        device=result.paths.path_length_m.device,
        dtype=torch.float32,
    )
    # F1/R5 + F5e/F5f: the diffracted field carries the z-hat short-dipole sin
    # pattern (projected onto the rx polarization) and the monotone finite-edge
    # truncation with the boundary-distance odd blend, which reshape the
    # per-edge gains; edges 0/1 sit at near-null while edges 2/5 stay dominant.
    expected_gain = torch.tensor(
        [
            2.980571e-18,
            2.680186e-11,
            1.471983e-08,
            6.686395e-09,
        ],
        device=result.paths.path_gain.device,
        dtype=torch.float32,
    )
    torch.testing.assert_close(result.paths.path_length_m, expected_length, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(result.paths.path_gain, expected_gain, rtol=5.0e-3, atol=1.0e-8)
    torch.testing.assert_close(
        result.path_gain.reshape(-1).sum(),
        reference.a[reference.valid].abs().square().sum(),
        rtol=5.0e-3,
        atol=1.0e-8,
    )


def test_vertical_only_edge_policy_filters_horizontal_edges():
    """The scene's edge policy must govern path generation (audit DF-4)."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    from witwin.channel_native.core.edge_policy import EdgePolicy

    scene = wedge_diffraction_scene()
    scene.metadata["sionna_import_edge_policy"] = EdgePolicy(edge_selection_mode="vertical_only")

    result = solve(scene, Config(components={"diffraction"}, coherent=False, export_paths=True, return_field=False))

    assert result.paths is not None
    # The horizontal outline edges (|dz|/length = 0) must not produce paths;
    # the vertical shared wedge edge and the slanted outline edges
    # (|dz|/length ~ 0.83 > 0.7) survive.
    assert int(result.paths.valid.numel()) == 3
    baseline = solve(
        wedge_diffraction_scene(),
        Config(components={"diffraction"}, coherent=False, export_paths=True, return_field=False),
    )
    # F5e/F5f (utd-continuity-fix-design): the finite-edge truncation pushes
    # the weak horizontal edge that F2 had admitted back below the existence
    # floor, so the unfiltered baseline emits 4 paths; vertical_only keeps 3.
    assert int(baseline.paths.valid.numel()) == 4


def test_diffraction_path_field_export_uses_native_complex_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    scene = wedge_diffraction_scene()
    result = solve(scene, Config(components={"diffraction"}, coherent=True, export_paths=True))
    reference = solve_paths(scene, PathConfig(components={"diffraction"}))

    assert result.paths is not None
    path_field = torch.complex(result.paths.field_real, result.paths.field_imag)
    expected_phase = torch.remainder(-torch.angle(path_field), 2.0 * torch.pi)
    expected_field = reference.a[..., 0][reference.valid]
    assert torch.all(result.paths.valid)
    assert torch.count_nonzero(path_field.abs() > 0.0).item() == result.paths.valid.numel()
    torch.testing.assert_close(path_field, expected_field, rtol=5.0e-4, atol=1.0e-7)
    torch.testing.assert_close(result.paths.path_gain, path_field.abs().square(), rtol=2.0e-4, atol=1.0e-10)
    torch.testing.assert_close(result.paths.phase_rad, expected_phase, rtol=2.0e-4, atol=1.0e-6)
    torch.testing.assert_close(result.path_gain, result.field.abs().square(), rtol=2.0e-4, atol=1.0e-10)
