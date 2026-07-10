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
    torch.testing.assert_close(result.paths.edge_id, reference.edge_id)
    # The shared wedge edge is one merged record (audit D-6): the historical
    # expectation carried a duplicate half-plane entry (old ids 0 and 3).
    torch.testing.assert_close(
        result.paths.edge_id,
        torch.tensor([0, 1, 2, 4, 5], device=result.paths.edge_id.device, dtype=torch.int32),
    )
    expected_length = torch.tensor(
        [
            3.6502816677093506,
            5.004337787628174,
            3.8284270763397217,
            5.3027753829956055,
            4.162277698516846,
        ],
        device=result.paths.path_length_m.device,
        dtype=torch.float32,
    )
    expected_gain = torch.tensor(
        [
            1.42285853144e-05,
            3.796661076194141e-06,
            2.850105011020787e-05,
            3.1767988275532844e-06,
            2.2800833903602324e-05,
        ],
        device=result.paths.path_gain.device,
        dtype=torch.float32,
    )
    torch.testing.assert_close(result.paths.path_length_m, expected_length, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(result.paths.path_gain, expected_gain, rtol=5.0e-3, atol=1.0e-8)
    torch.testing.assert_close(result.path_gain.reshape(-1).sum(), reference.path_gain.sum(), rtol=5.0e-3, atol=1.0e-8)


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
    # The two horizontal outline edges (|dz|/length = 0) must not produce
    # paths; the vertical shared wedge edge and the two slanted outline edges
    # (|dz|/length ~ 0.83 > 0.7) survive.
    assert int(result.paths.valid.numel()) == 3
    baseline = solve(
        wedge_diffraction_scene(),
        Config(components={"diffraction"}, coherent=False, export_paths=True, return_field=False),
    )
    assert int(baseline.paths.valid.numel()) == 5


def test_diffraction_path_field_export_uses_native_complex_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    result = solve(wedge_diffraction_scene(), Config(components={"diffraction"}, coherent=True, export_paths=True))

    assert result.paths is not None
    path_field = torch.complex(result.paths.field_real, result.paths.field_imag)
    expected_phase = torch.remainder(-torch.angle(path_field), 2.0 * torch.pi)
    # Merged wedge record (audit D-6): one entry for the shared edge with the
    # 3*pi/2 exterior-angle weight instead of two full half-plane duplicates.
    expected_field_real = torch.tensor(
        [
            -3.71350278147e-03,
            0.00171906349715,
            -0.00198933575302,
            0.00163817277644,
            -0.00276799290441,
        ],
        device=result.paths.field_real.device,
        dtype=torch.float32,
    )
    expected_field_imag = torch.tensor(
        [
            6.62179896608e-04,
            -0.000917323108297,
            -0.00495414901525,
            -0.000702273973729,
            0.00389089318924,
        ],
        device=result.paths.field_imag.device,
        dtype=torch.float32,
    )
    assert torch.all(result.paths.valid)
    assert torch.count_nonzero(path_field.abs() > 0.0).item() == result.paths.valid.numel()
    torch.testing.assert_close(result.paths.field_real, expected_field_real, rtol=5.0e-4, atol=1.0e-7)
    torch.testing.assert_close(result.paths.field_imag, expected_field_imag, rtol=5.0e-4, atol=1.0e-7)
    torch.testing.assert_close(result.paths.path_gain, path_field.abs().square(), rtol=2.0e-4, atol=1.0e-10)
    torch.testing.assert_close(result.paths.phase_rad, expected_phase, rtol=2.0e-4, atol=1.0e-6)
    torch.testing.assert_close(result.path_gain, result.field.abs().square(), rtol=2.0e-4, atol=1.0e-10)
