# Copyright Xingyu Chen.
# Tests diffraction single wedge.

import pytest
import torch
from witwin.core import Scene

from tests.support.scenes import wedge_diffraction_scene
from witwin.channel.deployment import build_info
from witwin.channel.deterministic import Config, solve
from witwin.channel.path import Config as PathConfig
from witwin.channel.path import solve as solve_paths


def test_single_wedge_diffraction_matches_path_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diffraction")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    scene = wedge_diffraction_scene()
    result = solve(
        scene,
        Config(
            components={"diffraction"},
            coherent=False,
            export_paths=True,
            return_field=False,
        ),
        reference_frequency_hz=3.0e9,
    )
    reference = solve_paths(
        scene, PathConfig(components={"diffraction"}), reference_frequency_hz=3.0e9
    )

    assert result.paths is not None
    assert result.paths.valid.numel() == reference.valid.numel()
    torch.testing.assert_close(
        result.paths.edge_id, reference.primitive_id[reference.valid, 0]
    )
    # The shared wedge emits one row for each surviving edge. Finite-edge
    # truncation and boundary blending push horizontal edge 4 below the 1e-30
    # floor. The receiver lies behind both faces of the shared edge 0 (their
    # normals point -x and -y), and RayD reports its edge-to-receiver segment
    # as occluded, so edge 0 emits no row. That leaves edge IDs 1, 2, and 5 in
    # both solvers.
    torch.testing.assert_close(
        result.paths.edge_id,
        torch.tensor(
            [1, 2, 5], device=result.paths.edge_id.device, dtype=torch.int32
        ),
    )
    expected_length = torch.tensor(
        [
            4.744879723,
            3.768759727,
            4.046976566,
        ],
        device=result.paths.path_length_m.device,
        dtype=torch.float32,
    )
    # / /: the diffracted field carries the z-hat short-dipole sin
    # pattern (projected onto the rx polarization) and the monotone finite-edge
    # truncation with the boundary-distance odd blend, which reshape the
    # per-edge gains; edge 1 sits at near-null while edges 2/5 stay dominant.
    expected_gain = torch.tensor(
        [
            2.680186e-11,
            1.471983e-08,
            6.686395e-09,
        ],
        device=result.paths.path_gain.device,
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        result.paths.path_length_m, expected_length, rtol=1.0e-5, atol=1.0e-6
    )
    torch.testing.assert_close(
        result.paths.path_gain, expected_gain, rtol=5.0e-3, atol=1.0e-8
    )
    torch.testing.assert_close(
        result.path_gain.reshape(-1).sum(),
        reference.a[reference.valid].abs().square().sum(),
        rtol=5.0e-3,
        atol=1.0e-8,
    )


def test_vertical_only_edge_policy_filters_horizontal_edges():
    """The scene's edge policy must govern path generation."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diffraction")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    from witwin.channel.scene.resources import EdgePolicy

    base_scene = wedge_diffraction_scene()
    scene = Scene(
        structures=base_scene.structures,
        endpoints=base_scene.endpoints,
        versions=base_scene.versions,
        metadata={
            **base_scene.metadata,
            "imported_edge_policy": EdgePolicy(
                edge_selection_mode="vertical_only"
            ),
        },
    )

    result = solve(
        scene,
        Config(
            components={"diffraction"},
            coherent=False,
            export_paths=True,
            return_field=False,
        ),
        reference_frequency_hz=3.0e9,
    )

    assert result.paths is not None
    # The horizontal outline edges (|dz|/length = 0) must not produce paths;
    # the slanted outline edges (|dz|/length ~ 0.83 > 0.7) survive. The vertical
    # shared wedge edge passes the policy but is occluded from the receiver.
    assert result.paths.edge_id.tolist() == [2, 5]
    baseline = solve(
        wedge_diffraction_scene(),
        Config(
            components={"diffraction"},
            coherent=False,
            export_paths=True,
            return_field=False,
        ),
        reference_frequency_hz=3.0e9,
    )
    # The finite-edge truncation pushes the weak horizontal edge 4 below the
    # existence floor, so the unfiltered baseline emits edges 1, 2 and 5;
    # vertical_only drops horizontal edge 1.
    assert baseline.paths.edge_id.tolist() == [1, 2, 5]


def test_diffraction_path_field_export_uses_native_complex_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diffraction")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    scene = wedge_diffraction_scene()
    result = solve(
        scene,
        Config(components={"diffraction"}, coherent=True, export_paths=True),
        reference_frequency_hz=3.0e9,
    )
    reference = solve_paths(
        scene, PathConfig(components={"diffraction"}), reference_frequency_hz=3.0e9
    )

    assert result.paths is not None
    path_field = torch.complex(result.paths.field_real, result.paths.field_imag)
    expected_phase = torch.remainder(-torch.angle(path_field), 2.0 * torch.pi)
    expected_field = reference.a[..., 0][reference.valid]
    assert torch.all(result.paths.valid)
    assert (
        torch.count_nonzero(path_field.abs() > 0.0).item() == result.paths.valid.numel()
    )
    torch.testing.assert_close(path_field, expected_field, rtol=5.0e-4, atol=1.0e-7)
    torch.testing.assert_close(
        result.paths.path_gain, path_field.abs().square(), rtol=2.0e-4, atol=1.0e-10
    )
    torch.testing.assert_close(
        result.paths.phase_rad, expected_phase, rtol=2.0e-4, atol=1.0e-6
    )
    torch.testing.assert_close(
        result.path_gain, result.field.abs().square(), rtol=2.0e-4, atol=1.0e-10
    )

def test_shared_wedge_edge_diffracts_to_a_receiver_in_front_of_the_faces():
    """Edge 0 is dropped above only because that receiver is behind both faces."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diffraction")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    scene = wedge_diffraction_scene(rx=torch.tensor([1.0, 1.0, 0.5]))
    result = solve(
        scene,
        Config(
            components={"diffraction"},
            coherent=False,
            export_paths=True,
            return_field=False,
        ),
        reference_frequency_hz=3.0e9,
    )

    assert result.paths is not None
    assert 0 in result.paths.edge_id.tolist()
