import cmath
import math

import pytest
import torch

from witwin.channel.propagation.fields.kernels import functional as ops


def _slab_coefficient(
    *,
    eps_r: float,
    sigma_e: float,
    mu_r: float,
    gain: float,
    thickness: float,
    frequency: float,
) -> complex:
    epsilon0 = 8.8541878128e-12
    omega = 2.0 * math.pi * frequency
    wavelength = 299792458.0 / frequency
    eta = complex(eps_r, -sigma_e / (omega * epsilon0))
    root = cmath.sqrt(mu_r * eta)
    interface = (mu_r - root) / (mu_r + root)
    q = 2.0 * math.pi * thickness / wavelength * root
    phase = cmath.exp(-2j * q)
    return gain * interface * (1.0 - phase) / (1.0 - interface * interface * phase)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_free_space_complex3_matches_analytic_phase_and_power_contract():
    frequency = 3.0e9
    distance = 10.0
    source = torch.tensor([[0.0, 0.0, 0.0]], device="cuda")
    target = torch.tensor([[distance, 0.0, 0.0]], device="cuda")
    polarization = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")

    unit = ops.field_free_space(
        source,
        target,
        torch.tensor([1.0], device="cuda"),
        polarization,
        polarization,
        frequency_hz=frequency,
    )
    powered = ops.field_free_space(
        source,
        target,
        torch.tensor([4.0], device="cuda"),
        polarization,
        polarization,
        frequency_hz=frequency,
    )

    wavelength = 299792458.0 / frequency
    expected = wavelength / (4.0 * math.pi * distance) * cmath.exp(
        -1j * 2.0 * math.pi * distance / wavelength
    )
    torch.testing.assert_close(
        unit["coefficient"],
        torch.tensor([expected], device="cuda", dtype=torch.complex64),
        rtol=1.0e-4,
        atol=1.0e-8,
    )
    phase_error = torch.angle(
        unit["coefficient"]
        / torch.tensor([expected], device="cuda", dtype=torch.complex64)
    ).abs()
    assert phase_error.item() <= 1.0e-3
    torch.testing.assert_close(powered["coefficient"], unit["coefficient"])
    torch.testing.assert_close(powered["path_field"], 2.0 * unit["path_field"])
    torch.testing.assert_close(powered["path_gain"], 4.0 * unit["path_gain"])
    torch.testing.assert_close(unit["field_vector"][:, 2], unit["coefficient"])
    torch.testing.assert_close(
        unit["field_vector"][:, :2], torch.zeros_like(unit["field_vector"][:, :2])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_free_space_receiver_projection_and_global_rotation_are_invariant():
    source = torch.tensor([[0.0, 0.0, 0.0]], device="cuda")
    target = torch.tensor([[3.0, 4.0, 1.0]], device="cuda")
    tx_pol = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
    rx_pol = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device="cuda",
    )

    base = ops.field_free_space(
        source,
        target,
        torch.ones(1, device="cuda"),
        tx_pol,
        rx_pol,
        frequency_hz=28.0e9,
    )
    rotated = ops.field_free_space(
        source @ rotation.T,
        target @ rotation.T,
        torch.ones(1, device="cuda"),
        tx_pol @ rotation.T,
        rx_pol @ rotation.T,
        frequency_hz=28.0e9,
    )

    torch.testing.assert_close(rotated["coefficient"], base["coefficient"])
    torch.testing.assert_close(
        rotated["field_vector"], base["field_vector"] @ rotation.T.to(torch.complex64)
    )
    cross = ops.field_project_complex3(
        base["field_vector"],
        base["direction"],
        torch.tensor([[0.0, 1.0, 0.0]], device="cuda"),
    )
    assert cross["path_gain"].item() < base["path_gain"].item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_lossy_slab_reflection_matches_complex_reference():
    frequency = 3.5e9
    source = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
    target = torch.tensor([[0.0, 0.0, 2.0]], device="cuda")
    hit = torch.tensor([[[0.0, 0.0, 0.0]]], device="cuda")
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda")
    polarization = torch.tensor([[1.0, 0.0, 0.0]], device="cuda")
    result = ops.field_reflection_sequence(
        source,
        target,
        hit,
        normal,
        torch.tensor([1.0], device="cuda"),
        polarization,
        polarization,
        torch.tensor([[4.0]], device="cuda"),
        torch.tensor([[0.025]], device="cuda"),
        torch.tensor([[1.0]], device="cuda"),
        torch.tensor([[0.8]], device="cuda"),
        torch.tensor([[0.12]], device="cuda"),
        frequency_hz=frequency,
    )

    length = 3.0
    wavelength = 299792458.0 / frequency
    expected = (
        wavelength
        / (4.0 * math.pi * length)
        * cmath.exp(-2j * math.pi * length / wavelength)
        * _slab_coefficient(
            eps_r=4.0,
            sigma_e=0.025,
            mu_r=1.0,
            gain=0.8,
            thickness=0.12,
            frequency=frequency,
        )
    )
    expected_tensor = torch.tensor([expected], device="cuda", dtype=torch.complex64)
    torch.testing.assert_close(
        result["coefficient"], expected_tensor, rtol=1.0e-4, atol=1.0e-8
    )
    assert torch.angle(result["coefficient"] / expected_tensor).abs().item() <= 1.0e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_non_coplanar_double_reflection_is_rotation_invariant():
    root2 = math.sqrt(0.5)
    source = torch.tensor([[0.0, 0.0, 0.0]], device="cuda")
    target = torch.tensor([[1.0, 1.0, 1.0]], device="cuda")
    hits = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]], device="cuda")
    normals = torch.tensor(
        [[[root2, -root2, 0.0], [0.0, root2, -root2]]], device="cuda"
    )
    tx_pol = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
    rx_pol = torch.tensor([[1.0, 0.0, 0.0]], device="cuda")
    rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda"
    )

    def solve(rotated: bool):
        matrix = rotation if rotated else torch.eye(3, device="cuda")
        return ops.field_reflection_sequence(
            source @ matrix.T,
            target @ matrix.T,
            hits @ matrix.T,
            normals @ matrix.T,
            torch.ones(1, device="cuda"),
            tx_pol @ matrix.T,
            rx_pol @ matrix.T,
            torch.full((1, 2), 1.0, device="cuda"),
            torch.full((1, 2), 1.0e9, device="cuda"),
            torch.ones((1, 2), device="cuda"),
            torch.ones((1, 2), device="cuda"),
            torch.full((1, 2), 0.1, device="cuda"),
            frequency_hz=6.0e9,
        )

    base = solve(False)
    rotated = solve(True)
    torch.testing.assert_close(rotated["coefficient"], base["coefficient"])
    torch.testing.assert_close(
        rotated["field_vector"], base["field_vector"] @ rotation.T.to(torch.complex64)
    )
