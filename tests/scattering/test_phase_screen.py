"""Phase-screen runtime vs the CPU complex128 oracle.

The runtime samples heights from a texture (bilinear, edge clamp) and
integrates the complex phasor over mean-plane triangles; the oracle
integrates the analytic height function over the same parallelogram.
"""

import math

import numpy as np
import torch

from witwin.core import PhaseScreen, SurfaceRoughness
from tests.reference.em_oracle import (
    C0,
    phase_screen_patch_integral as oracle_patch_integral,
)
from witwin.channel.scene.resources import (
    PhaseScreenRuntime,
    generate_gaussian_realization,
    patch_phase_integral,
    realization_seed,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
F0 = 6.0e9
K0 = 2.0 * math.pi * F0 / C0


def _rect_patch(size: float):
    """Unit-square-parametrized rectangle in z = 0 split into 2 triangles.

    Returns (oracle parallelogram corners [4,3], triangle vertices [2,3,3],
    triangle uv [2,3,2]); uv equals the patch (u, v) parametrization so the
    texture convention matches oracle ``height_fn(u, v)``.
    """

    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [size, 0.0, 0.0],
            [size, size, 0.0],
            [0.0, size, 0.0],
        ]
    )
    p = torch.tensor(corners, dtype=torch.float32)
    tris = torch.stack(
        (
            torch.stack((p[0], p[1], p[2])),
            torch.stack((p[0], p[2], p[3])),
        )
    )
    uv = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    return corners, tris, uv


def _wave_vectors():
    """Near-specular pair: the integral stays O(area), so the comparison is
    not dominated by float32 rounding of a heavily cancelling carrier."""

    theta_i = math.radians(25.0)
    theta_s = math.radians(28.0)
    k_i = K0 * np.array([math.sin(theta_i), 0.0, -math.cos(theta_i)])
    k_s = K0 * np.array([math.sin(theta_s), 0.02, math.sqrt(1.0 - math.sin(theta_s) ** 2 - 0.0004)])
    return k_i, k_s


def _texture_from_fn(height_fn, resolution: int) -> torch.Tensor:
    """Sample ``height_fn(u, v)`` at texel centers (runtime convention)."""

    centers = (np.arange(resolution) + 0.5) / resolution
    u, v = np.meshgrid(centers, centers, indexing="xy")
    return torch.tensor(height_fn(u, v), dtype=torch.float32)


def test_constant_height_analytic_phase():
    """A constant height texture reproduces exp(-j*q_n*h0) exactly."""

    size = 0.5
    corners, tris, uv = _rect_patch(size)
    k_i, k_s = _wave_vectors()
    q_n = (k_s - k_i)[2]
    h0 = 0.003
    screen = PhaseScreen(height=torch.ones(8, 8), height_scale_m=h0)
    runtime = PhaseScreenRuntime(screen, device=DEVICE)
    result = patch_phase_integral(
        runtime, tris, uv, torch.tensor(k_i), torch.tensor(k_s), F0, n_quad=48
    )
    flat = oracle_patch_integral(
        lambda u, v: np.zeros_like(u), corners, k_i, k_s, F0, n_quad=96
    )
    expected = np.exp(-1j * q_n * h0) * flat
    assert abs(complex(result.item()) - expected) < 1e-4 * abs(expected)


def test_sinusoidal_texture_matches_oracle():
    """Sinusoidal height sampled to a texture matches the oracle integral."""

    size = 0.5
    amplitude = 0.002
    def height_fn(u, v):
        return amplitude * np.sin(2.0 * math.pi * 3.0 * u)

    corners, tris, uv = _rect_patch(size)
    k_i, k_s = _wave_vectors()
    screen = PhaseScreen(height=_texture_from_fn(height_fn, 512), height_scale_m=1.0)
    runtime = PhaseScreenRuntime(screen, device=DEVICE)
    result = complex(
        patch_phase_integral(
            runtime, tris, uv, torch.tensor(k_i), torch.tensor(k_s), F0, n_quad=64
        ).item()
    )
    expected = oracle_patch_integral(height_fn, corners, k_i, k_s, F0, n_quad=96)
    assert abs(result - expected) < 1e-3 * abs(expected)


def test_quadrature_refinement_converges():
    size = 0.5
    amplitude = 0.002
    def height_fn(u, v):
        return amplitude * np.sin(2.0 * math.pi * 3.0 * u) * np.cos(
            2.0 * math.pi * 2.0 * v
        )

    _, tris, uv = _rect_patch(size)
    k_i, k_s = _wave_vectors()
    screen = PhaseScreen(height=_texture_from_fn(height_fn, 256), height_scale_m=1.0)
    runtime = PhaseScreenRuntime(screen, device=DEVICE)

    def integral(n):
        return complex(
            patch_phase_integral(
                runtime, tris, uv, torch.tensor(k_i), torch.tensor(k_s), F0, n_quad=n
            ).item()
        )

    ref = integral(96)
    assert abs(integral(16) - ref) > abs(integral(48) - ref)
    assert abs(integral(48) - ref) < 1e-3 * abs(ref)


def test_realization_rms_matches_sigma():
    """Spectral synthesis reproduces sigma_h within 5% (spatial average)."""

    sigma_h = 2e-3
    rough = SurfaceRoughness(rms_height_m=sigma_h, correlation_length_x_m=0.02, correlation_length_y_m=0.02)
    stds = []
    for realization_id in range(3):
        seed = realization_seed(2024, 7, realization_id)
        field = generate_gaussian_realization(rough, 1.28, 512, seed, device=DEVICE)
        assert field.shape == (512, 512)
        stds.append(float(field.std()))
    mean_std = sum(stds) / len(stds)
    assert abs(mean_std - sigma_h) < 0.05 * sigma_h


def test_realization_seed_reproducible_and_decorrelated():
    rough = SurfaceRoughness(rms_height_m=1e-3, correlation_length_x_m=0.02, correlation_length_y_m=0.02)
    seed_a = realization_seed(11, 3, 0)
    field_a = generate_gaussian_realization(rough, 0.64, 256, seed_a, device=DEVICE)
    field_a_again = generate_gaussian_realization(rough, 0.64, 256, seed_a, device=DEVICE)
    assert torch.equal(field_a, field_a_again)
    # Different realization_id must give a decorrelated surface.
    seed_b = realization_seed(11, 3, 1)
    assert seed_a != seed_b
    field_b = generate_gaussian_realization(rough, 0.64, 256, seed_b, device=DEVICE)
    stacked = torch.stack((field_a.reshape(-1), field_b.reshape(-1)))
    corr = torch.corrcoef(stacked)[0, 1]
    assert abs(float(corr)) < 0.1


def test_footprint_averaging_on_phasor_not_height():
    """Mip-averaging heights before exponentiation is WRONG; the runtime must
    integrate phasors. For a high-variance patch the two differ strongly."""

    size = 0.5
    _, tris, uv = _rect_patch(size)
    # Specular geometry (q_par = 0): the flat/mip integral stays fully
    # coherent (magnitude = area), making the contrast with the decorrelated
    # phasor integral unambiguous.
    theta_i = math.radians(25.0)
    k_i = K0 * np.array([math.sin(theta_i), 0.0, -math.cos(theta_i)])
    k_s = K0 * np.array([math.sin(theta_i), 0.0, math.cos(theta_i)])
    q = k_s - k_i
    q_n = q[2]
    # Zero-mean rough heights with q_n*sigma ~ pi: phasor decorrelation is
    # severe while the mean height stays ~0.
    rng = np.random.default_rng(5)
    sigma = math.pi / abs(q_n)
    heights = torch.tensor(
        rng.normal(0.0, sigma, size=(64, 64)), dtype=torch.float32
    )
    screen = PhaseScreen(height=heights, height_scale_m=1.0)
    runtime = PhaseScreenRuntime(screen, device=DEVICE)
    correct = complex(
        patch_phase_integral(
            runtime, tris, uv, torch.tensor(k_i), torch.tensor(k_s), F0, n_quad=96
        ).item()
    )
    # The (wrong) mip answer: replace the texture by its average height.
    mean_height = float(runtime.heights_m.mean())
    corners = np.array(
        [[0.0, 0.0, 0.0], [size, 0.0, 0.0], [size, size, 0.0], [0.0, size, 0.0]]
    )
    mip = oracle_patch_integral(
        lambda u, v: np.full_like(u, mean_height), corners, k_i, k_s, F0, n_quad=96
    )
    # Phasor averaging destroys coherence: |correct| << |mip| and the two
    # values disagree by far more than any quadrature tolerance.
    assert abs(correct - mip) > 0.5 * abs(mip)
    assert abs(correct) < 0.5 * abs(mip)


def test_sample_height_bilinear_edges():
    """Texel-center convention and edge clamp of the manual bilinear gather."""

    heights = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    screen = PhaseScreen(height=heights, height_scale_m=1.0)
    runtime = PhaseScreenRuntime(screen, device=DEVICE)
    uv = torch.tensor(
        [
            [0.25, 0.25],  # texel (0, 0) center
            [0.75, 0.75],  # texel (1, 1) center
            [0.5, 0.5],  # midpoint of all four texels
            [0.0, 0.0],  # clamped corner
            [1.0, 1.0],  # clamped corner
        ]
    )
    values = runtime.sample_height(uv).cpu()
    torch.testing.assert_close(
        values, torch.tensor([0.0, 3.0, 1.5, 0.0, 3.0]), atol=1e-6, rtol=0.0
    )


def test_phasor_convention():
    """phasor() implements exp(-j*q_n*h) under the e^{+jwt} convention."""

    screen = PhaseScreen(height=torch.full((4, 4), 2.0), height_scale_m=1e-3)
    runtime = PhaseScreenRuntime(screen, device=DEVICE)
    q_n = 100.0
    value = runtime.phasor(torch.tensor([[0.5, 0.5]]), q_n)
    expected = complex(math.cos(-q_n * 2e-3), math.sin(-q_n * 2e-3))
    assert value.dtype == torch.complex64
    assert abs(complex(value.item()) - expected) < 1e-6
