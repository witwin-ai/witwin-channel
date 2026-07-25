"""Realization-coherent phase-screen runtime (contract section 6, plan 6.7).

Heights NEVER displace geometry: RayD intersects the mean plane and the
sampled metric height ``h(u, v)`` enters only through the complex phasor
``exp(-j*q_n*h)`` with ``q_n = (k_s - k_i) . n_hat`` (``e^{+j w t}`` /
``e^{-j k r}`` conventions, matching ``core.field_state.PHASE_CONVENTION``
and the CPU oracle). Footprint averaging must happen on the PHASOR, never
on heights: ``E[exp(-j*q_n*h)] != exp(-j*q_n*E[h])``.

Texture convention: ``height[iy, ix]`` samples the UV point
``u = (ix + 0.5)/W``, ``v = (iy + 0.5)/H`` (texel centers). Bilinear
interpolation between texel centers with EDGE CLAMP at the borders (no
wrap): UV coordinates outside the half-texel margin reuse the border texel
value. Implemented with explicit gathers (not ``grid_sample``) so the
border behavior is exact and documented.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from witwin.core import PhaseScreen, SurfaceRoughness
from witwin.channel.constants import C0

__all__ = [
    "PhaseScreenRuntime",
    "generate_gaussian_realization",
    "patch_phase_integral",
    "realization_seed",
]


class PhaseScreenRuntime:
    """GPU height texture + phasor sampling for one surface realization."""

    def __init__(self, screen: PhaseScreen, device: torch.device | str = "cuda"):
        if not isinstance(screen, PhaseScreen):
            raise ValueError("screen must be a PhaseScreen")
        device = torch.device(device)
        # Metric heights [m]: stored texture * scale + offset, float32.
        heights = screen.height.to(device=device, dtype=torch.float32)
        self.heights_m = (
            heights * float(screen.height_scale_m) + float(screen.height_offset_m)
        ).contiguous()
        self.screen = screen
        self.device = device

    def sample_height(self, uv: torch.Tensor) -> torch.Tensor:
        """Bilinear metric height [m] at ``uv`` ([..., 2], u right, v down).

        Manual gather with edge clamp (see module docstring for the texel
        convention).
        """

        if uv.shape[-1] != 2:
            raise ValueError("uv must have trailing dimension 2")
        h_rows, w_cols = self.heights_m.shape
        uv = uv.to(device=self.device, dtype=torch.float32)
        # Edge clamp: clamp the CONTINUOUS texel coordinate into the span of
        # texel centers before flooring, so UV outside the half-texel margin
        # reproduces the border texel exactly (no border interpolation).
        tx = (uv[..., 0] * w_cols - 0.5).clamp(0.0, float(w_cols - 1))
        ty = (uv[..., 1] * h_rows - 0.5).clamp(0.0, float(h_rows - 1))
        x0 = torch.floor(tx)
        y0 = torch.floor(ty)
        wx = tx - x0
        wy = ty - y0
        ix0 = x0.long()
        iy0 = y0.long()
        ix1 = (ix0 + 1).clamp(max=w_cols - 1)
        iy1 = (iy0 + 1).clamp(max=h_rows - 1)
        flat = self.heights_m.reshape(-1)

        def tex(iy: torch.Tensor, ix: torch.Tensor) -> torch.Tensor:
            return flat[iy * w_cols + ix]

        top = tex(iy0, ix0) * (1.0 - wx) + tex(iy0, ix1) * wx
        bot = tex(iy1, ix0) * (1.0 - wx) + tex(iy1, ix1) * wx
        return top * (1.0 - wy) + bot * wy

    def phasor(self, uv: torch.Tensor, q_n: torch.Tensor | float) -> torch.Tensor:
        """Complex64 phase-screen factor ``exp(-j*q_n*h(u, v))``."""

        h = self.sample_height(uv)
        if not isinstance(q_n, torch.Tensor):
            q_n = torch.as_tensor(q_n, dtype=torch.float32, device=self.device)
        phase = -(q_n * h)
        return torch.polar(torch.ones_like(phase), phase).to(torch.complex64)


def realization_seed(scene_seed: int, surface_id: int, realization_id: int) -> int:
    """Deterministic 64-bit seed for ``(scene_seed, surface_id, realization_id)``.

    SplitMix64-style avalanche over the packed inputs so nearby ids give
    decorrelated seeds while the mapping stays reproducible across runs and
    platforms (pure integer arithmetic, no RNG state involved).
    """

    mask = (1 << 64) - 1
    z = (int(scene_seed) & mask)
    for salt in (int(surface_id), int(realization_id)):
        z = (z + 0x9E3779B97F4A7C15 + (salt & mask)) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        z = z ^ (z >> 31)
    return z


def generate_gaussian_realization(
    roughness: SurfaceRoughness,
    extent_m: tuple[float, float] | float,
    resolution: tuple[int, int] | int,
    seed: int,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Periodic Gaussian random height field with Gaussian correlation.

    FFT spectral synthesis of the section-6.1 correlation
    ``C(x, y) = sigma_h^2 * exp(-(x/lx)^2 - (y/ly)^2)`` whose PSD (plan
    section 6.3 Fourier convention, ``(1/(2*pi)^2) Int W d^2q = sigma_h^2``)
    is ``W(qx, qy) = pi*sigma_h^2*lx*ly*exp(-(qx^2*lx^2 + qy^2*ly^2)/4)``.

    Normalization: on a periodic domain ``Lx x Ly`` each Fourier mode gets
    variance ``E[|h_k|^2] = W(q_k)/(Lx*Ly)`` (the Riemann sum of the PSD
    integral over the discrete mode lattice ``dq = 2*pi/L``). Modes are
    drawn as independent circular complex Gaussians and the real part is
    taken, which halves per-mode variance, so amplitudes carry a
    compensating ``sqrt(2)``. Total variance then approximates ``sigma_h^2``
    up to spectral truncation at Nyquist: for ``dx << l`` and ``L >> l`` the
    sampled RMS height matches ``sigma_h`` within a few percent.

    ``extent_m``/``resolution`` are ``(x, y)`` pairs (scalars broadcast).
    Returns float32 heights [m] of shape ``(ny, nx)`` on ``device``
    (row = v/y axis, matching :class:`PhaseScreenRuntime`).
    """

    sigma_h = float(roughness.rms_height_m)
    lx = float(roughness.correlation_length_x_m)
    ly = float(roughness.correlation_length_y_m)
    if isinstance(extent_m, (int, float)):
        extent_m = (float(extent_m), float(extent_m))
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    length_x, length_y = float(extent_m[0]), float(extent_m[1])
    nx, ny = int(resolution[0]), int(resolution[1])
    if length_x <= 0.0 or length_y <= 0.0 or nx < 2 or ny < 2:
        raise ValueError("extent_m must be positive and resolution >= 2")

    qx = 2.0 * np.pi * np.fft.fftfreq(nx, d=length_x / nx)
    qy = 2.0 * np.pi * np.fft.fftfreq(ny, d=length_y / ny)
    psd = (
        np.pi
        * sigma_h**2
        * lx
        * ly
        * np.exp(-((qx[None, :] * lx) ** 2 + (qy[:, None] * ly) ** 2) / 4.0)
    )
    amplitude = np.sqrt(2.0 * psd / (length_x * length_y))
    rng = np.random.default_rng(int(seed) & ((1 << 64) - 1))
    xi = (rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))) / math.sqrt(2.0)
    # h(x_m) = Re sum_k A_k xi_k exp(+j q_k . x_m); numpy ifft2 divides by
    # nx*ny, so multiply it back to get the plain Fourier-series sum.
    field = np.fft.ifft2(amplitude * xi) * (nx * ny)
    heights = np.real(field)
    return torch.from_numpy(np.ascontiguousarray(heights, dtype=np.float32)).to(
        torch.device(device)
    )


def patch_phase_integral(
    runtime: PhaseScreenRuntime,
    patch_vertices: torch.Tensor,
    uv_vertices: torch.Tensor,
    k_i_vec: torch.Tensor,
    k_s_vec: torch.Tensor,
    frequency_hz: float,
    n_quad: int = 16,
) -> torch.Tensor:
    """Triangle-domain quadrature of the Kirchhoff phase integral (GPU).

    Evaluates ``sum_T Int_T exp(-j*(k_s - k_i).x) * exp(-j*q_n*h(u, v)) dA``
    over triangles ``patch_vertices`` [T, 3, 3] with matching UVs
    ``uv_vertices`` [T, 3, 2] (a single triangle may omit the leading dim).
    Positions stay on the mean plane; heights enter only the phase, matching
    ``oracle.phase_screen_patch_integral`` for the same height field.

    Quadrature: Duffy-mapped tensor-product Gauss-Legendre with ``n_quad``
    points per axis  -  the unit square ``(xi, eta)`` maps to barycentric
    ``(a, b) = (xi, eta*(1 - xi))`` with Jacobian ``(1 - xi)``, so
    refinement in ``n_quad`` converges to the exact triangle integral.
    Returns a 0-dim complex64 tensor on the runtime device.
    """

    device = runtime.device
    tri = patch_vertices.to(device=device, dtype=torch.float32)
    uv = uv_vertices.to(device=device, dtype=torch.float32)
    if tri.ndim == 2:
        tri = tri.unsqueeze(0)
    if uv.ndim == 2:
        uv = uv.unsqueeze(0)
    if tri.shape[1:] != (3, 3) or uv.shape[1:] != (3, 2) or tri.shape[0] != uv.shape[0]:
        raise ValueError("patch_vertices must be [T, 3, 3] and uv_vertices [T, 3, 2]")
    k_i = k_i_vec.to(device=device, dtype=torch.float32).reshape(3)
    k_s = k_s_vec.to(device=device, dtype=torch.float32).reshape(3)
    k0 = 2.0 * math.pi * float(frequency_hz) / C0
    for name, vec in (("k_i_vec", k_i), ("k_s_vec", k_s)):
        if abs(float(torch.linalg.vector_norm(vec)) - k0) > 1e-5 * k0:
            raise ValueError(f"|{name}| does not match 2*pi*frequency_hz/c0")
    q = k_s - k_i

    # Per-triangle mean-plane frame: edges, normal, area.
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    normal = torch.cross(e1, e2, dim=1)
    double_area = torch.linalg.vector_norm(normal, dim=1)
    n_hat = normal / double_area.unsqueeze(1).clamp(min=1e-30)
    q_n = n_hat @ q  # [T] normal wavenumber transfer per triangle

    # Duffy-mapped Gauss points on the unit square (float64 nodes -> f32).
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    xi = torch.from_numpy(0.5 * (nodes + 1.0)).to(device=device, dtype=torch.float32)
    w1 = torch.from_numpy(0.5 * weights).to(device=device, dtype=torch.float32)
    a = xi[:, None].expand(n_quad, n_quad)  # barycentric a = xi
    b = xi[None, :] * (1.0 - xi[:, None])  # barycentric b = eta*(1 - xi)
    w2d = (w1[:, None] * w1[None, :]) * (1.0 - xi[:, None])  # Duffy Jacobian
    a = a.reshape(-1)
    b = b.reshape(-1)
    w2d = w2d.reshape(-1)

    # Quadrature positions/UVs: x = p0 + a*e1 + b*e2 (same barycentric
    # interpolation for UV), batched [T, Q, ...].
    pos = (
        tri[:, 0, None, :]
        + a[None, :, None] * e1[:, None, :]
        + b[None, :, None] * e2[:, None, :]
    )
    uv_pts = (
        uv[:, 0, None, :]
        + a[None, :, None] * (uv[:, 1] - uv[:, 0])[:, None, :]
        + b[None, :, None] * (uv[:, 2] - uv[:, 0])[:, None, :]
    )
    heights = runtime.sample_height(uv_pts)  # [T, Q]
    phase = pos @ q + q_n[:, None] * heights
    phasor = torch.polar(torch.ones_like(phase), -phase)
    # dA = 2*Area * da db; the simplex measure is carried by w2d.
    contrib = (phasor * w2d[None, :]).sum(dim=1) * double_area.to(phasor.real.dtype)
    return contrib.sum().to(torch.complex64)
