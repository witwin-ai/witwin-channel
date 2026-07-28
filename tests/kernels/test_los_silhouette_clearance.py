"""Native ISB-taper (ADR-017) LoS clearance kernel contract.

Pins the native ``los_silhouette_clearance`` op to the exact reference formulas
(signed-AABB clearance at the segment's closest-approach sample, Fresnel
penumbra w_F = sqrt(lambda d1 d2 / (d1 + d2)), and the C1 tau smoothstep) from
artifacts/isb-taper/common.py / stage2.py, and checks ``los_taper_apply`` scales
a LoS field bundle by tau (power by tau^2).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from witwin.channel.propagation.geometry import (
    apply_los_taper,
    los_clearance_factor,
)

_C0 = 299792458.0
_SAMPLES = 400


def _reference_tau(
    source: np.ndarray,
    target: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
    wavelength: float,
    width: float,
) -> float:
    ts = np.linspace(0.0, 1.0, _SAMPLES)
    pts = source[None, :] + ts[:, None] * (target - source)[None, :]
    q = np.maximum(box_min[None, :] - pts, pts - box_max[None, :])
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(q.max(axis=1), 0.0)
    sd = outside + inside
    j = int(np.argmin(np.abs(sd)))
    c = float(sd[j])
    g = pts[j]
    d1 = float(np.linalg.norm(g - source))
    d2 = float(np.linalg.norm(target - g))
    # Shadow magnification: the occluder-plane miss distance c projects into the
    # receiver plane enlarged by (d1 + d2) / d1 (point-source shadow), matching
    # the in-receiver-plane clearance the accepted stage2 projection scores.
    c_plane = c * (d1 + d2) / max(d1, 1e-6)
    w_f = np.sqrt(max(wavelength * d1 * d2 / max(d1 + d2, 1e-12), 0.0))
    w = max(width * w_f, 1e-6)
    t = np.clip(0.5 * (c_plane / w + 1.0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the LoS clearance kernel")


def test_silhouette_clearance_matches_reference_formulas():
    _require_cuda()
    device = torch.device("cuda")
    frequency = 5.0e9
    width = 0.5
    wavelength = _C0 / frequency
    box_min = np.array([-0.1, -0.1, -0.1], dtype=np.float64)
    box_max = np.array([0.1, 0.1, 0.1], dtype=np.float64)

    source = np.array(
        [
            [0.0, 0.0, 1.0],   # straight down through the box -> deep shadow
            [0.02, 0.0, 1.0],  # near the silhouette edge -> penumbra
            [0.5, 0.0, 1.0],   # far to the side -> fully lit
            [0.11, 0.0, 1.0],  # just outside the box face -> lit margin
        ],
        dtype=np.float64,
    )
    target = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.02, 0.0, -1.0],
            [0.5, 0.0, -1.0],
            [0.11, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            _reference_tau(source[i], target[i], box_min, box_max, wavelength, width)
            for i in range(source.shape[0])
        ]
    )

    tau = los_clearance_factor(
        torch.tensor(source, dtype=torch.float32, device=device),
        torch.tensor(target, dtype=torch.float32, device=device),
        torch.tensor(box_min[None, :], dtype=torch.float32, device=device),
        torch.tensor(box_max[None, :], dtype=torch.float32, device=device),
        frequency_hz=frequency,
        width=width,
    )

    assert tau.shape == (4,)
    assert tau.dtype == torch.float32
    np.testing.assert_allclose(tau.cpu().numpy(), expected, rtol=2e-4, atol=2e-4)
    # Deep shadow tapers to ~0, fully lit to ~1, penumbra strictly between.
    values = tau.cpu().numpy()
    assert values[2] > 0.99
    assert 0.0 <= values[0] < 0.5
    assert 0.0 < values[1] < 1.0


def test_los_taper_apply_scales_bundle_by_tau():
    _require_cuda()
    device = torch.device("cuda")
    rows = 5
    field_vector = torch.randn(rows, 3, dtype=torch.complex64, device=device)
    coefficient = torch.randn(rows, dtype=torch.complex64, device=device)
    path_field = torch.randn(rows, dtype=torch.complex64, device=device)
    path_gain = torch.rand(rows, dtype=torch.float32, device=device)
    tau = torch.linspace(0.0, 1.0, rows, dtype=torch.float32, device=device)

    out = apply_los_taper(field_vector, coefficient, path_field, path_gain, tau)

    scale = tau[:, None]
    torch.testing.assert_close(out["field_vector"], field_vector * scale)
    torch.testing.assert_close(out["coefficient"], coefficient * tau)
    torch.testing.assert_close(out["path_field"], path_field * tau)
    torch.testing.assert_close(out["path_gain"], path_gain * tau * tau)
