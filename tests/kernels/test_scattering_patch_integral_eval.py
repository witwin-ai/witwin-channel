"""Lockstep and contract tests for ADR-010 op 2 (phase-screen patch integral)."""

import math

import pytest
import torch

from witwin.channel_native.core.materials import PhaseScreen
from witwin.channel_native.physics.conventions import C0
from witwin.channel_native.scattering import PhaseScreenRuntime
from witwin.channel_native.scattering.kernels import functional as ops
from witwin.channel_native.runtime import symbols

from tests.reference import phase_screen_realization as reference

_FREQUENCY = 3.0e9
_K0 = 2.0 * math.pi * _FREQUENCY / C0


def _random_case(patches, rows, *, device, seed, resolution=64, height_rms=0.004):
    g = torch.Generator(device=device).manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32)

    # Patches: small near-planar triangles with jitter, UVs in [0, 1].
    base = rn(patches, 1, 3) * 0.5
    jitter = rn(patches, 3, 3) * 0.08
    jitter[..., 2] *= 0.1
    patch_tris = base + jitter
    patch_uvs = torch.rand(patches, 3, 2, generator=g, device=device)

    cpu_generator = torch.Generator().manual_seed(seed + 1)
    heights = torch.randn(resolution, resolution, generator=cpu_generator) * height_rms
    screen = PhaseScreen(height=heights, height_scale_m=1.0)
    runtime = PhaseScreenRuntime(screen, device=device)

    row_index = torch.randperm(patches, generator=g, device=device)[:rows].to(torch.int64)
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    d_i = torch.nn.functional.normalize(rn(rows, 3) * 0.3 - up, dim=-1)
    d_o = torch.nn.functional.normalize(rn(rows, 3) * 0.3 + up, dim=-1)
    n_rows = torch.nn.functional.normalize(rn(rows, 3) * 0.15 + up, dim=-1)
    r_te = torch.complex(rn(rows) * 0.4, rn(rows) * 0.4)
    r_tm = torch.complex(rn(rows) * 0.4, rn(rows) * 0.4)
    pol_t = torch.nn.functional.normalize(rn(3), dim=-1)
    pol_r = torch.nn.functional.normalize(rn(3), dim=-1)
    r1_rows = rn(rows).abs() + 1.0
    r2_rows = rn(rows).abs() + 1.0
    centroids = patch_tris[row_index].mean(dim=1)
    return {
        "runtime": runtime,
        "patch_tris": patch_tris.contiguous(),
        "patch_uvs": patch_uvs.contiguous(),
        "rows": row_index.contiguous(),
        "d_i": d_i.contiguous(),
        "d_o": d_o.contiguous(),
        "n_rows": n_rows.contiguous(),
        "r_te": r_te.contiguous(),
        "r_tm": r_tm.contiguous(),
        "pol_t": pol_t.contiguous(),
        "pol_r": pol_r.contiguous(),
        "r1_rows": r1_rows.contiguous(),
        "r2_rows": r2_rows.contiguous(),
        "centroids": centroids.contiguous(),
    }


def _native(case):
    return ops.scattering_patch_integral_eval(
        case["patch_tris"], case["patch_uvs"], case["rows"], case["d_i"],
        case["d_o"], case["n_rows"], case["r_te"], case["r_tm"], case["pol_t"],
        case["pol_r"], case["r1_rows"], case["r2_rows"], case["centroids"],
        case["runtime"].heights_m, k0=_K0,
    )


def _reference(case):
    return reference.realization_patch_total(
        case["runtime"], case["patch_tris"], case["patch_uvs"], case["rows"],
        case["d_i"], case["d_o"], case["n_rows"], case["r_te"], case["r_tm"],
        case["pol_t"], case["pol_r"], case["r1_rows"], case["r2_rows"],
        case["centroids"], _K0, _FREQUENCY,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("seed", [2, 11, 29, 101])
def test_patch_integral_total_matches_reference(seed):
    case = _random_case(48, 32, device="cuda", seed=seed)
    native = _native(case)
    ref_total, ref_integrals = _reference(case)
    # Per-row integrals meet the ADR-010 1e-5 budget (with an absolute floor
    # for heavily cancelling individual rows).
    d = (native["integral"] - ref_integrals).abs()
    floor = 1.0e-5 * float(ref_integrals.abs().max())
    bad = (d > 1.0e-5 * ref_integrals.abs()) & (d > floor)
    assert int(bad.sum()) == 0
    # Random-phase rows cancel arbitrarily strongly in the total, so the
    # correctly-propagated bound is 1e-5 of the summed row mass (the strict
    # max-rel <= 1e-5 on the total itself is asserted on the coherent
    # near-specular case below and on the canonical production fixture).
    total_n = complex(native["total"].item())
    total_r = complex(ref_total.item())
    mass = float(native["row_value"].abs().sum())
    assert abs(total_n - total_r) <= 1.0e-5 * mass


def _specular_case(patches, *, device, seed, resolution=64, height_rms=0.002):
    """Near-specular coherent geometry: the production realization regime."""

    g = torch.Generator(device=device).manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32)

    grid = int(math.isqrt(patches))
    patches = grid * grid
    # A flat z=0 plate split into small patches with matching UVs.
    xs = torch.linspace(-0.4, 0.4, grid + 1, device=device)
    tris = []
    uvs = []
    for i in range(grid):
        for j in range(grid):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = xs[j], xs[j + 1]
            tris.append(torch.tensor(
                [[x0, y0, 0.0], [x1, y0, 0.0], [x0, y1, 0.0]], device=device))
            u0 = (x0 + 0.4) / 0.8
            u1 = (x1 + 0.4) / 0.8
            v0 = (y0 + 0.4) / 0.8
            v1 = (y1 + 0.4) / 0.8
            uvs.append(torch.tensor(
                [[u0, v0], [u1, v0], [u0, v1]], device=device))
    patch_tris = torch.stack(tris)
    patch_uvs = torch.stack(uvs)
    rows = torch.arange(patches, device=device, dtype=torch.int64)

    cpu_generator = torch.Generator().manual_seed(seed + 1)
    heights = torch.randn(resolution, resolution, generator=cpu_generator) * height_rms
    runtime = PhaseScreenRuntime(
        PhaseScreen(height=heights, height_scale_m=1.0), device=device
    )

    centroids = patch_tris[rows].mean(dim=1)
    tx = torch.tensor([0.0, -1.0, 1.2], device=device)
    rx = torch.tensor([0.0, 1.0, 1.2], device=device)
    to_tx = tx[None, :] - centroids
    r1_rows = torch.linalg.vector_norm(to_tx, dim=-1)
    d_i = -(to_tx / r1_rows[:, None])
    to_rx = rx[None, :] - centroids
    r2_rows = torch.linalg.vector_norm(to_rx, dim=-1)
    d_o = to_rx / r2_rows[:, None]
    n_rows = torch.zeros_like(centroids)
    n_rows[:, 2] = 1.0
    r_te = torch.complex(
        torch.full((patches,), -0.6, device=device), rn(patches) * 0.01
    )
    r_tm = torch.complex(
        torch.full((patches,), -0.4, device=device), rn(patches) * 0.01
    )
    pol_t = torch.tensor([0.0, 0.0, 1.0], device=device)
    pol_r = torch.tensor([0.0, 0.0, 1.0], device=device)
    return {
        "runtime": runtime,
        "patch_tris": patch_tris.contiguous(),
        "patch_uvs": patch_uvs.contiguous(),
        "rows": rows.contiguous(),
        "d_i": d_i.contiguous(),
        "d_o": d_o.contiguous(),
        "n_rows": n_rows.contiguous(),
        "r_te": r_te.contiguous(),
        "r_tm": r_tm.contiguous(),
        "pol_t": pol_t.contiguous(),
        "pol_r": pol_r.contiguous(),
        "r1_rows": r1_rows.contiguous(),
        "r2_rows": r2_rows.contiguous(),
        "centroids": centroids.contiguous(),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("seed", [3, 17])
def test_patch_integral_coherent_total_meets_strict_gate(seed):
    case = _specular_case(64, device="cuda", seed=seed)
    native = _native(case)
    ref_total, _ref_integrals = _reference(case)
    total_n = complex(native["total"].item())
    total_r = complex(ref_total.item())
    # ADR-010 op 2 gate: realization total max-rel <= 1e-5 in the coherent
    # (production) regime.
    assert abs(total_n - total_r) <= 1.0e-5 * abs(total_r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_patch_integral_total_is_run_to_run_deterministic():
    case = _random_case(48, 32, device="cuda", seed=5)
    first = _native(case)
    second = _native(case)
    assert torch.equal(first["total"], second["total"])
    assert torch.equal(first["integral"], second["integral"])
    assert torch.equal(first["row_value"], second["row_value"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_patch_integral_contract_shapes_and_dtypes():
    case = _random_case(16, 9, device="cuda", seed=8)
    out = _native(case)
    assert out["total"].shape == ()
    assert out["total"].dtype == torch.complex64
    assert out["total"].is_cuda
    assert out["integral"].shape == (9,)
    assert out["integral"].dtype == torch.complex64
    assert out["row_value"].shape == (9,)
    assert torch.isfinite(out["total"].real) and torch.isfinite(out["total"].imag)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_patch_integral_requires_native_kernel(monkeypatch):
    case = _random_case(8, 4, device="cuda", seed=13)
    monkeypatch.setattr(symbols, "native_extension", lambda: None)
    with pytest.raises(
        RuntimeError, match="scattering_patch_integral_eval CUDA kernel is required"
    ):
        _native(case)
