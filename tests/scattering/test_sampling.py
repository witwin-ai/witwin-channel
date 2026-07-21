"""Kirchhoff table direction sampling: sampler/pdf consistency and measure."""

import math

import pytest
import torch
from scipy.stats import chi2

from witwin.channel_native.core.materials import Roughness
from witwin.channel_native.scattering import (
    build_kirchhoff_table,
    pdf,
    pdf_reverse,
    sample_directions,
)
from witwin.channel_native.runtime import symbols

_EPS0 = 8.8541878128e-12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def table():
    frequency_hz = 60e9
    sigma_e = 0.1 * 2.0 * math.pi * frequency_hz * _EPS0
    rough = Roughness(rms_height_m=1e-3, corr_length_x_m=10e-3, corr_length_y_m=10e-3)
    return build_kirchhoff_table(
        rough, [(0.1, 4.0, sigma_e, 1.0)], frequency_hz, device=DEVICE
    )


def _fixed_wi(table, ti: int) -> torch.Tensor:
    cos_i = float(table.cos_theta_i[ti])
    sin_i = math.sqrt(1.0 - cos_i**2)
    return torch.tensor([[sin_i, 0.0, cos_i]], device=table.device)


def _all_valid(rows: int, *, device: torch.device) -> torch.Tensor:
    return torch.ones(rows, dtype=torch.bool, device=device)


def test_sample_matches_pdf_binned(table):
    """Binned frequencies of 2e5 samples match the sampling masses.

    The outgoing grid (32 x 64) aggregates exactly into coarse 16 x 16 bins
    (2 x 4 fine bins each); a chi-square test against the expected bin
    masses must not reject at p = 0.001.
    """

    n = 200_000
    ti = 20
    wi = _fixed_wi(table, ti).expand(n, 3).contiguous()
    gen = torch.Generator(device="cpu").manual_seed(12345)
    u1 = torch.rand(n, generator=gen).to(table.device)
    u2 = torch.rand(n, generator=gen).to(table.device)
    wo, _ = sample_directions(table, _all_valid(n, device=wi.device), wi, u1, u2)

    # Coarse bin indices of the samples.
    cos_o = wo[:, 2].clamp(0.0, 1.0 - 1e-7)
    phi_o = torch.atan2(wo[:, 1], wo[:, 0])
    phi_o = torch.where(phi_o < 0.0, phi_o + 2.0 * math.pi, phi_o)
    bin_cos = (cos_o * 16).long().clamp(0, 15)
    bin_phi = (phi_o / (2.0 * math.pi / 16)).long().clamp(0, 15)
    counts = torch.zeros(16 * 16, device=table.device)
    counts.scatter_add_(
        0, bin_cos * 16 + bin_phi, torch.ones(n, device=table.device)
    )

    # Expected masses: aggregate the fine sampling density into 16 x 16.
    mass = table.sample_density[ti, 0] * table.bin_solid_angle  # [32, 64]
    expected = mass.reshape(16, 2, 16, 4).sum(dim=(1, 3)).reshape(-1) * n

    keep = expected >= 5.0
    stat = float((((counts - expected) ** 2 / expected)[keep]).sum())
    dof = int(keep.sum()) - 1
    assert chi2.sf(stat, dof) > 0.001


def test_pdf_integrates_to_one(table):
    """The sampling density integrates to 1 over the hemisphere within 1e-2."""

    for ti in (5, 20, 31):
        wi = _fixed_wi(table, ti)
        cos_c = table.cos_theta_o
        phi_c = table.phi_o
        cg, pg = torch.meshgrid(cos_c, phi_c, indexing="ij")
        sg = torch.sqrt((1.0 - cg**2).clamp(min=0.0))
        wo = torch.stack(
            (sg * torch.cos(pg), sg * torch.sin(pg), cg), dim=-1
        ).reshape(-1, 3)
        wi_rows = wi.expand(wo.shape[0], 3).contiguous()
        density = pdf(table, _all_valid(wo.shape[0], device=wi.device), wi_rows, wo)
        integral = float((density * table.bin_solid_angle).sum())
        assert abs(integral - 1.0) < 1e-2


def test_sample_returns_its_own_pdf(table):
    """(wo, pdf) from the sampler agrees with pdf(table, wi, wo) exactly."""

    n = 20_000
    wi = _fixed_wi(table, 14).expand(n, 3).contiguous()
    gen = torch.Generator(device="cpu").manual_seed(7)
    u1 = torch.rand(n, generator=gen).to(table.device)
    u2 = torch.rand(n, generator=gen).to(table.device)
    valid = _all_valid(n, device=wi.device)
    wo, density = sample_directions(table, valid, wi, u1, u2)
    lookup = pdf(table, valid, wi, wo)
    # Identical up to samples landing exactly on a bin edge (measure zero;
    # allow a vanishing mismatch fraction from float rounding).
    mismatch = (density != lookup).float().mean().item()
    assert mismatch < 1e-3
    assert bool((density > 0.0).all())


def test_pdf_reverse_is_pdf_with_swapped_args(table):
    gen = torch.Generator(device="cpu").manual_seed(3)

    def rand_dirs(n: int) -> torch.Tensor:
        cos = torch.rand(n, generator=gen) * 0.97 + 0.02
        phi = torch.rand(n, generator=gen) * 2.0 * math.pi
        sin = torch.sqrt(1.0 - cos**2)
        return torch.stack(
            (sin * torch.cos(phi), sin * torch.sin(phi), cos), dim=1
        ).to(table.device)

    wi = rand_dirs(512)
    wo = rand_dirs(512)
    valid = _all_valid(wi.shape[0], device=wi.device)
    assert torch.equal(pdf_reverse(table, valid, wo, wi), pdf(table, valid, wo, wi))


def test_pdf_zero_below_horizon(table):
    wi = _fixed_wi(table, 20)
    wo_down = torch.tensor([[0.0, 0.5, -0.5]], device=table.device)
    assert float(pdf(table, _all_valid(1, device=wi.device), wi, wo_down)) == 0.0


def test_runtime_table_ops_have_no_pytorch_fallback(table, monkeypatch):
    wi = _fixed_wi(table, 12)
    monkeypatch.setattr(symbols, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="scattering_table_eval CUDA kernel is required"):
        from witwin.channel_native.scattering import eval_bsdf

        eval_bsdf(table, _all_valid(1, device=wi.device), wi, wi)
    with pytest.raises(RuntimeError, match="scattering_table_pdf CUDA kernel is required"):
        pdf(table, _all_valid(1, device=wi.device), wi, wi)
