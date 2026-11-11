# Copyright Xingyu Chen.
# Kirchhoff table direction sampling: sampler/pdf consistency and measure.

"""Kirchhoff table direction sampling: sampler/pdf consistency and measure."""

import math

import pytest
import torch

from witwin.core import SurfaceRoughness
from witwin.channel.scene.resources import (
    build_kirchhoff_table,
    pdf,
    pdf_reverse,
)
from witwin.channel import runtime

_EPS0 = 8.8541878128e-12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def table():
    frequency_hz = 60e9
    sigma_e = 0.1 * 2.0 * math.pi * frequency_hz * _EPS0
    rough = SurfaceRoughness(rms_height_m=1e-3, correlation_length_x_m=10e-3, correlation_length_y_m=10e-3)
    return build_kirchhoff_table(
        rough, [(0.1, 4.0, sigma_e, 1.0)], frequency_hz, device=DEVICE
    )


def _fixed_wi(table, ti: int) -> torch.Tensor:
    cos_i = float(table.cos_theta_i[ti])
    sin_i = math.sqrt(1.0 - cos_i**2)
    return torch.tensor([[sin_i, 0.0, cos_i]], device=table.device)


def _all_valid(rows: int, *, device: torch.device) -> torch.Tensor:
    return torch.ones(rows, dtype=torch.bool, device=device)


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
    monkeypatch.setattr(runtime, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="scattering_table_pdf CUDA kernel is required"):
        pdf(table, _all_valid(1, device=wi.device), wi, wi)