import math

import pytest
import torch

from witwin.channel.scattering.kernels import functional as ops
from witwin.channel.runtime import symbols


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scattering_event_probabilities_match_energy_split():
    cos_theta = torch.tensor([0.2, 0.5, 0.9, 0.5], device="cuda")
    material_id = torch.tensor([0, 0, 0, 1], device="cuda", dtype=torch.int32)
    cap_r_te = torch.tensor([0.3, 0.4, 0.6, 0.4], device="cuda")
    cap_r_tm = torch.tensor([0.2, 0.3, 0.5, 0.3], device="cuda")
    cap_t_te = torch.tensor([0.5, 0.4, 0.2, 0.4], device="cuda")
    cap_t_tm = torch.tensor([0.6, 0.5, 0.3, 0.5], device="cuda")
    rough_sigma = torch.tensor([1.0e-3, 0.0], device="cuda")
    scatter_model = torch.tensor([1, 0], device="cuda", dtype=torch.int32)

    out = ops.scattering_event_probabilities(
        cos_theta,
        material_id,
        cap_r_te,
        cap_r_tm,
        cap_t_te,
        cap_t_tm,
        rough_sigma,
        scatter_model,
        frequency_hz=60.0e9,
        probability_floor=0.05,
    )

    k0 = 2.0 * math.pi * 60.0e9 / 299_792_458.0
    coherent = torch.exp(-2.0 * (k0 * cos_theta[:3] * 1.0e-3) ** 2)
    r_bar = 0.5 * (cap_r_te[:3] + cap_r_tm[:3])
    t_bar = 0.5 * (cap_t_te[:3] + cap_t_tm[:3])
    shares = torch.stack(
        (r_bar * coherent.square(), r_bar * (1.0 - coherent.square()), t_bar)
    )
    shares /= shares.sum(dim=0, keepdim=True)
    shares = shares.clamp_min(0.05)
    shares /= shares.sum(dim=0, keepdim=True)

    torch.testing.assert_close(out["r_coh_amplitude"][:3], coherent, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(out["p_scatter"][:3], shares[1], rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(out["p_transmit"][:3], shares[2], rtol=2e-6, atol=2e-7)
    assert out["rough"].tolist() == [True, True, True, False]
    assert out["p_scatter"][3].item() == 0.0
    assert out["p_transmit"][3].item() == 0.0
    assert out["r_coh_amplitude"][3].item() == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scattering_event_probabilities_require_native_kernel(monkeypatch):
    monkeypatch.setattr(symbols, "native_extension", lambda: None)
    x = torch.ones(1, device="cuda")
    material = torch.zeros(1, device="cuda", dtype=torch.int32)
    with pytest.raises(
        RuntimeError, match="scattering_event_probabilities CUDA kernel is required"
    ):
        ops.scattering_event_probabilities(
            x,
            material,
            x,
            x,
            x,
            x,
            x,
            material,
            frequency_hz=1.0e9,
            probability_floor=0.05,
        )
