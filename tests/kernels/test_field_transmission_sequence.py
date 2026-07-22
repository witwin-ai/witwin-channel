"""Endpoint-connection specular transmission (contract section 4).

The decisive invariant: a thin_sheet wall made of a single vacuum layer must
reproduce the no-wall free-space complex field (amplitude AND phase) at normal
and oblique incidence.
"""

import math

import pytest
import torch

from witwin.channel.propagation.fields.kernels import functional as ops
from witwin.channel.materials.kernels import functional as material_functional

C0 = 299792458.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA torch is required"
)


def _csr(materials: list[list[tuple[float, float, float, float]]]) -> dict[str, torch.Tensor]:
    offsets: list[int] = []
    counts: list[int] = []
    thickness: list[float] = []
    eps_r: list[float] = []
    sigma_e: list[float] = []
    mu_r: list[float] = []
    for layers in materials:
        offsets.append(len(thickness))
        counts.append(len(layers))
        for layer in layers:
            thickness.append(layer[0])
            eps_r.append(layer[1])
            sigma_e.append(layer[2])
            mu_r.append(layer[3])
    return {
        "layer_offset": torch.tensor(offsets, device="cuda", dtype=torch.int32),
        "layer_count": torch.tensor(counts, device="cuda", dtype=torch.int32),
        "layer_thickness_m": torch.tensor(thickness, device="cuda", dtype=torch.float32),
        "layer_eps_r": torch.tensor(eps_r, device="cuda", dtype=torch.float32),
        "layer_sigma_e": torch.tensor(sigma_e, device="cuda", dtype=torch.float32),
        "layer_mu_r": torch.tensor(mu_r, device="cuda", dtype=torch.float32),
    }


def _transmission(
    source: list[float],
    target: list[float],
    normal: list[float],
    hit: list[float],
    polarization: list[float],
    materials: list[list[tuple[float, float, float, float]]],
    frequency_hz: float,
    material_index: int = 0,
    path_valid: bool = True,
) -> dict[str, torch.Tensor]:
    return ops.field_transmission_sequence(
        torch.tensor([path_valid], device="cuda", dtype=torch.bool),
        torch.tensor([source], device="cuda", dtype=torch.float32),
        torch.tensor([target], device="cuda", dtype=torch.float32),
        torch.tensor([[hit]], device="cuda", dtype=torch.float32),
        torch.tensor([[normal]], device="cuda", dtype=torch.float32),
        torch.tensor([[material_index]], device="cuda", dtype=torch.int32),
        torch.tensor([[True]], device="cuda", dtype=torch.bool),
        torch.tensor([1.0], device="cuda", dtype=torch.float32),
        torch.tensor([polarization], device="cuda", dtype=torch.float32),
        torch.tensor([polarization], device="cuda", dtype=torch.float32),
        **_csr(materials),
        frequency_hz=frequency_hz,
    )


def test_invalid_path_short_circuits_poisoned_payload():
    out = _transmission(
        [math.nan, math.nan, math.nan],
        [0.0, 0.0, -1.0],
        [math.nan, math.nan, math.nan],
        [math.nan, math.nan, math.nan],
        [0.0, 1.0, 0.0],
        [[(0.1, 4.0, 0.05, 1.0)]],
        3.5e9,
        material_index=2**31 - 1,
        path_valid=False,
    )
    torch.cuda.synchronize()
    for tensor in out.values():
        assert torch.count_nonzero(tensor).item() == 0


def _free_space(
    source: list[float],
    target: list[float],
    polarization: list[float],
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    return ops.field_free_space(
        torch.tensor([source], device="cuda", dtype=torch.float32),
        torch.tensor([target], device="cuda", dtype=torch.float32),
        torch.tensor([1.0], device="cuda", dtype=torch.float32),
        torch.tensor([polarization], device="cuda", dtype=torch.float32),
        torch.tensor([polarization], device="cuda", dtype=torch.float32),
        frequency_hz=frequency_hz,
    )


@pytest.mark.parametrize(
    ("source", "target", "hit"),
    (
        # Normal incidence through the z = 0 wall.
        ([0.0, 0.0, 2.0], [0.0, 0.0, -1.5], [0.0, 0.0, 0.0]),
        # 45 degree oblique incidence through the z = 0 wall.
        ([-1.5, 0.0, 1.5], [1.5, 0.0, -1.5], [0.0, 0.0, 0.0]),
    ),
)
def test_vacuum_wall_reproduces_free_space_field(source, target, hit):
    frequency = 3.0e9
    polarization = [0.0, 1.0, 0.0]
    vacuum = [[(0.3, 1.0, 0.0, 1.0)]]

    through = _transmission(
        source, target, [0.0, 0.0, 1.0], hit, polarization, vacuum, frequency
    )
    free = _free_space(source, target, polarization, frequency)

    ratio = through["coefficient"] / free["coefficient"]
    assert torch.abs(torch.abs(ratio) - 1.0).item() <= 1.0e-5
    assert torch.abs(torch.angle(ratio)).item() <= 2.0e-5
    torch.testing.assert_close(
        through["field_vector"], free["field_vector"], rtol=1.0e-4, atol=1.0e-9
    )
    torch.testing.assert_close(through["direction"], free["direction"])
    torch.testing.assert_close(through["path_length_m"], free["path_length_m"])


def test_lossy_wall_attenuates_consistently_with_layer_stack_eval():
    frequency = 3.5e9
    polarization = [0.0, 1.0, 0.0]
    lossy = [[(0.1, 4.0, 0.05, 1.0)]]
    source = [0.0, 0.0, 2.0]
    target = [0.0, 0.0, -2.0]

    through = _transmission(
        source, target, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], polarization, lossy, frequency
    )
    free = _free_space(source, target, polarization, frequency)
    stack = material_functional.em_layer_stack_eval(
        torch.tensor([1.0], device="cuda", dtype=torch.float32),
        torch.tensor([0], device="cuda", dtype=torch.int32),
        **_csr(lossy),
        frequency_hz=frequency,
    )

    # Normal incidence with y-polarization is pure TE: the field magnitude
    # scales by |t_te| relative to free space.
    t_magnitude = math.hypot(
        stack["t_te_real"][0].item(), stack["t_te_imag"][0].item()
    )
    assert t_magnitude < 0.999  # the wall really attenuates
    observed = (
        torch.abs(through["coefficient"]) / torch.abs(free["coefficient"])
    ).item()
    assert abs(observed - t_magnitude) <= 1.0e-5 * max(t_magnitude, 1.0)
    assert through["path_gain"][0].item() < free["path_gain"][0].item()


def test_path_length_and_delay_report_full_straight_length():
    frequency = 2.4e9
    source = [1.0, -2.0, 3.0]
    target = [-1.0, 2.0, -1.0]
    length = math.dist(source, target)
    through = _transmission(
        source,
        target,
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [[(0.2, 3.0, 0.01, 1.0)]],
        frequency,
    )
    assert through["path_length_m"][0].item() == pytest.approx(length, rel=1.0e-6)
    assert through["delay_s"][0].item() == pytest.approx(length / C0, rel=1.0e-6)
