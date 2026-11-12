# Copyright Xingyu Chen.
# Cell accumulation gives the same bits for any row order and stays near the exact sum.

from __future__ import annotations

import pytest
import torch

from witwin.channel.kernels.deterministic import (
    deterministic_accumulate_flat,
    deterministic_accumulate_flat_jvp,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

_NUM_TX = 2
_NUM_RX = 5
_ROWS = 20_000


def _rows(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = _ROWS
    # Mixed magnitudes and signs, many rows per cell: a float atomic sum of these is order dependent.
    magnitude = torch.pow(10.0, torch.empty(count).uniform_(-9.0, -3.0, generator=generator))
    sign = torch.where(torch.rand(count, generator=generator) < 0.5, -1.0, 1.0)
    rows = {
        "valid": torch.rand(count, generator=generator) < 0.95,
        "tx_id": torch.randint(0, _NUM_TX, (count,), generator=generator, dtype=torch.int32),
        "rx_id": torch.randint(0, _NUM_RX, (count,), generator=generator, dtype=torch.int32),
        "component_id": torch.tensor([0, 1, 2, 3, 5, 6, 7], dtype=torch.int32)[
            torch.randint(0, 7, (count,), generator=generator)
        ],
        "path_gain": magnitude.float(),
        "field_real": (sign * magnitude.sqrt()).float(),
        "field_imag": (torch.randn(count, generator=generator) * magnitude.sqrt()).float(),
    }
    return {name: value.cuda() for name, value in rows.items()}


def _accumulate(rows: dict[str, torch.Tensor], order: torch.Tensor, coherent: bool) -> dict:
    permuted = {name: value[order].contiguous() for name, value in rows.items()}
    return deterministic_accumulate_flat(
        permuted["valid"], permuted["tx_id"], permuted["rx_id"], permuted["component_id"],
        permuted["path_gain"], permuted["field_real"], permuted["field_imag"],
        num_tx=_NUM_TX, num_rx=_NUM_RX, coherent=coherent,
    )


@pytest.mark.parametrize("coherent", [True, False])
def test_cell_sums_are_identical_for_every_row_order_and_repeat(coherent):
    rows = _rows(7)
    generator = torch.Generator(device="cpu").manual_seed(11)
    identity = torch.arange(_ROWS, device="cuda")
    reference = _accumulate(rows, identity, coherent)
    orders = [identity, identity, torch.flip(identity, (0,))]
    orders += [torch.randperm(_ROWS, generator=generator).cuda() for _ in range(3)]
    for order in orders:
        result = _accumulate(rows, order, coherent)
        for name, value in reference.items():
            assert torch.equal(result[name], value), name


def test_cell_sums_match_the_exact_sum_to_float_rounding():
    rows = _rows(3)
    result = _accumulate(rows, torch.arange(_ROWS, device="cuda"), coherent=True)
    slot_of = {0: 0, 1: 1, 2: 2, 3: 5, 5: 3, 6: 4, 7: 5}
    exact = torch.zeros(6, _NUM_TX, _NUM_RX, dtype=torch.float64)
    cpu = {name: value.cpu() for name, value in rows.items()}
    keep = cpu["valid"]
    slots = torch.tensor([slot_of[int(c)] for c in cpu["component_id"][keep]])
    index = (slots, cpu["tx_id"][keep].long(), cpu["rx_id"][keep].long())
    exact.index_put_(index, cpu["field_real"][keep].double(), accumulate=True)
    got = result["component_field_real"].cpu().double()
    # Each cell differs from the exact sum by at most one float rounding of the result plus half a
    # fixed-point quantum per term; the quantum is 2^(e - 47) for 20,000 rows, with 2^e at most
    # twice the cell's largest magnitude.
    flat = (index[0] * _NUM_TX + index[1]) * _NUM_RX + index[2]
    values = cpu["field_real"][keep].double()
    largest = torch.zeros(exact.numel(), dtype=torch.float64).scatter_reduce(
        0, flat, values.abs(), reduce="amax").view_as(exact)
    terms = torch.bincount(flat, minlength=exact.numel()).double().view_as(exact)
    bound = exact.abs() * 2.0**-24 + largest * terms * 2.0**-47
    assert torch.all((got - exact).abs() <= bound + 1e-30)


def test_tangent_cell_sums_are_identical_for_every_row_order():
    rows = _rows(5)
    identity = torch.arange(_ROWS, device="cuda")
    generator = torch.Generator(device="cpu").manual_seed(2)
    primal = _accumulate(rows, identity, coherent=True)
    tangents = {
        name: torch.randn(_ROWS, generator=generator).float().cuda()
        for name in ("path_gain", "field_real", "field_imag")
    }

    def jvp(order: torch.Tensor) -> dict:
        def pick(value: torch.Tensor) -> torch.Tensor:
            return value[order].contiguous()

        return deterministic_accumulate_flat_jvp(
            pick(rows["valid"]), pick(rows["tx_id"]), pick(rows["rx_id"]),
            pick(rows["component_id"]), primal["component_field_real"],
            primal["component_field_imag"], primal["power_total"],
            tangent_path_gain=pick(tangents["path_gain"]),
            tangent_field_real=pick(tangents["field_real"]),
            tangent_field_imag=pick(tangents["field_imag"]),
            num_tx=_NUM_TX, num_rx=_NUM_RX, coherent=True,
        )

    reference = jvp(identity)
    for _ in range(3):
        result = jvp(torch.randperm(_ROWS, generator=generator).cuda())
        for name, value in reference.items():
            assert torch.equal(result[name], value), name
