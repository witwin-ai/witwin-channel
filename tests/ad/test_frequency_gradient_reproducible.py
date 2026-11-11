# Copyright Xingyu Chen.
# Frequency cotangents reduce in a fixed order: bitwise reproducible and equal to the row sum.

"""Frequency cotangents reduce in a fixed order: bitwise reproducible and equal to the row sum."""

import pytest
import torch

from witwin.channel.kernels.fields import field_free_space_backward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _rows(count: int):
    generator = torch.Generator(device="cuda").manual_seed(7)
    source = torch.rand((count, 3), device="cuda", generator=generator) * 10.0
    target = torch.rand((count, 3), device="cuda", generator=generator) * 10.0 + 20.0
    power = torch.ones((count,), device="cuda")
    pol = torch.zeros((count, 3), device="cuda")
    pol[:, 2] = 1.0
    grad_gain = torch.rand((count,), device="cuda", generator=generator) - 0.5
    return source, target, power, pol, grad_gain


def _grad_frequency(rows, count: int) -> torch.Tensor:
    source, target, power, pol, grad_gain = rows
    out = field_free_space_backward(
        source[:count], target[:count], power[:count], pol[:count], pol[:count],
        frequency_hz=3.0e9, grad_path_gain=grad_gain[:count].contiguous(),
    )
    return out["grad_frequency"]


def test_free_space_frequency_gradient_is_bitwise_reproducible() -> None:
    count = 1 << 21
    rows = _rows(count)
    first = _grad_frequency(rows, count)
    for _ in range(4):
        assert torch.equal(_grad_frequency(rows, count), first)


def test_free_space_frequency_gradient_equals_the_sum_of_single_rows() -> None:
    count = 4096
    rows = _rows(count)
    total = _grad_frequency(rows, count)
    source, target, power, pol, grad_gain = rows
    per_row = torch.stack([
        field_free_space_backward(
            source[i : i + 1], target[i : i + 1], power[i : i + 1], pol[i : i + 1],
            pol[i : i + 1], frequency_hz=3.0e9, grad_path_gain=grad_gain[i : i + 1].contiguous(),
        )["grad_frequency"][0]
        for i in range(0, count, 64)
    ])
    sub_total = _grad_frequency(
        tuple(value[::64].contiguous() for value in rows), count // 64
    )
    torch.testing.assert_close(sub_total[0], per_row.double().sum().float(), rtol=1e-5, atol=0.0)
    assert torch.isfinite(total).all()
