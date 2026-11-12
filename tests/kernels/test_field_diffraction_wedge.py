# Copyright Xingyu Chen.
# Tests the primal wedge diffraction facade against its differentiable twin.

"""Tests the primal wedge diffraction facade against its differentiable twin."""

from __future__ import annotations

import math

import pytest
import torch

from witwin.channel.kernels import fields as field_kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _knife_edge_rows() -> tuple[torch.Tensor, ...]:
    """Three rows on a dielectric knife edge along +z; the middle row is inactive."""

    device = "cuda"
    rows = 3

    def column(value: float) -> torch.Tensor:
        return torch.full((rows,), value, device=device, dtype=torch.float32)

    def vectors(*values: tuple[float, float, float]) -> torch.Tensor:
        return torch.tensor(values, device=device, dtype=torch.float32)

    valid = torch.tensor([True, False, True], device=device)
    source = vectors((-2.0, 1.0, 0.0), (-2.0, 1.0, 0.0), (-2.0, 1.0, 0.0))
    target = vectors((2.0, -1.5, 0.3), (2.0, -1.5, 0.3), (1.5, -2.0, -0.2))
    edge = (
        torch.zeros((rows, 3), device=device),
        vectors(*([(0.0, 0.0, 1.0)] * rows)),
        column(-5.0),
        column(5.0),
        vectors(*([(0.0, 1.0, 0.0)] * rows)),
        vectors(*([(0.0, -1.0, 0.0)] * rows)),
        column(2.0 * math.pi),
    )
    face = (
        torch.ones((rows,), device=device, dtype=torch.bool),
        column(5.0),
        column(0.01),
        column(1.0),
        column(1.0),
    )
    return (valid, source, target, *edge, *face, *face, column(1.0))


def test_primal_wedge_facade_matches_the_differentiable_primal_bitwise():
    rows = _knife_edge_rows()
    primal = field_kernels.field_diffraction_wedge(*rows, frequency_hz=3.0e9)
    differentiable = field_kernels.field_diffraction_wedge_ad(*rows, frequency=3.0e9)

    assert set(primal) == {"field_vector", "direction"}
    for name, value in primal.items():
        assert torch.equal(value, differentiable[name]), name
    field = primal["field_vector"]
    assert torch.isfinite(torch.view_as_real(field)).all()
    assert bool((field[1] == 0).all()) and bool((primal["direction"][1] == 0).all())
    active = field[[0, 2]]
    assert bool((active.abs().sum(dim=1) > 0).all())
    norms = torch.linalg.vector_norm(primal["direction"][[0, 2]], dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_primal_wedge_facade_rejects_a_valid_mask_of_the_wrong_length():
    rows = list(_knife_edge_rows())
    rows[0] = rows[0][:2]
    with pytest.raises(ValueError, match="valid must have shape"):
        field_kernels.field_diffraction_wedge(*rows, frequency_hz=3.0e9)
