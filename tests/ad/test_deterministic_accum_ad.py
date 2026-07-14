"""Direct AD checks of the native deterministic flat-accumulation Function.

Strict float64 torch.autograd.gradcheck of
ops.deterministic_accumulate_flat_ad on a small path fixture (coherent
|sum field|^2 totals and incoherent power sums with the sqrt pseudo-field),
a jvp-vs-vjp inner-product duality check, float32 forward parity against the
primal native accumulator, and the frozen-gate contract: rows outside the
materialized component slots or the tx/rx ranges receive exactly zero
gradient.
"""

from __future__ import annotations

import pytest
import torch

from witwin.channel_native.core.kernels import ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for accumulator AD"
)

_NUM_TX = 2
_NUM_RX = 2

_OUTPUT_FIELDS = (
    "power_total",
    "field_total_real",
    "field_total_imag",
    "component_power",
    "component_field_real",
    "component_field_imag",
)

# Rows 0-4 and 8 are kept (cell (0, 0) carries a same-slot collision on slot
# 0 plus a second slot, so the cell nonlinearities mix several paths). Row 5
# has a component id outside the three materialized slots (a transmission
# row the Python side accumulates separately), row 6 an out-of-range tx and
# row 7 an out-of-range rx: the accumulator must drop all three.
_TX_ID = (0, 0, 0, 1, 1, 0, -1, 0, 0)
_RX_ID = (0, 0, 1, 1, 0, 1, 0, 3, 0)
_COMPONENT_ID = (0, 1, 1, 2, 0, 5, 1, 2, 0)
_DROPPED_ROWS = (5, 6, 7)


def _fixture(dtype: torch.dtype):
    device = torch.device("cuda")
    tx_id = torch.tensor(_TX_ID, device=device, dtype=torch.int32)
    rx_id = torch.tensor(_RX_ID, device=device, dtype=torch.int32)
    component_id = torch.tensor(_COMPONENT_ID, device=device, dtype=torch.int32)
    generator = torch.Generator(device="cpu").manual_seed(3)
    count = int(tx_id.numel())
    path_gain = (
        torch.rand((count,), generator=generator, dtype=torch.float64) + 0.25
    ).to(device=device, dtype=dtype)
    field_real = torch.randn(
        (count,), generator=generator, dtype=torch.float64
    ).to(device=device, dtype=dtype)
    field_imag = torch.randn(
        (count,), generator=generator, dtype=torch.float64
    ).to(device=device, dtype=dtype)
    return tx_id, rx_id, component_id, path_gain, field_real, field_imag


def _apply(tx_id, rx_id, component_id, gain, real, imag, coherent):
    out = ops.deterministic_accumulate_flat_ad(
        tx_id,
        rx_id,
        component_id,
        gain,
        real,
        imag,
        num_tx=_NUM_TX,
        num_rx=_NUM_RX,
        coherent=coherent,
    )
    return tuple(out[name] for name in _OUTPUT_FIELDS)


@pytest.mark.parametrize("coherent", (True, False))
def test_accumulate_flat_gradcheck_strict_float64(coherent):
    tx_id, rx_id, component_id, path_gain, field_real, field_imag = _fixture(
        torch.float64
    )

    def func(gain, real, imag):
        return _apply(tx_id, rx_id, component_id, gain, real, imag, coherent)

    leaves = tuple(
        value.clone().requires_grad_(True)
        for value in (path_gain, field_real, field_imag)
    )
    assert torch.autograd.gradcheck(
        func, leaves, eps=1.0e-6, atol=1.0e-9, rtol=1.0e-5, nondet_tol=1.0e-10
    )
    assert torch.autograd.gradcheck(
        func,
        leaves,
        eps=1.0e-6,
        atol=1.0e-9,
        rtol=1.0e-5,
        nondet_tol=1.0e-10,
        check_forward_ad=True,
        check_backward_ad=False,
        check_undefined_grad=False,
        check_batched_grad=False,
    )


@pytest.mark.parametrize("coherent", (True, False))
def test_accumulate_flat_jvp_vjp_duality(coherent):
    """<u, J t> from forward mode must equal <J^T u, t> from reverse mode."""

    tx_id, rx_id, component_id, path_gain, field_real, field_imag = _fixture(
        torch.float64
    )
    generator = torch.Generator(device="cpu").manual_seed(11)

    def randn_like(reference):
        return torch.randn(
            reference.shape, generator=generator, dtype=torch.float64
        ).cuda()

    tangents = tuple(
        randn_like(value) for value in (path_gain, field_real, field_imag)
    )

    leaves = tuple(
        value.clone().requires_grad_(True)
        for value in (path_gain, field_real, field_imag)
    )
    outputs = _apply(tx_id, rx_id, component_id, *leaves, coherent)
    cotangents = tuple(randn_like(value) for value in outputs)
    loss = sum(
        (cotangent * output).sum()
        for cotangent, output in zip(cotangents, outputs, strict=True)
    )
    loss.backward()
    vjp_inner = sum(
        (leaf.grad * tangent).sum()
        for leaf, tangent in zip(leaves, tangents, strict=True)
    )

    with torch.autograd.forward_ad.dual_level():
        duals = tuple(
            torch.autograd.forward_ad.make_dual(value.clone(), tangent)
            for value, tangent in zip(
                (path_gain, field_real, field_imag), tangents, strict=True
            )
        )
        dual_outputs = _apply(tx_id, rx_id, component_id, *duals, coherent)
        jvp_inner = sum(
            (
                cotangent
                * torch.autograd.forward_ad.unpack_dual(output).tangent
            ).sum()
            for cotangent, output in zip(cotangents, dual_outputs, strict=True)
        )

    torch.testing.assert_close(jvp_inner, vjp_inner, rtol=1.0e-9, atol=1.0e-12)


@pytest.mark.parametrize("coherent", (True, False))
def test_accumulate_flat_ad_forward_matches_primal_float32(coherent):
    tx_id, rx_id, component_id, path_gain, field_real, field_imag = _fixture(
        torch.float32
    )
    ad_out = ops.deterministic_accumulate_flat_ad(
        tx_id,
        rx_id,
        component_id,
        path_gain,
        field_real,
        field_imag,
        num_tx=_NUM_TX,
        num_rx=_NUM_RX,
        coherent=coherent,
    )
    primal = ops.deterministic_accumulate_flat(
        tx_id,
        rx_id,
        component_id,
        path_gain,
        field_real,
        field_imag,
        num_tx=_NUM_TX,
        num_rx=_NUM_RX,
        coherent=coherent,
    )
    for name in _OUTPUT_FIELDS:
        # Same kernel, same inputs; the only permitted difference is the
        # atomic scatter's reassociation order inside a shared cell.
        torch.testing.assert_close(
            ad_out[name], primal[name], rtol=1.0e-6, atol=1.0e-7
        )


@pytest.mark.parametrize("coherent", (True, False))
def test_accumulate_flat_dropped_rows_get_exact_zero_gradient(coherent):
    tx_id, rx_id, component_id, path_gain, field_real, field_imag = _fixture(
        torch.float64
    )
    leaves = tuple(
        value.clone().requires_grad_(True)
        for value in (path_gain, field_real, field_imag)
    )
    outputs = _apply(tx_id, rx_id, component_id, *leaves, coherent)
    sum(output.sum() for output in outputs).backward()
    dropped = torch.tensor(_DROPPED_ROWS, device="cuda", dtype=torch.int64)
    for leaf in leaves:
        assert leaf.grad is not None
        assert bool((leaf.grad[dropped] == 0.0).all())
    # The kept rows of at least one leaf must carry a live gradient.
    assert any(float(leaf.grad.abs().max()) > 0.0 for leaf in leaves)
