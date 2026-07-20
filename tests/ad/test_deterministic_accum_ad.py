"""Direct AD checks of the native deterministic flat-accumulation Function.

Strict float64 torch.autograd.gradcheck of
ops.deterministic_accumulate_flat_ad on a small path fixture covering all
six materialized slots (coherent |sum field|^2 totals over the los /
reflection / diffraction / transmission / coupled field slots, the
power-domain scattering slot, and incoherent power sums with the sqrt
pseudo-field), a jvp-vs-vjp inner-product duality check, float32 forward
parity against the primal native accumulator, and the frozen-gate contract:
invalid capacity rows are skipped before poison IDs/numerical payloads, and
invalid or out-of-range rows receive exactly zero gradient/tangent contribution.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from witwin.channel_native.deterministic.kernels import accumulation as ops

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

# Rows 0-5, 8-9 and 10-11 are kept and cover every materialized slot: cell
# (0, 0) carries a same-slot collision on slot 0 plus reflection and
# scattering rows, so the cell nonlinearities mix several paths; row 5 is a
# transmission row (slot 3, part of the coherent field sum) and row 9 a
# scattering row (slot 4, power-domain in both modes). Rows 10 (cid 3, R->D)
# and 11 (cid 4, D->R) are coupled rows that both land in the coherent coupled
# slot 5 and collide in cell (1, 1), exercising the E_RD + E_DR sum (ADR-011).
# Row 6 has an out-of-range tx and row 7 an out-of-range rx: the accumulator
# must drop both.
_TX_ID = (0, 0, 0, 1, 1, 0, -1, 0, 0, 0, 1, 1)
_RX_ID = (0, 0, 1, 1, 0, 1, 0, 3, 0, 0, 1, 1)
_COMPONENT_ID = (0, 1, 1, 2, 0, 5, 1, 2, 0, 6, 3, 4)
_DROPPED_ROWS = (6, 7)

_ALL_VALID_BASELINE_SHA256 = {
    (
        torch.float32,
        True,
    ): "29c4e0ef1406dfa4e145967dc088535730e59e9538fee1ab4c6fcfe42a380fbe",
    (
        torch.float32,
        False,
    ): "d68266d382aeb6ae7fdf6479660e29eea039065bb7645eeafcc7aad25ebf303d",
    (
        torch.float64,
        True,
    ): "74d28d783c3f5d6613a305eeb66f63644e3373000a6d36272b22409813bfe7e7",
    (
        torch.float64,
        False,
    ): "70ca8b66f1bbc6a97fb39582ca7cd02cc1037a55a9e9891e4b3a5261453b669e",
}


def _fixture(dtype: torch.dtype):
    device = torch.device("cuda")
    valid = torch.ones((len(_TX_ID),), device=device, dtype=torch.bool)
    tx_id = torch.tensor(_TX_ID, device=device, dtype=torch.int32)
    rx_id = torch.tensor(_RX_ID, device=device, dtype=torch.int32)
    component_id = torch.tensor(_COMPONENT_ID, device=device, dtype=torch.int32)
    generator = torch.Generator(device="cpu").manual_seed(3)
    count = int(tx_id.numel())
    path_gain = (
        torch.rand((count,), generator=generator, dtype=torch.float64) + 0.25
    ).to(device=device, dtype=dtype)
    field_real = torch.randn((count,), generator=generator, dtype=torch.float64).to(
        device=device, dtype=dtype
    )
    field_imag = torch.randn((count,), generator=generator, dtype=torch.float64).to(
        device=device, dtype=dtype
    )
    return valid, tx_id, rx_id, component_id, path_gain, field_real, field_imag


def _apply(valid, tx_id, rx_id, component_id, gain, real, imag, coherent):
    out = ops.deterministic_accumulate_flat_ad(
        valid,
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


def _output_sha256(out: dict[str, torch.Tensor]) -> str:
    payload = b"".join(
        out[name].detach().contiguous().cpu().numpy().tobytes()
        for name in _OUTPUT_FIELDS
    )
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("coherent", (True, False))
def test_accumulate_flat_all_valid_preserves_frozen_exact_hash(dtype, coherent):
    fixture = _fixture(dtype)
    if dtype == torch.float32:
        out = ops.deterministic_accumulate_flat(
            *fixture,
            num_tx=_NUM_TX,
            num_rx=_NUM_RX,
            coherent=coherent,
        )
    else:
        out = ops.deterministic_accumulate_flat_ad(
            *fixture,
            num_tx=_NUM_TX,
            num_rx=_NUM_RX,
            coherent=coherent,
        )
    assert _output_sha256(out) == _ALL_VALID_BASELINE_SHA256[(dtype, coherent)]


@pytest.mark.parametrize("coherent", (True, False))
def test_accumulate_flat_gradcheck_strict_float64(coherent):
    valid, tx_id, rx_id, component_id, path_gain, field_real, field_imag = _fixture(
        torch.float64
    )

    def func(gain, real, imag):
        return _apply(valid, tx_id, rx_id, component_id, gain, real, imag, coherent)

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

    valid, tx_id, rx_id, component_id, path_gain, field_real, field_imag = _fixture(
        torch.float64
    )
    generator = torch.Generator(device="cpu").manual_seed(11)

    def randn_like(reference):
        return torch.randn(
            reference.shape, generator=generator, dtype=torch.float64
        ).cuda()

    tangents = tuple(randn_like(value) for value in (path_gain, field_real, field_imag))

    leaves = tuple(
        value.clone().requires_grad_(True)
        for value in (path_gain, field_real, field_imag)
    )
    outputs = _apply(valid, tx_id, rx_id, component_id, *leaves, coherent)
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
        dual_outputs = _apply(valid, tx_id, rx_id, component_id, *duals, coherent)
        jvp_inner = sum(
            (cotangent * torch.autograd.forward_ad.unpack_dual(output).tangent).sum()
            for cotangent, output in zip(cotangents, dual_outputs, strict=True)
        )

    torch.testing.assert_close(jvp_inner, vjp_inner, rtol=1.0e-9, atol=1.0e-12)


@pytest.mark.parametrize("coherent", (True, False))
def test_accumulate_flat_ad_forward_matches_primal_float32(coherent):
    valid, tx_id, rx_id, component_id, path_gain, field_real, field_imag = _fixture(
        torch.float32
    )
    ad_out = ops.deterministic_accumulate_flat_ad(
        valid,
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
        valid,
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
        torch.testing.assert_close(ad_out[name], primal[name], rtol=1.0e-6, atol=1.0e-7)


@pytest.mark.parametrize("coherent", (True, False))
def test_accumulate_flat_dropped_rows_get_exact_zero_gradient(coherent):
    valid, tx_id, rx_id, component_id, path_gain, field_real, field_imag = _fixture(
        torch.float64
    )
    leaves = tuple(
        value.clone().requires_grad_(True)
        for value in (path_gain, field_real, field_imag)
    )
    outputs = _apply(valid, tx_id, rx_id, component_id, *leaves, coherent)
    sum(output.sum() for output in outputs).backward()
    dropped = torch.tensor(_DROPPED_ROWS, device="cuda", dtype=torch.int64)
    for leaf in leaves:
        assert leaf.grad is not None
        assert bool((leaf.grad[dropped] == 0.0).all())
    # The kept rows of at least one leaf must carry a live gradient.
    assert any(float(leaf.grad.abs().max()) > 0.0 for leaf in leaves)


@pytest.mark.parametrize("coherent", (True, False))
def test_accumulate_flat_sparse_validity_skips_poison_before_row_reads(coherent):
    valid = torch.tensor([True, False, True, False], device="cuda")
    tx_id = torch.tensor([0, -(2**31), 1, 2**31 - 1], device="cuda", dtype=torch.int32)
    rx_id = torch.tensor([0, 2**31 - 1, 1, -(2**31)], device="cuda", dtype=torch.int32)
    component_id = torch.tensor(
        [0, -(2**31), 1, 2**31 - 1], device="cuda", dtype=torch.int32
    )
    gain = torch.tensor([1.25, float("nan"), 2.5, float("inf")], device="cuda")
    real = torch.tensor([0.5, float("inf"), -0.75, float("nan")], device="cuda")
    imag = torch.tensor([-0.25, float("nan"), 1.5, float("-inf")], device="cuda")
    out = ops.deterministic_accumulate_flat(
        valid,
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
    live = ops.deterministic_accumulate_flat(
        valid[valid].contiguous(),
        tx_id[valid].contiguous(),
        rx_id[valid].contiguous(),
        component_id[valid].contiguous(),
        gain[valid].contiguous(),
        real[valid].contiguous(),
        imag[valid].contiguous(),
        num_tx=_NUM_TX,
        num_rx=_NUM_RX,
        coherent=coherent,
    )
    for name in _OUTPUT_FIELDS:
        assert torch.equal(out[name], live[name])
        assert bool(torch.isfinite(out[name]).all())


@pytest.mark.parametrize("coherent", (True, False))
def test_accumulate_flat_all_invalid_is_zero_with_poison_primal_and_ad(coherent):
    valid = torch.zeros((3,), device="cuda", dtype=torch.bool)
    tx_id = torch.tensor([-(2**31), 2**31 - 1, -1], device="cuda", dtype=torch.int32)
    rx_id = torch.tensor([2**31 - 1, -(2**31), -1], device="cuda", dtype=torch.int32)
    component_id = torch.tensor(
        [2**31 - 1, -(2**31), -1], device="cuda", dtype=torch.int32
    )
    leaves = tuple(
        tensor.requires_grad_(True)
        for tensor in (
            torch.tensor(
                [float("nan"), float("inf"), float("-inf")],
                device="cuda",
                dtype=torch.float64,
            ),
            torch.tensor(
                [float("inf"), float("nan"), float("-inf")],
                device="cuda",
                dtype=torch.float64,
            ),
            torch.tensor(
                [float("-inf"), float("inf"), float("nan")],
                device="cuda",
                dtype=torch.float64,
            ),
        )
    )
    outputs = _apply(valid, tx_id, rx_id, component_id, *leaves, coherent)
    for output in outputs:
        assert bool((output == 0.0).all())
    sum(output.sum() for output in outputs).backward()
    for leaf in leaves:
        assert leaf.grad is not None
        assert bool((leaf.grad == 0.0).all())


@pytest.mark.parametrize("coherent", (True, False))
def test_accumulate_flat_invalid_tangents_contribute_exact_zero(coherent):
    valid, tx_id, rx_id, component_id, gain, real, imag = _fixture(torch.float64)
    valid = valid.clone()
    valid[1::2] = False
    leaves = tuple(
        value.clone().requires_grad_(True) for value in (gain, real, imag)
    )
    outputs = _apply(valid, tx_id, rx_id, component_id, *leaves, coherent)
    sum(output.sum() for output in outputs).backward()
    for leaf in leaves:
        assert leaf.grad is not None
        assert bool((leaf.grad[~valid] == 0.0).all())

    tangents = tuple(
        torch.linspace(0.25, 1.75, gain.numel(), device="cuda", dtype=torch.float64)
        for _ in range(3)
    )
    poisoned = tuple(tangent.clone() for tangent in tangents)
    poisoned[0][~valid] = float("nan")
    poisoned[1][~valid] = float("inf")
    poisoned[2][~valid] = float("-inf")

    def jvp_values(tangent_values):
        with torch.autograd.forward_ad.dual_level():
            duals = tuple(
                torch.autograd.forward_ad.make_dual(primal, tangent)
                for primal, tangent in zip(
                    (gain, real, imag), tangent_values, strict=True
                )
            )
            dual_outputs = _apply(valid, tx_id, rx_id, component_id, *duals, coherent)
            return tuple(
                torch.autograd.forward_ad.unpack_dual(output).tangent.clone()
                for output in dual_outputs
            )

    zeroed = tuple(tangent.masked_fill(~valid, 0.0) for tangent in tangents)
    poison_jvp = jvp_values(poisoned)
    zero_jvp = jvp_values(zeroed)
    for actual, expected in zip(poison_jvp, zero_jvp, strict=True):
        assert torch.equal(actual, expected)
        assert bool(torch.isfinite(actual).all())


def test_accumulate_flat_uses_current_cuda_stream():
    valid, tx_id, rx_id, component_id, gain, real, imag = _fixture(torch.float32)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = ops.deterministic_accumulate_flat(
            valid,
            tx_id,
            rx_id,
            component_id,
            gain,
            real,
            imag,
            num_tx=_NUM_TX,
            num_rx=_NUM_RX,
            coherent=True,
        )
        finished = torch.cuda.Event()
        finished.record(stream)
    finished.synchronize()
    assert _output_sha256(out) == _ALL_VALID_BASELINE_SHA256[(torch.float32, True)]


def test_accumulate_flat_missing_native_symbol_fails_without_fallback(monkeypatch):
    valid, tx_id, rx_id, component_id, gain, real, imag = _fixture(torch.float32)
    monkeypatch.setattr(ops, "native_extension", lambda: None)
    with pytest.raises(RuntimeError, match="CUDA kernel is required"):
        ops.deterministic_accumulate_flat(
            valid,
            tx_id,
            rx_id,
            component_id,
            gain,
            real,
            imag,
            num_tx=_NUM_TX,
            num_rx=_NUM_RX,
            coherent=True,
        )
