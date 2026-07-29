# Copyright Xingyu Chen.
# Tests field source amplitude scale.

"""Direct contract tests for the ADR-039 source-amplitude scale.

The operation is ``path_field_vector = field_vector * sqrt(max(tx_power, 0))``
with the same amplitude expression the field transport kernels use. It is
linear in the field vector and its amplitude is real, so the VJP and the JVP
are the same scale, and ``tx_power`` carries no derivative.
"""

from __future__ import annotations

import pytest
import torch

from witwin.channel.kernels import fields as field_kernels
from witwin.channel import runtime


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _case(rows: int = 6, seed: int = 11):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    field_vector = torch.complex(
        torch.randn(rows, 3, generator=generator, device="cuda"),
        torch.randn(rows, 3, generator=generator, device="cuda"),
    )
    tx_power = torch.rand(rows, generator=generator, device="cuda") * 9.0
    return field_vector, tx_power


def test_source_amplitude_scale_contract_shapes_and_dtypes() -> None:
    field_vector, tx_power = _case()

    out = field_kernels.field_source_amplitude_scale(field_vector, tx_power)

    assert set(out) == {"path_field_vector"}
    scaled = out["path_field_vector"]
    assert scaled.shape == field_vector.shape
    assert scaled.dtype == torch.complex64
    assert scaled.device == field_vector.device
    assert scaled.is_contiguous()


def test_source_amplitude_scale_applies_the_transport_amplitude() -> None:
    field_vector, tx_power = _case()

    scaled = field_kernels.field_source_amplitude_scale(field_vector, tx_power)[
        "path_field_vector"
    ]

    amplitude = tx_power.clamp_min(0.0).sqrt()
    torch.testing.assert_close(scaled, field_vector * amplitude[:, None])


def test_zero_and_negative_power_publish_an_inert_row() -> None:
    field_vector, tx_power = _case()
    tx_power = tx_power.clone()
    tx_power[0] = 0.0
    tx_power[1] = -3.0

    scaled = field_kernels.field_source_amplitude_scale(field_vector, tx_power)[
        "path_field_vector"
    ]

    assert torch.count_nonzero(scaled[:2]).item() == 0


def test_source_amplitude_scale_rejects_a_row_count_mismatch() -> None:
    field_vector, tx_power = _case()

    with pytest.raises(ValueError):
        field_kernels.field_source_amplitude_scale(field_vector, tx_power[:-1])


def test_backward_and_jvp_reproduce_the_forward_scale() -> None:
    field_vector, tx_power = _case()

    forward = field_kernels.field_source_amplitude_scale(field_vector, tx_power)[
        "path_field_vector"
    ]
    backward = field_kernels.field_source_amplitude_scale_backward(
        tx_power, field_vector
    )["grad_field_vector"]
    jvp = field_kernels.field_source_amplitude_scale_jvp(tx_power, field_vector)[
        "tangent_path_field_vector"
    ]

    torch.testing.assert_close(backward, forward, rtol=0.0, atol=0.0)
    torch.testing.assert_close(jvp, forward, rtol=0.0, atol=0.0)


def test_autograd_vjp_matches_the_native_backward() -> None:
    field_vector, tx_power = _case()
    leaf = field_vector.clone().requires_grad_(True)

    scaled = field_kernels.field_source_amplitude_scale_ad(leaf, tx_power)
    cotangent = torch.complex(
        torch.randn_like(field_vector.real), torch.randn_like(field_vector.real)
    )
    scaled.backward(cotangent)

    expected = field_kernels.field_source_amplitude_scale_backward(
        tx_power, cotangent
    )["grad_field_vector"]
    assert leaf.grad is not None
    torch.testing.assert_close(leaf.grad, expected, rtol=0.0, atol=0.0)


def test_autograd_jvp_matches_the_native_jvp() -> None:
    field_vector, tx_power = _case()
    tangent = torch.complex(
        torch.randn_like(field_vector.real), torch.randn_like(field_vector.real)
    )

    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(field_vector, tangent)
        scaled = field_kernels.field_source_amplitude_scale_ad(dual, tx_power)
        primal, out_tangent = torch.autograd.forward_ad.unpack_dual(scaled)

    expected = field_kernels.field_source_amplitude_scale_jvp(tx_power, tangent)[
        "tangent_path_field_vector"
    ]
    assert out_tangent is not None
    torch.testing.assert_close(out_tangent, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        primal,
        field_kernels.field_source_amplitude_scale(field_vector, tx_power)[
            "path_field_vector"
        ],
        rtol=0.0,
        atol=0.0,
    )


def test_tx_power_carries_no_derivative() -> None:
    field_vector, tx_power = _case()
    power_leaf = tx_power.clone().requires_grad_(True)

    scaled = field_kernels.field_source_amplitude_scale_ad(field_vector, power_leaf)
    with pytest.raises(NotImplementedError):
        scaled.real.sum().backward()

    tangent = torch.ones_like(tx_power)
    with torch.autograd.forward_ad.dual_level():
        dual_power = torch.autograd.forward_ad.make_dual(tx_power, tangent)
        with pytest.raises(NotImplementedError):
            field_kernels.field_source_amplitude_scale_ad(field_vector, dual_power)


def test_source_amplitude_scale_requires_the_native_symbol(monkeypatch) -> None:
    field_vector, tx_power = _case()
    monkeypatch.setattr(
        runtime, "required_symbol", _missing_symbol, raising=True
    )
    monkeypatch.setattr(
        field_kernels, "_required_native_op", _missing_symbol, raising=True
    )

    with pytest.raises(RuntimeError):
        field_kernels.field_source_amplitude_scale(field_vector, tx_power)


def _missing_symbol(name: str):
    raise RuntimeError(f"native symbol {name} is unavailable")