from __future__ import annotations

import inspect

import pytest
import torch

from witwin.channel.montecarlo.basic.kernels import capacity as capacity_kernels
from witwin.channel.montecarlo.basic.kernels.capacity import (
    mc_capacity_failure_component_maps_sanitize,
)
from witwin.channel.runtime import create_capacity_failure_state


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_FIELDS = ("los", "reflection", "diffraction", "transmission", "scattering")


def _maps(*, requires_grad: bool = False) -> tuple[torch.Tensor, ...]:
    return tuple(
        (
            torch.arange(12, device="cuda", dtype=torch.float32).reshape(1, 3, 4)
            + index
        ).requires_grad_(requires_grad)
        for index in range(5)
    )


def _noncontiguous_maps() -> tuple[torch.Tensor, ...]:
    return tuple(
        (
            torch.arange(12, device="cuda", dtype=torch.float32).reshape(1, 4, 3)
            + 10.0 * index
        ).transpose(1, 2)
        for index in range(5)
    )


def test_component_map_sanitizer_success_is_exact_on_current_stream() -> None:
    values = _maps()
    state = create_capacity_failure_state(values[0])
    bits_pointer = state.bits.data_ptr()
    stream = torch.cuda.Stream()
    # The op launches on the caller's current stream; input readiness is the
    # caller's responsibility. The maps and the failure state were produced on
    # the default stream, so a correct caller orders the side stream behind it
    # before launching - without this the kernel raced its own inputs and the
    # test failed roughly once per thousand full-suite runs.
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        output = mc_capacity_failure_component_maps_sanitize(
            *values, failure_state=state
        )
    stream.synchronize()

    assert tuple(output) == _FIELDS
    assert state.bits.data_ptr() == bits_pointer
    assert state.bits.tolist() == [0]
    for name, value in zip(_FIELDS, values, strict=True):
        assert output[name].is_contiguous()
        assert torch.equal(output[name], value)


def test_component_map_sanitizer_failure_makes_all_five_maps_positive_zero() -> None:
    values = _maps()
    state = create_capacity_failure_state(values[0])
    state.bits.fill_(1)
    output = mc_capacity_failure_component_maps_sanitize(*values, failure_state=state)

    for name in _FIELDS:
        assert torch.count_nonzero(output[name]).item() == 0
        assert not torch.signbit(output[name]).any().item()
    assert state.bits.tolist() == [1]


@pytest.mark.parametrize("failed", [False, True])
def test_component_map_sanitizer_vjp_jvp_and_duality(failed: bool) -> None:
    values = _maps(requires_grad=True)
    state = create_capacity_failure_state(values[0])
    if failed:
        state.bits.fill_(1)
    weights = tuple(
        torch.full_like(value, index + 0.25) for index, value in enumerate(values)
    )
    tangents = tuple(
        torch.full_like(value, 0.1 * (index + 1)) for index, value in enumerate(values)
    )

    output = mc_capacity_failure_component_maps_sanitize(*values, failure_state=state)
    objective = sum(
        (output[name] * weight).sum()
        for name, weight in zip(_FIELDS, weights, strict=True)
    )
    gradients = torch.autograd.grad(objective, values)

    with torch.autograd.forward_ad.dual_level():
        dual_values = tuple(
            torch.autograd.forward_ad.make_dual(value.detach(), tangent)
            for value, tangent in zip(values, tangents, strict=True)
        )
        dual_output = mc_capacity_failure_component_maps_sanitize(
            *dual_values, failure_state=state
        )
        output_tangents = tuple(
            torch.autograd.forward_ad.unpack_dual(dual_output[name]).tangent
            for name in _FIELDS
        )

    if failed:
        for gradient, tangent in zip(gradients, output_tangents, strict=True):
            assert torch.count_nonzero(gradient).item() == 0
            assert tangent is not None
            assert torch.count_nonzero(tangent).item() == 0
    else:
        for gradient, weight in zip(gradients, weights, strict=True):
            assert torch.equal(gradient, weight)
        for actual, expected in zip(output_tangents, tangents, strict=True):
            assert actual is not None
            assert torch.equal(actual, expected)

    lhs = sum(
        (tangent * weight).sum()
        for tangent, weight in zip(output_tangents, weights, strict=True)
        if tangent is not None
    )
    rhs = sum(
        (gradient * tangent).sum()
        for gradient, tangent in zip(gradients, tangents, strict=True)
    )
    torch.testing.assert_close(lhs, rhs, rtol=0.0, atol=0.0)


def test_component_map_sanitizer_zero_rows_preserve_family_shapes() -> None:
    values = tuple(
        torch.empty((0, 3, 4), device="cuda", dtype=torch.float32).requires_grad_()
        for _ in _FIELDS
    )
    state = create_capacity_failure_state(values[0])

    output = mc_capacity_failure_component_maps_sanitize(*values, failure_state=state)
    gradients = torch.autograd.grad(
        tuple(output[name] for name in _FIELDS),
        values,
        grad_outputs=tuple(torch.empty_like(value) for value in output.values()),
    )
    with torch.autograd.forward_ad.dual_level():
        dual_values = tuple(
            torch.autograd.forward_ad.make_dual(value.detach(), torch.empty_like(value))
            for value in values
        )
        dual_output = mc_capacity_failure_component_maps_sanitize(
            *dual_values, failure_state=state
        )
        output_tangents = tuple(
            torch.autograd.forward_ad.unpack_dual(dual_output[name]).tangent
            for name in _FIELDS
        )
    torch.cuda.synchronize()

    assert state.bits.tolist() == [0]
    for name in _FIELDS:
        assert output[name].shape == (0, 3, 4)
        assert output[name].is_contiguous()
    for gradient, tangent in zip(gradients, output_tangents, strict=True):
        assert gradient.shape == (0, 3, 4)
        assert gradient.is_contiguous()
        assert tangent is not None
        assert tangent.shape == (0, 3, 4)
        assert tangent.is_contiguous()


@pytest.mark.parametrize("failed", [False, True])
def test_component_map_sanitizer_noncontiguous_vjp_jvp_are_exact(
    failed: bool,
) -> None:
    values = _maps(requires_grad=True)
    state = create_capacity_failure_state(values[0])
    if failed:
        state.bits.fill_(1)
    gradients_in = _noncontiguous_maps()
    tangents_in = _noncontiguous_maps()
    assert all(not value.is_contiguous() for value in (*gradients_in, *tangents_in))

    output = mc_capacity_failure_component_maps_sanitize(*values, failure_state=state)
    gradients = torch.autograd.grad(
        tuple(output[name] for name in _FIELDS),
        values,
        grad_outputs=gradients_in,
    )
    with torch.autograd.forward_ad.dual_level():
        dual_values = tuple(
            torch.autograd.forward_ad.make_dual(value.detach(), tangent)
            for value, tangent in zip(values, tangents_in, strict=True)
        )
        dual_output = mc_capacity_failure_component_maps_sanitize(
            *dual_values, failure_state=state
        )
        output_tangents = tuple(
            torch.autograd.forward_ad.unpack_dual(dual_output[name]).tangent
            for name in _FIELDS
        )

    if failed:
        for gradient, tangent in zip(gradients, output_tangents, strict=True):
            assert torch.count_nonzero(gradient).item() == 0
            assert tangent is not None
            assert torch.count_nonzero(tangent).item() == 0
    else:
        for actual, expected in zip(gradients, gradients_in, strict=True):
            assert torch.equal(actual, expected)
        for actual, expected in zip(output_tangents, tangents_in, strict=True):
            assert actual is not None
            assert torch.equal(actual, expected)


def test_component_map_sanitizer_forwards_exact_failure_storage_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _maps()
    state = create_capacity_failure_state(values[0])
    calls: list[tuple[str, tuple[object, ...]]] = []

    def required(name: str):
        def native(*args: object) -> dict[str, torch.Tensor]:
            calls.append((name, args))
            outputs = tuple(value.clone() for value in args[1:])
            return dict(zip(_FIELDS, outputs, strict=True))

        return native

    monkeypatch.setattr(capacity_kernels, "_required_native_op", required)
    output = mc_capacity_failure_component_maps_sanitize(*values, failure_state=state)

    assert tuple(output) == _FIELDS
    assert len(calls) == 1
    assert calls[0][0] == "mc_capacity_failure_component_maps_sanitize"
    assert calls[0][1][0] is state.bits


@pytest.mark.parametrize(
    "missing_symbol",
    (
        "mc_capacity_failure_component_maps_sanitize",
        "mc_capacity_failure_component_maps_sanitize_backward",
        "mc_capacity_failure_component_maps_sanitize_jvp",
    ),
)
def test_component_map_sanitizer_family_has_no_fallback(
    missing_symbol: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = capacity_kernels._required_native_op

    def required(name: str):
        if name == missing_symbol:
            raise RuntimeError(f"missing required native symbol {name}")
        return original(name)

    monkeypatch.setattr(capacity_kernels, "_required_native_op", required)
    values = _maps(requires_grad=missing_symbol.endswith("backward"))
    state = create_capacity_failure_state(values[0])

    with pytest.raises(RuntimeError, match=missing_symbol):
        if missing_symbol.endswith("backward"):
            output = mc_capacity_failure_component_maps_sanitize(
                *values, failure_state=state
            )
            sum(output.values()).sum().backward()
        elif missing_symbol.endswith("jvp"):
            with torch.autograd.forward_ad.dual_level():
                dual_values = tuple(
                    torch.autograd.forward_ad.make_dual(value, torch.ones_like(value))
                    for value in values
                )
                mc_capacity_failure_component_maps_sanitize(
                    *dual_values, failure_state=state
                )
        else:
            mc_capacity_failure_component_maps_sanitize(*values, failure_state=state)


def test_component_map_sanitizer_custom_function_uses_only_native_family() -> None:
    source = inspect.getsource(
        capacity_kernels._McCapacityFailureComponentMapsSanitizeFunction
    )
    assert "_mc_capacity_failure_component_maps_sanitize_native(" in source
    assert "_mc_capacity_failure_component_maps_sanitize_backward_native(" in source
    assert "_mc_capacity_failure_component_maps_sanitize_jvp_native(" in source
    for forbidden in ("torch.where", "torch.zeros", ".item(", ".cpu(", ".numpy("):
        assert forbidden not in source
