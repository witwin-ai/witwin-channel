from __future__ import annotations

import inspect

import pytest

from witwin.channel.deterministic.kernels import accumulation
from witwin.channel.propagation import fields as public_fields
from witwin.channel.propagation.fields.kernels import deterministic as fields
from witwin.channel.runtime import (
    autograd_contracts,
    symbols,
    tensor_contracts,
    torch_compat,
)


_OWNER_NAMES = (
    "deterministic_delay_to_path_length",
    "deterministic_diffraction_vector_field",
    "deterministic_field_from_power_phase",
    "deterministic_los_field",
    "deterministic_pack_complex",
    "deterministic_phase_from_field",
    "deterministic_phase_from_length",
    "deterministic_reflection_field",
    "deterministic_reflection_sequence_field",
    "deterministic_zero_field_phase",
)

_ACCUMULATION_OWNER_NAMES = (
    "_DeterministicAccumulateFlatAdFunction",
    "deterministic_accumulate_flat",
    "deterministic_accumulate_flat_ad",
    "deterministic_accumulate_flat_backward",
    "deterministic_accumulate_flat_jvp",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_deterministic_fields_is_the_single_object_owner(name: str):
    owner = getattr(fields, name)

    assert owner.__module__ == fields.__name__
    assert getattr(public_fields, name) is owner


def test_deterministic_fields_uses_canonical_runtime_dependencies():
    assert fields.native_extension is symbols.native_extension
    assert fields.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor


@pytest.mark.parametrize("name", _ACCUMULATION_OWNER_NAMES)
def test_deterministic_accumulation_is_the_single_object_owner(name: str):
    owner = getattr(accumulation, name)

    assert owner.__module__ == accumulation.__name__


def test_deterministic_accumulation_uses_canonical_runtime_dependencies():
    assert accumulation.native_extension is symbols.native_extension
    assert accumulation._required_native_op is symbols.required_symbol
    assert accumulation.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert accumulation.torch_compat is torch_compat
    assert (
        accumulation._ad_native_tangent_or_none
        is autograd_contracts._ad_native_tangent_or_none
    )
    assert accumulation._ad_native_tensor is autograd_contracts._ad_native_tensor
    assert accumulation._DETERMINISTIC_ACCUM_FIELDS


def test_deterministic_accumulation_ad_methods_resolve_canonical_siblings():
    function = accumulation._DeterministicAccumulateFlatAdFunction

    assert function.forward.__globals__ is accumulation.__dict__
    assert inspect.unwrap(function.backward).__globals__ is accumulation.__dict__
    assert function.jvp.__globals__ is accumulation.__dict__
    assert (
        inspect.unwrap(function.backward).__globals__[
            "deterministic_accumulate_flat_backward"
        ]
        is accumulation.deterministic_accumulate_flat_backward
    )
    assert (
        function.jvp.__globals__["deterministic_accumulate_flat_jvp"]
        is accumulation.deterministic_accumulate_flat_jvp
    )
    assert (
        accumulation.deterministic_accumulate_flat_ad.__globals__[
            "_DeterministicAccumulateFlatAdFunction"
        ]
        is function
    )
