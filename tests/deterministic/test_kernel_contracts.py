# Copyright Xingyu Chen.
# Tests kernel contracts.

from __future__ import annotations

import inspect

import pytest

from witwin.channel.kernels import deterministic as accumulation
from witwin.channel.kernels import fields as field_kernels
from witwin.channel.propagation import fields as public_fields
from witwin.channel import runtime


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
    owner = getattr(field_kernels, name)

    assert owner.__module__ == field_kernels.__name__
    assert getattr(public_fields, name) is owner


def test_deterministic_fields_uses_canonical_runtime_dependencies():
    assert field_kernels._required_native_op is runtime.required_symbol
    assert not hasattr(field_kernels, "native_extension")
    assert field_kernels.validate_cuda_tensor is runtime.validate_cuda_tensor


@pytest.mark.parametrize("name", _ACCUMULATION_OWNER_NAMES)
def test_deterministic_accumulation_is_the_single_object_owner(name: str):
    owner = getattr(accumulation, name)

    assert owner.__module__ == accumulation.__name__


def test_deterministic_accumulation_uses_canonical_runtime_dependencies():
    # The raw accessor is not a solver dependency: every probe goes through
    # runtime.required_symbol, and ci/check_import_graph.py rejects a solver
    # that imports ``native_extension`` at all.
    assert not hasattr(accumulation, "native_extension")
    assert accumulation._required_native_op is runtime.required_symbol
    assert accumulation.validate_cuda_tensor is runtime.validate_cuda_tensor
    assert accumulation.disable_functorch is runtime.disable_functorch
    assert (
        accumulation._ad_native_tangent_or_none
        is runtime._ad_native_tangent_or_none
    )
    assert accumulation._ad_native_tensor is runtime._ad_native_tensor
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