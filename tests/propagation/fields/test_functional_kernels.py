# Copyright Xingyu Chen.
# Tests functional kernels.

from __future__ import annotations

import inspect

import pytest

from witwin.channel import materials
from witwin.channel.kernels import fields as field_kernels
from witwin.channel.kernels import materials as material_contracts
from witwin.channel.propagation import fields
from witwin.channel import runtime


_OWNER_NAMES = (
    "field_coupled_rd",
    "field_diffraction_wedge",
    "field_free_space",
    "field_free_space_backward",
    "field_free_space_jvp",
    "field_project_complex3",
    "field_reflection_sequence",
    "field_reflection_sequence_backward",
    "field_reflection_sequence_jvp",
    "field_transmission_sequence",
    "field_transmission_sequence_backward",
    "field_transmission_sequence_jvp",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_fields_functional_is_the_single_object_owner(name: str):
    owner = getattr(field_kernels, name)

    assert owner.__module__ == field_kernels.__name__
    assert getattr(fields, name) is owner


def test_fields_functional_uses_canonical_dependencies():
    owner = material_contracts._validate_layer_csr

    assert field_kernels._required_native_op is runtime.required_symbol
    assert field_kernels.validate_cuda_tensor is runtime.validate_cuda_tensor
    assert field_kernels._validate_layer_csr is owner
    assert material_contracts.validate_layer_csr is owner
    assert materials.validate_layer_csr is owner


def test_transmission_family_requires_explicit_path_valid_first():
    for name in (
        "field_transmission_sequence",
        "field_transmission_sequence_backward",
        "field_transmission_sequence_jvp",
    ):
        parameters = tuple(inspect.signature(getattr(field_kernels, name)).parameters)
        assert parameters[0] == "path_valid"


def test_diffraction_wedge_requires_explicit_valid_first():
    parameters = tuple(inspect.signature(field_kernels.field_diffraction_wedge).parameters)
    assert parameters[0] == "valid"