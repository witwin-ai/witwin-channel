from __future__ import annotations


import pytest

from witwin.channel_native import materials
from witwin.channel_native.propagation.fields.kernels import functional as ops
from witwin.channel_native.materials.kernels import contracts as material_contracts
from witwin.channel_native.propagation import fields
from witwin.channel_native.propagation.fields import kernels
from witwin.channel_native.propagation.fields.kernels import functional
from witwin.channel_native.runtime import symbols, tensor_contracts


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
    owner = getattr(functional, name)

    assert owner.__module__ == functional.__name__
    assert getattr(kernels, name) is owner
    assert getattr(fields, name) is owner
    assert getattr(ops, name) is owner


def test_fields_functional_uses_canonical_dependencies():
    owner = material_contracts._validate_layer_csr

    assert functional._required_native_op is symbols.required_symbol
    assert functional.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert functional._validate_layer_csr is owner
    assert material_contracts.validate_layer_csr is owner
    assert materials.validate_layer_csr is owner
