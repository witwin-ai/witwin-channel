from __future__ import annotations


import pytest

from witwin.channel_native.scattering import kernels
from witwin.channel_native.scattering.kernels import functional
from witwin.channel_native.runtime import symbols, tensor_contracts


_OWNER_NAMES = (
    "scattering_event_probabilities",
    "scattering_table_eval",
    "scattering_table_pdf",
    "scattering_table_sample",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_scattering_functional_is_the_single_object_owner(name: str):
    owner = getattr(functional, name)

    assert owner.__module__ == functional.__name__
    assert getattr(kernels, name) is owner


def test_scattering_functional_uses_canonical_runtime_dependencies():
    assert functional._required_native_op is symbols.required_symbol
    assert functional.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
