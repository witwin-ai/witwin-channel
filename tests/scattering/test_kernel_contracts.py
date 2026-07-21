from __future__ import annotations


import inspect

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

_VALID_FIRST_NAMES = (
    "scattering_table_eval",
    "scattering_table_eval_backward",
    "scattering_table_eval_jvp",
    "scattering_table_pdf",
    "scattering_table_sample",
    "scattering_ensemble_eval",
    "scattering_ensemble_eval_backward",
    "scattering_ensemble_eval_jvp",
    "scattering_patch_integral_eval",
    "scattering_patch_integral_eval_backward",
    "scattering_patch_integral_eval_jvp",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_scattering_functional_is_the_single_object_owner(name: str):
    owner = getattr(functional, name)

    assert owner.__module__ == functional.__name__
    assert getattr(kernels, name) is owner


@pytest.mark.parametrize("name", _VALID_FIRST_NAMES)
def test_non_chain_row_operations_require_explicit_valid_first(name: str):
    parameters = inspect.signature(getattr(functional, name)).parameters

    assert next(iter(parameters)) == "valid"


def test_scattering_functional_uses_canonical_runtime_dependencies():
    assert functional._required_native_op is symbols.required_symbol
    assert functional.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
