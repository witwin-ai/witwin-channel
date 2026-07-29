# Copyright Xingyu Chen.
# Tests kernel contracts.

from __future__ import annotations


import inspect

import pytest

from witwin.channel.kernels import scattering as kernels
from witwin.channel import runtime


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
    owner = getattr(kernels, name)

    assert owner.__module__ == kernels.__name__
    assert kernels.__all__.count(name) == 1


@pytest.mark.parametrize("name", _VALID_FIRST_NAMES)
def test_non_chain_row_operations_require_explicit_valid_first(name: str):
    parameters = inspect.signature(getattr(kernels, name)).parameters

    assert next(iter(parameters)) == "valid"


def test_scattering_functional_uses_canonical_runtime_dependencies():
    assert kernels._required_native_op is runtime.required_symbol
    assert kernels.validate_cuda_tensor is runtime.validate_cuda_tensor