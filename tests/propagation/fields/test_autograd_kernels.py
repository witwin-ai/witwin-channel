from __future__ import annotations

import inspect

import pytest

from witwin.channel_native.propagation.fields.kernels import autograd as ops
from witwin.channel_native.propagation import fields
from witwin.channel_native.propagation.fields import kernels
from witwin.channel_native.propagation.fields.kernels import (
    autograd,
    autograd_projection,
    functional,
)
from witwin.channel_native.runtime import autograd_contracts, symbols, torch_compat


_OWNER_NAMES = (
    "_CoupledRdPrepareAdFunction",
    "_FieldCoupledRdAdFunction",
    "_FieldDiffractionWedgeAdFunction",
    "_FieldFreeSpaceAdFunction",
    "_FieldReflectionSequenceAdFunction",
    "_FieldTransmissionSequenceAdFunction",
    "coupled_rd_prepare_ad",
    "field_coupled_rd_ad",
    "field_diffraction_wedge_ad",
    "field_free_space_ad",
    "field_reflection_sequence_ad",
    "field_transmission_sequence_ad",
)

_PROJECTION_OWNER_NAMES = (
    "_FieldProjectComplex3AdFunction",
    "field_project_complex3_ad",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_fields_autograd_is_the_single_object_owner(name: str):
    owner = getattr(autograd, name)

    assert owner.__module__ == autograd.__name__
    assert getattr(kernels, name) is owner
    assert getattr(fields, name) is owner
    assert getattr(ops, name) is owner


@pytest.mark.parametrize("name", _PROJECTION_OWNER_NAMES)
def test_fields_projection_autograd_is_the_single_object_owner(name: str):
    owner = getattr(autograd_projection, name)

    assert owner.__module__ == autograd_projection.__name__
    assert getattr(kernels, name) is owner
    assert getattr(fields, name) is owner
    assert not hasattr(autograd, name)


def test_fields_autograd_uses_canonical_runtime_dependencies():
    assert autograd._required_native_op is symbols.required_symbol
    assert autograd.torch_compat is torch_compat
    for name in (
        "_ad_checked_tangent",
        "_ad_frequency_grad",
        "_ad_frequency_tangent",
        "_ad_frequency_value",
        "_ad_geometry_live",
        "_ad_geometry_tangent",
        "_ad_native_tangent_or_none",
        "_ad_native_tensor",
        "_ad_reject_fixed_inputs",
        "_ad_reject_fixed_tangents",
    ):
        assert getattr(autograd, name) is getattr(autograd_contracts, name)


def test_fields_autograd_uses_canonical_functional_companions():
    for name in (
        "field_free_space_backward",
        "field_free_space_jvp",
        "field_reflection_sequence_backward",
        "field_reflection_sequence_jvp",
        "field_transmission_sequence_backward",
        "field_transmission_sequence_jvp",
    ):
        assert getattr(autograd, name) is getattr(functional, name)
    for name in (
        "_COUPLED_OUTPUT_FIELDS",
        "_FIELD_AD_OUTPUT_FIELDS",
        "_FIELD_AD_TANGENT_FIELDS",
        "_WEDGE_OUTPUT_FIELDS",
    ):
        assert getattr(autograd, name) is getattr(functional, name)


def test_transmission_autograd_requires_explicit_path_valid_first():
    parameters = tuple(
        inspect.signature(autograd.field_transmission_sequence_ad).parameters
    )
    assert parameters[0] == "path_valid"
    function_parameters = tuple(
        inspect.signature(
            autograd._FieldTransmissionSequenceAdFunction.forward
        ).parameters
    )
    assert function_parameters[0] == "path_valid"


def test_diffraction_wedge_autograd_requires_explicit_valid_first():
    parameters = tuple(
        inspect.signature(autograd.field_diffraction_wedge_ad).parameters
    )
    assert parameters[0] == "valid"
    function_parameters = tuple(
        inspect.signature(autograd._FieldDiffractionWedgeAdFunction.forward).parameters
    )
    assert function_parameters[0] == "valid"
