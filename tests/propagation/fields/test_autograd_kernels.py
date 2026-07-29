# Copyright Xingyu Chen.
# Tests autograd kernels.

from __future__ import annotations

import inspect

import pytest

from witwin.channel.kernels import fields as field_kernels
from witwin.channel.propagation import fields
from witwin.channel import runtime


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
    owner = getattr(field_kernels, name)

    assert owner.__module__ == field_kernels.__name__
    assert getattr(fields, name) is owner


@pytest.mark.parametrize("name", _PROJECTION_OWNER_NAMES)
def test_fields_projection_autograd_is_the_single_object_owner(name: str):
    owner = getattr(field_kernels, name)

    assert owner.__module__ == field_kernels.__name__
    assert getattr(fields, name) is owner


def test_fields_autograd_uses_canonical_runtime_dependencies():
    assert field_kernels._required_native_op is runtime.required_symbol
    assert field_kernels.disable_functorch is runtime.disable_functorch
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
        assert getattr(field_kernels, name) is getattr(runtime, name)


def test_fields_autograd_uses_canonical_functional_companions():
    # The forward companions and the shared field tuples are defined in the
    # same module as the wrappers that dispatch them, so the wrappers reach
    # them through module globals rather than through a second binding.
    for name in (
        "field_free_space_backward",
        "field_free_space_jvp",
        "field_reflection_sequence_backward",
        "field_reflection_sequence_jvp",
        "field_transmission_sequence_backward",
        "field_transmission_sequence_jvp",
    ):
        assert getattr(field_kernels, name).__globals__ is field_kernels.__dict__
    for name in (
        "_COUPLED_OUTPUT_FIELDS",
        "_FIELD_AD_OUTPUT_FIELDS",
        "_FIELD_AD_TANGENT_FIELDS",
        "_WEDGE_OUTPUT_FIELDS",
    ):
        assert isinstance(getattr(field_kernels, name), tuple)


def test_transmission_autograd_requires_explicit_path_valid_first():
    parameters = tuple(
        inspect.signature(field_kernels.field_transmission_sequence_ad).parameters
    )
    assert parameters[0] == "path_valid"
    function_parameters = tuple(
        inspect.signature(
            field_kernels._FieldTransmissionSequenceAdFunction.forward
        ).parameters
    )
    assert function_parameters[0] == "path_valid"


def test_diffraction_wedge_autograd_requires_explicit_valid_first():
    parameters = tuple(
        inspect.signature(field_kernels.field_diffraction_wedge_ad).parameters
    )
    assert parameters[0] == "valid"
    function_parameters = tuple(
        inspect.signature(field_kernels._FieldDiffractionWedgeAdFunction.forward).parameters
    )
    assert function_parameters[0] == "valid"