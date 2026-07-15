from __future__ import annotations

from pathlib import Path

import pytest

from ci import check_ops_migration as migration
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.propagation import fields
from witwin.channel_native.propagation.fields import kernels
from witwin.channel_native.propagation.fields.kernels import autograd, functional
from witwin.channel_native.runtime import autograd_contracts, symbols, torch_compat


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "ops_migration_manifest.json"

_OWNER_NAMES = (
    "_CoupledRdPrepareAdFunction",
    "_FieldCoupledRdAdFunction",
    "_FieldDiffractionWedgeAdFunction",
    "_FieldFreeSpaceAdFunction",
    "_FieldProjectComplex3AdFunction",
    "_FieldReflectionSequenceAdFunction",
    "_FieldTransmissionSequenceAdFunction",
    "coupled_rd_prepare_ad",
    "field_coupled_rd_ad",
    "field_diffraction_wedge_ad",
    "field_free_space_ad",
    "field_project_complex3_ad",
    "field_reflection_sequence_ad",
    "field_transmission_sequence_ad",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_fields_autograd_is_the_single_object_owner(name: str):
    owner = getattr(autograd, name)

    assert owner.__module__ == autograd.__name__
    assert getattr(kernels, name) is owner
    assert getattr(fields, name) is owner
    assert getattr(ops, name) is owner


def test_fields_autograd_preserves_all_frozen_body_contracts():
    manifest = migration.load_manifest(MANIFEST_PATH)
    contracts = {entry["id"]: entry for entry in manifest["contracts"]}
    prefix = f"{autograd.__name__}."
    definitions = [
        item
        for item in migration.scan_definitions(REPOSITORY_ROOT)
        if item.qualified_name.startswith(prefix)
    ]

    assert len(definitions) == 36
    for definition in definitions:
        contract = contracts[definition.terminal_name]
        assert definition.signature == contract["signature"]
        assert definition.body_sha256 == contract["body_sha256"]
        assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]


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
