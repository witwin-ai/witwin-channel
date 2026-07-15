from __future__ import annotations

from pathlib import Path

import pytest

from ci import check_ops_migration as migration
from witwin.channel_native import materials
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.materials.kernels import contracts as material_contracts
from witwin.channel_native.propagation import fields
from witwin.channel_native.propagation.fields import kernels
from witwin.channel_native.propagation.fields.kernels import functional
from witwin.channel_native.runtime import symbols, tensor_contracts


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "ops_migration_manifest.json"

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


def test_fields_functional_preserves_all_frozen_body_contracts():
    manifest = migration.load_manifest(MANIFEST_PATH)
    contracts = {entry["id"]: entry for entry in manifest["contracts"]}
    prefix = f"{functional.__name__}."
    definitions = [
        item
        for item in migration.scan_definitions(REPOSITORY_ROOT)
        if item.qualified_name.startswith(prefix)
    ]

    assert len(definitions) == 12
    for definition in definitions:
        contract = contracts[definition.terminal_name]
        assert definition.signature == contract["signature"]
        assert definition.body_sha256 == contract["body_sha256"]
        assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]


def test_fields_functional_uses_canonical_dependencies():
    owner = material_contracts._validate_layer_csr

    assert functional._required_native_op is symbols.required_symbol
    assert functional.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert functional._validate_layer_csr is owner
    assert material_contracts.validate_layer_csr is owner
    assert materials.validate_layer_csr is owner
