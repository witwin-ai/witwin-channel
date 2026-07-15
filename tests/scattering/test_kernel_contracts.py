from __future__ import annotations

from pathlib import Path

import pytest

from ci import check_ops_migration as migration
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.scattering import kernels
from witwin.channel_native.scattering.kernels import functional
from witwin.channel_native.runtime import symbols, tensor_contracts


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "ops_migration_manifest.json"

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
    assert getattr(ops, name) is owner


def test_scattering_functional_preserves_all_frozen_body_contracts():
    manifest = migration.load_manifest(MANIFEST_PATH)
    contracts = {entry["id"]: entry for entry in manifest["contracts"]}
    prefix = f"{functional.__name__}."
    definitions = [
        item
        for item in migration.scan_definitions(REPOSITORY_ROOT)
        if item.qualified_name.startswith(prefix)
    ]

    assert len(definitions) == 4
    for definition in definitions:
        contract = contracts[definition.terminal_name]
        assert definition.signature == contract["signature"]
        assert definition.body_sha256 == contract["body_sha256"]
        assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]


def test_scattering_functional_uses_canonical_runtime_dependencies():
    assert functional._required_native_op is symbols.required_symbol
    assert functional.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
