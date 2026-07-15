from __future__ import annotations

from pathlib import Path

import pytest

from ci import check_ops_migration as migration
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.propagation.topology import kernels
from witwin.channel_native.propagation.topology.kernels import blocks, construction
from witwin.channel_native.runtime import symbols, tensor_contracts


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "ops_migration_manifest.json"

_OWNER_NAMES = (
    "deterministic_face_anchor_points",
    "deterministic_face_sequence_chunk",
    "deterministic_los_topology_block",
    "deterministic_mapped_face_sequence_chunk",
    "deterministic_pad_topology_sequences",
    "deterministic_reflection_epc_input_batch",
    "deterministic_repeat_range",
    "deterministic_topology_base_fields",
    "deterministic_topology_default_fields",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_topology_construction_is_the_single_object_owner(name: str):
    owner = getattr(construction, name)

    assert owner.__module__ == construction.__name__
    assert getattr(kernels, name) is owner
    assert getattr(ops, name) is owner


def test_topology_construction_preserves_all_frozen_body_contracts():
    manifest = migration.load_manifest(MANIFEST_PATH)
    contracts = {entry["id"]: entry for entry in manifest["contracts"]}
    prefix = f"{construction.__name__}."
    definitions = [
        item
        for item in migration.scan_definitions(REPOSITORY_ROOT)
        if item.qualified_name.startswith(prefix)
    ]

    assert {item.terminal_name for item in definitions} == set(_OWNER_NAMES)
    for definition in definitions:
        contract = contracts[definition.terminal_name]
        assert definition.signature == contract["signature"]
        assert definition.body_sha256 == contract["body_sha256"]
        assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]


def test_topology_construction_uses_only_canonical_dependencies():
    assert construction._required_native_op is symbols.required_symbol
    assert construction.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert construction._validate_deterministic_topology_block is (
        blocks._validate_deterministic_topology_block
    )
    assert construction._validate_path_block is blocks._validate_path_block
    assert (
        construction.deterministic_los_topology_block.__globals__[
            "_validate_deterministic_topology_block"
        ]
        is blocks._validate_deterministic_topology_block
    )
    assert (
        construction.deterministic_topology_base_fields.__globals__[
            "_validate_path_block"
        ]
        is blocks._validate_path_block
    )
    for name in _OWNER_NAMES:
        assert getattr(construction, name).__globals__ is construction.__dict__
    assert "ops" not in construction.__dict__
