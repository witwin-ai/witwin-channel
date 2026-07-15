from __future__ import annotations

from pathlib import Path

import pytest

from ci import check_ops_migration as migration
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.propagation.topology import kernels
from witwin.channel_native.propagation.topology.kernels import primitives
from witwin.channel_native.runtime import symbols, tensor_contracts


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "ops_migration_manifest.json"

_OWNER_NAMES = (
    "core_pack_int2",
    "deterministic_component_counts",
    "deterministic_diffraction_state_pack",
    "deterministic_diffraction_state_pack_selected",
    "deterministic_selected_edge_count",
    "mc_selected_edge_indices",
    "path_concat_vec3",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_topology_primitives_is_the_single_object_owner(name: str):
    owner = getattr(primitives, name)

    assert owner.__module__ == primitives.__name__
    assert getattr(kernels, name) is owner
    assert getattr(ops, name) is owner


def test_topology_primitives_preserve_all_frozen_body_contracts():
    manifest = migration.load_manifest(MANIFEST_PATH)
    contracts = {entry["id"]: entry for entry in manifest["contracts"]}
    prefix = f"{primitives.__name__}."
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


def test_topology_primitives_use_only_canonical_runtime_dependencies():
    assert primitives._required_native_op is symbols.required_symbol
    assert primitives.native_extension is symbols.native_extension
    assert primitives.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    for name in _OWNER_NAMES:
        assert getattr(primitives, name).__globals__ is primitives.__dict__
    assert "ops" not in primitives.__dict__
