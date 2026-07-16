from __future__ import annotations

from pathlib import Path

import pytest

from ci import check_ops_migration as migration
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.propagation.topology import kernels
from witwin.channel_native.propagation.topology.kernels import blocks, candidates
from witwin.channel_native.runtime import symbols, tensor_contracts
from witwin.channel_native.scene import native_handles
from witwin.channel_native.scene.kernels import rayd_scene


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "ops_migration_manifest.json"

_OWNER_NAMES = (
    "path_diffraction_paths_order1",
    "path_reflection_candidates",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_topology_candidates_is_the_single_object_owner(name: str):
    owner = getattr(candidates, name)

    assert owner.__module__ == candidates.__name__
    assert getattr(kernels, name) is owner
    assert getattr(ops, name) is owner


def test_topology_candidates_preserve_all_frozen_body_contracts():
    manifest = migration.load_manifest(MANIFEST_PATH)
    contracts = {entry["id"]: entry for entry in manifest["contracts"]}
    projections = {
        entry["id"]: entry for entry in manifest["approved_body_projections"]
    }
    prefix = f"{candidates.__name__}."
    definitions = [
        item
        for item in migration.scan_definitions(REPOSITORY_ROOT)
        if item.qualified_name.startswith(prefix)
    ]

    assert {item.terminal_name for item in definitions} == set(_OWNER_NAMES)
    for definition in definitions:
        contract = contracts[definition.terminal_name]
        assert definition.signature == contract["signature"]
        projection = projections.get(definition.terminal_name)
        if projection is None:
            assert definition.body_sha256 == contract["body_sha256"]
            assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]
        else:
            assert definition.projected_native_symbol == projection["native_symbol"]
            assert definition.projected_body_sha256 == contract["body_sha256"]
            assert (
                definition.projected_normalized_ast_sha256
                == contract["normalized_ast_sha256"]
            )


def test_topology_candidates_use_only_canonical_dependencies():
    assert candidates._required_native_op is symbols.required_symbol
    assert candidates.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert candidates._validate_path_block is blocks._validate_path_block
    assert (
        candidates._validate_path_reflection_candidates
        is blocks._validate_path_reflection_candidates
    )
    assert "_raydn_module_handle" not in candidates.__dict__
    assert candidates._raydn_scene_handle_id is native_handles._raydn_scene_handle_id
    for name in _OWNER_NAMES:
        assert getattr(candidates, name).__globals__ is candidates.__dict__
    assert "ops" not in candidates.__dict__


def test_scene_handle_normalizer_is_a_same_object_reexport():
    assert native_handles.__all__ == ["_raydn_scene_handle_id"]
    assert ops._raydn_scene_handle_id is native_handles._raydn_scene_handle_id
    assert rayd_scene._raydn_scene_handle_id is native_handles._raydn_scene_handle_id
