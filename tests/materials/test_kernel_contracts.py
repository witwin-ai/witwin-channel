from __future__ import annotations

from pathlib import Path

from ci import check_ops_migration as migration
import witwin.channel_native.materials as public_materials
from witwin.channel_native.core import materials as core_materials
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.materials import kernels
from witwin.channel_native.materials.kernels import contracts
from witwin.channel_native.runtime import tensor_contracts


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "ops_migration_manifest.json"


def test_material_kernel_contract_has_one_same_object_owner():
    owner = contracts._validate_layer_csr

    assert owner.__module__ == contracts.__name__
    assert kernels._validate_layer_csr is owner
    assert ops._validate_layer_csr is owner


def test_material_kernel_contract_preserves_the_frozen_body():
    manifest = migration.load_manifest(MANIFEST_PATH)
    contracts_by_id = {entry["id"]: entry for entry in manifest["contracts"]}
    prefix = f"{contracts.__name__}."
    definitions = [
        item
        for item in migration.scan_definitions(REPOSITORY_ROOT)
        if item.qualified_name.startswith(prefix)
    ]

    assert len(definitions) == 1
    definition = definitions[0]
    contract = contracts_by_id[definition.terminal_name]
    assert definition.signature == contract["signature"]
    assert definition.body_sha256 == contract["body_sha256"]
    assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]


def test_material_kernel_contract_uses_canonical_tensor_validation():
    assert contracts.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor


def test_materials_package_preserves_public_object_identity():
    expected_exports = (
        "DebyeModel",
        "Dielectric",
        "DispersiveMaterial",
        "ITUMaterial",
        "Layer",
        "LossyDielectric",
        "PerfectConductor",
        "PhaseScreen",
        "PhysicalSurface",
        "Roughness",
        "SurfaceAssignment",
        "TabulatedPermittivity",
    )

    assert tuple(public_materials.__all__) == expected_exports
    for name in expected_exports:
        assert getattr(public_materials, name) is getattr(core_materials, name)
