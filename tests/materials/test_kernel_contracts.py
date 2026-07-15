from __future__ import annotations

from pathlib import Path

from ci import check_ops_migration as migration
import pytest
import witwin.channel_native.materials as public_materials
from witwin.channel_native.core import materials as core_materials
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.materials import kernels
from witwin.channel_native.materials.kernels import autograd, contracts, functional
from witwin.channel_native.runtime import (
    autograd_contracts,
    symbols,
    tensor_contracts,
    torch_compat,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "ops_migration_manifest.json"

_FUNCTIONAL_OWNER_NAMES = (
    "bdpt_face_material_tensors",
    "bdpt_face_material_tensors_from_host",
    "em_layer_stack_backward",
    "em_layer_stack_eval",
    "em_layer_stack_jvp",
    "mc_face_material_tensors",
)

_AUTOGRAD_OWNER_NAMES = (
    "_EmLayerStackAdFunction",
    "em_layer_stack_ad",
)


def test_material_kernel_contract_has_one_same_object_owner():
    owner = contracts._validate_layer_csr

    assert owner.__module__ == contracts.__name__
    assert kernels._validate_layer_csr is owner
    assert ops._validate_layer_csr is owner
    assert contracts.validate_layer_csr is owner
    assert kernels.validate_layer_csr is owner
    assert public_materials.validate_layer_csr is owner


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


@pytest.mark.parametrize("name", _FUNCTIONAL_OWNER_NAMES)
def test_material_functional_is_the_single_object_owner(name: str):
    owner = getattr(functional, name)

    assert owner.__module__ == functional.__name__
    assert getattr(kernels, name) is owner
    assert getattr(ops, name) is owner


@pytest.mark.parametrize("name", _AUTOGRAD_OWNER_NAMES)
def test_material_autograd_is_the_single_object_owner(name: str):
    owner = getattr(autograd, name)

    assert owner.__module__ == autograd.__name__
    assert getattr(kernels, name) is owner
    assert getattr(ops, name) is owner


@pytest.mark.parametrize(
    ("module", "expected_count"),
    ((functional, 6), (autograd, 5)),
)
def test_material_kernels_preserve_all_frozen_body_contracts(module, expected_count):
    manifest = migration.load_manifest(MANIFEST_PATH)
    contracts_by_id = {entry["id"]: entry for entry in manifest["contracts"]}
    prefix = f"{module.__name__}."
    definitions = [
        item
        for item in migration.scan_definitions(REPOSITORY_ROOT)
        if item.qualified_name.startswith(prefix)
    ]

    assert len(definitions) == expected_count
    for definition in definitions:
        contract = contracts_by_id[definition.terminal_name]
        assert definition.signature == contract["signature"]
        assert definition.body_sha256 == contract["body_sha256"]
        assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]


def test_material_functional_uses_canonical_dependencies():
    assert functional._required_native_op is symbols.required_symbol
    assert functional.native_extension is symbols.native_extension
    assert functional.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert functional._validate_layer_csr is contracts._validate_layer_csr


def test_material_autograd_uses_canonical_dependencies():
    assert autograd.torch_compat is torch_compat
    for name in (
        "_ad_checked_tangent",
        "_ad_frequency_grad",
        "_ad_frequency_tangent",
        "_ad_frequency_value",
        "_ad_native_tangent_or_none",
        "_ad_native_tensor",
        "_ad_reject_fixed_inputs",
        "_ad_reject_fixed_tangents",
    ):
        assert getattr(autograd, name) is getattr(autograd_contracts, name)
    for name in (
        "_EM_LAYER_STACK_FIELDS",
        "em_layer_stack_backward",
        "em_layer_stack_eval",
        "em_layer_stack_jvp",
    ):
        assert getattr(autograd, name) is getattr(functional, name)


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
