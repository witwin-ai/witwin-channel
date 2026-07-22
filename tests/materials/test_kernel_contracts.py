from __future__ import annotations


import pytest
import torch
import witwin.channel.materials as public_materials
from witwin.channel.core import materials as core_materials
from witwin.channel.materials import kernels
from witwin.channel.materials.kernels import autograd, contracts, functional
from witwin.channel.runtime import (
    autograd_contracts,
    symbols,
    tensor_contracts,
    torch_compat,
)


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
    assert contracts.validate_layer_csr is owner
    assert kernels.validate_layer_csr is owner
    assert public_materials.validate_layer_csr is owner


def test_material_kernel_contract_uses_canonical_tensor_validation():
    assert contracts.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor


@pytest.mark.parametrize("name", _FUNCTIONAL_OWNER_NAMES)
def test_material_functional_is_the_single_object_owner(name: str):
    owner = getattr(functional, name)

    assert owner.__module__ == functional.__name__
    assert getattr(kernels, name) is owner


@pytest.mark.parametrize("name", _AUTOGRAD_OWNER_NAMES)
def test_material_autograd_is_the_single_object_owner(name: str):
    owner = getattr(autograd, name)

    assert owner.__module__ == autograd.__name__
    assert getattr(kernels, name) is owner


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_bdpt_face_material_tensor_facade_dispatches_and_validates_schema():
    exported = functional.bdpt_face_material_tensors(
        torch.tensor([2.5, 4.0], device="cuda", dtype=torch.float32),
        torch.tensor([0.01, 0.02], device="cuda", dtype=torch.float32),
        torch.tensor([1.0, 1.2], device="cuda", dtype=torch.float32),
        torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32),
    )

    assert set(exported) == {"eps_r", "sigma_e", "mu_r", "gain", "valid"}
    assert all(value.shape == (3,) for value in exported.values())
    assert exported["valid"].dtype == torch.bool


def test_host_material_facade_rejects_inconsistent_contracts():
    with pytest.raises(ValueError, match="must not be empty"):
        functional.bdpt_face_material_tensors_from_host((), (), (), ())
    with pytest.raises(ValueError, match="sigma_e must match"):
        functional.bdpt_face_material_tensors_from_host((2.5,), (), (1.0,), (0,))
    with pytest.raises(ValueError, match="mu_r must match"):
        functional.bdpt_face_material_tensors_from_host((2.5,), (0.01,), (), (0,))
    with pytest.raises(ValueError, match="entries must reference"):
        functional.bdpt_face_material_tensors_from_host(
            (2.5,), (0.01,), (1.0,), (1,)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_mc_face_material_facade_rejects_mismatched_sigma_shape():
    with pytest.raises(ValueError, match="sigma_e must match"):
        functional.mc_face_material_tensors(
            torch.tensor([2.5], device="cuda", dtype=torch.float32),
            torch.tensor([0.01, 0.02], device="cuda", dtype=torch.float32),
            torch.tensor([1.0], device="cuda", dtype=torch.float32),
            torch.tensor([0], device="cuda", dtype=torch.int32),
        )


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
