# Copyright Xingyu Chen.
# Tests kernel contracts.

from __future__ import annotations


import pytest
import torch
import witwin.channel.materials as public_materials
from witwin.channel.kernels import materials as kernels
from witwin.channel.kernels import spec
from witwin.channel import runtime


_FUNCTIONAL_OWNER_NAMES = (
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
    owner = kernels._validate_layer_csr

    assert owner.__module__ == kernels.__name__
    assert kernels.validate_layer_csr is owner
    assert "validate_layer_csr" not in vars(public_materials)


def test_material_kernel_contract_uses_canonical_tensor_validation():
    assert kernels.validate_cuda_tensor is runtime.validate_cuda_tensor


@pytest.mark.parametrize("name", _FUNCTIONAL_OWNER_NAMES)
def test_material_functional_is_the_single_object_owner(name: str):
    owner = getattr(kernels, name)

    assert owner.__module__ == kernels.__name__
    assert kernels.__all__.count(name) == 1


@pytest.mark.parametrize("name", _AUTOGRAD_OWNER_NAMES)
def test_material_autograd_is_the_single_object_owner(name: str):
    owner = getattr(kernels, name)

    assert owner.__module__ == kernels.__name__
    assert kernels.__all__.count(name) == 1


def test_material_functional_uses_canonical_dependencies():
    assert kernels._required_native_op is runtime.required_symbol
    assert not hasattr(kernels, "native_extension")
    assert kernels.validate_cuda_tensor is runtime.validate_cuda_tensor


def test_material_autograd_uses_canonical_dependencies():
    assert kernels.disable_functorch is runtime.disable_functorch
    # The AD plumbing is runtime's, used directly or through the declared AD inputs.
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
        owners = [module for module in (kernels, spec) if hasattr(module, name)]
        assert owners
        assert all(getattr(module, name) is getattr(runtime, name) for module in owners)
    for name in (
        "_EM_LAYER_STACK_FIELDS",
        "em_layer_stack_backward",
        "em_layer_stack_eval",
        "em_layer_stack_jvp",
    ):
        assert hasattr(kernels, name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_mc_face_material_facade_rejects_mismatched_sigma_shape():
    with pytest.raises(ValueError, match="sigma_e must match"):
        kernels.mc_face_material_tensors(
            torch.tensor([2.5], device="cuda", dtype=torch.float32),
            torch.tensor([0.01, 0.02], device="cuda", dtype=torch.float32),
            torch.tensor([1.0], device="cuda", dtype=torch.float32),
            torch.tensor([0], device="cuda", dtype=torch.int32),
        )


def test_materials_module_reexports_neither_kernels_nor_core_world_types():
    assert not hasattr(public_materials, "__all__")
    assert "validate_layer_csr" not in vars(public_materials)
    for name in (
        "Dielectric",
        "Layer",
        "PhaseScreen",
        "PhysicalSurface",
        "SurfaceAssignment",
    ):
        assert not hasattr(public_materials, name)