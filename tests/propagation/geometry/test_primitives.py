from __future__ import annotations


import pytest

from witwin.channel.propagation.geometry.kernels import primitives as ops
from witwin.channel.propagation.geometry import kernels
from witwin.channel.propagation.geometry.kernels import primitives
from witwin.channel.runtime import symbols, tensor_contracts


_OWNER_NAMES = (
    "core_diffraction_edge_count",
    "deterministic_face_groups",
    "deterministic_normalize_vec3",
    "deterministic_reflect_points",
    "deterministic_surface_face_groups",
    "mc_diffraction_edge_geometry",
    "mc_surface_group_edge_candidates",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_geometry_primitives_is_the_single_object_owner(name: str):
    owner = getattr(primitives, name)

    assert owner.__module__ == primitives.__name__
    assert getattr(kernels, name) is owner
    assert getattr(ops, name) is owner


def test_geometry_primitives_use_only_canonical_runtime_dependencies():
    assert primitives._required_native_op is symbols.required_symbol
    assert primitives.native_extension is symbols.native_extension
    assert primitives.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    for name in _OWNER_NAMES:
        assert getattr(primitives, name).__globals__ is primitives.__dict__
    assert "ops" not in primitives.__dict__
