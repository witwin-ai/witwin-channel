# Copyright Xingyu Chen.
# Tests primitives.

from __future__ import annotations


import pytest

from witwin.channel.kernels import geometry
from witwin.channel import runtime


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
    owner = getattr(geometry, name)

    assert owner.__module__ == geometry.__name__


def test_geometry_primitives_use_only_canonical_runtime_dependencies():
    assert geometry._required_native_op is runtime.required_symbol
    assert not hasattr(geometry, "native_extension")
    assert geometry.validate_cuda_tensor is runtime.validate_cuda_tensor
    for name in _OWNER_NAMES:
        assert getattr(geometry, name).__globals__ is geometry.__dict__
    assert "ops" not in geometry.__dict__