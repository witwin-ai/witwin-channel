from __future__ import annotations


import pytest

from witwin.channel.propagation.topology.kernels import primitives as ops
from witwin.channel.propagation.topology import kernels
from witwin.channel.propagation.topology.kernels import primitives
from witwin.channel import runtime


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


def test_topology_primitives_use_only_canonical_runtime_dependencies():
    assert primitives._required_native_op is runtime.required_symbol
    assert primitives.native_extension is runtime.native_extension
    assert primitives.validate_cuda_tensor is runtime.validate_cuda_tensor
    for name in _OWNER_NAMES:
        assert getattr(primitives, name).__globals__ is primitives.__dict__
    assert "ops" not in primitives.__dict__
