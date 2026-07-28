from __future__ import annotations


import pytest

from witwin.channel.kernels import topology
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
    owner = getattr(topology, name)

    assert owner.__module__ == topology.__name__


def test_topology_primitives_use_only_canonical_runtime_dependencies():
    assert topology._required_native_op is runtime.required_symbol
    assert topology.native_extension is runtime.native_extension
    assert topology.validate_cuda_tensor is runtime.validate_cuda_tensor
    for name in _OWNER_NAMES:
        assert getattr(topology, name).__globals__ is topology.__dict__
    assert "ops" not in topology.__dict__
