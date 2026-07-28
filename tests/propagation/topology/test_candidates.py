from __future__ import annotations


import pytest

from witwin.channel.kernels import topology
from witwin.channel import runtime


_OWNER_NAMES = (
    "path_reflection_candidates",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_topology_candidates_is_the_single_object_owner(name: str):
    owner = getattr(topology, name)

    assert owner.__module__ == topology.__name__


def test_topology_candidates_use_only_canonical_dependencies():
    assert topology._required_native_op is runtime.required_symbol
    assert topology.validate_cuda_tensor is runtime.validate_cuda_tensor
    for name in _OWNER_NAMES:
        assert getattr(topology, name).__globals__ is topology.__dict__
    assert "ops" not in topology.__dict__
