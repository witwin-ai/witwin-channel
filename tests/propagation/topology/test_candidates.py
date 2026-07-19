from __future__ import annotations


import pytest

from witwin.channel_native.propagation.topology import kernels
from witwin.channel_native.propagation.topology.kernels import blocks, candidates
from witwin.channel_native.runtime import symbols, tensor_contracts


_OWNER_NAMES = (
    "path_reflection_candidates",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_topology_candidates_is_the_single_object_owner(name: str):
    owner = getattr(candidates, name)

    assert owner.__module__ == candidates.__name__
    assert getattr(kernels, name) is owner


def test_topology_candidates_use_only_canonical_dependencies():
    assert candidates._required_native_op is symbols.required_symbol
    assert candidates.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert (
        candidates._validate_path_reflection_candidates
        is blocks._validate_path_reflection_candidates
    )
    for name in _OWNER_NAMES:
        assert getattr(candidates, name).__globals__ is candidates.__dict__
    assert "ops" not in candidates.__dict__
