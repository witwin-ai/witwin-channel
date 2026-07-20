from __future__ import annotations


import pytest

from witwin.channel_native.propagation.topology.kernels import compaction as ops
from witwin.channel_native.propagation.topology import kernels
from witwin.channel_native.propagation.topology.kernels import compaction
from witwin.channel_native.runtime import symbols, tensor_contracts


_OWNER_NAMES = (
    "deterministic_capacity_finalize",
    "deterministic_diffraction_order1_capacity_block",
    "deterministic_diffraction_order1_compact",
    "deterministic_reflection_order1_compact",
    "deterministic_reflection_sequence_compact",
    "deterministic_sort_order",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_topology_compaction_is_the_single_object_owner(name: str):
    owner = getattr(compaction, name)

    assert owner.__module__ == compaction.__name__
    assert getattr(kernels, name) is owner
    assert getattr(ops, name) is owner


def test_topology_compaction_uses_only_canonical_runtime_dependencies():
    assert compaction._required_native_op is symbols.required_symbol
    assert compaction.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    for name in _OWNER_NAMES:
        assert getattr(compaction, name).__globals__ is compaction.__dict__
    assert "ops" not in compaction.__dict__
