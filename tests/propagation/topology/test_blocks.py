# Copyright Xingyu Chen.
# Tests blocks.

from __future__ import annotations


import pytest

from witwin.channel.kernels import topology
from witwin.channel import runtime


_OWNER_NAMES = (
    "_validate_deterministic_topology_block",
    "_validate_path_block",
    "_validate_topology_extra_fields",
    "deterministic_concat_topology_blocks",
    "path_los_export",
    "path_los_visibility_inputs",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_topology_blocks_is_the_single_object_owner(name: str):
    owner = getattr(topology, name)

    assert owner.__module__ == topology.__name__


def test_topology_blocks_owns_the_shared_schemas():
    for name in ("_PATH_BLOCK_SCHEMA", "_DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA"):
        assert name in topology.__dict__


def test_topology_blocks_use_only_canonical_dependencies():
    assert topology._required_native_op is runtime.required_symbol
    assert topology.validate_cuda_tensor is runtime.validate_cuda_tensor
    for name in _OWNER_NAMES:
        assert getattr(topology, name).__globals__ is topology.__dict__
    assert "ops" not in topology.__dict__


def test_topology_block_helpers_resolve_siblings_in_the_canonical_owner():
    assert (
        topology._validate_deterministic_topology_block.__globals__["_validate_path_block"]
        is topology._validate_path_block
    )
    assert (
        topology._validate_deterministic_topology_block.__globals__[
            "_validate_topology_extra_fields"
        ]
        is topology._validate_topology_extra_fields
    )
    assert (
        topology.deterministic_concat_topology_blocks.__globals__["_validate_path_block"]
        is topology._validate_path_block
    )
