from __future__ import annotations


import pytest

from witwin.channel.propagation.topology.kernels import blocks as ops
from witwin.channel.propagation.topology import kernels
from witwin.channel.propagation.topology.kernels import blocks
from witwin.channel import runtime


_OWNER_NAMES = (
    "_validate_deterministic_topology_block",
    "_validate_path_block",
    "_validate_path_reflection_candidates",
    "_validate_topology_extra_fields",
    "deterministic_concat_topology_blocks",
    "deterministic_gather_topology_block",
    "path_diffraction_block",
    "path_filter_block",
    "path_filter_los",
    "path_finalize_blocks",
    "path_los_export",
    "path_los_visibility_inputs",
    "path_merge_blocks",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_topology_blocks_is_the_single_object_owner(name: str):
    owner = getattr(blocks, name)

    assert owner.__module__ == blocks.__name__
    assert getattr(kernels, name) is owner
    assert getattr(ops, name) is owner


def test_topology_blocks_owns_the_shared_schemas():
    for name in ("_PATH_BLOCK_SCHEMA", "_DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA"):
        owner = getattr(blocks, name)

        assert getattr(kernels, name) is owner
        assert getattr(ops, name) is owner


def test_topology_blocks_use_only_canonical_dependencies():
    assert blocks._required_native_op is runtime.required_symbol
    assert blocks.validate_cuda_tensor is runtime.validate_cuda_tensor
    for name in _OWNER_NAMES:
        assert getattr(blocks, name).__globals__ is blocks.__dict__
    assert "ops" not in blocks.__dict__


def test_topology_block_helpers_resolve_siblings_in_the_canonical_owner():
    assert (
        blocks._validate_deterministic_topology_block.__globals__["_validate_path_block"]
        is blocks._validate_path_block
    )
    assert (
        blocks._validate_deterministic_topology_block.__globals__[
            "_validate_topology_extra_fields"
        ]
        is blocks._validate_topology_extra_fields
    )
    assert (
        blocks._validate_path_reflection_candidates.__globals__["_validate_path_block"]
        is blocks._validate_path_block
    )
    for name in (
        "deterministic_concat_topology_blocks",
        "deterministic_gather_topology_block",
        "path_diffraction_block",
        "path_filter_block",
        "path_filter_los",
        "path_finalize_blocks",
        "path_merge_blocks",
    ):
        assert getattr(blocks, name).__globals__["_validate_path_block"] is blocks._validate_path_block
