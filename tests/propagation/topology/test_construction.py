from __future__ import annotations


import pytest

from witwin.channel.propagation.topology.kernels import construction as ops
from witwin.channel.propagation.topology import kernels
from witwin.channel.propagation.topology.kernels import blocks, construction
from witwin.channel.runtime import symbols, tensor_contracts


_OWNER_NAMES = (
    "deterministic_face_anchor_points",
    "deterministic_face_sequence_chunk",
    "deterministic_los_topology_block",
    "deterministic_mapped_face_sequence_chunk",
    "deterministic_pad_topology_sequences",
    "deterministic_reflection_epc_input_batch",
    "deterministic_repeat_range",
    "deterministic_topology_base_fields",
    "deterministic_topology_default_fields",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_topology_construction_is_the_single_object_owner(name: str):
    owner = getattr(construction, name)

    assert owner.__module__ == construction.__name__
    assert getattr(kernels, name) is owner
    assert getattr(ops, name) is owner


def test_topology_construction_uses_only_canonical_dependencies():
    assert construction._required_native_op is symbols.required_symbol
    assert construction.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert construction._validate_deterministic_topology_block is (
        blocks._validate_deterministic_topology_block
    )
    assert construction._validate_path_block is blocks._validate_path_block
    assert (
        construction.deterministic_los_topology_block.__globals__[
            "_validate_deterministic_topology_block"
        ]
        is blocks._validate_deterministic_topology_block
    )
    assert (
        construction.deterministic_topology_base_fields.__globals__[
            "_validate_path_block"
        ]
        is blocks._validate_path_block
    )
    for name in _OWNER_NAMES:
        assert getattr(construction, name).__globals__ is construction.__dict__
    assert "ops" not in construction.__dict__
