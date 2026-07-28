from __future__ import annotations


import pytest

from witwin.channel.kernels import topology
from witwin.channel import runtime


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
    owner = getattr(topology, name)

    assert owner.__module__ == topology.__name__


def test_topology_construction_uses_only_canonical_dependencies():
    assert topology._required_native_op is runtime.required_symbol
    assert topology.validate_cuda_tensor is runtime.validate_cuda_tensor
    assert (
        topology.deterministic_los_topology_block.__globals__[
            "_validate_deterministic_topology_block"
        ]
        is topology._validate_deterministic_topology_block
    )
    assert (
        topology.deterministic_topology_base_fields.__globals__[
            "_validate_path_block"
        ]
        is topology._validate_path_block
    )
    for name in _OWNER_NAMES:
        assert getattr(topology, name).__globals__ is topology.__dict__
    assert "ops" not in topology.__dict__
