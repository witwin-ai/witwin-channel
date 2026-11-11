# Copyright Xingyu Chen.
# Tests the interaction-chain discovery helpers: component ids, half lists and the edge join.

from __future__ import annotations

import pytest
import torch

from witwin.channel.interactions import coupled


@pytest.mark.parametrize(
    "prefix,suffix,expected",
    [(1, 0, 3), (2, 0, 3), (0, 1, 4), (0, 3, 4), (1, 1, 9), (2, 1, 10), (1, 2, 9)],
)
def test_chain_component_id_places_the_diffraction(prefix, suffix, expected):
    assert coupled.chain_component_id(prefix, suffix) == expected


def test_direct_half_lists_visible_endpoints_edge_major():
    visible = torch.tensor(
        [
            [True, False, True],
            [False, False, True],
            [True, False, False],
        ]
    )
    endpoints = torch.zeros((3, 3))
    half = coupled._direct_half(visible, endpoints)
    assert half.edge_offsets.tolist() == [0, 2, 2, 4]
    assert half.endpoint.tolist() == [0, 2, 0, 1]
    assert half.node.tolist() == [-1, -1, -1, -1]
    assert half.depth == 0


def test_lane_offsets_join_every_entry_pair_of_a_shared_edge():
    a = coupled._direct_half(torch.tensor([[True, True], [True, False]]), torch.zeros((2, 3)))
    b = coupled._direct_half(torch.tensor([[True, True], [False, True], [False, True]]), torch.zeros((3, 3)))
    offsets, total = coupled._lane_offsets(a, b)
    # Edge 0: 2 transmitter entries x 1 receiver entry; edge 1: 1 x 3.
    assert offsets.tolist() == [0, 2, 5]
    assert total == 5


def test_endpoint_edge_lists_are_csr_per_endpoint():
    offsets, edges = coupled._endpoint_edge_lists(
        torch.tensor([[False, True, True], [False, False, False], [True, False, True]])
    )
    assert offsets.tolist() == [0, 2, 2, 4]
    assert edges.tolist() == [1, 2, 0, 2]
