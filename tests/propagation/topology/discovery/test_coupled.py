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


def _canyon_chain_inputs():
    import numpy as np

    from benchmarks.fullwave_validation import reference_set as rs
    from witwin.channel.propagation.geometry import receiver_positions_and_layout
    from witwin.channel.scene.compiler import compile as compile_scene
    from witwin.channel.scene.endpoints import bind_solver_scene

    case = rs.load_reference_case("canyon_z_3g5")
    planes, _ = rs.load_reference(case)
    plane = dict(planes[0], x=np.linspace(-1.2, 1.2, 9), y=np.linspace(-1.1, 1.3, 7))
    compiled = compile_scene(rs.build_channel_scene(case, plane), reference_frequency_hz=case.frequency_hz)
    scene = bind_solver_scene(compiled)
    device = torch.device("cuda")
    tx = torch.tensor([case.tx_position], device=device, dtype=torch.float32)
    rx, _ = receiver_positions_and_layout(scene, device=device)
    return scene, compiled, tx, rx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="chain discovery needs CUDA")
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_unmaterialized_deepest_half_gives_the_materialized_join(depth):
    """The deepest receiver half tested as children of the level above equals the full join."""

    scene, compiled, tx, rx = _canyon_chain_inputs()
    context = coupled._prepare_chain_context(scene, compiled, tx, rx)
    assert context is not None
    a = coupled._HalfSource(context, tx).half(0)
    source = coupled._HalfSource(context, rx)
    parents = source.frontier(depth - 1)
    leaf = list(coupled._iter_leaf_candidates(context, a, parents, source.endpoints))
    full = list(coupled._iter_chain_candidates(context, a, source.half(depth)))
    assert full, "the scene must hold chain candidates of this depth"
    for key in ("tx", "rx", "edge", "a_node", "point", "b_sequence"):
        got = torch.cat([batch[key] for batch in leaf])
        want = torch.cat([batch[key] for batch in full])
        assert got.shape == want.shape, key
        assert torch.equal(got, want), key
