# Copyright Xingyu Chen.
# Tests coupled.

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from witwin.channel.interactions import coupled


def test_prepare_coupled_candidate_plan_freezes_counts_and_tensor_identity():
    representative_faces = torch.tensor([5, 8], dtype=torch.int32)
    selected_edges = torch.tensor([7, 9, 11], dtype=torch.int32)

    plan = coupled.prepare_coupled_candidate_plan(
        tx_count=2,
        rx_count=3,
        representative_faces=representative_faces,
        selected_edges=selected_edges,
        candidate_limit=200,
        chunk_size=5,
    )

    assert [field.name for field in fields(plan)] == [
        "tx_count",
        "rx_count",
        "representative_faces",
        "selected_edges",
        "edge_count",
        "candidates_per_pair",
        "dd_candidates_per_pair",
        "base_candidate_count",
        "dd_base_candidate_count",
        "theoretical_candidate_count",
        "chunk_size",
        "dd_chunk_size",
    ]
    assert plan.representative_faces is representative_faces
    assert plan.selected_edges is selected_edges
    assert plan.edge_count == 3
    assert plan.candidates_per_pair == 6
    # coupled double diffraction: cid-7 ordered edge pairs e1 != e2 -> edges*(edges-1).
    assert plan.dd_candidates_per_pair == 6
    assert plan.base_candidate_count == 36
    assert plan.dd_base_candidate_count == 36
    # Budget union: both R->D / D->R directions (x2) plus one-direction D->D.
    assert plan.theoretical_candidate_count == 36 * 2 + 36
    assert plan.chunk_size == 5
    # The D->D stream carries its own (larger) chunk so it collapses to one
    # native launch per receiver block (coupled double diffraction); the R->D / D->R stream
    # stays at chunk_size, keeping cid-3/4 row identity byte-identical.
    assert plan.dd_chunk_size == coupled._COUPLED_DD_CANDIDATE_CHUNK_SIZE
    with pytest.raises(FrozenInstanceError):
        plan.chunk_size = 6


@pytest.mark.parametrize(
    ("tx_count", "rx_count", "candidate_limit", "expected_limit"),
    (
        (2, 2, 23, 23),
        (500_001, 1, 2_000_000, coupled._MAX_COUPLED_CANDIDATES),
    ),
)
def test_candidate_guard_runs_during_planning_before_iteration(
    monkeypatch,
    tx_count,
    rx_count,
    candidate_limit,
    expected_limit,
):
    monkeypatch.setattr(
        coupled.torch,
        "arange",
        lambda *_args, **_kwargs: pytest.fail("candidate iterator ran before guard"),
    )
    representative_faces = torch.tensor([5, 8], dtype=torch.int32)
    selected_edges = torch.tensor([7, 9, 11], dtype=torch.int32)
    # Union budget: 2*groups*edges (R->D / D->R) + edges*(edges-1) (D->D).
    theoretical = tx_count * rx_count * (2 * 2 * 3 + 3 * 2)

    with pytest.raises(
        RuntimeError,
        match=(
            rf"requires {theoretical} candidates, exceeding "
            rf"coupled_candidate_limit={expected_limit}"
        ),
    ):
        coupled.prepare_coupled_candidate_plan(
            tx_count=tx_count,
            rx_count=rx_count,
            representative_faces=representative_faces,
            selected_edges=selected_edges,
            candidate_limit=candidate_limit,
        )


def test_candidate_requests_are_lazy_chunked_and_rd_first(monkeypatch):
    representative_faces = torch.tensor([5], dtype=torch.int32)
    selected_edges = torch.tensor([7, 9], dtype=torch.int32)
    plan = coupled.prepare_coupled_candidate_plan(
        tx_count=2,
        rx_count=2,
        representative_faces=representative_faces,
        selected_edges=selected_edges,
        candidate_limit=32,
        chunk_size=3,
    )
    arange = torch.arange
    calls: list[tuple[int, int]] = []

    def recording_arange(start, end, **kwargs):
        calls.append((start, end))
        return arange(start, end, **kwargs)

    monkeypatch.setattr(coupled.torch, "arange", recording_arange)
    requests = coupled.iter_coupled_candidate_requests(
        plan,
        device=torch.device("cpu"),
    )

    assert calls == []
    rd = next(requests)
    assert calls == [(0, 3)]
    dr = next(requests)
    assert calls == [(0, 3)]
    assert (rd.reverse, rd.component_id) == (False, 3)
    assert (dr.reverse, dr.component_id) == (True, 4)
    assert (rd.chunk_start, rd.chunk_end, rd.candidate_count) == (0, 3, 3)
    torch.testing.assert_close(rd.linear, torch.tensor([0, 1, 2]))
    torch.testing.assert_close(rd.tx_slot, torch.tensor([0, 0, 0]))
    torch.testing.assert_close(rd.rx_slot, torch.tensor([0, 0, 1]))
    torch.testing.assert_close(rd.face_id, torch.tensor([5, 5, 5], dtype=torch.int32))
    torch.testing.assert_close(rd.edge_id, torch.tensor([7, 9, 7], dtype=torch.int32))
    for name in ("linear", "tx_slot", "rx_slot", "face_id", "edge_id"):
        rd_tensor = getattr(rd, name)
        dr_tensor = getattr(dr, name)
        assert dr_tensor is rd_tensor
        assert dr_tensor.data_ptr() == rd_tensor.data_ptr()
        assert dr_tensor.stride() == rd_tensor.stride()
    with pytest.raises(FrozenInstanceError):
        rd.reverse = True

    representative_faces[0] = 13
    selected_edges[0] = 17
    next_rd = next(requests)
    assert calls == [(0, 3), (3, 6)]
    assert (next_rd.reverse, next_rd.component_id) == (False, 3)
    assert (next_rd.chunk_start, next_rd.chunk_end, next_rd.candidate_count) == (
        3,
        6,
        3,
    )
    torch.testing.assert_close(
        next_rd.face_id,
        torch.tensor([13, 13, 13], dtype=torch.int32),
    )
    torch.testing.assert_close(
        next_rd.edge_id,
        torch.tensor([9, 17, 9], dtype=torch.int32),
    )


@pytest.mark.parametrize(
    ("representative_faces", "selected_edges"),
    (
        (torch.empty((0,), dtype=torch.int32), torch.tensor([7], dtype=torch.int32)),
        (torch.tensor([5], dtype=torch.int32), torch.empty((0,), dtype=torch.int32)),
    ),
)
def test_zero_candidate_plan_yields_without_prefetch(
    monkeypatch,
    representative_faces,
    selected_edges,
):
    monkeypatch.setattr(
        coupled.torch,
        "arange",
        lambda *_args, **_kwargs: pytest.fail("zero plan allocated a candidate chunk"),
    )
    plan = coupled.prepare_coupled_candidate_plan(
        tx_count=2,
        rx_count=3,
        representative_faces=representative_faces,
        selected_edges=selected_edges,
        candidate_limit=1,
    )
    requests = coupled.iter_coupled_candidate_requests(
        plan,
        device=torch.device("cpu"),
    )

    assert plan.base_candidate_count == 0
    assert plan.theoretical_candidate_count == 0
    with pytest.raises(StopIteration):
        next(requests)


def test_dd_candidate_requests_enumerate_ordered_pairs():
    representative_faces = torch.tensor([5], dtype=torch.int32)
    selected_edges = torch.tensor([7, 9, 11], dtype=torch.int32)
    plan = coupled.prepare_coupled_candidate_plan(
        tx_count=1,
        rx_count=1,
        representative_faces=representative_faces,
        selected_edges=selected_edges,
        candidate_limit=100,
        chunk_size=100,
    )

    assert plan.dd_candidates_per_pair == 6
    assert plan.dd_base_candidate_count == 6

    requests = list(
        coupled.iter_coupled_dd_candidate_requests(plan, device=torch.device("cpu"))
    )
    assert len(requests) == 1
    request = requests[0]
    assert request.component_id == 7
    assert (request.chunk_start, request.chunk_end, request.candidate_count) == (0, 6, 6)
    # Ordered pairs (e1, e2) with e1 != e2 by index; the second index steps over
    # its own position, so the diagonal is skipped.
    torch.testing.assert_close(
        request.edge1_id, torch.tensor([7, 7, 9, 9, 11, 11], dtype=torch.int32)
    )
    torch.testing.assert_close(
        request.edge2_id, torch.tensor([9, 11, 7, 11, 7, 9], dtype=torch.int32)
    )
    assert bool((request.edge1_id != request.edge2_id).all())
    torch.testing.assert_close(request.tx_slot, torch.zeros(6, dtype=torch.int64))
    torch.testing.assert_close(request.rx_slot, torch.zeros(6, dtype=torch.int64))


def test_dd_candidate_requests_decompose_tx_and_are_lazy(monkeypatch):
    representative_faces = torch.tensor([5], dtype=torch.int32)
    selected_edges = torch.tensor([7, 9], dtype=torch.int32)
    plan = coupled.prepare_coupled_candidate_plan(
        tx_count=2,
        rx_count=1,
        representative_faces=representative_faces,
        selected_edges=selected_edges,
        candidate_limit=100,
        chunk_size=3,
        # The D->D stream is chunked by dd_chunk_size (its own knob), not the
        # R->D / D->R chunk_size; drive it small here to exercise laziness.
        dd_chunk_size=3,
    )
    arange = torch.arange
    calls: list[tuple[int, int]] = []

    def recording_arange(start, end, **kwargs):
        calls.append((start, end))
        return arange(start, end, **kwargs)

    monkeypatch.setattr(coupled.torch, "arange", recording_arange)
    requests = coupled.iter_coupled_dd_candidate_requests(
        plan, device=torch.device("cpu")
    )

    assert calls == []
    first = next(requests)
    assert calls == [(0, 3)]
    assert first.component_id == 7
    torch.testing.assert_close(first.tx_slot, torch.tensor([0, 0, 1]))
    torch.testing.assert_close(first.rx_slot, torch.tensor([0, 0, 0]))
    torch.testing.assert_close(
        first.edge1_id, torch.tensor([7, 9, 7], dtype=torch.int32)
    )
    torch.testing.assert_close(
        first.edge2_id, torch.tensor([9, 7, 9], dtype=torch.int32)
    )
    second = next(requests)
    assert calls == [(0, 3), (3, 4)]
    assert (second.chunk_start, second.chunk_end, second.candidate_count) == (3, 4, 1)
    torch.testing.assert_close(second.tx_slot, torch.tensor([1]))
    torch.testing.assert_close(
        second.edge1_id, torch.tensor([9], dtype=torch.int32)
    )
    torch.testing.assert_close(
        second.edge2_id, torch.tensor([7], dtype=torch.int32)
    )
    with pytest.raises(StopIteration):
        next(requests)


def test_dd_candidate_stream_is_empty_below_two_edges(monkeypatch):
    representative_faces = torch.tensor([5], dtype=torch.int32)
    selected_edges = torch.tensor([7], dtype=torch.int32)
    plan = coupled.prepare_coupled_candidate_plan(
        tx_count=3,
        rx_count=4,
        representative_faces=representative_faces,
        selected_edges=selected_edges,
        candidate_limit=1000,
    )

    assert plan.dd_candidates_per_pair == 0
    assert plan.dd_base_candidate_count == 0

    monkeypatch.setattr(
        coupled.torch,
        "arange",
        lambda *_args, **_kwargs: pytest.fail("single-edge DD plan allocated a chunk"),
    )
    requests = coupled.iter_coupled_dd_candidate_requests(
        plan, device=torch.device("cpu")
    )
    with pytest.raises(StopIteration):
        next(requests)