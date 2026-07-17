from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from witwin.channel_native.propagation.topology.discovery import coupled


def test_prepare_coupled_candidate_plan_freezes_counts_and_tensor_identity():
    representative_faces = torch.tensor([5, 8], dtype=torch.int32)
    selected_edges = torch.tensor([7, 9, 11], dtype=torch.int32)

    plan = coupled.prepare_coupled_candidate_plan(
        tx_count=2,
        rx_count=3,
        representative_faces=representative_faces,
        selected_edges=selected_edges,
        candidate_limit=100,
        chunk_size=5,
    )

    assert [field.name for field in fields(plan)] == [
        "tx_count",
        "rx_count",
        "representative_faces",
        "selected_edges",
        "edge_count",
        "candidates_per_pair",
        "base_candidate_count",
        "theoretical_candidate_count",
        "chunk_size",
    ]
    assert plan.representative_faces is representative_faces
    assert plan.selected_edges is selected_edges
    assert plan.edge_count == 3
    assert plan.candidates_per_pair == 6
    assert plan.base_candidate_count == 36
    assert plan.theoretical_candidate_count == 72
    assert plan.chunk_size == 5
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
    theoretical = tx_count * rx_count * 2 * 3 * 2

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
        candidate_limit=16,
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
