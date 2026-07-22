from __future__ import annotations

from pathlib import Path

import pytest
import torch

from witwin.channel.propagation.models import CoupledCandidateCapacity
from witwin.channel.propagation.topology.kernels import coupled
from witwin.channel.runtime import (
    CapacityFailureBit,
    CapacityFailureState,
    create_capacity_failure_state,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _theoretical(tx_count: int, rx_count: int, groups: int, edges: int) -> int:
    return tx_count * rx_count * (2 * groups * edges + edges * (edges - 1))


def _block(
    representative_faces: torch.Tensor,
    selected_edges: torch.Tensor,
    *,
    tx_count: int,
    rx_count: int,
    rx_id_offset: int = 0,
    capacity: int | None = None,
    candidate_limit: int = 1_000_000,
    failure_state: CapacityFailureState | None = None,
) -> CoupledCandidateCapacity:
    theoretical = _theoretical(
        tx_count,
        rx_count,
        int(representative_faces.shape[0]),
        int(selected_edges.shape[0]),
    )
    failure_state = failure_state or create_capacity_failure_state(
        representative_faces
    )
    return coupled.coupled_candidate_capacity_block(
        representative_faces,
        selected_edges,
        failure_state=failure_state,
        tx_count=tx_count,
        rx_count=rx_count,
        rx_id_offset=rx_id_offset,
        candidate_capacity=theoretical if capacity is None else capacity,
        candidate_limit=candidate_limit,
    )


def _legacy_rows(
    faces: list[int],
    edges: list[int],
    *,
    tx_count: int,
    rx_count: int,
    rx_id_offset: int,
    rd_chunk_size: int = 65_536,
) -> list[tuple[int, int, int, int, int, int]]:
    base = tx_count * rx_count * len(faces) * len(edges)
    rows: list[tuple[int, int, int, int, int, int]] = []
    candidates_per_pair = len(faces) * len(edges)
    for start in range(0, base, rd_chunk_size):
        end = min(start + rd_chunk_size, base)
        for component_id in (3, 4):
            for linear in range(start, end):
                pair_slot, local_slot = divmod(linear, candidates_per_pair)
                tx_slot, rx_slot = divmod(pair_slot, rx_count)
                face_slot, edge_slot = divmod(local_slot, len(edges))
                rows.append(
                    (
                        tx_slot,
                        rx_id_offset + rx_slot,
                        component_id,
                        faces[face_slot],
                        edges[edge_slot],
                        -1,
                    )
                )
    dd_per_pair = len(edges) * (len(edges) - 1)
    for linear in range(tx_count * rx_count * dd_per_pair):
        pair_slot, local_slot = divmod(linear, dd_per_pair)
        tx_slot, rx_slot = divmod(pair_slot, rx_count)
        first_slot, remainder_slot = divmod(local_slot, len(edges) - 1)
        second_slot = (
            remainder_slot if remainder_slot < first_slot else remainder_slot + 1
        )
        rows.append(
            (
                tx_slot,
                rx_id_offset + rx_slot,
                7,
                -1,
                edges[first_slot],
                edges[second_slot],
            )
        )
    return rows


def test_coupled_candidate_capacity_matches_frozen_rd_dd_order() -> None:
    faces = torch.tensor([5, 8], device="cuda", dtype=torch.int32)
    edges = torch.tensor([7, 9, 11], device="cuda", dtype=torch.int32)
    block = _block(faces, edges, tx_count=2, rx_count=2, rx_id_offset=4)
    expected = _legacy_rows([5, 8], [7, 9, 11], tx_count=2, rx_count=2, rx_id_offset=4)

    assert block.candidate_capacity == len(expected)
    assert block.device.type == "cuda"
    assert block.candidate_count.tolist() == [len(expected)]
    assert block.overflow.tolist() == [False]
    assert block.valid.tolist() == [True] * len(expected)
    actual = list(
        zip(
            block.tx_id.tolist(),
            block.rx_id.tolist(),
            block.component_id.tolist(),
            block.face_id.tolist(),
            block.edge1_id.tolist(),
            block.edge2_id.tolist(),
            strict=True,
        )
    )
    assert actual == expected


def test_coupled_candidate_capacity_preserves_65536_chunk_interleave() -> None:
    group_count = 65_537
    faces = torch.arange(group_count, device="cuda", dtype=torch.int32)
    edges = torch.tensor([17], device="cuda", dtype=torch.int32)
    block = _block(faces, edges, tx_count=1, rx_count=1)

    assert block.candidate_capacity == 2 * group_count
    assert block.component_id[0].item() == 3
    assert block.component_id[65_535].item() == 3
    assert block.component_id[65_536].item() == 4
    assert block.component_id[131_071].item() == 4
    assert block.component_id[131_072].item() == 3
    assert block.component_id[131_073].item() == 4
    assert block.face_id[[0, 65_535, 65_536, 131_071, 131_072, 131_073]].tolist() == [
        0,
        65_535,
        0,
        65_535,
        65_536,
        65_536,
    ]


def test_coupled_candidate_capacity_zero_and_inert_tail() -> None:
    empty = _block(
        torch.empty(0, device="cuda", dtype=torch.int32),
        torch.empty(0, device="cuda", dtype=torch.int32),
        tx_count=0,
        rx_count=3,
        capacity=0,
    )
    assert empty.valid.shape == (0,)
    assert empty.candidate_count.tolist() == [0]
    assert empty.overflow.tolist() == [False]

    faces = torch.tensor([5], device="cuda", dtype=torch.int32)
    edges = torch.tensor([7], device="cuda", dtype=torch.int32)
    padded = _block(faces, edges, tx_count=1, rx_count=1, capacity=5)
    assert padded.valid.tolist() == [True, True, False, False, False]
    assert padded.candidate_count.tolist() == [2]
    assert padded.overflow.tolist() == [False]
    for tensor in (
        padded.tx_id,
        padded.rx_id,
        padded.component_id,
        padded.face_id,
        padded.edge1_id,
        padded.edge2_id,
    ):
        assert tensor[2:].tolist() == [-1, -1, -1]


def test_coupled_candidate_capacity_uses_current_cuda_stream() -> None:
    faces = torch.tensor([5], device="cuda", dtype=torch.int32)
    edges = torch.tensor([7, 9], device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        block = _block(faces, edges, tx_count=1, rx_count=2, rx_id_offset=6)
    stream.synchronize()
    assert block.candidate_count.tolist() == [12]
    assert block.rx_id.unique(sorted=True).tolist() == [6, 7]


def test_coupled_candidate_guard_precedes_native_dispatch(monkeypatch) -> None:
    faces = torch.tensor([5, 8], device="cuda", dtype=torch.int32)
    edges = torch.tensor([7, 9, 11], device="cuda", dtype=torch.int32)
    monkeypatch.setattr(
        coupled,
        "_required_native_op",
        lambda _name: pytest.fail("native dispatch occurred before candidate guard"),
    )
    with pytest.raises(
        RuntimeError,
        match="requires 18 candidates, exceeding coupled_candidate_limit=17",
    ):
        _block(
            faces,
            edges,
            tx_count=1,
            rx_count=1,
            capacity=17,
            candidate_limit=17,
        )


def test_coupled_candidate_capacity_has_no_fallback(monkeypatch) -> None:
    faces = torch.tensor([5], device="cuda", dtype=torch.int32)
    edges = torch.tensor([7], device="cuda", dtype=torch.int32)

    def missing(_name: str):
        raise RuntimeError("required native symbol is missing")

    monkeypatch.setattr(coupled, "_required_native_op", missing)
    with pytest.raises(RuntimeError, match="required native symbol is missing"):
        _block(faces, edges, tx_count=1, rx_count=1)


def test_coupled_candidate_capacity_overflow_sets_state_and_is_inert() -> None:
    faces = torch.tensor([5], device="cuda", dtype=torch.int32)
    edges = torch.tensor([7, 9], device="cuda", dtype=torch.int32)
    failure_state = create_capacity_failure_state(faces)

    block = _block(
        faces,
        edges,
        tx_count=1,
        rx_count=1,
        capacity=5,
        candidate_limit=100,
        failure_state=failure_state,
    )

    assert block.failure_state is failure_state
    assert failure_state.bits.tolist() == [
        int(CapacityFailureBit.COUPLED_CANDIDATE_OVERFLOW)
    ]
    assert block.candidate_count.tolist() == [0]
    assert block.overflow.tolist() == [True]
    assert block.valid.tolist() == [False] * 5
    assert block.tx_id.tolist() == [-1] * 5


def test_coupled_candidate_capacity_source_has_no_host_transfer_or_public_cap() -> None:
    root = Path(__file__).resolve().parents[3]
    native = (
        root / "native/channel/kernels/coupled_candidate_capacity.cu"
    ).read_text(encoding="utf-8")
    facade = (
        root / "src/witwin/channel/propagation/topology/kernels/coupled.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "cudaMemcpy",
        "cudaStreamSynchronize",
        ".cpu()",
        ".numpy()",
        "path_capacity_per_pair",
        "max_paths",
    ):
        assert forbidden not in native
        assert forbidden not in facade
    for forbidden in ("torch.arange", "torch.div", "torch.remainder", "torch.where"):
        assert forbidden not in facade
    assert "kRdChunkSize = 65'536" in native
    assert "trap;" not in native
    assert "atomicOr" in native
