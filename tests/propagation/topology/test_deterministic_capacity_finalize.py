from __future__ import annotations

from pathlib import Path

import pytest
import torch

from witwin.channel_native.propagation.models import CapacityPathSelection
from witwin.channel_native.propagation.topology.kernels import compaction
from witwin.channel_native.runtime import (
    CapacityFailureBit,
    CapacityFailureState,
    create_capacity_failure_state,
)
from witwin.channel_native.runtime.symbols import required_symbol


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _finalize(
    valid: torch.Tensor,
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    *,
    pair_count: int,
    num_tx: int,
    num_rx: int,
    capacity: int,
    failure_state: CapacityFailureState | None = None,
) -> CapacityPathSelection:
    failure_state = failure_state or create_capacity_failure_state(valid)
    return compaction.deterministic_capacity_finalize(
        failure_state=failure_state,
        valid=valid,
        tx_id=tx_id,
        rx_id=rx_id,
        pair_count=pair_count,
        num_tx=num_tx,
        num_rx=num_rx,
        path_capacity_per_pair=capacity,
    )


def test_capacity_finalize_is_pair_major_stable_and_poison_safe() -> None:
    valid = torch.tensor(
        [True, False, True, True, True, False, True],
        device="cuda",
        dtype=torch.bool,
    )
    poison = torch.iinfo(torch.int32).min
    tx_id = torch.tensor(
        [2, poison, 1, 2, 1, poison, 0], device="cuda", dtype=torch.int32
    )
    rx_id = torch.tensor(
        [1, poison, 0, 1, 0, poison, 0], device="cuda", dtype=torch.int32
    )

    selection = _finalize(
        valid, tx_id, rx_id, pair_count=6, num_tx=3, num_rx=2, capacity=3
    )

    assert selection.pair_count == 6
    assert selection.path_capacity_per_pair == 3
    assert selection.selected_row_index.dtype == torch.int64
    assert selection.selected_row_index.tolist() == [
        6,
        -1,
        -1,
        2,
        4,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
        3,
        -1,
    ]
    assert selection.valid.tolist() == [
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
    ]
    assert selection.num_paths.tolist() == [1, 2, 0, 0, 0, 2]
    assert selection.overflow.tolist() == [False]


def test_capacity_finalize_handles_zero_sparse_and_exact_capacity() -> None:
    empty = _finalize(
        torch.empty(0, device="cuda", dtype=torch.bool),
        torch.empty(0, device="cuda", dtype=torch.int32),
        torch.empty(0, device="cuda", dtype=torch.int32),
        pair_count=0,
        num_tx=0,
        num_rx=4,
        capacity=4,
    )
    assert empty.selected_row_index.shape == (0,)
    assert empty.valid.shape == (0,)
    assert empty.num_paths.shape == (0,)
    assert empty.overflow.tolist() == [False]

    empty_tx_only = _finalize(
        torch.empty(0, device="cuda", dtype=torch.bool),
        torch.empty(0, device="cuda", dtype=torch.int32),
        torch.empty(0, device="cuda", dtype=torch.int32),
        pair_count=0,
        num_tx=4,
        num_rx=0,
        capacity=4,
    )
    assert empty_tx_only.selected_row_index.shape == (0,)
    assert empty_tx_only.num_paths.shape == (0,)
    assert empty_tx_only.overflow.tolist() == [False]

    inert = _finalize(
        torch.zeros(3, device="cuda", dtype=torch.bool),
        torch.full((3,), -2_000_000_000, device="cuda", dtype=torch.int32),
        torch.full((3,), 2_000_000_000, device="cuda", dtype=torch.int32),
        pair_count=2,
        num_tx=1,
        num_rx=2,
        capacity=0,
    )
    assert inert.selected_row_index.shape == (0,)
    assert inert.valid.shape == (0,)
    assert inert.num_paths.tolist() == [0, 0]
    assert inert.overflow.tolist() == [False]

    exact = _finalize(
        torch.ones(4, device="cuda", dtype=torch.bool),
        torch.tensor([1, 0, 1, 0], device="cuda", dtype=torch.int32),
        torch.tensor([1, 0, 1, 0], device="cuda", dtype=torch.int32),
        pair_count=4,
        num_tx=2,
        num_rx=2,
        capacity=2,
    )
    assert exact.selected_row_index.tolist() == [1, 3, -1, -1, -1, -1, 0, 2]
    assert exact.num_paths.tolist() == [2, 0, 0, 2]
    assert exact.overflow.tolist() == [False]


def test_capacity_finalize_uses_the_current_cuda_stream() -> None:
    valid = torch.tensor([True, False, True], device="cuda", dtype=torch.bool)
    tx_id = torch.tensor([0, -1, 0], device="cuda", dtype=torch.int32)
    rx_id = torch.tensor([0, -1, 1], device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        selection = _finalize(
            valid, tx_id, rx_id, pair_count=2, num_tx=1, num_rx=2, capacity=1
        )
    stream.synchronize()

    assert selection.selected_row_index.tolist() == [0, 2]
    assert selection.valid.tolist() == [True, True]
    assert selection.num_paths.tolist() == [1, 1]


def test_capacity_finalize_rejects_bad_host_layout_metadata() -> None:
    empty_bool = torch.empty(0, device="cuda", dtype=torch.bool)
    empty_int = torch.empty(0, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match=r"pair_count must equal num_tx \* num_rx"):
        _finalize(
            empty_bool,
            empty_int,
            empty_int,
            pair_count=3,
            num_tx=2,
            num_rx=2,
            capacity=1,
        )


@pytest.mark.parametrize(
    ("pair_count", "num_tx", "num_rx", "capacity", "message"),
    [
        (-1, 0, 0, 0, "pair_count must be non-negative"),
        (0, -1, 0, 0, "num_tx must be non-negative"),
        (0, 0, -1, 0, "num_rx must be non-negative"),
        (0, 0, 0, -1, "path_capacity_per_pair must be non-negative"),
    ],
)
def test_native_capacity_finalize_rejects_negative_metadata_before_allocation(
    pair_count: int,
    num_tx: int,
    num_rx: int,
    capacity: int,
    message: str,
) -> None:
    empty_bool = torch.empty(0, device="cuda", dtype=torch.bool)
    empty_int = torch.empty(0, device="cuda", dtype=torch.int32)
    native = required_symbol("deterministic_capacity_finalize")
    failure_state = create_capacity_failure_state(empty_bool)
    with pytest.raises(RuntimeError, match=message):
        native(
            failure_state.bits,
            empty_bool,
            empty_int,
            empty_int,
            pair_count,
            num_tx,
            num_rx,
            capacity,
        )


def test_capacity_finalize_overflow_sets_state_and_returns_inert() -> None:
    valid = torch.ones(3, device="cuda", dtype=torch.bool)
    failure_state = create_capacity_failure_state(valid)

    selection = _finalize(
        valid,
        torch.zeros(3, device="cuda", dtype=torch.int32),
        torch.zeros(3, device="cuda", dtype=torch.int32),
        pair_count=1,
        num_tx=1,
        num_rx=1,
        capacity=2,
        failure_state=failure_state,
    )

    assert failure_state.bits.tolist() == [
        int(CapacityFailureBit.PAIR_CAPACITY_OVERFLOW)
    ]
    assert selection.layout.failure_state is failure_state
    assert selection.selected_row_index.tolist() == [-1, -1]
    assert selection.valid.tolist() == [False, False]
    assert selection.num_paths.tolist() == [0]
    assert selection.overflow.tolist() == [True]


def test_capacity_finalize_valid_bad_id_sets_state_and_returns_inert() -> None:
    valid = torch.tensor([True, False], device="cuda", dtype=torch.bool)
    failure_state = create_capacity_failure_state(valid)

    selection = _finalize(
        valid,
        torch.tensor([2, -2147483648], device="cuda", dtype=torch.int32),
        torch.tensor([0, -2147483648], device="cuda", dtype=torch.int32),
        pair_count=1,
        num_tx=1,
        num_rx=1,
        capacity=2,
        failure_state=failure_state,
    )

    assert failure_state.bits.tolist() == [int(CapacityFailureBit.PAIR_CONTRACT_ERROR)]
    assert selection.selected_row_index.tolist() == [-1, -1]
    assert selection.valid.tolist() == [False, False]
    assert selection.num_paths.tolist() == [0]
    assert selection.overflow.tolist() == [False]


def test_capacity_finalize_has_no_host_count_transfer_or_atomic_rank() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "native"
        / "channel_native"
        / "kernels"
        / "deterministic_capacity_finalize.cu"
    ).read_text(encoding="utf-8")
    for forbidden in ("cudaMemcpy", "cudaStreamSynchronize", ".item", ".cpu"):
        assert forbidden not in source
    assert "DeviceRadixSort::SortPairs" in source
    assert "position - pair_start[pair]" in source
    assert "trap;" not in source
    assert "failure_state[0] != 0" in source
