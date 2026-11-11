# Copyright Xingyu Chen.
# Tests solve capacity transaction.

from __future__ import annotations

import pytest
import torch

from witwin.channel.runtime import (
    CapacityExecutionCounts,
    SolveCapacityTransaction,
    create_solve_capacity_transaction,
)
from witwin.channel import runtime


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_solve_capacity_transaction_reads_the_record_once(monkeypatch) -> None:
    reference = torch.empty(1, device="cuda")
    transaction = create_solve_capacity_transaction(reference)
    observed: list[object] = []

    def read(state):
        observed.append(state)
        return 0, 3

    monkeypatch.setattr(runtime, "capacity_failure_state_read", read)

    assert transaction.observed is False
    assert transaction.terminal_check() == 3
    assert transaction.terminal_check() == 3
    assert observed == [transaction.failure_state]
    assert transaction.observed is True
    assert transaction.dropped_rows == 3


def test_solve_capacity_transaction_compact_observation_skips_the_read(monkeypatch) -> None:
    transaction = create_solve_capacity_transaction(torch.empty(1, device="cuda"))
    monkeypatch.setattr(
        runtime,
        "capacity_failure_state_read",
        lambda _state: pytest.fail("an observed transaction read the record again"),
    )

    transaction.observe(0, 5)
    assert transaction.terminal_check() == 5
    with pytest.raises(RuntimeError, match="already observed"):
        transaction.observe(0, 5)


def test_solve_capacity_transaction_raises_named_capacity_error() -> None:
    transaction = create_solve_capacity_transaction(torch.empty(1, device="cuda"))
    bits = int(
        runtime.CapacityFailureBit.SEGMENT_PENETRATION_FAILURE
        | runtime.CapacityFailureBit.PAIR_CONTRACT_ERROR
    )
    with pytest.raises(runtime.CapacityError) as caught:
        transaction.observe(bits, 0)
    assert caught.value.bits == bits
    assert caught.value.bit_names == ("PAIR_CONTRACT_ERROR", "SEGMENT_PENETRATION_FAILURE")
    assert "SEGMENT_PENETRATION_FAILURE" in str(caught.value)
    assert "returned no result" in str(caught.value)


def test_unknown_failure_bits_are_named_by_value() -> None:
    assert runtime.decode_capacity_failure_bits(1 << 12) == ("UNKNOWN_0x00001000",)
    assert runtime.decode_capacity_failure_bits(0) == ()


def test_capacity_execution_counts_keep_actual_counts_on_device() -> None:
    transaction = create_solve_capacity_transaction(torch.empty(1, device="cuda"))
    candidate_count = torch.tensor([7], device="cuda", dtype=torch.int32)
    guardrail_count = torch.tensor([2], device="cuda", dtype=torch.int32)
    counts = CapacityExecutionCounts(
        candidate_capacity=16,
        failure_state=transaction.failure_state,
        device_candidate_count=candidate_count,
        device_guardrail_count=guardrail_count,
    )

    assert counts.candidate_capacity == 16
    assert counts.device_candidate_count is candidate_count
    assert counts.device_guardrail_count is guardrail_count
    assert counts.device == candidate_count.device


def test_solve_capacity_transaction_rejects_untyped_state() -> None:
    with pytest.raises(TypeError, match="CapacityFailureState"):
        SolveCapacityTransaction(  # type: ignore[arg-type]
            failure_state=torch.zeros(1, device="cuda", dtype=torch.int32)
        )