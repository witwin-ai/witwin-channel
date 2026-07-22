from __future__ import annotations

import pytest
import torch

from witwin.channel.propagation.models.capacity import CapacityExecutionCounts
from witwin.channel.runtime import capacity as capacity_runtime
from witwin.channel.runtime.capacity import (
    SolveCapacityTransaction,
    create_solve_capacity_transaction,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_solve_capacity_transaction_owns_state_and_one_terminal(monkeypatch) -> None:
    reference = torch.empty(1, device="cuda")
    transaction = create_solve_capacity_transaction(reference)
    observed: list[object] = []
    monkeypatch.setattr(
        capacity_runtime,
        "capacity_failure_terminal_check",
        lambda state: observed.append(state),
    )

    transaction.terminal_check()
    assert observed == [transaction.failure_state]
    assert transaction.terminal_enqueued is True
    with pytest.raises(RuntimeError, match="already enqueued"):
        transaction.terminal_check()


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
