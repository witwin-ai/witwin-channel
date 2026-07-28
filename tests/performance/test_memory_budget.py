from __future__ import annotations

import pytest

from witwin.channel.runtime import (
    MemoryBudgetError,
    checked_product,
    enforce_memory_budget,
    estimate_monte_carlo_memory,
)


def test_maintained_configuration_fits_16_gib_budget():
    estimate = estimate_monte_carlo_memory(
        samples=4096, transmitters=16, receivers=1024, depth=3
    )

    enforce_memory_budget(
        estimate,
        budget_bytes=16 << 30,
        headroom_bytes=1 << 30,
        workload="maintained BDPT",
    )


def test_100m_sample_artifact_fails_before_allocation_with_actionable_error():
    estimate = estimate_monte_carlo_memory(
        samples=100_000_000, transmitters=1, receivers=1024, depth=3
    )

    with pytest.raises(
        MemoryBudgetError,
        match="before launch.*Reduce samples, TX/RX count, depth",
    ):
        enforce_memory_budget(
            estimate,
            budget_bytes=16 << 30,
            headroom_bytes=1 << 30,
            workload="100M MC",
        )


def test_memory_estimate_rejects_integer_overflow():
    with pytest.raises(MemoryBudgetError, match="overflows"):
        checked_product(1 << 62, 8, label="test")
