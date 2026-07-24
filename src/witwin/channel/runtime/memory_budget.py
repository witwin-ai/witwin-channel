from __future__ import annotations

from dataclasses import dataclass


_MAX_SIGNED_BYTES = (1 << 63) - 1


class MemoryBudgetError(RuntimeError):
    """Raised before launch when a requested workload cannot fit its budget."""


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    persistent_bytes: int = 0
    temporary_bytes: int = 0
    output_bytes: int = 0
    tape_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "persistent_bytes",
            "temporary_bytes",
            "output_bytes",
            "tape_bytes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def total_bytes(self) -> int:
        return _checked_sum(
            self.persistent_bytes,
            self.temporary_bytes,
            self.output_bytes,
            self.tape_bytes,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "persistent_bytes": self.persistent_bytes,
            "temporary_bytes": self.temporary_bytes,
            "output_bytes": self.output_bytes,
            "tape_bytes": self.tape_bytes,
            "total_bytes": self.total_bytes,
        }


def _checked_sum(*values: int) -> int:
    total = 0
    for value in values:
        total += int(value)
        if total > _MAX_SIGNED_BYTES:
            raise MemoryBudgetError(
                "memory estimate exceeds the supported signed 64-bit byte range"
            )
    return total


def checked_product(*values: int, label: str = "workload") -> int:
    product = 1
    for value in values:
        value = int(value)
        if value < 0:
            raise ValueError(f"{label} dimensions must be non-negative")
        if value and product > _MAX_SIGNED_BYTES // value:
            raise MemoryBudgetError(
                f"{label} memory estimate overflows the signed 64-bit byte range"
            )
        product *= value
    return product


def estimate_monte_carlo_memory(
    *,
    samples: int,
    transmitters: int,
    receivers: int,
    depth: int,
    bytes_per_path_state: int = 192,
    output_bytes_per_pair: int = 16,
    persistent_bytes: int = 0,
    tape_bytes: int = 0,
) -> MemoryEstimate:
    """Conservative, allocation-free estimate for MC/BDPT scale sweeps.

    The per-path default covers two complex3 states, geometry/PDF state, and
    compaction indices. Callers may override it with a measured solver value.
    """

    for name, value in {
        "samples": samples,
        "transmitters": transmitters,
        "receivers": receivers,
        "depth": depth,
        "bytes_per_path_state": bytes_per_path_state,
        "output_bytes_per_pair": output_bytes_per_pair,
        "persistent_bytes": persistent_bytes,
        "tape_bytes": tape_bytes,
    }.items():
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative")

    active_depth = max(1, int(depth))
    temporary = checked_product(
        samples,
        transmitters,
        active_depth,
        bytes_per_path_state,
        label="Monte Carlo path state",
    )
    output = checked_product(
        transmitters,
        receivers,
        output_bytes_per_pair,
        label="Monte Carlo output",
    )
    return MemoryEstimate(
        persistent_bytes=int(persistent_bytes),
        temporary_bytes=temporary,
        output_bytes=output,
        tape_bytes=int(tape_bytes),
    )


def enforce_memory_budget(
    estimate: MemoryEstimate,
    *,
    budget_bytes: int,
    workload: str,
    headroom_bytes: int = 0,
) -> None:
    """Fail with an actionable error before any workload allocation occurs."""

    if budget_bytes < 0 or headroom_bytes < 0:
        raise ValueError("budget_bytes and headroom_bytes must be non-negative")
    required = _checked_sum(estimate.total_bytes, headroom_bytes)
    if required <= int(budget_bytes):
        return
    raise MemoryBudgetError(
        f"{workload} exceeds the GPU memory budget before launch: estimated "
        f"{estimate.total_bytes} bytes plus {headroom_bytes} bytes headroom "
        f"requires {required} bytes, budget is {int(budget_bytes)} bytes. "
        "Reduce samples, TX/RX count, depth, grid resolution, or exported paths."
    )
