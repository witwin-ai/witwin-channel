# Copyright Xingyu Chen.
# Tests the host-side observation of the capacity failure record.

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch

from witwin.channel.runtime import (
    CapacityFailureState,
    capacity_failure_state_read,
    create_capacity_failure_state,
)
from witwin.channel import runtime


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_capacity_failure_read_returns_bits_and_dropped_rows_on_current_stream() -> None:
    reference = torch.empty(1, device="cuda")
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        state = create_capacity_failure_state(reference)
        state.dropped_rows.fill_(4)
        assert capacity_failure_state_read(state) == (0, 4)


def test_capacity_failure_read_requires_typed_state_before_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime,
        "required_symbol",
        lambda _name: pytest.fail("native dispatch accepted an untyped state"),
    )
    with pytest.raises(TypeError, match="CapacityFailureState"):
        runtime.capacity_failure_state_read(
            torch.zeros(2, device="cuda", dtype=torch.int32)  # type: ignore[arg-type]
        )


def test_capacity_failure_read_native_bridge_rejects_bad_metadata() -> None:
    native = runtime.required_symbol("capacity_failure_state_read")

    with pytest.raises(RuntimeError, match="CUDA tensor"):
        native(torch.zeros(2, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="wrong dtype"):
        native(torch.zeros(2, device="cuda", dtype=torch.int64))
    with pytest.raises(RuntimeError, match=r"shape \(2,\)"):
        native(torch.zeros(1, device="cuda", dtype=torch.int32))


def test_capacity_failure_read_missing_symbol_has_no_fallback(monkeypatch) -> None:
    state = CapacityFailureState(torch.zeros(2, device="cuda", dtype=torch.int32))
    requested: list[str] = []

    def missing(name: str):
        requested.append(name)
        raise runtime.NativeSymbolError("observation native symbol is required")

    monkeypatch.setattr(runtime, "required_symbol", missing)
    with pytest.raises(runtime.NativeSymbolError, match="observation native symbol is required"):
        runtime.capacity_failure_state_read(state)

    assert requested == ["capacity_failure_state_read"]


def test_failed_transaction_raises_capacity_error_and_keeps_the_context_usable() -> None:
    code = textwrap.dedent(
        """
        import torch

        from witwin.channel.runtime import (
            CapacityError,
            CapacityFailureBit,
            create_solve_capacity_transaction,
        )

        transaction = create_solve_capacity_transaction(torch.empty(1, device="cuda"))
        transaction.failure_state.bits.fill_(int(CapacityFailureBit.SEGMENT_PENETRATION_FAILURE))
        try:
            transaction.terminal_check()
        except CapacityError as error:
            print("NAMES", ",".join(error.bit_names), flush=True)
        else:
            raise AssertionError("a failed transaction did not raise")
        assert float(torch.ones(3, device="cuda").sum()) == 3.0
        print("CONTEXT_USABLE", flush=True)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert "NAMES SEGMENT_PENETRATION_FAILURE" in completed.stdout
    assert "CONTEXT_USABLE" in completed.stdout


def test_native_channel_sources_contain_no_device_trap() -> None:
    trap_sources = sorted(
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in (REPOSITORY_ROOT / "native/channel").rglob("*")
        if path.is_file()
        and path.suffix in {".cu", ".cuh", ".cpp", ".h"}
        and ("trap;" in (text := path.read_text(encoding="utf-8")) or "__trap" in text)
    )
    assert trap_sources == []
