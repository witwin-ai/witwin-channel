# Copyright Xingyu Chen.
# Tests capacity failure terminal.

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
    capacity_failure_terminal_check,
    create_capacity_failure_state,
)
from witwin.channel import runtime


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_capacity_failure_terminal_zero_state_is_current_stream_noop() -> None:
    reference = torch.empty(1, device="cuda")
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        state = create_capacity_failure_state(reference)
        result = capacity_failure_terminal_check(state)
    stream.synchronize()

    assert result is None
    assert state.bits.tolist() == [0]


def test_capacity_failure_terminal_requires_typed_state_before_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime,
        "required_symbol",
        lambda _name: pytest.fail("native dispatch accepted an untyped state"),
    )
    with pytest.raises(TypeError, match="CapacityFailureState"):
        runtime.capacity_failure_terminal_check(
            torch.zeros(1, device="cuda", dtype=torch.int32)  # type: ignore[arg-type]
        )


def test_capacity_failure_terminal_native_bridge_rejects_bad_metadata() -> None:
    native = runtime.required_symbol("capacity_failure_terminal_check")

    with pytest.raises(RuntimeError, match="CUDA tensor"):
        native(torch.zeros(1, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="wrong dtype"):
        native(torch.zeros(1, device="cuda", dtype=torch.int64))
    with pytest.raises(RuntimeError, match=r"shape \(1,\)"):
        native(torch.zeros(2, device="cuda", dtype=torch.int32))


def test_capacity_failure_terminal_missing_symbol_has_no_fallback(monkeypatch) -> None:
    state = CapacityFailureState(
        torch.zeros(1, device="cuda", dtype=torch.int32)
    )
    requested: list[str] = []

    def missing(name: str):
        requested.append(name)
        raise runtime.NativeSymbolError("terminal native symbol is required")

    monkeypatch.setattr(runtime, "required_symbol", missing)
    with pytest.raises(
        runtime.NativeSymbolError, match="terminal native symbol is required"
    ):
        runtime.capacity_failure_terminal_check(state)

    assert requested == ["capacity_failure_terminal_check"]


def test_capacity_failure_terminal_trap_isolated_to_subprocess() -> None:
    code = textwrap.dedent(
        """
        import sys
        import torch

        sys.meta_path = [
            finder
            for finder in sys.meta_path
            if "_witwin_channel_editable" not in type(finder).__module__
        ]

        from witwin.channel.runtime import (
            CapacityFailureBit,
            capacity_failure_terminal_check,
            create_capacity_failure_state,
        )

        reference = torch.tensor([5], device="cuda", dtype=torch.int32)
        state = create_capacity_failure_state(reference)
        expected = int(CapacityFailureBit.SEGMENT_PENETRATION_FAILURE)
        state.bits.fill_(expected)

        # Test-only observation before the terminal launch proves the failure
        # state itself is a plain device value and did not poison CUDA.
        torch.cuda.synchronize()
        assert state.bits.tolist() == [expected]

        terminal_stream = torch.cuda.Stream()
        with torch.cuda.stream(terminal_stream):
            capacity_failure_terminal_check(state)
        print("TERMINAL_ENQUEUED", flush=True)
        try:
            terminal_stream.synchronize()
        except RuntimeError:
            print("TERMINAL_SYNC_ERROR", flush=True)
        else:
            raise AssertionError("terminal failure was not exposed at synchronization")
        """
    )
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT)
    core_root = str(REPOSITORY_ROOT.parent / "core-radar-architecture-stage1")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (core_root, source_root, environment.get("PYTHONPATH"))
        if value
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert "TERMINAL_ENQUEUED" in completed.stdout
    assert "TERMINAL_SYNC_ERROR" in completed.stdout


def test_capacity_failure_terminal_source_has_one_async_device_observer() -> None:
    native = (
        REPOSITORY_ROOT
        / "native/channel/kernels/capacity.cu"
    ).read_text(encoding="utf-8")
    facade = (
        REPOSITORY_ROOT
        / "witwin/channel/runtime.py"
    ).read_text(encoding="utf-8")
    terminal = native.split(
        "// ==== Section: Capacity failure terminal check ====", 1
    )[1].split(
        "// ==== Section: Enumerated capacity sanitization ====", 1
    )[0]

    assert terminal.count('asm volatile("trap;")') == 1
    assert "getCurrentCUDAStream" in terminal
    assert "const int *__restrict__ failure_state" in terminal
    assert "failure_state[0] != 0" in terminal
    assert "failure_state[0] =" not in terminal
    assert "at::empty" not in terminal
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" not in terminal
    assert "cudaGetLastError" not in terminal
    assert "cudaPeekAtLastError" not in terminal
    for forbidden in (
        "cudaMemcpy",
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
        ".item(",
        ".cpu(",
        ".numpy(",
    ):
        assert forbidden not in terminal
        assert forbidden not in facade

    trap_sources = sorted(
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in (REPOSITORY_ROOT / "native/channel").rglob("*")
        if path.is_file()
        and path.suffix in {".cu", ".cuh", ".cpp", ".h"}
        and "trap;" in path.read_text(encoding="utf-8")
    )
    assert trap_sources == [
        "native/channel/kernels/capacity.cu"
    ]

    production_mentions = sorted(
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in (REPOSITORY_ROOT / "witwin/channel").rglob("*.py")
        if "capacity_failure_terminal_check" in path.read_text(encoding="utf-8")
    )
    assert production_mentions == ["witwin/channel/runtime.py"]