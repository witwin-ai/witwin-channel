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
from witwin.channel.runtime import capacity as capacity_runtime
from witwin.channel.runtime import symbols


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


def test_capacity_failure_terminal_requires_typed_state_before_dispatch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        symbols,
        "required_symbol",
        lambda _name: pytest.fail("native dispatch accepted an untyped state"),
    )
    with pytest.raises(TypeError, match="CapacityFailureState"):
        capacity_runtime.capacity_failure_terminal_check(
            torch.zeros(1, device="cuda", dtype=torch.int32)  # type: ignore[arg-type]
        )


def test_capacity_failure_terminal_native_bridge_rejects_bad_metadata() -> None:
    native = symbols.required_symbol("capacity_failure_terminal_check")

    with pytest.raises(RuntimeError, match="CUDA tensor"):
        native(torch.zeros(1, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="wrong dtype"):
        native(torch.zeros(1, device="cuda", dtype=torch.int64))
    with pytest.raises(RuntimeError, match=r"shape \(1,\)"):
        native(torch.zeros(2, device="cuda", dtype=torch.int32))


def test_capacity_failure_terminal_missing_symbol_has_no_fallback(
    monkeypatch,
) -> None:
    state = CapacityFailureState(
        torch.zeros(1, device="cuda", dtype=torch.int32)
    )
    requested: list[str] = []

    def missing(name: str):
        requested.append(name)
        raise symbols.NativeSymbolError("terminal native symbol is required")

    monkeypatch.setattr(symbols, "required_symbol", missing)
    with pytest.raises(
        symbols.NativeSymbolError, match="terminal native symbol is required"
    ):
        capacity_runtime.capacity_failure_terminal_check(state)

    assert requested == ["capacity_failure_terminal_check"]


def test_capacity_failure_terminal_trap_isolated_to_subprocess() -> None:
    code = textwrap.dedent(
        """
        import torch

        from witwin.channel.propagation.topology.kernels import coupled
        from witwin.channel.runtime import (
            CapacityFailureBit,
            capacity_failure_terminal_check,
            create_capacity_failure_state,
        )

        faces = torch.tensor([5], device="cuda", dtype=torch.int32)
        edges = torch.tensor([7], device="cuda", dtype=torch.int32)
        state = create_capacity_failure_state(faces)
        block = coupled.coupled_candidate_capacity_block(
            faces,
            edges,
            failure_state=state,
            tx_count=1,
            rx_count=1,
            rx_id_offset=0,
            candidate_capacity=1,
            candidate_limit=100,
        )

        # Test-only observation before the terminal launch proves the producer
        # completed canonical sanitization and did not itself poison CUDA.
        torch.cuda.synchronize()
        expected = int(CapacityFailureBit.COUPLED_CANDIDATE_OVERFLOW)
        assert state.bits.tolist() == [expected]
        assert block.candidate_count.tolist() == [0]
        assert block.overflow.tolist() == [True]
        assert block.valid.tolist() == [False]
        for identifiers in (
            block.tx_id,
            block.rx_id,
            block.component_id,
            block.face_id,
            block.edge1_id,
            block.edge2_id,
        ):
            assert identifiers.tolist() == [-1]

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
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_root, environment.get("PYTHONPATH"))
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
        / "native/channel/kernels/capacity_failure_terminal.cu"
    ).read_text(encoding="utf-8")
    facade = (
        REPOSITORY_ROOT
        / "src/witwin/channel/runtime/capacity.py"
    ).read_text(encoding="utf-8")

    assert native.count('asm volatile("trap;")') == 1
    assert "getCurrentCUDAStream" in native
    assert "const int *__restrict__ failure_state" in native
    assert "failure_state[0] != 0" in native
    assert "failure_state[0] =" not in native
    assert "at::empty" not in native
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" not in native
    assert "cudaGetLastError" not in native
    assert "cudaPeekAtLastError" not in native
    for forbidden in (
        "cudaMemcpy",
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
        ".item(",
        ".cpu(",
        ".numpy(",
    ):
        assert forbidden not in native
        assert forbidden not in facade

    trap_sources = sorted(
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in (REPOSITORY_ROOT / "native/channel").rglob("*")
        if path.is_file()
        and path.suffix in {".cu", ".cuh", ".cpp", ".h"}
        and "trap;" in path.read_text(encoding="utf-8")
    )
    assert trap_sources == [
        "native/channel/kernels/capacity_failure_terminal.cu"
    ]

    production_mentions = sorted(
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in (REPOSITORY_ROOT / "src/witwin/channel").rglob("*.py")
        if "capacity_failure_terminal_check" in path.read_text(encoding="utf-8")
    )
    assert production_mentions == [
        "src/witwin/channel/runtime/__init__.py",
        "src/witwin/channel/runtime/capacity.py",
    ]
