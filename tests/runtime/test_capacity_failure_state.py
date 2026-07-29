# Copyright Xingyu Chen.
# Tests capacity failure state.

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from witwin.channel.runtime import CapacityFailureState, create_capacity_failure_state


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _native_section(path: Path, section_name: str) -> str:
    text = path.read_text(encoding="utf-8")
    marker = f"// ==== Section: {section_name} ===="
    start = text.index(marker)
    end = text.find("// ==== Section: ", start + len(marker))
    return text[start:] if end < 0 else text[start:end]


def test_capacity_failure_state_is_native_zeroed_on_current_stream() -> None:
    reference = torch.empty(4, device="cuda")
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        state = create_capacity_failure_state(reference)
    stream.synchronize()

    assert state.bits.dtype == torch.int32
    assert state.bits.shape == (1,)
    assert state.bits.is_contiguous()
    assert state.bits.device == reference.device
    assert state.bits.tolist() == [0]


def test_capacity_failure_state_rejects_bad_metadata() -> None:
    with pytest.raises(ValueError, match="CUDA tensor"):
        CapacityFailureState(torch.zeros(1, dtype=torch.int32))
    with pytest.raises(ValueError, match=r"shape \(1,\)"):
        CapacityFailureState(torch.zeros(2, device="cuda", dtype=torch.int32))
    with pytest.raises(TypeError, match="torch.int32"):
        CapacityFailureState(torch.zeros(1, device="cuda", dtype=torch.int64))


def test_capacity_intermediates_have_no_trap_or_host_synchronization() -> None:
    root = Path(__file__).resolve().parents[2]
    sources = (
        ("capacity.cu", "Enumerated capacity sanitization"),
        ("evaluated_paths.cu", "Capacity-pack AD"),
        (
            "capacity.cu",
            "Monte Carlo capacity sanitization",
        ),
    )
    kernels = root / "native" / "channel" / "kernels"
    for owner, section_name in sources:
        source = _native_section(kernels / owner, section_name)
        for forbidden in (
            "trap;",
            "cudaMemcpy",
            "cudaStreamSynchronize",
            ".item",
            ".cpu",
        ):
            assert forbidden not in source, f"{section_name} contains {forbidden}"

    initializer = _native_section(
        kernels / "capacity.cu",
        "Capacity failure state",
    )
    assert "cudaMemsetAsync" in initializer
    assert "getCurrentCUDAStream" in initializer