from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest
import torch

from witwin.channel_native.propagation.models import CapacityPathLayout
from witwin.channel_native.runtime import (
    CapacityFailureState,
    create_capacity_failure_state,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _layout_inputs(
    *, pair_count: int = 2, capacity: int = 3
) -> dict[str, int | torch.Tensor | CapacityFailureState]:
    device = torch.device("cuda:0")
    reference = torch.empty(0, device=device)
    return {
        "pair_count": pair_count,
        "path_capacity_per_pair": capacity,
        "failure_state": create_capacity_failure_state(reference),
        "valid": torch.zeros(pair_count * capacity, dtype=torch.bool, device=device),
        "num_paths": torch.zeros(pair_count, dtype=torch.int32, device=device),
        "overflow": torch.zeros(1, dtype=torch.bool, device=device),
    }


@pytest.mark.parametrize(
    ("pair_count", "capacity", "valid_values", "count_values"),
    [
        (2, 0, [], [0, 0]),
        (2, 3, [True, False, False, False, True, False], [1, 1]),
        (2, 3, [True, True, True, True, True, True], [3, 3]),
    ],
)
def test_capacity_layout_accepts_zero_sparse_and_dense_rows(
    pair_count: int,
    capacity: int,
    valid_values: list[bool],
    count_values: list[int],
) -> None:
    inputs = _layout_inputs(pair_count=pair_count, capacity=capacity)
    inputs["valid"] = torch.tensor(valid_values, dtype=torch.bool, device="cuda")
    inputs["num_paths"] = torch.tensor(
        count_values, dtype=torch.int32, device="cuda"
    )

    layout = CapacityPathLayout(**inputs)  # type: ignore[arg-type]

    assert layout.pair_count == pair_count
    assert layout.path_capacity_per_pair == capacity
    assert layout.row_capacity == pair_count * capacity
    assert layout.device.type == "cuda"
    assert layout.valid is inputs["valid"]
    assert layout.failure_state is inputs["failure_state"]
    assert layout.failure_state.bits.data_ptr() == inputs["failure_state"].bits.data_ptr()
    assert layout.num_paths is inputs["num_paths"]
    assert layout.overflow is inputs["overflow"]
    with pytest.raises(FrozenInstanceError):
        layout.pair_count = 0  # type: ignore[misc]


@pytest.mark.parametrize("name", ["pair_count", "path_capacity_per_pair"])
def test_capacity_layout_rejects_non_integer_host_counts(name: str) -> None:
    inputs = _layout_inputs()
    inputs[name] = True

    with pytest.raises(TypeError, match=rf"{name} must be an int"):
        CapacityPathLayout(**inputs)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["pair_count", "path_capacity_per_pair"])
def test_capacity_layout_rejects_negative_host_counts(name: str) -> None:
    inputs = _layout_inputs()
    inputs[name] = -1

    with pytest.raises(ValueError, match=rf"{name} must be non-negative"):
        CapacityPathLayout(**inputs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "replacement", "error"),
    [
        (
            "valid",
            lambda: torch.zeros(5, dtype=torch.bool, device="cuda"),
            r"valid must have shape \(6,\)",
        ),
        (
            "num_paths",
            lambda: torch.zeros(3, dtype=torch.int32, device="cuda"),
            r"num_paths must have shape \(2,\)",
        ),
        (
            "overflow",
            lambda: torch.zeros(2, dtype=torch.bool, device="cuda"),
            r"overflow must have shape \(1,\)",
        ),
        (
            "valid",
            lambda: torch.zeros(6, dtype=torch.float32, device="cuda"),
            "valid must use torch.bool",
        ),
        (
            "num_paths",
            lambda: torch.zeros(2, dtype=torch.int64, device="cuda"),
            "num_paths must use torch.int32",
        ),
        (
            "overflow",
            lambda: torch.zeros(1, dtype=torch.uint8, device="cuda"),
            "overflow must use torch.bool",
        ),
        (
            "valid",
            lambda: torch.zeros(12, dtype=torch.bool, device="cuda")[::2],
            "valid must be contiguous",
        ),
        (
            "num_paths",
            lambda: torch.zeros(4, dtype=torch.int32, device="cuda")[::2],
            "num_paths must be contiguous",
        ),
    ],
)
def test_capacity_layout_rejects_invalid_tensor_metadata(
    name: str,
    replacement: Callable[[], torch.Tensor],
    error: str,
) -> None:
    inputs = _layout_inputs()
    inputs[name] = replacement()

    with pytest.raises(ValueError, match=error):
        CapacityPathLayout(**inputs)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["valid", "num_paths", "overflow"])
def test_capacity_layout_requires_cuda_tensors(name: str) -> None:
    inputs = _layout_inputs()
    source = inputs[name]
    assert isinstance(source, torch.Tensor)
    inputs[name] = torch.empty_like(source, device="cpu")

    with pytest.raises(ValueError, match=rf"{name} must be a CUDA tensor"):
        CapacityPathLayout(**inputs)  # type: ignore[arg-type]


def test_capacity_layout_requires_one_cuda_device() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    inputs = _layout_inputs()
    inputs["num_paths"] = torch.zeros(2, dtype=torch.int32, device="cuda:1")

    with pytest.raises(ValueError, match="num_paths must be on cuda:0"):
        CapacityPathLayout(**inputs)  # type: ignore[arg-type]
