from __future__ import annotations

from pathlib import Path

import pytest
import torch

from witwin.channel.propagation.topology.kernels import compaction
from witwin.channel.runtime import (
    CapacityFailureBit,
    CapacityFailureState,
    create_capacity_failure_state,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _inputs(valid_values: tuple[bool, ...]) -> dict[str, torch.Tensor]:
    capacity = len(valid_values)
    rows = torch.arange(capacity, device="cuda", dtype=torch.float32)
    return {
        "count": torch.tensor(
            [sum(valid_values)], device="cuda", dtype=torch.int32
        ),
        "valid": torch.tensor(valid_values, device="cuda", dtype=torch.bool),
        "rx_id": torch.arange(capacity, device="cuda", dtype=torch.int32) + 10,
        "depth": torch.arange(capacity, device="cuda", dtype=torch.int32) + 20,
        "edge_id": torch.arange(capacity, device="cuda", dtype=torch.int32) + 30,
        "delay_s": rows + 40.0,
        "x_re": rows + 50.0,
        "x_im": rows + 60.0,
        "y_re": rows + 70.0,
        "y_im": rows + 80.0,
        "z_re": rows + 90.0,
        "z_im": rows + 100.0,
        "interaction_position": torch.stack(
            (rows + 110.0, rows + 120.0, rows + 130.0), dim=1
        ),
    }


def _produce(
    inputs: dict[str, torch.Tensor],
    *,
    output_capacity: int,
    failure_state: CapacityFailureState | None = None,
) -> compaction.DiffractionOrder1CapacityBlock:
    failure_state = failure_state or create_capacity_failure_state(inputs["valid"])
    return compaction.deterministic_diffraction_order1_capacity_block(
        failure_state=failure_state, **inputs, output_capacity=output_capacity
    )


@pytest.mark.parametrize(
    ("valid_values", "output_capacity", "expected_rows"),
    (
        ((False, False, False), 4, ()),
        ((False, True, False, True), 5, (1, 3)),
        ((True, True, True), 3, (0, 1, 2)),
    ),
)
def test_diffraction_order1_capacity_block_is_stable_and_canonical(
    valid_values: tuple[bool, ...],
    output_capacity: int,
    expected_rows: tuple[int, ...],
) -> None:
    inputs = _inputs(valid_values)

    block = _produce(inputs, output_capacity=output_capacity)

    selected_count = len(expected_rows)
    assert block.valid.tolist() == [True] * selected_count + [False] * (
        output_capacity - selected_count
    )
    assert block.num_paths.tolist() == [selected_count]
    assert block.overflow.tolist() == [False]
    if selected_count:
        index = torch.tensor(expected_rows, device="cuda", dtype=torch.long)
        for name in (
            "rx_id",
            "depth",
            "edge_id",
            "delay_s",
            "x_re",
            "x_im",
            "y_re",
            "y_im",
            "z_re",
            "z_im",
            "interaction_position",
        ):
            torch.testing.assert_close(
                getattr(block, name)[:selected_count], inputs[name][index]
            )
    if selected_count < output_capacity:
        invalid = slice(selected_count, None)
        assert block.rx_id[invalid].tolist() == [-1] * (
            output_capacity - selected_count
        )
        assert block.depth[invalid].tolist() == [0] * (
            output_capacity - selected_count
        )
        assert block.edge_id[invalid].tolist() == [-1] * (
            output_capacity - selected_count
        )
        assert block.delay_s[invalid].tolist() == [-1.0] * (
            output_capacity - selected_count
        )
        for tensor in (
            block.x_re,
            block.x_im,
            block.y_re,
            block.y_im,
            block.z_re,
            block.z_im,
            block.interaction_position,
        ):
            assert torch.count_nonzero(tensor[invalid]).item() == 0


def test_diffraction_order1_capacity_block_handles_empty_input_and_capacity() -> None:
    block = _produce(_inputs(()), output_capacity=0)

    assert block.valid.shape == (0,)
    assert block.interaction_position.shape == (0, 3)
    assert block.num_paths.tolist() == [0]
    assert block.overflow.tolist() == [False]


def test_diffraction_order1_capacity_block_uses_nondefault_stream() -> None:
    inputs = _inputs((False, True, True, False))
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        block = _produce(inputs, output_capacity=3)
    stream.synchronize()

    assert block.rx_id.tolist() == [11, 12, -1]
    assert block.valid.tolist() == [True, True, False]
    assert block.num_paths.tolist() == [2]


def test_diffraction_order1_capacity_block_never_reads_invalid_poison() -> None:
    inputs = _inputs((True, False, True))
    inputs["rx_id"][1] = torch.iinfo(torch.int32).max
    inputs["depth"][1] = torch.iinfo(torch.int32).min
    inputs["edge_id"][1] = torch.iinfo(torch.int32).max
    inputs["delay_s"][1] = float("nan")
    inputs["x_re"][1] = float("inf")
    inputs["x_im"][1] = float("-inf")
    inputs["y_re"][1] = float("nan")
    inputs["y_im"][1] = float("inf")
    inputs["z_re"][1] = float("-inf")
    inputs["z_im"][1] = float("nan")
    inputs["interaction_position"][1] = torch.tensor(
        [float("nan"), float("inf"), float("-inf")],
        device="cuda",
        dtype=torch.float32,
    )

    block = _produce(inputs, output_capacity=3)

    assert block.valid.tolist() == [True, True, False]
    assert block.rx_id.tolist() == [10, 12, -1]
    assert block.depth.tolist() == [20, 22, 0]
    assert block.edge_id.tolist() == [30, 32, -1]
    assert block.delay_s.tolist() == [40.0, 42.0, -1.0]
    for tensor in (
        block.x_re,
        block.x_im,
        block.y_re,
        block.y_im,
        block.z_re,
        block.z_im,
        block.interaction_position,
    ):
        assert torch.isfinite(tensor).all().item()
    assert torch.count_nonzero(block.interaction_position[2]).item() == 0


@pytest.mark.parametrize(
    ("failure", "bit"),
    (
        ("overflow", CapacityFailureBit.DIFFRACTION_PATH_OVERFLOW),
        ("count_mismatch", CapacityFailureBit.DIFFRACTION_PATH_CONTRACT_ERROR),
    ),
)
def test_diffraction_order1_capacity_failure_sets_state_and_is_inert(
    failure: str, bit: CapacityFailureBit
) -> None:
    inputs = _inputs((True, True, True))
    inputs["count"] = torch.tensor(
        [2 if failure == "count_mismatch" else 3],
        device="cuda",
        dtype=torch.int32,
    )
    output_capacity = 3 if failure == "count_mismatch" else 2
    failure_state = create_capacity_failure_state(inputs["valid"])

    block = _produce(
        inputs, output_capacity=output_capacity, failure_state=failure_state
    )

    assert failure_state.bits.tolist() == [int(bit)]
    assert block.num_paths.tolist() == [0]
    assert block.overflow.tolist() == [True]
    assert block.valid.tolist() == [False] * output_capacity
    assert block.rx_id.tolist() == [-1] * output_capacity


def test_diffraction_order1_capacity_owner_has_no_host_count_transfer() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "native"
        / "channel_native"
        / "kernels"
        / "diffraction_path_capacity.cu"
    ).read_text(encoding="utf-8")
    for forbidden in ("cudaMemcpy", "cudaStreamSynchronize", ".item", ".cpu"):
        assert forbidden not in source
    assert source.index("diffraction_path_capacity_init_kernel<<<") < source.index(
        "diffraction_path_capacity_status_kernel<<<"
    )
    assert "trap;" not in source
    assert "failure_state[0] != 0" in source
