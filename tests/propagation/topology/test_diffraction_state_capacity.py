from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from witwin.channel_native.propagation.topology.kernels import primitives


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _states(count: int) -> tuple[torch.Tensor, ...]:
    rows = torch.arange(count, device="cuda", dtype=torch.float32)
    vectors = torch.stack((rows, rows + 0.25, rows + 0.5), dim=1)
    return (
        torch.arange(count, device="cuda", dtype=torch.int32),
        vectors,
        vectors + 10.0,
        rows + 20.0,
        rows + 30.0,
        vectors + 40.0,
        vectors + 50.0,
        torch.arange(count, device="cuda", dtype=torch.int32) + 60,
        torch.arange(count, device="cuda", dtype=torch.int32) + 70,
        rows + 80.0,
        vectors + 90.0,
        rows + 100.0,
    )


def _select(
    active: torch.Tensor,
    states: tuple[torch.Tensor, ...],
    capacity: int,
) -> primitives.DiffractionStateCapacityBlock:
    return primitives.deterministic_diffraction_state_capacity_select(
        active=active,
        edge_index=states[0],
        edge_position=states[1],
        edge_direction=states[2],
        edge_t_min=states[3],
        edge_t_max=states[4],
        n0=states[5],
        n1=states[6],
        prim0=states[7],
        prim1=states[8],
        exterior_angle=states[9],
        source=states[10],
        source_power=states[11],
        state_capacity=capacity,
    )


@pytest.mark.parametrize(
    ("active_values", "capacity", "expected_indices"),
    (
        ((False, False, False), 3, ()),
        ((False, True, False), 3, (1,)),
        ((True, False, True, True), 3, (0, 2, 3)),
    ),
)
def test_diffraction_state_capacity_select_is_stable_and_inert(
    active_values: tuple[bool, ...],
    capacity: int,
    expected_indices: tuple[int, ...],
) -> None:
    states = _states(len(active_values))
    active = torch.tensor(active_values, device="cuda", dtype=torch.bool)

    block = _select(active, states, capacity)

    expected_count = len(expected_indices)
    assert block.edge_index.shape == (capacity,)
    assert block.edge_position.shape == (capacity, 3)
    assert block.actual_count.tolist() == [expected_count]
    assert block.overflow.tolist() == [False]
    assert block.valid.tolist() == [True] * expected_count + [False] * (
        capacity - expected_count
    )
    if expected_count:
        index = torch.tensor(expected_indices, device="cuda", dtype=torch.long)
        for actual, source in zip(
            (
                block.edge_index,
                block.edge_position,
                block.edge_direction,
                block.edge_t_min,
                block.edge_t_max,
                block.n0,
                block.n1,
                block.prim0,
                block.prim1,
                block.exterior_angle,
                block.source,
                block.source_power,
            ),
            states,
            strict=True,
        ):
            torch.testing.assert_close(actual[:expected_count], source[index])
    if expected_count < capacity:
        assert block.edge_index[expected_count:].tolist() == [-1] * (
            capacity - expected_count
        )
        assert block.prim0[expected_count:].tolist() == [-1] * (
            capacity - expected_count
        )
        assert block.prim1[expected_count:].tolist() == [-1] * (
            capacity - expected_count
        )
        for tensor in (
            block.edge_position,
            block.edge_direction,
            block.edge_t_min,
            block.edge_t_max,
            block.n0,
            block.n1,
            block.exterior_angle,
            block.source,
            block.source_power,
        ):
            assert torch.count_nonzero(tensor[expected_count:]).item() == 0


def test_diffraction_state_capacity_select_handles_zero_and_effective_capacity() -> (
    None
):
    empty = _select(
        torch.empty((0,), device="cuda", dtype=torch.bool),
        _states(0),
        8,
    )
    assert empty.edge_index.shape == (0,)
    assert empty.valid.shape == (0,)
    assert empty.actual_count.tolist() == [0]
    assert empty.overflow.tolist() == [False]

    one = _select(
        torch.ones((1,), device="cuda", dtype=torch.bool),
        _states(1),
        8,
    )
    assert one.edge_index.shape == (1,)
    assert one.valid.tolist() == [True]
    assert one.actual_count.tolist() == [1]


def test_diffraction_state_capacity_select_accepts_strided_inputs_and_stream() -> None:
    dense = _states(8)
    states = tuple(tensor[::2] for tensor in dense)
    active = torch.tensor([False, True, False, True], device="cuda", dtype=torch.bool)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        block = _select(active, states, 2)
    stream.synchronize()

    assert block.edge_index.tolist() == [2, 6]
    assert block.valid.tolist() == [True, True]
    assert block.actual_count.tolist() == [2]
    torch.testing.assert_close(block.edge_position, states[1][[1, 3]])


def test_diffraction_state_capacity_overflow_fails_asynchronously_in_subprocess() -> (
    None
):
    script = r"""
import torch
from witwin.channel_native.propagation.topology.kernels import primitives

n = 3
rows = torch.arange(n, device="cuda", dtype=torch.float32)
vec = torch.stack((rows, rows + 1, rows + 2), dim=1)
block = primitives.deterministic_diffraction_state_capacity_select(
    active=torch.ones((n,), device="cuda", dtype=torch.bool),
    edge_index=torch.arange(n, device="cuda", dtype=torch.int32),
    edge_position=vec,
    edge_direction=vec,
    edge_t_min=rows,
    edge_t_max=rows,
    n0=vec,
    n1=vec,
    prim0=torch.arange(n, device="cuda", dtype=torch.int32),
    prim1=torch.arange(n, device="cuda", dtype=torch.int32),
    exterior_angle=rows,
    source=vec,
    source_power=rows,
    state_capacity=2,
)
assert block.edge_index.shape == (2,)
try:
    torch.cuda.synchronize()
except RuntimeError:
    raise SystemExit(0)
raise SystemExit("expected asynchronous diffraction capacity overflow")
"""
    repo_root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root / "src"), str(repo_root), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_diffraction_state_capacity_selector_has_no_host_count_transfer() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "native"
        / "channel_native"
        / "kernels"
        / "diffraction_state_capacity.cu"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "cudaMemcpy",
        "cudaStreamSynchronize",
        ".item",
        ".cpu",
    ):
        assert forbidden not in source
