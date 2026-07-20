from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from witwin.channel_native.propagation.topology.kernels import compaction
from witwin.channel_native.propagation.topology.kernels import reflection


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _inputs(
    visible_values: tuple[bool, ...], *, depth: int
) -> dict[str, torch.Tensor | int | bool]:
    count = len(visible_values)
    rows = torch.arange(count, device="cuda", dtype=torch.float32)
    columns = torch.arange(depth, device="cuda", dtype=torch.int32)
    sequences = (
        torch.arange(count, device="cuda", dtype=torch.int32).reshape(-1, 1)
        + columns.reshape(1, -1)
    ) % 6
    hits = torch.empty((count, depth, 3), device="cuda", dtype=torch.float32)
    normals = torch.empty_like(hits)
    for column in range(depth):
        hits[:, column, 0] = rows + 10.0 * column
        hits[:, column, 1] = rows + 20.0 + 10.0 * column
        hits[:, column, 2] = rows + 40.0 + 10.0 * column
        normals[:, column, 0] = 1.0 + column
        normals[:, column, 1] = 2.0 + column
        normals[:, column, 2] = 3.0 + column
    return {
        "visible": torch.tensor(visible_values, device="cuda", dtype=torch.bool),
        "epc_sequences": sequences,
        "epc_hits": hits,
        "epc_normals": normals,
        "sequence_batch": (sequences + 1) % 6,
        "rx_indices": torch.arange(count, device="cuda", dtype=torch.int32) % 4,
        "tx": torch.tensor([1.0, 2.0, 3.0], device="cuda"),
        "rx_positions": torch.arange(
            12, device="cuda", dtype=torch.float32
        ).reshape(4, 3),
        "tx_power": torch.tensor([7.0, 11.0], device="cuda"),
        "tx_index": 1,
        "face_eps_r": torch.arange(6, device="cuda", dtype=torch.float32) + 2.0,
        "face_sigma_e": torch.arange(6, device="cuda", dtype=torch.float32)
        + 12.0,
        "face_mu_r": torch.arange(6, device="cuda", dtype=torch.float32) + 22.0,
        "face_gain": torch.arange(6, device="cuda", dtype=torch.float32) + 32.0,
        "face_material_id": torch.arange(6, device="cuda", dtype=torch.int32)
        + 42,
        "grouped_export": True,
    }


def _produce(
    inputs: dict[str, torch.Tensor | int | bool], *, candidate_capacity: int
):
    return reflection.deterministic_reflection_candidate_capacity_block(
        **inputs,
        candidate_capacity=candidate_capacity,
    )


def _assert_inert(block: object, start: int) -> None:
    capacity = block.candidate_capacity
    invalid_count = capacity - start
    assert block.valid[start:].tolist() == [False] * invalid_count
    for tensor in (block.selected_rx_id, block.first_face, block.material_id):
        assert tensor[start:].tolist() == [-1] * invalid_count
    for tensor in (block.selected_sequences, block.material_sequence):
        assert torch.count_nonzero(tensor[start:] + 1).item() == 0
    for tensor in (
        block.selected_hits,
        block.selected_normals,
        block.selected_tx,
        block.selected_rx,
        block.tx_power,
        block.eps_r,
        block.sigma_e,
        block.mu_r,
        block.gain,
        block.first_hit,
        block.first_normal,
    ):
        assert torch.count_nonzero(tensor[start:]).item() == 0


@pytest.mark.parametrize("sequence_dtype", (torch.int32, torch.int64))
def test_reflection_candidate_capacity_matches_multibounce_compact_order(
    sequence_dtype: torch.dtype,
) -> None:
    inputs = _inputs((False, True, False, True, True), depth=2)
    inputs["epc_sequences"] = inputs["epc_sequences"].to(sequence_dtype)
    block = _produce(inputs, candidate_capacity=5)
    old = compaction.deterministic_reflection_sequence_compact(
        visible=inputs["visible"],
        epc_sequences=inputs["epc_sequences"],
        epc_hits=inputs["epc_hits"],
        epc_normals=inputs["epc_normals"],
        rx_indices=inputs["rx_indices"],
        tx=inputs["tx"],
        rx_positions=inputs["rx_positions"],
        tx_power=inputs["tx_power"],
        tx_index=inputs["tx_index"],
        face_eps_r=inputs["face_eps_r"],
        face_sigma_e=inputs["face_sigma_e"],
        face_mu_r=inputs["face_mu_r"],
        face_gain=inputs["face_gain"],
        face_material_id=inputs["face_material_id"],
        max_count=-1,
    )

    assert block.candidate_count.tolist() == [3]
    assert block.overflow.tolist() == [False]
    assert block.valid.tolist() == [True, True, True, False, False]
    for name in (
        "selected_sequences",
        "selected_hits",
        "selected_normals",
        "selected_rx_id",
        "selected_tx",
        "selected_rx",
        "tx_power",
        "eps_r",
        "sigma_e",
        "mu_r",
        "gain",
        "first_face",
        "material_id",
        "material_sequence",
        "first_hit",
        "first_normal",
    ):
        torch.testing.assert_close(getattr(block, name)[:3], old[name], rtol=0, atol=0)
    _assert_inert(block, 3)


@pytest.mark.parametrize("sequence_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("grouped_export", (False, True))
def test_reflection_candidate_capacity_matches_order1_face_semantics(
    grouped_export: bool,
    sequence_dtype: torch.dtype,
) -> None:
    inputs = _inputs((True, False, True, False), depth=1)
    inputs["epc_sequences"] = inputs["epc_sequences"].to(sequence_dtype)
    inputs["grouped_export"] = grouped_export
    block = _produce(inputs, candidate_capacity=4)
    old = compaction.deterministic_reflection_order1_compact(
        visible=inputs["visible"],
        epc_faces=inputs["epc_sequences"],
        epc_hits=inputs["epc_hits"],
        epc_normals=inputs["epc_normals"],
        sequence_batch=inputs["sequence_batch"],
        rx_indices=inputs["rx_indices"],
        tx=inputs["tx"],
        rx_positions=inputs["rx_positions"],
        tx_power=inputs["tx_power"],
        tx_index=inputs["tx_index"],
        face_eps_r=inputs["face_eps_r"],
        face_sigma_e=inputs["face_sigma_e"],
        face_mu_r=inputs["face_mu_r"],
        face_gain=inputs["face_gain"],
        face_material_id=inputs["face_material_id"],
        grouped_export=grouped_export,
    )

    torch.testing.assert_close(block.first_face[:2], old["selected_faces"])
    torch.testing.assert_close(block.first_hit[:2], old["selected_points"])
    torch.testing.assert_close(block.first_normal[:2], old["selected_normals"])
    torch.testing.assert_close(block.selected_rx_id[:2], old["selected_rx_id"])
    torch.testing.assert_close(block.selected_tx[:2], old["tx_keep"])
    torch.testing.assert_close(block.selected_rx[:2], old["rx_keep"])
    torch.testing.assert_close(block.tx_power[:2], old["tx_power"])
    for name in ("eps_r", "sigma_e", "mu_r", "gain"):
        torch.testing.assert_close(getattr(block, name)[:2, 0], old[name])
    torch.testing.assert_close(block.material_id[:2], old["material_id"])
    _assert_inert(block, 2)


def test_reflection_candidate_capacity_handles_empty_input_and_capacity() -> None:
    block = _produce(_inputs((), depth=2), candidate_capacity=0)

    assert block.valid.shape == (0,)
    assert block.selected_sequences.shape == (0, 2)
    assert block.selected_hits.shape == (0, 2, 3)
    assert block.candidate_count.tolist() == [0]
    assert block.overflow.tolist() == [False]


def test_reflection_candidate_capacity_empty_input_initializes_nonzero_capacity() -> None:
    block = _produce(_inputs((), depth=2), candidate_capacity=3)

    assert block.candidate_count.tolist() == [0]
    assert block.overflow.tolist() == [False]
    _assert_inert(block, 0)


def test_reflection_candidate_capacity_zero_capacity_accepts_zero_visible_rows() -> None:
    block = _produce(_inputs((False, False, False), depth=2), candidate_capacity=0)

    assert block.valid.shape == (0,)
    assert block.candidate_count.tolist() == [0]
    assert block.overflow.tolist() == [False]


def test_reflection_candidate_capacity_never_reads_invalid_poison() -> None:
    inputs = _inputs((True, False, False, True), depth=2)
    inputs["epc_sequences"][1] = torch.iinfo(torch.int32).max
    inputs["sequence_batch"][1] = torch.iinfo(torch.int32).min
    inputs["epc_hits"][1] = float("nan")
    inputs["epc_normals"][1] = float("inf")
    inputs["rx_indices"][1] = torch.iinfo(torch.int32).max
    inputs["epc_sequences"][2] = 5
    inputs["sequence_batch"][2] = 5
    inputs["epc_hits"][2] = float("-inf")
    inputs["epc_normals"][2] = float("nan")
    inputs["rx_indices"][2] = torch.iinfo(torch.int32).min
    inputs["face_eps_r"][5] = float("nan")
    inputs["face_sigma_e"][5] = float("inf")
    inputs["face_mu_r"][5] = float("-inf")
    inputs["face_gain"][5] = float("nan")
    inputs["face_material_id"][5] = torch.iinfo(torch.int32).max

    block = _produce(inputs, candidate_capacity=4)

    assert block.valid.tolist() == [True, True, False, False]
    assert block.candidate_count.tolist() == [2]
    for tensor in (
        block.selected_hits,
        block.selected_normals,
        block.selected_tx,
        block.selected_rx,
        block.tx_power,
        block.eps_r,
        block.sigma_e,
        block.mu_r,
        block.gain,
        block.first_hit,
        block.first_normal,
    ):
        assert torch.isfinite(tensor).all().item()
    _assert_inert(block, 2)


def test_reflection_candidate_capacity_uses_nondefault_stream() -> None:
    inputs = _inputs((False, True, True, False), depth=2)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        block = _produce(inputs, candidate_capacity=4)
    stream.synchronize()

    assert block.valid.tolist() == [True, True, False, False]
    assert block.candidate_count.tolist() == [2]
    assert block.selected_rx_id.tolist() == [1, 2, -1, -1]


@pytest.mark.parametrize(
    ("visible_values", "candidate_capacity"),
    (((True, True, True), 2), ((True,), 0)),
)
def test_reflection_candidate_capacity_overflow_is_asynchronous_in_subprocess(
    visible_values: tuple[bool, ...], candidate_capacity: int
) -> None:
    script = f"""
import torch
from witwin.channel_native.propagation.topology.kernels import reflection

n = {len(visible_values)}
d = 2
seq = torch.zeros((n, d), device="cuda", dtype=torch.int32)
vec = torch.zeros((n, d, 3), device="cuda", dtype=torch.float32)
block = reflection.deterministic_reflection_candidate_capacity_block(
    visible=torch.tensor({visible_values!r}, device="cuda", dtype=torch.bool),
    epc_sequences=seq,
    epc_hits=vec,
    epc_normals=vec,
    sequence_batch=seq,
    rx_indices=torch.zeros((n,), device="cuda", dtype=torch.int32),
    tx=torch.zeros((3,), device="cuda", dtype=torch.float32),
    rx_positions=torch.zeros((1, 3), device="cuda", dtype=torch.float32),
    tx_power=torch.ones((1,), device="cuda", dtype=torch.float32),
    tx_index=0,
    face_eps_r=torch.ones((1,), device="cuda", dtype=torch.float32),
    face_sigma_e=torch.zeros((1,), device="cuda", dtype=torch.float32),
    face_mu_r=torch.ones((1,), device="cuda", dtype=torch.float32),
    face_gain=torch.ones((1,), device="cuda", dtype=torch.float32),
    face_material_id=torch.zeros((1,), device="cuda", dtype=torch.int32),
    grouped_export=True,
    candidate_capacity={candidate_capacity},
)
assert block.valid.shape == ({candidate_capacity},)
try:
    torch.cuda.synchronize()
except RuntimeError:
    raise SystemExit(0)
raise SystemExit("expected asynchronous reflection candidate capacity failure")
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


def test_reflection_candidate_capacity_owner_has_no_host_count_transfer() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "native"
        / "channel_native"
        / "kernels"
        / "reflection_candidate_capacity.cu"
    ).read_text(encoding="utf-8")
    for forbidden in ("cudaMemcpy", "cudaStreamSynchronize", ".item", ".cpu"):
        assert forbidden not in source
    assert "path_capacity_per_pair" not in source
    assert source.index("reflection_candidate_init_kernel<<<") < source.index(
        "reflection_candidate_status_kernel<<<"
    )
    assert source.index("reflection_candidate_gather_kernel<int64_t><<<") < source.index(
        "reflection_candidate_overflow_kernel<<<"
    )
    assert "if (overflow[0])" in source
    assert source.index("if (flags[row] == 0)") < source.index(
        "const int rx_id = rx_indices[row]"
    )
