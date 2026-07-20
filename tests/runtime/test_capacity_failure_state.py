from __future__ import annotations

from pathlib import Path

import pytest
import torch

from witwin.channel_native.propagation.topology.kernels import (
    compaction,
    coupled,
    reflection,
)
from witwin.channel_native.runtime import (
    CapacityFailureBit,
    CapacityFailureState,
    create_capacity_failure_state,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _reflection_overflow(
    failure_state: CapacityFailureState,
) -> object:
    count = 2
    depth = 1
    sequence = torch.zeros((count, depth), device="cuda", dtype=torch.int32)
    vectors = torch.zeros((count, depth, 3), device="cuda")
    return reflection.deterministic_reflection_candidate_capacity_block(
        failure_state=failure_state,
        visible=torch.ones(count, device="cuda", dtype=torch.bool),
        epc_sequences=sequence,
        epc_hits=vectors,
        epc_normals=vectors,
        sequence_batch=sequence,
        rx_indices=torch.zeros(count, device="cuda", dtype=torch.int32),
        tx=torch.zeros(3, device="cuda"),
        rx_positions=torch.zeros((1, 3), device="cuda"),
        tx_power=torch.ones(1, device="cuda"),
        tx_index=0,
        face_eps_r=torch.ones(1, device="cuda"),
        face_sigma_e=torch.zeros(1, device="cuda"),
        face_mu_r=torch.ones(1, device="cuda"),
        face_gain=torch.ones(1, device="cuda"),
        face_material_id=torch.zeros(1, device="cuda", dtype=torch.int32),
        grouped_export=True,
        candidate_capacity=1,
    )


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


def test_capacity_failure_bits_accumulate_and_every_block_stays_inert() -> None:
    reference = torch.empty(0, device="cuda")
    state = create_capacity_failure_state(reference)

    reflection_block = _reflection_overflow(state)
    coupled_block = coupled.coupled_candidate_capacity_block(
        torch.tensor([5], device="cuda", dtype=torch.int32),
        torch.tensor([7, 9], device="cuda", dtype=torch.int32),
        failure_state=state,
        tx_count=1,
        rx_count=1,
        rx_id_offset=0,
        candidate_capacity=5,
        candidate_limit=100,
    )

    expected = (
        CapacityFailureBit.REFLECTION_CANDIDATE_OVERFLOW
        | CapacityFailureBit.COUPLED_CANDIDATE_OVERFLOW
    )
    assert state.bits.tolist() == [int(expected)]
    assert reflection_block.failure_state is state
    assert coupled_block.failure_state is state
    assert reflection_block.valid.tolist() == [False]
    assert reflection_block.candidate_count.tolist() == [0]
    assert coupled_block.valid.tolist() == [False] * 5
    assert coupled_block.candidate_count.tolist() == [0]


def test_upstream_failure_blocks_poison_payload_and_local_status_reads() -> None:
    reference = torch.empty(0, device="cuda")
    state = create_capacity_failure_state(reference)
    _reflection_overflow(state)
    poison_int = torch.tensor(
        [torch.iinfo(torch.int32).max], device="cuda", dtype=torch.int32
    )
    poison_float = torch.tensor([float("nan")], device="cuda")

    block = compaction.deterministic_diffraction_order1_capacity_block(
        failure_state=state,
        count=poison_int,
        valid=torch.ones(1, device="cuda", dtype=torch.bool),
        rx_id=poison_int,
        depth=poison_int,
        edge_id=poison_int,
        delay_s=poison_float,
        x_re=poison_float,
        x_im=poison_float,
        y_re=poison_float,
        y_im=poison_float,
        z_re=poison_float,
        z_im=poison_float,
        interaction_position=torch.full((1, 3), float("nan"), device="cuda"),
        output_capacity=1,
    )

    assert state.bits.tolist() == [
        int(CapacityFailureBit.REFLECTION_CANDIDATE_OVERFLOW)
    ]
    assert block.failure_state is state
    assert block.num_paths.tolist() == [0]
    assert block.overflow.tolist() == [False]
    assert block.valid.tolist() == [False]
    assert block.rx_id.tolist() == [-1]
    assert torch.isfinite(block.interaction_position).all().item()


def test_capacity_intermediates_have_no_trap_or_host_synchronization() -> None:
    root = Path(__file__).resolve().parents[2]
    sources = (
        "diffraction_state_capacity.cu",
        "diffraction_path_capacity.cu",
        "coupled_candidate_capacity.cu",
        "reflection_candidate_capacity.cu",
        "deterministic_capacity_finalize.cu",
        "evaluated_paths_capacity_pack.cu",
    )
    for name in sources:
        source = (
            root / "native" / "channel_native" / "kernels" / name
        ).read_text(encoding="utf-8")
        for forbidden in (
            "trap;",
            "cudaMemcpy",
            "cudaStreamSynchronize",
            ".item",
            ".cpu",
        ):
            assert forbidden not in source, f"{name} contains {forbidden}"

    initializer = (
        root
        / "native"
        / "channel_native"
        / "kernels"
        / "capacity_failure_state.cu"
    ).read_text(encoding="utf-8")
    assert "cudaMemsetAsync" in initializer
    assert "getCurrentCUDAStream" in initializer
