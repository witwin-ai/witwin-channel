import math

import pytest
import torch

from witwin.channel_native.core.kernels import ops
from witwin.channel_native.propagation.fields.kernels import (
    deterministic as deterministic_fields,
)
from witwin.channel_native.runtime import symbols
from witwin.channel_native.materials.kernels import functional as material_functional
from witwin.channel_native.montecarlo.basic.kernels import sampling as mc_sampling
from witwin.channel_native.montecarlo.basic.kernels import maps as mc_maps
from witwin.channel_native.runtime import native_buffers
from witwin.channel_native.runtime import symbols as runtime_symbols
from witwin.channel_native.propagation.topology.kernels import (
    primitives as topology_primitives,
)
from witwin.channel_native.propagation.topology.kernels import (
    sampling as topology_sampling,
)


def test_validate_cuda_tensor_accepts_matching_cuda_tensor_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for positive CUDA tensor validation")

    tensor = torch.zeros((2, 3), device="cuda", dtype=torch.float32)

    assert ops.validate_cuda_tensor(
        "points", tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    ) is tensor


def test_validate_cuda_tensor_rejects_cpu_tensor():
    tensor = torch.zeros((2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="points must be a CUDA tensor"):
        ops.validate_cuda_tensor(
            "points", tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )


def test_validate_cuda_tensor_rejects_wrong_dtype_before_shape():
    tensor = torch.zeros((2, 3), dtype=torch.float64)

    with pytest.raises(TypeError, match="points must have dtype torch.float32"):
        ops.validate_cuda_tensor(
            "points", tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )


def test_validate_cuda_tensor_rejects_trailing_shape():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to isolate shape validation after device validation")

    tensor = torch.zeros((2, 2), device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="points must end with shape"):
        ops.validate_cuda_tensor(
            "points", tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )


def test_noop_metadata_reports_valid_schema():
    metadata = ops.noop_metadata(accumulation_strategy="atomic_add")

    assert metadata["primitive"] == "noop_metadata"
    assert metadata["accumulation_strategy"] == "atomic_add"
    ops.validate_metadata(metadata)


def test_path_los_export_returns_cuda_path_tensors_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path LoS export")

    tx_positions = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    tx_power = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor([[3.0, 4.0, 0.0]], device="cuda", dtype=torch.float32)

    result = ops.path_los_export(
        tx_positions,
        tx_power,
        rx_positions,
        frequency_hz=3.0e9,
    )

    assert result["tx_id"].is_cuda
    assert result["rx_id"].is_cuda
    assert result["path_length_m"].is_cuda
    assert result["path_gain"].is_cuda
    assert result["path_length_m"].item() == pytest.approx(5.0)


def test_path_los_export_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path LoS export")

    tx_positions = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    tx_power = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    monkeypatch.setattr(runtime_symbols, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="path_los_export CUDA kernel is required"):
        ops.path_los_export(tx_positions, tx_power, rx_positions, frequency_hz=3.0e9)


def test_path_reflection_candidates_generate_native_segments():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path reflection candidates")

    vertices = torch.tensor(
        [[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 0.0, 1.0]],
        device="cuda",
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    face_normals = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    face_gain = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    tx_positions = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    tx_power = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)

    candidates = ops.path_reflection_candidates(
        vertices,
        faces,
        face_normals,
        face_gain,
        tx_positions,
        tx_power,
        rx_positions,
        frequency_hz=3.0e9,
    )

    assert candidates["valid"].is_cuda
    assert bool(candidates["valid"].item())
    assert int(candidates["component_id"].item()) == 1
    assert int(candidates["primitive_id"].item()) == 0
    assert candidates["seg0_start"].shape == (1, 3)
    torch.testing.assert_close(
        candidates["path_length_m"],
        torch.tensor([2.0], device="cuda", dtype=torch.float32),
        rtol=2.0e-6,
        atol=1.0e-6,
    )


def test_path_diffraction_block_and_merge_use_native_compaction():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path diffraction block compaction")

    capacity = 3
    out = (
        torch.tensor([capacity], device="cuda", dtype=torch.int32),
        torch.tensor([True, False, True], device="cuda", dtype=torch.bool),
        torch.zeros((capacity,), device="cuda", dtype=torch.int32),
        torch.tensor([7, 8, 9], device="cuda", dtype=torch.int32),
        torch.tensor([1, 1, 1], device="cuda", dtype=torch.int32),
        torch.tensor([11, 12, 13], device="cuda", dtype=torch.int32),
        torch.zeros((capacity,), device="cuda", dtype=torch.int32),
        torch.zeros((capacity,), device="cuda", dtype=torch.int32),
        torch.tensor([1.0e-9, 2.0e-9, 3.0e-9], device="cuda", dtype=torch.float32),
        torch.tensor([1.0, 9.0, 2.0], device="cuda", dtype=torch.float32),
        torch.tensor([2.0, 9.0, 3.0], device="cuda", dtype=torch.float32),
        torch.tensor([0.5, 9.0, 1.0], device="cuda", dtype=torch.float32),
        torch.tensor([0.25, 9.0, 4.0], device="cuda", dtype=torch.float32),
        torch.tensor([0.0, 9.0, 5.0], device="cuda", dtype=torch.float32),
        torch.tensor([0.75, 9.0, 6.0], device="cuda", dtype=torch.float32),
        torch.empty((capacity, 3), device="cuda", dtype=torch.float32),
        torch.empty((capacity, 3), device="cuda", dtype=torch.float32),
        torch.empty((capacity, 3), device="cuda", dtype=torch.float32),
    )

    block = ops.path_diffraction_block(out, tx_index=2)
    merged = ops.path_merge_blocks([block], tx_count=3, max_depth=1)

    assert block["valid"].shape == (2,)
    torch.testing.assert_close(block["tx_id"], torch.tensor([2, 2], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(block["rx_id"], torch.tensor([7, 9], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(block["edge_id"], torch.tensor([11, 13], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(
        block["path_gain"],
        torch.tensor([5.875, 91.0], device="cuda", dtype=torch.float32),
    )
    torch.testing.assert_close(merged["path_gain"], block["path_gain"])


def test_deterministic_los_field_matches_python_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic LoS field")

    from witwin.channel_native.deterministic.field import free_space_complex_field

    path_gain = torch.tensor([1.0e-4, 2.5e-5, 0.0], device="cuda", dtype=torch.float32)
    path_length = torch.tensor([1.0, 3.25, 9.5], device="cuda", dtype=torch.float32)

    result = ops.deterministic_los_field(
        path_gain=path_gain,
        path_length_m=path_length,
        frequency_hz=3.0e9,
    )

    field = torch.complex(result["field_real"], result["field_imag"])
    expected = free_space_complex_field(path_gain, path_length, 3.0e9)
    torch.testing.assert_close(field, expected, rtol=2.0e-5, atol=1.0e-7)
    torch.testing.assert_close(result["path_gain"], path_gain, rtol=0.0, atol=0.0)


def test_deterministic_los_topology_block_compacts_and_fills_extended_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic LoS topology block")

    tx_id = torch.tensor([0, 0, 1], device="cuda", dtype=torch.int32)
    rx_id = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int32)
    path_length = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
    delay = path_length / 299_792_458.0
    path_gain = torch.tensor([1.0e-4, 2.0e-4, 3.0e-4], device="cuda", dtype=torch.float32)
    visible = torch.tensor([True, False, True], device="cuda", dtype=torch.bool)

    block = ops.deterministic_los_topology_block(
        tx_id,
        rx_id,
        path_length,
        delay,
        path_gain,
        visible,
        frequency_hz=3.0e9,
        sequence_width=2,
    )

    assert block["valid"].shape == (2,)
    torch.testing.assert_close(block["tx_id"], torch.tensor([0, 1], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(block["rx_id"], torch.tensor([0, 0], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(block["depth"], torch.zeros((2,), device="cuda", dtype=torch.int32))
    torch.testing.assert_close(block["component_id"], torch.zeros((2,), device="cuda", dtype=torch.int32))
    torch.testing.assert_close(block["primitive_id"], torch.full((2,), -1, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(block["edge_id"], torch.full((2,), -1, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(block["path_length_m"], torch.tensor([1.0, 3.0], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(block["path_gain"], torch.tensor([1.0e-4, 3.0e-4], device="cuda", dtype=torch.float32))
    assert block["path_field"].is_cuda
    assert block["path_field"].dtype == torch.complex64
    assert block["interaction_position"].shape == (2, 3)
    assert block["primitive_sequence"].shape == (2, 2)
    assert block["interaction_positions"].shape == (2, 2, 3)
    torch.testing.assert_close(
        block["primitive_sequence"],
        torch.full((2, 2), -1, device="cuda", dtype=torch.int32),
    )


def test_deterministic_los_topology_block_all_visible_does_not_require_python_mask():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic LoS topology block")

    tx_id = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    rx_id = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    path_length = torch.tensor([4.0, 5.0], device="cuda", dtype=torch.float32)
    delay = path_length / 299_792_458.0
    path_gain = torch.tensor([4.0e-4, 5.0e-4], device="cuda", dtype=torch.float32)

    block = ops.deterministic_los_topology_block(
        tx_id,
        rx_id,
        path_length,
        delay,
        path_gain,
        None,
        frequency_hz=3.0e9,
        sequence_width=0,
    )

    assert block["valid"].shape == (2,)
    torch.testing.assert_close(block["tx_id"], tx_id)
    torch.testing.assert_close(block["rx_id"], rx_id)
    torch.testing.assert_close(block["path_length_m"], path_length)
    assert block["primitive_sequence"].shape == (2, 0)
    assert block["interaction_positions"].shape == (2, 0, 3)


def test_deterministic_selected_edge_count_uses_native_unique_count():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic selected-edge counting")

    edge_id = torch.tensor([-1, 7, 11, 7, -1, 11], device="cuda", dtype=torch.int32)

    assert ops.deterministic_selected_edge_count(edge_id) == 2


def test_deterministic_zero_field_phase_uses_native_storage_fill():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic zero field/phase storage")

    reference = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)

    exported = ops.deterministic_zero_field_phase(reference)

    assert exported["path_field"].is_cuda
    assert exported["path_field"].dtype == torch.complex64
    assert exported["phase_rad"].is_cuda
    assert exported["phase_rad"].dtype == torch.float32
    assert exported["path_field"].shape == reference.shape
    assert exported["phase_rad"].shape == reference.shape
    torch.testing.assert_close(exported["path_field"].abs(), reference * 0.0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(exported["phase_rad"], reference * 0.0, rtol=0.0, atol=0.0)


def test_deterministic_topology_default_fields_uses_native_fill():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic topology default fields")

    reference = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)

    exported = ops.deterministic_topology_default_fields(reference)

    assert exported["interaction_position"].shape == (2, 3)
    assert exported["interaction_normal"].shape == (2, 3)
    assert exported["material_id"].shape == (2,)
    assert exported["path_field"].shape == (2,)
    torch.testing.assert_close(
        exported["interaction_position"],
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        exported["interaction_normal"],
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        exported["material_id"],
        torch.tensor([-1, -1], device="cuda", dtype=torch.int32),
        rtol=0.0,
        atol=0,
    )
    torch.testing.assert_close(
        exported["path_field"],
        torch.tensor([0.0 + 0.0j, 0.0 + 0.0j], device="cuda", dtype=torch.complex64),
        rtol=0.0,
        atol=0.0,
    )


def test_deterministic_pad_topology_sequences_uses_native_defaults_and_copy():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic topology sequence padding")

    depth = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    primitive_id = torch.tensor([-1, 5, 9], device="cuda", dtype=torch.int32)
    material_id = torch.tensor([-1, 2, 3], device="cuda", dtype=torch.int32)
    interaction_position = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        device="cuda",
        dtype=torch.float32,
    )
    interaction_normal = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    empty_i32 = torch.empty((3, 0), device="cuda", dtype=torch.int32)
    empty_vec = torch.empty((3, 0, 3), device="cuda", dtype=torch.float32)

    defaults = ops.deterministic_pad_topology_sequences(
        depth=depth,
        primitive_id=primitive_id,
        material_id=material_id,
        interaction_position=interaction_position,
        interaction_normal=interaction_normal,
        primitive_sequence=empty_i32,
        material_sequence=empty_i32,
        interaction_positions=empty_vec,
        interaction_normals=empty_vec,
        width=2,
    )

    torch.testing.assert_close(
        defaults["primitive_sequence"],
        torch.tensor([[-1, -1], [5, -1], [9, -1]], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(
        defaults["material_sequence"],
        torch.tensor([[-1, -1], [2, -1], [3, -1]], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(defaults["interaction_positions"][1, 0], interaction_position[1])
    torch.testing.assert_close(defaults["interaction_normals"][2, 0], interaction_normal[2])

    source_i32 = torch.tensor([[11], [12], [13]], device="cuda", dtype=torch.int32)
    source_vec = torch.tensor(
        [[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]],
        device="cuda",
        dtype=torch.float32,
    )
    copied = ops.deterministic_pad_topology_sequences(
        depth=depth,
        primitive_id=primitive_id,
        material_id=material_id,
        interaction_position=interaction_position,
        interaction_normal=interaction_normal,
        primitive_sequence=source_i32,
        material_sequence=source_i32,
        interaction_positions=source_vec,
        interaction_normals=source_vec,
        width=2,
    )

    torch.testing.assert_close(
        copied["primitive_sequence"],
        torch.tensor([[11, -1], [12, -1], [13, -1]], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(copied["interaction_positions"][:, 0, :], source_vec[:, 0, :])


def test_deterministic_topology_base_fields_fill_constants_and_sources():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic topology base fields")

    rx_id = torch.tensor([3, 4], device="cuda", dtype=torch.int32)
    path_length = torch.tensor([1.5, 2.5], device="cuda", dtype=torch.float32)
    delay = torch.tensor([1.0e-9, 2.0e-9], device="cuda", dtype=torch.float32)
    path_gain = torch.tensor([0.25, 0.5], device="cuda", dtype=torch.float32)
    depth_source = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    primitive_source = torch.tensor([7, 8], device="cuda", dtype=torch.int32)
    empty_i32 = torch.empty((0,), device="cuda", dtype=torch.int32)

    block = ops.deterministic_topology_base_fields(
        rx_id=rx_id,
        path_length_m=path_length,
        delay_s=delay,
        path_gain=path_gain,
        tx_index=2,
        component_id=1,
        depth_source=depth_source,
        depth_value=0,
        primitive_source=primitive_source,
        primitive_value=-1,
        edge_source=empty_i32,
        edge_value=-1,
    )

    torch.testing.assert_close(block["valid"], torch.tensor([True, True], device="cuda", dtype=torch.bool))
    torch.testing.assert_close(block["tx_id"], torch.tensor([2, 2], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(block["rx_id"], rx_id)
    torch.testing.assert_close(block["depth"], depth_source)
    torch.testing.assert_close(block["component_id"], torch.tensor([1, 1], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(block["primitive_id"], primitive_source)
    torch.testing.assert_close(block["edge_id"], torch.tensor([-1, -1], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(block["path_length_m"], path_length)
    torch.testing.assert_close(block["delay_s"], delay)
    torch.testing.assert_close(block["path_gain"], path_gain)


def test_deterministic_repeat_range_generates_native_repeated_indices():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic repeated range generation")

    reference = torch.empty((1,), device="cuda", dtype=torch.float32)

    repeated = ops.deterministic_repeat_range(reference, start=4, end=7, repeats=2)

    torch.testing.assert_close(
        repeated,
        torch.tensor([4, 4, 5, 5, 6, 6], device="cuda", dtype=torch.int32),
    )


def test_deterministic_reflection_epc_input_batch_generates_native_pairs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection EPC batch generation")

    tx = torch.tensor([9.0, 8.0, 7.0], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    sequences = torch.tensor([[2, 0], [1, 2]], device="cuda", dtype=torch.long)
    tri_a = torch.tensor(
        [[10.0, 10.5, 11.0], [20.0, 20.5, 21.0], [30.0, 30.5, 31.0]],
        device="cuda",
        dtype=torch.float32,
    )
    normals = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        device="cuda",
        dtype=torch.float32,
    )

    batch = ops.deterministic_reflection_epc_input_batch(
        tx=tx,
        rx_positions=rx_positions,
        sequences=sequences,
        tri_a=tri_a,
        normals=normals,
        rx_start=4,
        rx_end=6,
    )

    expected_sequence_batch = sequences.repeat(2, 1).to(dtype=torch.int32)
    torch.testing.assert_close(batch["tx_batch"], tx.expand(4, 3).contiguous())
    torch.testing.assert_close(batch["rx_batch"], rx_positions[4:6].repeat_interleave(2, dim=0).contiguous())
    torch.testing.assert_close(batch["rx_indices"], torch.tensor([4, 4, 5, 5], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(batch["sequence_batch"], expected_sequence_batch)
    torch.testing.assert_close(batch["direct_plane_points"], tri_a[expected_sequence_batch.to(dtype=torch.long)])
    torch.testing.assert_close(batch["direct_plane_normals"], normals[expected_sequence_batch.to(dtype=torch.long)])


def test_deterministic_face_anchor_points_gathers_first_vertices_native():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic face anchor gather")

    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [2, 1, 0],
            [3, 2, 1],
            [1, 0, 3],
        ],
        device="cuda",
        dtype=torch.int32,
    )

    anchors = ops.deterministic_face_anchor_points(vertices, faces)

    expected = torch.tensor(
        [
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [1.0, 2.0, 3.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    torch.testing.assert_close(anchors, expected)


def test_deterministic_face_sequence_chunk_generates_base_digits():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic face sequence generation")

    reference = torch.empty((1,), device="cuda", dtype=torch.float32)

    sequences = ops.deterministic_face_sequence_chunk(
        reference,
        face_count=3,
        depth=2,
        start=1,
        end=6,
    )

    expected = torch.tensor(
        [[0, 1], [0, 2], [1, 0], [1, 1], [1, 2]],
        device="cuda",
        dtype=torch.int32,
    )
    torch.testing.assert_close(sequences, expected)


def test_deterministic_mapped_face_sequence_chunk_maps_digits_to_face_ids():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic mapped face sequence generation")

    face_ids = torch.tensor([4, 7, 9], device="cuda", dtype=torch.int64)

    sequences = ops.deterministic_mapped_face_sequence_chunk(
        face_ids,
        depth=2,
        start=1,
        end=6,
    )

    expected = torch.tensor(
        [[4, 7], [4, 9], [7, 4], [7, 7], [7, 9]],
        device="cuda",
        dtype=torch.int32,
    )
    torch.testing.assert_close(sequences, expected)


def test_deterministic_mapped_face_sequence_chunk_can_skip_adjacent_repeats():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic mapped face sequence generation")

    face_ids = torch.tensor([4, 7, 9], device="cuda", dtype=torch.int64)

    sequences = ops.deterministic_mapped_face_sequence_chunk(
        face_ids,
        depth=2,
        start=0,
        end=6,
        adjacent_distinct=True,
    )

    expected = torch.tensor(
        [[4, 7], [4, 9], [7, 4], [7, 9], [9, 4], [9, 7]],
        device="cuda",
        dtype=torch.int32,
    )
    torch.testing.assert_close(sequences, expected)


def test_deterministic_reflection_order1_compact_selects_visible_material_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection order1 compaction")

    visible = torch.tensor([False, True, True], device="cuda", dtype=torch.bool)
    epc_faces = torch.tensor([[9], [2], [1]], device="cuda", dtype=torch.int32)
    sequence_batch = torch.tensor([[0], [1], [2]], device="cuda", dtype=torch.int32)
    epc_hits = torch.tensor(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    epc_normals = epc_hits + 10.0
    rx_indices = torch.tensor([4, 4, 5], device="cuda", dtype=torch.int32)
    tx = torch.tensor([7.0, 8.0, 9.0], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [40.0, 41.0, 42.0],
            [50.0, 51.0, 52.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    tx_power = torch.tensor([0.5, 1.5], device="cuda", dtype=torch.float32)
    face_eps_r = torch.tensor([10.0, 11.0, 12.0], device="cuda", dtype=torch.float32)
    face_sigma_e = face_eps_r + 100.0
    face_mu_r = face_eps_r + 200.0
    face_gain = face_eps_r + 300.0
    face_material_id = torch.tensor([3, 4, 5], device="cuda", dtype=torch.int32)

    compacted = ops.deterministic_reflection_order1_compact(
        visible=visible,
        epc_faces=epc_faces,
        epc_hits=epc_hits,
        epc_normals=epc_normals,
        sequence_batch=sequence_batch,
        rx_indices=rx_indices,
        tx=tx,
        rx_positions=rx_positions,
        tx_power=tx_power,
        tx_index=1,
        face_eps_r=face_eps_r,
        face_sigma_e=face_sigma_e,
        face_mu_r=face_mu_r,
        face_gain=face_gain,
        face_material_id=face_material_id,
        grouped_export=False,
    )

    torch.testing.assert_close(compacted["selected_faces"], torch.tensor([1, 2], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(compacted["selected_points"], epc_hits[1:, 0, :])
    torch.testing.assert_close(compacted["selected_normals"], epc_normals[1:, 0, :])
    torch.testing.assert_close(compacted["selected_rx_id"], torch.tensor([4, 5], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(compacted["tx_keep"], tx.expand(2, 3).contiguous())
    torch.testing.assert_close(compacted["rx_keep"], rx_positions[4:6])
    torch.testing.assert_close(compacted["tx_power"], torch.tensor([1.5, 1.5], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["eps_r"], torch.tensor([11.0, 12.0], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["sigma_e"], torch.tensor([111.0, 112.0], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["mu_r"], torch.tensor([211.0, 212.0], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["gain"], torch.tensor([311.0, 312.0], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["material_id"], torch.tensor([4, 5], device="cuda", dtype=torch.int32))


def test_deterministic_reflection_sequence_compact_selects_visible_sequences_with_limit():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection sequence compaction")

    visible = torch.tensor([True, False, True], device="cuda", dtype=torch.bool)
    epc_sequences = torch.tensor([[0, 1], [1, 2], [2, 0]], device="cuda", dtype=torch.int32)
    epc_hits = torch.tensor(
        [
            [[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]],
            [[3.0, 3.5, 4.0], [4.0, 4.5, 5.0]],
            [[5.0, 5.5, 6.0], [6.0, 6.5, 7.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    epc_normals = epc_hits + 20.0
    rx_indices = torch.tensor([4, 4, 5], device="cuda", dtype=torch.int32)
    tx = torch.tensor([7.0, 8.0, 9.0], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [40.0, 41.0, 42.0],
            [50.0, 51.0, 52.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    tx_power = torch.tensor([0.5, 1.5], device="cuda", dtype=torch.float32)
    face_eps_r = torch.tensor([10.0, 11.0, 12.0], device="cuda", dtype=torch.float32)
    face_sigma_e = face_eps_r + 100.0
    face_mu_r = face_eps_r + 200.0
    face_gain = face_eps_r + 300.0
    face_material_id = torch.tensor([3, 4, 5], device="cuda", dtype=torch.int32)

    compacted = ops.deterministic_reflection_sequence_compact(
        visible=visible,
        epc_sequences=epc_sequences,
        epc_hits=epc_hits,
        epc_normals=epc_normals,
        rx_indices=rx_indices,
        tx=tx,
        rx_positions=rx_positions,
        tx_power=tx_power,
        tx_index=1,
        face_eps_r=face_eps_r,
        face_sigma_e=face_sigma_e,
        face_mu_r=face_mu_r,
        face_gain=face_gain,
        face_material_id=face_material_id,
        max_count=1,
    )

    torch.testing.assert_close(compacted["selected_sequences"], torch.tensor([[0, 1]], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(compacted["selected_hits"], epc_hits[:1])
    torch.testing.assert_close(compacted["selected_normals"], epc_normals[:1])
    torch.testing.assert_close(compacted["first_hit"], epc_hits[:1, 0, :])
    torch.testing.assert_close(compacted["first_normal"], epc_normals[:1, 0, :])
    torch.testing.assert_close(compacted["selected_rx_id"], torch.tensor([4], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(compacted["selected_tx"], tx.reshape(1, 3))
    torch.testing.assert_close(compacted["selected_rx"], rx_positions[4:5])
    torch.testing.assert_close(compacted["tx_power"], torch.tensor([1.5], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["eps_r"], torch.tensor([[10.0, 11.0]], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["sigma_e"], torch.tensor([[110.0, 111.0]], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["mu_r"], torch.tensor([[210.0, 211.0]], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["gain"], torch.tensor([[310.0, 311.0]], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["first_face"], torch.tensor([0], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(compacted["material_id"], torch.tensor([3], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(compacted["material_sequence"], torch.tensor([[3, 4]], device="cuda", dtype=torch.int32))


def test_deterministic_normalize_vec3_normalizes_rows_with_native_kernel():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic vec3 normalization")

    values = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)

    normalized = ops.deterministic_normalize_vec3(values, eps=1.0e-6)

    expected = torch.tensor([[0.6, 0.8, 0.0], [0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    torch.testing.assert_close(normalized, expected)


def test_deterministic_reflect_points_reflects_about_planes_with_native_kernel():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic point reflection")

    points = torch.tensor([[1.0, 2.0, 3.0], [2.0, -1.0, 0.0]], device="cuda", dtype=torch.float32)
    plane_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]], device="cuda", dtype=torch.float32)
    normals = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32)

    reflected = ops.deterministic_reflect_points(points, plane_points, normals)

    expected = torch.tensor([[-1.0, 2.0, 3.0], [2.0, -1.0, 0.0]], device="cuda", dtype=torch.float32)
    torch.testing.assert_close(reflected, expected)


def test_deterministic_face_groups_matches_canonical_plane_keys():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic face grouping")

    tri_a = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    normals = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    surface_ids = torch.tensor([1, 1, 1, 2, 1], device="cuda", dtype=torch.int64)

    groups = ops.deterministic_face_groups(tri_a, normals, surface_ids, quantization=1.0e-4)

    assert groups["group_count"] == 4
    torch.testing.assert_close(
        groups["face_group_id"],
        torch.tensor([0, 0, 1, 3, 2], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(
        groups["representative_faces"],
        torch.tensor([0, 2, 4, 3], device="cuda", dtype=torch.int64),
    )
    torch.testing.assert_close(
        groups["surface_group_size"],
        torch.tensor([2, 1, 1, 1], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(
        groups["surface_group_members"],
        torch.tensor([0, 1, 2, -1, 4, -1, 3, -1], device="cuda", dtype=torch.int32),
    )


def test_deterministic_surface_face_groups_groups_by_surface_id_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic surface grouping")

    surface_ids = torch.tensor([3, 3, 2, 3, 2], device="cuda", dtype=torch.int64)

    groups = ops.deterministic_surface_face_groups(surface_ids)

    assert groups["group_count"] == 2
    torch.testing.assert_close(
        groups["face_group_id"],
        torch.tensor([1, 1, 0, 1, 0], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(
        groups["representative_faces"],
        torch.tensor([2, 0], device="cuda", dtype=torch.int64),
    )
    torch.testing.assert_close(
        groups["surface_group_size"],
        torch.tensor([2, 3], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(
        groups["surface_group_members"],
        torch.tensor([2, 4, -1, 0, 1, 3], device="cuda", dtype=torch.int32),
    )


def test_deterministic_concat_topology_blocks_concatenates_full_schema():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic topology concat")

    def make_block(offset: int, count: int) -> dict[str, torch.Tensor]:
        i32 = torch.arange(offset, offset + count, device="cuda", dtype=torch.int32)
        f32 = i32.to(dtype=torch.float32)
        return {
            "valid": torch.ones((count,), device="cuda", dtype=torch.bool),
            "tx_id": i32,
            "rx_id": i32 + 10,
            "depth": torch.full((count,), 1, device="cuda", dtype=torch.int32),
            "component_id": torch.full((count,), 2, device="cuda", dtype=torch.int32),
            "primitive_id": i32 + 20,
            "edge_id": i32 + 30,
            "path_length_m": f32 + 0.25,
            "delay_s": f32 + 0.5,
            "path_gain": f32 + 0.75,
            "path_field": torch.complex(f32 + 1.0, f32 + 2.0),
            "interaction_position": torch.stack((f32, f32 + 1.0, f32 + 2.0), dim=1),
            "interaction_normal": torch.stack((f32 + 3.0, f32 + 4.0, f32 + 5.0), dim=1),
            "material_id": i32 + 40,
            "primitive_sequence": torch.stack((i32 + 50, i32 + 60), dim=1),
            "material_sequence": torch.stack((i32 + 70, i32 + 80), dim=1),
            "interaction_positions": torch.stack(
                (
                    torch.stack((f32, f32 + 1.0, f32 + 2.0), dim=1),
                    torch.stack((f32 + 3.0, f32 + 4.0, f32 + 5.0), dim=1),
                ),
                dim=1,
            ),
            "interaction_normals": torch.stack(
                (
                    torch.stack((f32 + 6.0, f32 + 7.0, f32 + 8.0), dim=1),
                    torch.stack((f32 + 9.0, f32 + 10.0, f32 + 11.0), dim=1),
                ),
                dim=1,
            ),
        }

    block0 = make_block(0, 2)
    block1 = make_block(10, 1)

    out = ops.deterministic_concat_topology_blocks([block0, block1], sequence_width=2)

    for key, tensor in block0.items():
        expected = torch.cat((tensor, block1[key]), dim=0).contiguous()
        torch.testing.assert_close(out[key], expected)


def test_deterministic_concat_topology_blocks_concatenates_unpadded_schema():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic topology concat")

    def make_block(offset: int, count: int) -> dict[str, torch.Tensor]:
        i32 = torch.arange(offset, offset + count, device="cuda", dtype=torch.int32)
        f32 = i32.to(dtype=torch.float32)
        return {
            "valid": torch.ones((count,), device="cuda", dtype=torch.bool),
            "tx_id": i32,
            "rx_id": i32 + 10,
            "depth": torch.full((count,), 1, device="cuda", dtype=torch.int32),
            "component_id": torch.full((count,), 2, device="cuda", dtype=torch.int32),
            "primitive_id": i32 + 20,
            "edge_id": i32 + 30,
            "path_length_m": f32 + 0.25,
            "delay_s": f32 + 0.5,
            "path_gain": f32 + 0.75,
            "path_field": torch.complex(f32 + 1.0, f32 + 2.0),
            "interaction_position": torch.stack((f32, f32 + 1.0, f32 + 2.0), dim=1),
            "interaction_normal": torch.stack((f32 + 3.0, f32 + 4.0, f32 + 5.0), dim=1),
            "material_id": i32 + 40,
        }

    block0 = make_block(0, 2)
    block1 = make_block(10, 1)

    out = ops.deterministic_concat_topology_blocks([block0, block1], sequence_width=0)

    assert "primitive_sequence" not in out
    assert "material_sequence" not in out
    assert "interaction_positions" not in out
    assert "interaction_normals" not in out
    for key, tensor in block0.items():
        expected = torch.cat((tensor, block1[key]), dim=0).contiguous()
        torch.testing.assert_close(out[key], expected)


def test_deterministic_gather_topology_block_orders_and_truncates_full_schema():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic topology gather")

    count = 4
    i32 = torch.arange(count, device="cuda", dtype=torch.int32)
    f32 = i32.to(dtype=torch.float32)
    block = {
        "valid": torch.ones((count,), device="cuda", dtype=torch.bool),
        "tx_id": i32,
        "rx_id": i32 + 10,
        "depth": i32 + 20,
        "component_id": i32 + 30,
        "primitive_id": i32 + 40,
        "edge_id": i32 + 50,
        "path_length_m": f32 + 0.25,
        "delay_s": f32 + 0.5,
        "path_gain": f32 + 0.75,
        "path_field": torch.complex(f32 + 1.0, f32 + 2.0),
        "interaction_position": torch.stack((f32, f32 + 1.0, f32 + 2.0), dim=1),
        "interaction_normal": torch.stack((f32 + 3.0, f32 + 4.0, f32 + 5.0), dim=1),
        "material_id": i32 + 60,
        "primitive_sequence": torch.stack((i32 + 70, i32 + 80), dim=1),
        "material_sequence": torch.stack((i32 + 90, i32 + 100), dim=1),
        "interaction_positions": torch.stack(
            (
                torch.stack((f32, f32 + 1.0, f32 + 2.0), dim=1),
                torch.stack((f32 + 3.0, f32 + 4.0, f32 + 5.0), dim=1),
            ),
            dim=1,
        ),
        "interaction_normals": torch.stack(
            (
                torch.stack((f32 + 6.0, f32 + 7.0, f32 + 8.0), dim=1),
                torch.stack((f32 + 9.0, f32 + 10.0, f32 + 11.0), dim=1),
            ),
            dim=1,
        ),
    }
    order = torch.tensor([2, 0, 3], device="cuda", dtype=torch.long)

    out = ops.deterministic_gather_topology_block(block, order, max_count=2, sequence_width=2)

    expected_order = order[:2]
    for key, tensor in block.items():
        torch.testing.assert_close(out[key], tensor[expected_order].contiguous())


def test_deterministic_sort_order_matches_topology_key_priority():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic topology sorting")

    valid = torch.ones(4, device="cuda", dtype=torch.bool)
    tx_id = torch.tensor([0, 0, 0, 0], device="cuda", dtype=torch.int32)
    rx_id = torch.tensor([1, 0, 0, 0], device="cuda", dtype=torch.int32)
    depth = torch.tensor([1, 1, 1, 0], device="cuda", dtype=torch.int32)
    component_id = torch.tensor([1, 1, 0, 1], device="cuda", dtype=torch.int32)
    primitive_id = torch.tensor([2, 1, 3, 4], device="cuda", dtype=torch.int32)
    edge_id = torch.tensor([-1, -1, -1, -1], device="cuda", dtype=torch.int32)
    primitive_sequence = torch.tensor(
        [[5, 2], [5, 1], [4, 9], [8, 8]],
        device="cuda",
        dtype=torch.int32,
    )

    order = ops.deterministic_sort_order(
        valid,
        tx_id,
        rx_id,
        depth,
        component_id,
        primitive_id,
        edge_id,
        primitive_sequence,
    )

    torch.testing.assert_close(order, torch.tensor([3, 2, 1, 0], device="cuda", dtype=torch.long))


def test_deterministic_diffraction_order1_compact_selects_valid_raydn_rows():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diffraction compaction")

    valid = torch.tensor([False, True, True, False], device="cuda", dtype=torch.bool)
    rx_id = torch.tensor([10, 11, 12, 13], device="cuda", dtype=torch.int32)
    depth = torch.tensor([0, 2, 3, 4], device="cuda", dtype=torch.int32)
    edge_id = torch.tensor([20, 21, 22, 23], device="cuda", dtype=torch.int32)
    delay = torch.tensor([0.1, 0.2, 0.3, 0.4], device="cuda", dtype=torch.float32)
    x_re = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=torch.float32)
    x_im = x_re + 10.0
    y_re = x_re + 20.0
    y_im = x_re + 30.0
    z_re = x_re + 40.0
    z_im = x_re + 50.0
    interaction_position = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    compacted = ops.deterministic_diffraction_order1_compact(
        valid=valid,
        rx_id=rx_id,
        depth=depth,
        edge_id=edge_id,
        delay_s=delay,
        x_re=x_re,
        x_im=x_im,
        y_re=y_re,
        y_im=y_im,
        z_re=z_re,
        z_im=z_im,
        interaction_position=interaction_position,
    )

    torch.testing.assert_close(compacted["rx_id"], torch.tensor([11, 12], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(compacted["depth"], torch.tensor([2, 3], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(compacted["edge_id"], torch.tensor([21, 22], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(compacted["delay_s"], torch.tensor([0.2, 0.3], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["x_re"], torch.tensor([2.0, 3.0], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["x_im"], torch.tensor([12.0, 13.0], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["y_re"], torch.tensor([22.0, 23.0], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["y_im"], torch.tensor([32.0, 33.0], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["z_re"], torch.tensor([42.0, 43.0], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["z_im"], torch.tensor([52.0, 53.0], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(compacted["interaction_position"], interaction_position[1:3])


def test_deterministic_diffraction_vector_field_matches_python_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diffraction field")

    from witwin.channel_native.deterministic.field import equivalent_field_from_vector_components

    x_re = torch.tensor([1.0, -0.5, 0.0], device="cuda", dtype=torch.float32)
    x_im = torch.tensor([0.5, 0.25, 0.0], device="cuda", dtype=torch.float32)
    y_re = torch.tensor([0.25, 0.75, 0.0], device="cuda", dtype=torch.float32)
    y_im = torch.tensor([-0.125, 0.5, 0.0], device="cuda", dtype=torch.float32)
    z_re = torch.tensor([0.125, -0.25, 0.0], device="cuda", dtype=torch.float32)
    z_im = torch.tensor([0.0625, -0.5, 0.0], device="cuda", dtype=torch.float32)

    result = ops.deterministic_diffraction_vector_field(
        x_re=x_re,
        x_im=x_im,
        y_re=y_re,
        y_im=y_im,
        z_re=z_re,
        z_im=z_im,
    )

    field = torch.complex(result["field_real"], result["field_imag"])
    expected_gain, expected_field = equivalent_field_from_vector_components(x_re, x_im, y_re, y_im, z_re, z_im)
    torch.testing.assert_close(result["path_gain"], expected_gain, rtol=2.0e-6, atol=1.0e-7)
    torch.testing.assert_close(field, expected_field, rtol=2.0e-6, atol=1.0e-7)


def test_deterministic_reflection_field_returns_native_complex_field():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection field")
    native = ops.native_extension()
    if native is None or not hasattr(native, "deterministic_reflection_field"):
        pytest.skip("native deterministic reflection field kernel is not built")

    tx_position = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    hit_position = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    rx_position = torch.tensor([[1.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    normal = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    one = torch.ones((1,), device="cuda", dtype=torch.float32)
    eps_r = torch.full((1,), 4.0, device="cuda", dtype=torch.float32)
    sigma_e = torch.zeros((1,), device="cuda", dtype=torch.float32)

    result = ops.deterministic_reflection_field(
        tx_position=tx_position,
        rx_position=rx_position,
        hit_position=hit_position,
        normal=normal,
        tx_power=one,
        eps_r=eps_r,
        sigma_e=sigma_e,
        mu_r=one,
        gain=one,
        frequency_hz=3.0e9,
    )

    assert result["path_gain"].is_cuda
    assert result["field_real"].is_cuda
    assert result["field_imag"].is_cuda
    assert result["path_length_m"].is_cuda
    assert result["delay_s"].is_cuda
    assert result["path_gain"].item() > 0.0
    expected_length = torch.tensor([1.0 + math.sqrt(2.0)], device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result["path_length_m"], expected_length, rtol=2.0e-6, atol=1.0e-6)
    torch.testing.assert_close(result["delay_s"], expected_length / 299_792_458.0, rtol=2.0e-6, atol=1.0e-12)
    field = torch.complex(result["field_real"], result["field_imag"])
    torch.testing.assert_close(result["path_gain"], field.abs().square(), rtol=2.0e-4, atol=1.0e-10)


def test_deterministic_reflection_field_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection field")

    tensor = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    one = torch.ones((1,), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(deterministic_fields, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="deterministic_reflection_field CUDA kernel is required"):
        ops.deterministic_reflection_field(
            tx_position=tensor,
            rx_position=tensor,
            hit_position=tensor,
            normal=tensor,
            tx_power=one,
            eps_r=one,
            sigma_e=one,
            mu_r=one,
            gain=one,
            frequency_hz=3.0e9,
        )


def test_deterministic_reflection_sequence_field_matches_python_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection sequence field")

    from witwin.channel_native.deterministic.field import reflection_sequence_complex_field

    tx_position = torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    hit_positions = torch.tensor(
        [
            [[2.0, 0.0, 1.0], [2.0, 2.0, 1.0]],
            [[2.0, 0.1, 1.0], [2.0, 2.0, 1.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    rx_position = torch.tensor([[0.0, 2.0, 1.0], [0.1, 2.0, 1.0]], device="cuda", dtype=torch.float32)
    normals = torch.tensor(
        [
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    tx_power = torch.tensor([1.0, 0.5], device="cuda", dtype=torch.float32)
    eps_r = torch.tensor([[3.0, 4.0], [3.5, 4.5]], device="cuda", dtype=torch.float32)
    sigma_e = torch.tensor([[0.005, 0.01], [0.004, 0.009]], device="cuda", dtype=torch.float32)
    mu_r = torch.ones((2, 2), device="cuda", dtype=torch.float32)
    gain = torch.ones((2, 2), device="cuda", dtype=torch.float32)

    result = ops.deterministic_reflection_sequence_field(
        tx_position=tx_position,
        rx_position=rx_position,
        hit_positions=hit_positions,
        normals=normals,
        tx_power=tx_power,
        eps_r=eps_r,
        sigma_e=sigma_e,
        mu_r=mu_r,
        gain=gain,
        frequency_hz=3.0e9,
    )
    expected_gain, expected_field, expected_length = reflection_sequence_complex_field(
        tx_position=tx_position,
        rx_position=rx_position,
        hit_positions=hit_positions,
        normals=normals,
        tx_power_w=tx_power,
        eps_r=eps_r,
        sigma_e=sigma_e,
        mu_r=mu_r,
        gain=gain,
        frequency_hz=3.0e9,
    )

    field = torch.complex(result["field_real"], result["field_imag"])
    torch.testing.assert_close(result["path_length_m"], expected_length, rtol=2.0e-5, atol=1.0e-6)
    torch.testing.assert_close(
        result["delay_s"],
        expected_length / 299_792_458.0,
        rtol=2.0e-5,
        atol=1.0e-12,
    )
    torch.testing.assert_close(result["path_gain"], expected_gain, rtol=5.0e-4, atol=1.0e-10)
    torch.testing.assert_close(field, expected_field, rtol=5.0e-4, atol=1.0e-7)


def test_deterministic_delay_to_path_length_uses_native_kernel():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic delay conversion")
    native = ops.native_extension()
    if native is None or not hasattr(native, "deterministic_delay_to_path_length"):
        pytest.skip("native deterministic delay conversion kernel is not built")

    delay = torch.tensor([0.0, 1.0e-9, 2.5e-9], device="cuda", dtype=torch.float32)
    result = ops.deterministic_delay_to_path_length(delay)
    expected = torch.tensor([0.0, 0.2997924685, 0.7494811416], device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result, expected, rtol=2.0e-6, atol=1.0e-7)


def test_mc_los_path_gain_backward_and_jvp_match_free_space_formula():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC LoS path-gain AD kernels")

    tx_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    tx_power = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor(
        [[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]],
        device="cuda",
        dtype=torch.float32,
    )
    grad_output = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=torch.float32).t()
    rx_tangent = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    power_tangent = torch.tensor([0.25, -0.5], device="cuda", dtype=torch.float32)
    tx_tangent = torch.zeros_like(tx_positions)

    frequency_hz = 3.0e9
    frequency_tangent = 2.5e5
    grad_tx, grad_power, grad_rx, grad_frequency = ops.mc_los_path_gain_backward(
        tx_positions,
        tx_power,
        rx_positions,
        grad_output,
        frequency_hz=frequency_hz,
    )
    jvp = ops.mc_los_path_gain_jvp(
        tx_positions,
        tx_power,
        rx_positions,
        tx_tangent,
        power_tangent,
        rx_tangent,
        False,
        True,
        True,
        frequency_hz=frequency_hz,
        frequency_tangent=frequency_tangent,
    )

    scale = (299_792_458.0 / frequency_hz / (4.0 * torch.pi)) ** 2
    scale_dfreq = -2.0 * scale / frequency_hz
    diff = tx_positions[:, None, :] - rx_positions[None, :, :]
    distance_sq = (diff * diff).sum(dim=-1)
    inv_d2 = 1.0 / distance_sq
    expected_grad_power = (grad_output * scale * inv_d2).sum(dim=1)
    expected_grad_frequency = (
        grad_output * tx_power[:, None] * scale_dfreq * inv_d2
    ).sum()
    coeff = grad_output * 2.0 * tx_power[:, None] * scale * inv_d2 * inv_d2
    expected_grad_rx = (coeff[:, :, None] * diff).sum(dim=0)
    expected_grad_tx = -(coeff[:, :, None] * diff).sum(dim=1)
    expected_jvp = power_tangent[:, None] * scale * inv_d2
    expected_jvp = expected_jvp + 2.0 * tx_power[:, None] * scale * inv_d2 * inv_d2 * (
        diff * rx_tangent[None, :, :]
    ).sum(dim=-1)
    expected_jvp = expected_jvp + (
        tx_power[:, None] * scale_dfreq * frequency_tangent * inv_d2
    )

    torch.testing.assert_close(grad_tx, expected_grad_tx, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(grad_power, expected_grad_power, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(grad_rx, expected_grad_rx, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(
        grad_frequency[0], expected_grad_frequency, rtol=1e-5, atol=1e-12
    )
    torch.testing.assert_close(jvp, expected_jvp, rtol=1e-6, atol=1e-9)


def test_mc_los_path_gain_ad_kernels_require_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC LoS path-gain AD kernels")

    tx_positions = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    tx_power = torch.ones((1,), device="cuda", dtype=torch.float32)
    rx_positions = torch.ones((1, 3), device="cuda", dtype=torch.float32)
    grad_output = torch.ones((1, 1), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(mc_maps, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_los_path_gain_backward CUDA kernel is required"):
        ops.mc_los_path_gain_backward(
            tx_positions,
            tx_power,
            rx_positions,
            grad_output,
            frequency_hz=3.0e9,
        )
    with pytest.raises(RuntimeError, match="mc_los_path_gain_jvp CUDA kernel is required"):
        ops.mc_los_path_gain_jvp(
            tx_positions,
            tx_power,
            rx_positions,
            tx_positions,
            tx_power,
            rx_positions,
            False,
            False,
            False,
            frequency_hz=3.0e9,
        )


def test_mc_finalize_component_maps_fuses_total_and_power_reductions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC finalize")

    los = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], device="cuda", dtype=torch.float32)
    reflection = torch.tensor([[[0.5, 0.0], [1.5, 2.0]]], device="cuda", dtype=torch.float32)
    diffraction = torch.tensor([[[0.25, 0.75], [0.0, 1.0]]], device="cuda", dtype=torch.float32)
    transmission = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], device="cuda", dtype=torch.float32)
    scattering = torch.tensor([[[0.05, 0.1], [0.15, 0.2]]], device="cuda", dtype=torch.float32)

    result = ops.mc_finalize_component_maps(
        los, reflection, diffraction, transmission, scattering
    )

    expected_total = (
        los + reflection + diffraction + transmission + scattering
    ).reshape(1, -1)
    torch.testing.assert_close(result["path_gain"], expected_total)
    torch.testing.assert_close(result["los_power"], los.sum())
    torch.testing.assert_close(result["reflection_power"], reflection.sum())
    torch.testing.assert_close(result["diffraction_power"], diffraction.sum())
    torch.testing.assert_close(result["transmission_power"], transmission.sum())
    torch.testing.assert_close(result["scattering_power"], scattering.sum())


def test_mc_finalize_component_maps_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC finalize")

    los = torch.zeros((1, 2, 2), device="cuda", dtype=torch.float32)
    reflection = torch.zeros_like(los)
    diffraction = torch.zeros_like(los)
    transmission = torch.zeros_like(los)
    scattering = torch.zeros_like(los)
    monkeypatch.setattr(mc_maps, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_finalize_component_maps CUDA kernel is required"):
        ops.mc_finalize_component_maps(
            los, reflection, diffraction, transmission, scattering
        )


def test_mc_component_map_buffer_and_store_kernels_write_tx_slots():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC component map stores")

    reference = torch.empty((1, 3), device="cuda", dtype=torch.float32)
    maps = ops.mc_component_map_buffer(reference, tx_count=2, dim0=2, dim1=2)
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=torch.float32)
    scale = torch.tensor([10.0, 0.5], device="cuda", dtype=torch.float32)

    ops.mc_store_component_map(maps, source, tx_index=0)
    ops.mc_store_scaled_component_map(maps, source, scale, tx_index=1, scale_index=1)

    expected = torch.stack((source, source * 0.5), dim=0)
    torch.testing.assert_close(maps, expected)


def test_mc_component_map_store_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC component map stores")

    maps = torch.zeros((1, 2, 2), device="cuda", dtype=torch.float32)
    source = torch.ones((2, 2), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(mc_maps, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_store_component_map CUDA kernel is required"):
        ops.mc_store_component_map(maps, source, tx_index=0)


def test_mc_sample_directions_matches_golden_ratio_sequence():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC sample directions")

    reference = torch.empty((1, 3), device="cuda", dtype=torch.float32)
    directions = ops.mc_sample_directions(5, reference)
    indices = torch.arange(5, device="cuda", dtype=torch.float64)
    golden_ratio = (1.0 + 5.0**0.5) / 2.0
    azimuth_u = torch.frac(indices / golden_ratio)
    elevation_v = indices / 4.0
    phi = 2.0 * torch.pi * azimuth_u
    z = 1.0 - 2.0 * elevation_v
    radial = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    expected = torch.stack((radial * torch.cos(phi), radial * torch.sin(phi), z), dim=1).to(torch.float32)

    assert directions.is_cuda
    assert directions.shape == (5, 3)
    torch.testing.assert_close(directions, expected, rtol=1e-6, atol=1e-6)


def test_mc_sample_directions_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC sample directions")

    reference = torch.empty((1, 3), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(topology_sampling, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_sample_directions CUDA kernel is required"):
        ops.mc_sample_directions(1, reference)


def test_mc_transmitter_tensors_creates_cuda_positions_and_power():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC transmitter tensors")

    result = ops.mc_transmitter_tensors(
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        (0.5, 2.0),
    )

    expected_positions = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        device="cuda",
        dtype=torch.float32,
    )
    expected_power = torch.tensor([0.5, 2.0], device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result["positions"], expected_positions)
    torch.testing.assert_close(result["power"], expected_power)


def test_mc_transmitter_tensors_requires_native_cuda_helper(monkeypatch):
    monkeypatch.setattr(native_buffers, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_transmitter_tensors CUDA helper is required"):
        ops.mc_transmitter_tensors((0.0, 0.0, 0.0), (1.0,))


def test_mc_pack_vec3_interleaves_cuda_vectors():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC vec3 packing")

    x = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)
    y = torch.tensor([3.0, 4.0], device="cuda", dtype=torch.float32)
    z = torch.tensor([5.0, 6.0], device="cuda", dtype=torch.float32)

    packed = ops.mc_pack_vec3(x, y, z)

    expected = torch.tensor(
        [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
        device="cuda",
        dtype=torch.float32,
    )
    torch.testing.assert_close(packed, expected)


def test_mc_pack_vec3_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC vec3 packing")

    x = torch.zeros((1,), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(native_buffers, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_pack_vec3 CUDA kernel is required"):
        ops.mc_pack_vec3(x, x, x)


def test_mc_los_component_maps_converts_to_public_grid_layout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC LoS component maps")

    los = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    maps = ops.mc_los_component_maps(los)

    torch.testing.assert_close(maps, los.transpose(1, 2).contiguous())


def test_mc_los_component_maps_from_matrix_uses_native_grid_layout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC LoS component maps")

    los = torch.arange(6, device="cuda", dtype=torch.float32).reshape(1, 6).contiguous()

    maps = ops.mc_los_component_maps_from_matrix(los, rows=2, cols=3)

    assert maps.shape == (1, 3, 2)
    expected = torch.tensor([[[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]], device="cuda")
    torch.testing.assert_close(maps, expected)


def test_mc_apply_los_visibility_masks_one_transmitter_in_public_layout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC LoS visibility")

    los = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cuda", dtype=torch.float32)
    maps = ops.mc_los_component_maps_from_matrix(los, rows=2, cols=2)
    visible = torch.tensor([True, False, False, True], device="cuda", dtype=torch.bool)

    out = ops.mc_apply_los_visibility(maps, los, visible, tx_index=0)

    expected = torch.tensor([[[1.0, 0.0], [0.0, 4.0]]], device="cuda", dtype=torch.float32)
    torch.testing.assert_close(out, expected)


def test_mc_apply_los_visibility_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC LoS visibility")

    los = torch.zeros((1, 4), device="cuda", dtype=torch.float32)
    maps = torch.zeros((1, 2, 2), device="cuda", dtype=torch.float32)
    visible = torch.ones((4,), device="cuda", dtype=torch.bool)
    monkeypatch.setattr(mc_maps, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_apply_los_visibility CUDA kernel is required"):
        ops.mc_apply_los_visibility(maps, los, visible, tx_index=0)


def test_mc_los_visibility_inputs_fill_start_and_active():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC LoS visibility inputs")

    tx_positions = torch.tensor(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        device="cuda",
        dtype=torch.float32,
    )

    result = ops.mc_los_visibility_inputs(tx_positions, tx_index=1, rx_count=4)

    expected_start = tx_positions[1].expand(4, 3).contiguous()
    torch.testing.assert_close(result["start"], expected_start)
    assert result["active"].dtype == torch.bool
    assert result["active"].all()


def test_mc_los_visibility_inputs_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC LoS visibility inputs")

    tx_positions = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(mc_maps, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_los_visibility_inputs CUDA kernel is required"):
        ops.mc_los_visibility_inputs(tx_positions, tx_index=0, rx_count=1)


def test_mc_receiver_grid_points_matches_receiver_grid_order():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC receiver grid points")

    reference = torch.empty((1, 3), device="cuda", dtype=torch.float32)

    points = ops.mc_receiver_grid_points(
        reference,
        origin=(1.0, 2.0, 3.0),
        x_axis=(1.0, 0.0, 0.0),
        y_axis=(0.0, 0.0, 1.0),
        shape=(2, 3),
        spacing=(0.5, 2.0),
    )

    expected = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 5.0],
            [1.0, 2.0, 7.0],
            [1.5, 2.0, 3.0],
            [1.5, 2.0, 5.0],
            [1.5, 2.0, 7.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    torch.testing.assert_close(points, expected)


def test_mc_receiver_grid_points_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC receiver grid points")

    reference = torch.empty((1, 3), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(native_buffers, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_receiver_grid_points CUDA kernel is required"):
        ops.mc_receiver_grid_points(
            reference,
            origin=(0.0, 0.0, 0.0),
            x_axis=(1.0, 0.0, 0.0),
            y_axis=(0.0, 1.0, 0.0),
            shape=(1, 1),
            spacing=(1.0, 1.0),
        )


def test_mc_reflection_launch_inputs_fill_ray_origin_active_and_polarization():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC reflection launch inputs")

    tx_positions = torch.tensor(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        device="cuda",
        dtype=torch.float32,
    )

    result = ops.mc_reflection_launch_inputs(tx_positions, tx_index=1, sample_count=4)

    torch.testing.assert_close(result["ray_o"], tx_positions[1].expand(4, 3).contiguous())
    assert result["ray_tmax"].shape == (0,)
    assert result["active"].dtype == torch.bool
    assert result["active"].all()
    expected_pol = torch.zeros((4, 3), device="cuda", dtype=torch.float32)
    expected_pol[:, 0] = 1.0
    torch.testing.assert_close(result["tx_pol"], expected_pol)


def test_mc_reflection_launch_inputs_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC reflection launch inputs")

    tx_positions = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(mc_sampling, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_reflection_launch_inputs CUDA kernel is required"):
        ops.mc_reflection_launch_inputs(tx_positions, tx_index=0, sample_count=1)


def test_mc_diffraction_state_wi_matches_normalized_edge_to_source_direction():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC diffraction state directions")

    edge_pos = torch.tensor(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]],
        device="cuda",
        dtype=torch.float32,
    )
    src = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device="cuda",
        dtype=torch.float32,
    )

    state_wi = ops.mc_diffraction_state_wi(edge_pos, src)

    expected = torch.nn.functional.normalize(edge_pos - src, dim=1, eps=1.0e-6)
    torch.testing.assert_close(state_wi, expected, rtol=1e-6, atol=1e-6)


def test_mc_diffraction_state_wi_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC diffraction state directions")

    edge_pos = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    src = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(mc_sampling, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_diffraction_state_wi CUDA kernel is required"):
        ops.mc_diffraction_state_wi(edge_pos, src)


def test_mc_selected_edge_indices_compacts_bool_mask_in_edge_order():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC selected edge compaction")

    selected = torch.tensor([False, True, True, False, True], device="cuda", dtype=torch.bool)

    indices = ops.mc_selected_edge_indices(selected)

    expected = torch.tensor([1, 2, 4], device="cuda", dtype=torch.int32)
    torch.testing.assert_close(indices, expected)


def test_mc_selected_edge_indices_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC selected edge compaction")

    selected = torch.ones((1,), device="cuda", dtype=torch.bool)
    monkeypatch.setattr(topology_primitives, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_selected_edge_indices CUDA kernel is required"):
        ops.mc_selected_edge_indices(selected)


def test_mc_diffraction_state_pack_gathers_edge_state_tensors():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC diffraction state packing")

    edge_indices = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    edge_pos = torch.tensor(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
        device="cuda",
        dtype=torch.float32,
    )
    edge_dir = edge_pos + 10.0
    line_min = torch.tensor([-1.0, -2.0, -3.0], device="cuda", dtype=torch.float32)
    line_max = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
    n0 = edge_pos + 20.0
    n1 = edge_pos + 30.0
    face0 = torch.tensor([10, 11, 12], device="cuda", dtype=torch.int32)
    face1 = torch.tensor([20, 21, 22], device="cuda", dtype=torch.int32)
    exterior_angle = torch.tensor([0.5, 1.5, 2.5], device="cuda", dtype=torch.float32)
    tx = torch.tensor([9.0, 8.0, 7.0], device="cuda", dtype=torch.float32)
    tx_power = torch.tensor(4.0, device="cuda", dtype=torch.float32)

    states = ops.mc_diffraction_state_pack(
        edge_indices,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
    )

    idx = edge_indices.to(dtype=torch.long)
    torch.testing.assert_close(states[0], edge_indices)
    torch.testing.assert_close(states[1], edge_pos[idx])
    torch.testing.assert_close(states[2], edge_dir[idx])
    torch.testing.assert_close(states[3], line_min[idx])
    torch.testing.assert_close(states[4], line_max[idx])
    torch.testing.assert_close(states[5], n0[idx])
    torch.testing.assert_close(states[6], n1[idx])
    torch.testing.assert_close(states[7], face0[idx])
    torch.testing.assert_close(states[8], face1[idx])
    torch.testing.assert_close(states[9], exterior_angle[idx])
    torch.testing.assert_close(states[10], tx.expand(2, 3).contiguous())
    torch.testing.assert_close(states[11], tx_power.expand(2).contiguous())


def test_deterministic_diffraction_state_pack_reads_power_by_native_index():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diffraction state packing")

    edge_indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    edge_pos = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], device="cuda", dtype=torch.float32)
    edge_dir = edge_pos + 10.0
    line_min = torch.tensor([-1.0, -2.0], device="cuda", dtype=torch.float32)
    line_max = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)
    n0 = edge_pos + 20.0
    n1 = edge_pos + 30.0
    face0 = torch.tensor([10, 11], device="cuda", dtype=torch.int32)
    face1 = torch.tensor([20, 21], device="cuda", dtype=torch.int32)
    exterior_angle = torch.tensor([0.5, 1.5], device="cuda", dtype=torch.float32)
    tx = torch.tensor([9.0, 8.0, 7.0], device="cuda", dtype=torch.float32)
    tx_power = torch.tensor([2.0, 4.0, 8.0], device="cuda", dtype=torch.float32)

    states = ops.deterministic_diffraction_state_pack(
        edge_indices,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        2,
    )

    torch.testing.assert_close(states[0], edge_indices)
    torch.testing.assert_close(states[1], torch.tensor([[3.0, 4.0, 5.0]], device="cuda", dtype=torch.float32))
    torch.testing.assert_close(states[10], tx.reshape(1, 3))
    torch.testing.assert_close(states[11], torch.tensor([8.0], device="cuda", dtype=torch.float32))


def test_deterministic_diffraction_state_pack_selected_keeps_device_sized_capacity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diffraction state packing")

    selected = torch.tensor([False, True, False], device="cuda", dtype=torch.bool)
    edge_pos = torch.tensor(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
        device="cuda",
        dtype=torch.float32,
    )
    edge_dir = edge_pos + 10.0
    line_min = torch.tensor([-1.0, -2.0, -3.0], device="cuda", dtype=torch.float32)
    line_max = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
    n0 = edge_pos + 20.0
    n1 = edge_pos + 30.0
    face0 = torch.tensor([10, 11, 12], device="cuda", dtype=torch.int32)
    face1 = torch.tensor([20, 21, 22], device="cuda", dtype=torch.int32)
    exterior_angle = torch.tensor([0.5, 1.5, 2.5], device="cuda", dtype=torch.float32)
    tx = torch.tensor([9.0, 8.0, 7.0], device="cuda", dtype=torch.float32)
    tx_power = torch.tensor([4.0, 6.0], device="cuda", dtype=torch.float32)

    states = ops.deterministic_diffraction_state_pack_selected(
        selected,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        tx_power_index=1,
    )

    torch.testing.assert_close(states[0], torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(states[1], edge_pos)
    torch.testing.assert_close(states[10], tx.expand(3, 3).contiguous())
    torch.testing.assert_close(states[11], torch.tensor([0.0, 6.0, 0.0], device="cuda", dtype=torch.float32))


def test_mc_diffraction_state_pack_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC diffraction state packing")

    edge_indices = torch.zeros((1,), device="cuda", dtype=torch.int32)
    edge_pos = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    line = torch.zeros((1,), device="cuda", dtype=torch.float32)
    face = torch.zeros((1,), device="cuda", dtype=torch.int32)
    tx = torch.zeros((3,), device="cuda", dtype=torch.float32)
    tx_power = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    monkeypatch.setattr(mc_sampling, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_diffraction_state_pack CUDA kernel is required"):
        ops.mc_diffraction_state_pack(
            edge_indices,
            edge_pos,
            edge_pos,
            line,
            line,
            edge_pos,
            edge_pos,
            face,
            face,
            line,
            tx,
            tx_power,
        )


def test_raydn_diffraction_discover_edges_counted_uses_gpu_hit_count():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RayDN counted diffraction edge discovery")

    symbols.native_extension()
    tx_pos = torch.tensor([0.0, 0.0, 0.0], device="cuda", dtype=torch.float32)
    ray_dir = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    prim_index = torch.tensor([0, 0, -1, -1], device="cuda", dtype=torch.int32)
    hit_p = torch.tensor(
        [
            [1.0, 0.2, 0.0],
            [1.0, 0.3, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    hit_n = torch.tensor([[0.0, 1.0, 0.0]] * 4, device="cuda", dtype=torch.float32)
    hit_geo_n = hit_n.contiguous()
    hit_count = torch.tensor([2], device="cuda", dtype=torch.int32)
    triangle_edge_count = torch.tensor([1], device="cuda", dtype=torch.int32)
    triangle_edge_indices = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    edge_pos = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    edge_dir = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    edge_n0 = torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32)
    edge_n1 = torch.tensor([[0.0, -1.0, 0.0]], device="cuda", dtype=torch.float32)
    line_min = torch.tensor([-1.0], device="cuda", dtype=torch.float32)
    line_max = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    face1 = torch.tensor([-1], device="cuda", dtype=torch.int32)

    sliced = ops.raydn_diffraction_discover_edges(
        tx_pos,
        ray_dir[:2].contiguous(),
        prim_index[:2].contiguous(),
        hit_p[:2].contiguous(),
        hit_n[:2].contiguous(),
        hit_geo_n[:2].contiguous(),
        triangle_edge_count,
        triangle_edge_indices,
        edge_pos,
        edge_dir,
        edge_n0,
        edge_n1,
        line_min,
        line_max,
        face1,
    )
    counted = ops.raydn_diffraction_discover_edges_counted(
        tx_pos,
        ray_dir,
        prim_index,
        hit_p,
        hit_n,
        hit_geo_n,
        hit_count,
        triangle_edge_count,
        triangle_edge_indices,
        edge_pos,
        edge_dir,
        edge_n0,
        edge_n1,
        line_min,
        line_max,
        face1,
    )

    torch.testing.assert_close(counted, sliced)


def test_mc_face_material_tensors_expands_material_ids():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC material expansion")

    eps_r = torch.tensor([1.0, 4.0], device="cuda", dtype=torch.float32)
    sigma_e = torch.tensor([0.0, 0.2], device="cuda", dtype=torch.float32)
    mu_r = torch.tensor([1.0, 1.5], device="cuda", dtype=torch.float32)
    face_material_id = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)

    result = ops.mc_face_material_tensors(eps_r, sigma_e, mu_r, face_material_id)

    torch.testing.assert_close(result["eps_r"], torch.tensor([4.0, 1.0, 4.0], device="cuda"))
    torch.testing.assert_close(result["sigma_e"], torch.tensor([0.2, 0.0, 0.2], device="cuda"))
    torch.testing.assert_close(result["mu_r"], torch.tensor([1.5, 1.0, 1.5], device="cuda"))
    torch.testing.assert_close(result["gain"], torch.ones((3,), device="cuda"))
    assert result["valid"].dtype == torch.bool
    assert result["valid"].all()


def test_mc_face_material_tensors_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC material expansion")

    eps_r = torch.ones((1,), device="cuda", dtype=torch.float32)
    sigma_e = torch.zeros((1,), device="cuda", dtype=torch.float32)
    mu_r = torch.ones((1,), device="cuda", dtype=torch.float32)
    face_material_id = torch.zeros((1,), device="cuda", dtype=torch.int32)
    monkeypatch.setattr(material_functional, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_face_material_tensors CUDA kernel is required"):
        ops.mc_face_material_tensors(eps_r, sigma_e, mu_r, face_material_id)


def test_deterministic_accumulate_flat_matches_torch_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic accumulation")

    # Component ids 0/1/2/5/6 map to slots 0/1/2/3/4; transmission (slot 3)
    # joins the coherent field sum while scattering (slot 4) folds into the
    # totals in the power domain and keeps its field as a diagnostic.
    tx_id = torch.tensor([0, 0, 0, 1, 0, 0], device="cuda", dtype=torch.int32)
    rx_id = torch.tensor([0, 0, 1, 1, 1, 0], device="cuda", dtype=torch.int32)
    component_id = torch.tensor([0, 1, 1, 2, 5, 6], device="cuda", dtype=torch.int32)
    path_gain = torch.tensor(
        [1.0, 4.0, 9.0, 16.0, 25.0, 0.5], device="cuda", dtype=torch.float32
    )
    field = torch.tensor(
        [1.0 + 0.0j, 0.0 + 2.0j, 3.0 + 0.0j, 0.0 + 4.0j, 5.0 + 1.0j, 0.5 + 0.0j],
        device="cuda",
        dtype=torch.complex64,
    )
    slot_of = {0: 0, 1: 1, 2: 2, 5: 3, 6: 4}

    result = ops.deterministic_accumulate_flat(
        tx_id,
        rx_id,
        component_id,
        path_gain,
        field.real.contiguous(),
        field.imag.contiguous(),
        num_tx=2,
        num_rx=2,
        coherent=True,
    )

    expected_component_field = torch.zeros((5, 2, 2), device="cuda", dtype=torch.complex64)
    expected_scattering_power = torch.zeros((2, 2), device="cuda", dtype=torch.float32)
    for index in range(int(tx_id.numel())):
        cid = int(component_id[index])
        tx = int(tx_id[index])
        rx = int(rx_id[index])
        expected_component_field[slot_of[cid], tx, rx] += field[index]
        if cid == 6:
            expected_scattering_power[tx, rx] += path_gain[index]
    expected_component_power = expected_component_field.abs().square()
    expected_component_power[4] = expected_scattering_power
    expected_field_total = expected_component_field[:4].sum(dim=0)
    expected_power_total = (
        expected_field_total.abs().square() + expected_scattering_power
    )

    torch.testing.assert_close(result["component_power"], expected_component_power)
    torch.testing.assert_close(torch.complex(result["component_field_real"], result["component_field_imag"]), expected_component_field)
    torch.testing.assert_close(result["power_total"], expected_power_total)
    torch.testing.assert_close(torch.complex(result["field_total_real"], result["field_total_imag"]), expected_field_total)


def test_deterministic_accumulate_flat_incoherent_sums_power():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic accumulation")

    tx_id = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    rx_id = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    component_id = torch.tensor([1, 1], device="cuda", dtype=torch.int32)
    path_gain = torch.tensor([4.0, 9.0], device="cuda", dtype=torch.float32)
    field_real = torch.tensor([2.0, 3.0], device="cuda", dtype=torch.float32)
    field_imag = torch.zeros((2,), device="cuda", dtype=torch.float32)

    result = ops.deterministic_accumulate_flat(
        tx_id,
        rx_id,
        component_id,
        path_gain,
        field_real,
        field_imag,
        num_tx=1,
        num_rx=1,
        coherent=False,
    )

    torch.testing.assert_close(result["component_power"][1], torch.tensor([[13.0]], device="cuda"))
    torch.testing.assert_close(result["power_total"], torch.tensor([[13.0]], device="cuda"))
    torch.testing.assert_close(result["field_total_real"], torch.sqrt(torch.tensor([[13.0]], device="cuda")))
    torch.testing.assert_close(result["field_total_imag"], torch.zeros((1, 1), device="cuda"))
