import pytest
import torch

from witwin.channel.propagation.topology import concatenate
from witwin.channel.propagation.topology.export import evaluated_paths_from_block
from witwin.channel.kernels.topology import (
    enumerated_canonical_compact,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for canonical selection"
)


def _mixed_block() -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    component_id = torch.tensor([3, 1, 2, 4, 3, 3, 1, 2], device=device, dtype=torch.int32)
    depth = torch.tensor([2, 1, 1, 2, 2, 2, 1, 1], device=device, dtype=torch.int32)
    rx_id = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], device=device, dtype=torch.int32)
    sequence = torch.tensor(
        [[5, 7], [5, -1], [-1, -1], [7, 5], [5, 7], [5, 8], [5, -1], [-1, -1]],
        device=device,
        dtype=torch.int32,
    )
    edge_id = torch.tensor([7, -1, 7, 7, 7, 8, -1, 7], device=device, dtype=torch.int32)
    count = int(component_id.numel())
    value = torch.arange(1, count + 1, device=device, dtype=torch.float32)
    positions = torch.stack((value, value + 10.0, value + 20.0), dim=-1)
    positions = positions[:, None, :].expand(-1, 2, -1).contiguous()
    normals = torch.zeros_like(positions)
    normals[..., 2] = 1.0
    return {
        "valid": torch.ones(count, device=device, dtype=torch.bool),
        "tx_id": torch.zeros(count, device=device, dtype=torch.int32),
        "rx_id": rx_id,
        "depth": depth,
        "component_id": component_id,
        "primitive_id": torch.where(component_id == 2, torch.full_like(component_id, -1), sequence[:, 0]),
        "edge_id": edge_id,
        "path_length_m": value,
        "delay_s": value * 1.0e-9,
        "path_gain": value.square(),
        "path_field": torch.complex(value, torch.zeros_like(value)),
        "field_xyz": torch.complex(positions[:, 0], torch.zeros_like(positions[:, 0])).contiguous(),
        "coefficient": torch.complex(value, torch.zeros_like(value)),
        "field_direction": torch.nn.functional.normalize(positions[:, 0], dim=-1).contiguous(),
        "interaction_position": positions[:, 0].contiguous(),
        "interaction_normal": normals[:, 0].contiguous(),
        "material_id": torch.zeros(count, device=device, dtype=torch.int32),
        "primitive_sequence": sequence,
        "material_sequence": torch.zeros((count, 2), device=device, dtype=torch.int32),
        "interaction_positions": positions,
        "interaction_normals": normals,
    }


def _permute(block: dict[str, torch.Tensor], order: torch.Tensor) -> dict[str, torch.Tensor]:
    count = int(order.numel())
    return {
        name: value[order] if isinstance(value, torch.Tensor) and value.shape[:1] == (count,) else value
        for name, value in block.items()
    }


def _identity(result) -> torch.Tensor:
    topology = result.topology
    block = {
        "valid": topology.valid,
        "tx_id": topology.tx_id,
        "rx_id": topology.rx_id,
        "depth": topology.depth,
        "component_id": topology.component_id,
        "primitive_sequence": topology.primitive_sequence,
        "edge_id": topology.edge_id,
    }
    return concatenate.canonical_sequence_key(block).reshape(
        topology.valid.numel(), -1
    )


def _select(block, *, max_paths=None):
    evaluated, _ = evaluated_paths_from_block(
        block,
        max_paths=max_paths,
        max_paths_scope="per_pair",
        tx_count=1,
        rx_count=2,
        max_depth=2,
        launch_count=0,
    )
    return evaluated


def test_mixed_component_order_and_dedup_are_invariant_to_input_shuffle():
    block = _mixed_block()
    baseline = _select(block)
    permutation = torch.tensor([7, 4, 2, 6, 0, 5, 3, 1], device="cuda")
    shuffled = _select(_permute(block, permutation))

    torch.testing.assert_close(_identity(shuffled), _identity(baseline))
    torch.testing.assert_close(
        shuffled.geometry.path_length_m, baseline.geometry.path_length_m
    )
    assert int(baseline.topology.valid.numel()) == 7


def test_coupled_edge_id_is_part_of_identity_and_dedup_precedes_pair_cap():
    selected = _select(_mixed_block(), max_paths=5)
    identity = _identity(selected)

    topology = selected.topology
    pair0 = topology.rx_id == 0
    assert int(pair0.sum()) == 5
    assert torch.unique(identity[pair0], dim=0).shape[0] == 5
    coupled_edges = topology.edge_id[pair0 & (topology.component_id == 3)]
    assert coupled_edges.tolist() == [7, 8]


def test_per_pair_cap_is_independent_at_pair_boundary():
    selected = _select(_mixed_block(), max_paths=2)
    assert selected.topology.rx_id.tolist() == [0, 0, 1, 1]
    endpoint_identity = torch.cat(
        (selected.topology.rx_id[:, None], _identity(selected)), dim=1
    )
    assert torch.unique(endpoint_identity, dim=0).shape[0] == 4


def test_exact_owner_carries_empty_pair_offsets_and_one_sync_ledger():
    evaluated, sidecars = evaluated_paths_from_block(
        _mixed_block(),
        max_paths=None,
        max_paths_scope="per_pair",
        tx_count=1,
        rx_count=3,
        max_depth=2,
        launch_count=0,
    )

    metadata = sidecars.compact_metadata
    assert metadata is not None
    assert metadata.path_count == evaluated.row_count == 7
    torch.testing.assert_close(
        metadata.pair_offsets,
        torch.tensor([0, 5, 7, 7], device="cuda", dtype=torch.int64),
    )
    assert metadata.count_d2h_copies == 1
    assert metadata.count_d2h_bytes == 8
    assert metadata.count_synchronizations == 1
    assert bool(evaluated.topology.valid.all())


def test_exact_owner_vjp_scatters_continuous_rows_to_unique_sources():
    block = _mixed_block()
    path_length = block["path_length_m"].detach().requires_grad_(True)
    interaction_positions = block["interaction_positions"].detach().requires_grad_(True)
    block["path_length_m"] = path_length
    block["interaction_positions"] = interaction_positions

    compact = enumerated_canonical_compact(
        block,
        pair_count=2,
        num_tx=1,
        num_rx=2,
        max_paths=None,
        max_paths_scope="per_pair",
        sequence_width=2,
    )
    loss = (
        compact.block["path_length_m"].sum()
        + compact.block["interaction_positions"].sum()
    )
    loss.backward()

    selected_row_index = compact.selected_row_index
    assert torch.unique(selected_row_index).numel() == selected_row_index.numel()
    assert path_length.grad is not None
    assert interaction_positions.grad is not None
    expected_path_grad = torch.zeros_like(path_length)
    expected_path_grad[selected_row_index] = 1
    expected_positions_grad = torch.zeros_like(interaction_positions)
    expected_positions_grad[selected_row_index] = 1
    torch.testing.assert_close(path_length.grad, expected_path_grad)
    torch.testing.assert_close(
        interaction_positions.grad,
        expected_positions_grad,
    )
    assert not compact.block["tx_id"].requires_grad
    assert not compact.pair_index.requires_grad


def test_exact_owner_jvp_gathers_path_and_interaction_tangents():
    block = _mixed_block()
    path_tangent = torch.arange(
        1, block["path_length_m"].numel() + 1, device="cuda", dtype=torch.float32
    )
    position_tangent = torch.arange(
        block["interaction_positions"].numel(),
        device="cuda",
        dtype=torch.float32,
    ).reshape_as(block["interaction_positions"])

    with torch.autograd.forward_ad.dual_level():
        block["path_length_m"] = torch.autograd.forward_ad.make_dual(
            block["path_length_m"], path_tangent
        )
        block["interaction_positions"] = torch.autograd.forward_ad.make_dual(
            block["interaction_positions"], position_tangent
        )
        compact = enumerated_canonical_compact(
            block,
            pair_count=2,
            num_tx=1,
            num_rx=2,
            max_paths=None,
            max_paths_scope="per_pair",
            sequence_width=2,
        )
        _, path_jvp = torch.autograd.forward_ad.unpack_dual(
            compact.block["path_length_m"]
        )
        _, positions_jvp = torch.autograd.forward_ad.unpack_dual(
            compact.block["interaction_positions"]
        )
        selected_row_index = compact.selected_row_index

    assert path_jvp is not None
    assert positions_jvp is not None
    torch.testing.assert_close(path_jvp, path_tangent[selected_row_index])
    torch.testing.assert_close(
        positions_jvp,
        position_tangent[selected_row_index],
    )
