import torch

from witwin.channel_native.path import PathResult, RaggedPathSoA


def test_explicit_antenna_and_time_dimensions_pack_without_shape_collapse():
    ragged = RaggedPathSoA.from_flat(
        num_rx=1,
        num_rx_ant=2,
        num_tx=1,
        num_tx_ant=2,
        rx_id=torch.tensor([0, 0], dtype=torch.int32),
        rx_ant_id=torch.tensor([0, 1], dtype=torch.int32),
        tx_id=torch.tensor([0, 0], dtype=torch.int32),
        tx_ant_id=torch.tensor([1, 0], dtype=torch.int32),
        field=torch.tensor([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=torch.complex64),
        delay_s=torch.tensor([1.0e-9, 2.0e-9]),
        theta_t=torch.zeros(2),
        phi_t=torch.zeros(2),
        theta_r=torch.zeros(2),
        phi_r=torch.zeros(2),
        interaction_type=torch.empty((2, 0), dtype=torch.int32),
        primitive_id=torch.empty((2, 0), dtype=torch.int32),
        material_id=torch.empty((2, 0), dtype=torch.int32),
        position=torch.empty((2, 0, 3)),
        normal=torch.empty((2, 0, 3)),
    )
    result = PathResult.from_ragged(ragged)

    assert result.a.shape == (1, 2, 1, 2, 1, 2)
    assert result.num_paths.tolist() == [[[[0, 1]], [[1, 0]]]]
    assert torch.equal(result.a[0, 0, 0, 1, 0], torch.tensor([1 + 2j, 3 + 4j]))
    assert torch.equal(result.a[0, 1, 0, 0, 0], torch.tensor([5 + 6j, 7 + 8j]))
