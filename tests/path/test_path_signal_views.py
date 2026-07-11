import math

import torch

from witwin.channel_native.path import InteractionType, PathResultV2, RaggedPathSoA


def _signal_result() -> PathResultV2:
    field = torch.tensor([[1.0 + 0.0j], [0.0 + 2.0j]], dtype=torch.complex64)
    types = torch.tensor(
        [[int(InteractionType.REFLECTION)], [int(InteractionType.DIFFRACTION)]],
        dtype=torch.int32,
    )
    ragged = RaggedPathSoA.from_flat(
        num_rx=1,
        num_rx_ant=1,
        num_tx=1,
        num_tx_ant=1,
        rx_id=torch.zeros(2, dtype=torch.int32),
        tx_id=torch.zeros(2, dtype=torch.int32),
        field=field,
        delay_s=torch.tensor([1.0e-9, 3.0e-9]),
        theta_t=torch.zeros(2),
        phi_t=torch.zeros(2),
        theta_r=torch.zeros(2),
        phi_r=torch.zeros(2),
        interaction_type=types,
        primitive_id=torch.tensor([[3], [7]], dtype=torch.int32),
        material_id=torch.tensor([[1], [2]], dtype=torch.int32),
        position=torch.zeros((2, 1, 3)),
        normal=torch.zeros((2, 1, 3)),
    )
    return PathResultV2.from_ragged(ragged)


def test_cir_masks_padding_and_normalizes_delay_per_pair():
    result = _signal_result()
    coeff, tau = result.cir()

    assert torch.equal(coeff, result.a)
    assert torch.allclose(tau.flatten(), torch.tensor([0.0, 2.0e-9]))


def test_cfr_matches_manual_complex_sum():
    result = _signal_result()
    frequency = torch.tensor([0.0, 125.0e6])
    cfr = result.cfr(frequency)
    expected_zero = torch.tensor(1.0 + 2.0j)
    expected_one = 1.0 + 2.0j * torch.exp(torch.tensor(-2.0j * math.pi * 0.25))

    assert cfr.shape == (1, 1, 1, 1, 1, 2)
    assert torch.allclose(cfr[..., 0].squeeze(), expected_zero, atol=1.0e-6)
    assert torch.allclose(cfr[..., 1].squeeze(), expected_one, atol=1.0e-5)


def test_taps_accumulate_paths_into_analytic_bins():
    result = _signal_result()
    taps = result.taps(bandwidth=1.0e9, num_taps=4)

    assert taps.shape == (1, 1, 1, 1, 1, 4)
    assert taps[..., 0].item() == 1.0 + 0.0j
    assert taps[..., 2].item() == 0.0 + 2.0j


def test_filter_by_type_compacts_paths_and_preserves_geometry():
    result = _signal_result()
    filtered = result.filter_by_type(InteractionType.DIFFRACTION)

    assert filtered.a.shape == (1, 1, 1, 1, 1, 1)
    assert filtered.num_paths.item() == 1
    assert filtered.a.item() == 0.0 + 2.0j
    assert filtered.primitive_id.item() == 7
    assert filtered.metadata["filtered_interaction_types"] == [2]


def test_filter_by_type_keeps_legacy_minimum_width_when_all_paths_are_removed():
    filtered = _signal_result().filter_by_type(InteractionType.TRANSMISSION)

    assert filtered.a.shape == (1, 1, 1, 1, 1, 1)
    assert filtered.num_paths.item() == 0
    assert not filtered.valid.any()
    assert filtered.tau.item() == -1.0
