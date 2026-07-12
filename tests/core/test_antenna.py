import math

import torch

from witwin.channel_native import AntennaArray, AntennaPattern, Transmitter
from witwin.channel_native.core.antenna import (
    apply_endpoint_weights,
    apply_precoding_combining,
    orientation_matrix,
    steering_vector,
)


def test_ula_steering_matches_analytic_half_wavelength_phase():
    array = AntennaArray.ula(2, 0.5)
    steering = steering_vector(
        array,
        torch.tensor([1.0, 0.0, 0.0]),
        frequency_hz=299_792_458.0,
    )

    expected = torch.tensor([-1.0j, 1.0j], dtype=torch.complex64)
    torch.testing.assert_close(steering, expected, atol=1.0e-6, rtol=1.0e-6)


def test_ura_is_centred_and_row_major():
    array = AntennaArray.ura(2, 3, (2.0, 1.0))

    assert array.positions.shape == (6, 3)
    torch.testing.assert_close(array.positions.mean(dim=0), torch.zeros(3))
    torch.testing.assert_close(array.positions[0], torch.tensor([-1.0, -1.0, 0.0]))
    torch.testing.assert_close(array.positions[3], torch.tensor([1.0, -1.0, 0.0]))


def test_orientation_rotates_array_and_polarization_consistently():
    orientation = torch.tensor([0.0, math.pi / 2.0, 0.0])
    transmitter = Transmitter(position=torch.zeros(3), orientation=orientation)
    rotation = orientation_matrix(orientation)

    torch.testing.assert_close(
        transmitter.polarization,
        torch.tensor([1.0, 0.0, 0.0]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        rotation @ torch.tensor([0.0, 0.0, 1.0]), transmitter.polarization
    )


def test_builtin_dipole_patterns_have_axis_nulls():
    vertical = AntennaPattern("vertical")
    horizontal = AntennaPattern("horizontal")
    directions = torch.eye(3)

    torch.testing.assert_close(
        vertical.field_response(directions),
        torch.tensor([1.0, 1.0, 0.0], dtype=torch.complex64),
    )
    torch.testing.assert_close(
        horizontal.field_response(directions),
        torch.tensor([0.0, 1.0, 1.0], dtype=torch.complex64),
    )


def test_precoding_and_combining_use_receiver_conjugate_weights():
    coefficients = torch.tensor(
        [[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]],
        dtype=torch.complex64,
    )
    value = apply_precoding_combining(
        coefficients,
        tx_weights=torch.tensor([1.0, 0.5], dtype=torch.complex64),
        rx_weights=torch.tensor([1.0, 1.0j], dtype=torch.complex64),
    )

    torch.testing.assert_close(value, torch.tensor(2.0 - 5.0j))


def test_endpoint_weights_support_multi_tx_rx_and_preserve_signal_dimensions():
    coefficients = torch.arange(2 * 2 * 2 * 2 * 3, dtype=torch.float32).reshape(
        2, 2, 2, 2, 3
    ).to(torch.complex64)
    tx_weights = torch.tensor([[1.0, 0.5j], [-0.25j, 2.0]], dtype=torch.complex64)
    rx_weights = torch.tensor([[1.0, -1.0j], [0.5j, 0.25]], dtype=torch.complex64)

    combined = apply_endpoint_weights(
        coefficients, tx_weights=tx_weights, rx_weights=rx_weights
    )
    expected = torch.empty((2, 2, 3), dtype=torch.complex64)
    for rx in range(2):
        for tx in range(2):
            expected[rx, tx] = torch.einsum(
                "abk,b,a->k",
                coefficients[rx, :, tx],
                tx_weights[tx],
                rx_weights[rx].conj(),
            )

    torch.testing.assert_close(combined, expected)
