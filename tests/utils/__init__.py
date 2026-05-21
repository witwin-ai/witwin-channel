from __future__ import annotations

from witwin.channel.core.runtime import Material, Rx, Tx, Wave


def test_runtime_endpoint_bundles_live_in_channel_utils():
    tx = Tx(position=(1.0, 2.0, 3.0), polarization=(0.0, 1.0, 0.0))
    rx = Rx(positions=(4.0, 5.0, 6.0))
    wave = Wave.from_frequency(3.0e9)
    material = Material(reflection_coef=0.5)

    assert tx.polarization_tuple == (0.0, 1.0, 0.0)
    assert rx.effective_polarization(tx) is tx.polarization
    assert wave.wavelength_scalar > 0.0
    assert material.gain_scalar == 0.5
