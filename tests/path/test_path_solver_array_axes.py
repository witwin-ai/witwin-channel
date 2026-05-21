from __future__ import annotations

import math

import torch
import witwin.channel as wc
from witwin.channel.core.numerics.tensors import drjit_to_torch_view


def _solve_los(scene: wc.Scene, *, synthetic_array: bool = True):
    return wc.path.solve(
        scene=scene,
        transmitter="tx",
        receiver="rx",
        config=wc.path.Config(
            max_num_paths=1,
            max_bounces=0,
            max_diffraction_order=0,
            synthetic_array=synthetic_array,
        ),
    )


def _los_array_scene() -> wc.Scene:
    return wc.Scene(
        transmitters=[
            wc.Transmitter(
                name="tx",
                position=(0.0, 0.0, 1.0),
                array=wc.ULA(num_elements=2, spacing=0.01, polarization="VH"),
            )
        ],
        receivers=[
            wc.Receiver(
                name="rx",
                position=(10.0, 0.0, 1.0),
                array=wc.ULA(num_elements=3, spacing=0.01),
            )
        ],
        frequency=3.0e9,
        device="cuda",
    )


def test_path_solver_expands_array_axes_and_time_axis():
    result = wc.path.solve(
        scene=_los_array_scene(),
        transmitter="tx",
        receiver="rx",
        config=wc.path.Config(
            max_num_paths=1,
            max_bounces=0,
            max_diffraction_order=0,
            synthetic_array=True,
        ),
    )

    assert result.num_tx_ant == 4
    assert result.num_rx_ant == 3
    assert result.num_time_steps == 1
    assert tuple(result.coeff_tensor().shape) == (1, 3, 1, 4, 1, 1)
    assert tuple(result.tau.shape) == (1, 3, 1, 4, 1)
    assert tuple(result.num_paths.shape) == (1, 3, 1, 4)


def test_synthetic_and_explicit_array_modes_match_in_far_field():
    scene = _los_array_scene()
    synthetic = _solve_los(scene, synthetic_array=True)
    explicit = _solve_los(scene, synthetic_array=False)

    synthetic_power = torch.abs(synthetic.coeff_tensor()).clamp_min(1e-30)
    explicit_power = torch.abs(explicit.coeff_tensor()).clamp_min(1e-30)
    delta_db = 20.0 * torch.log10(synthetic_power / explicit_power).abs()
    assert float(delta_db.max().item()) <= 0.1


def test_path_solver_applies_tr38901_pattern_gain_to_broadside_los():
    def _scene(pattern: str, rx_position: tuple[float, float, float]) -> wc.Scene:
        return wc.Scene(
            transmitters=[
                wc.Transmitter(
                    name="tx",
                    position=(0.0, 0.0, 0.0),
                    array=wc.AntennaArray(element_positions=[(0.0, 0.0, 0.0)], pattern=pattern),
                )
            ],
            receivers=[wc.Receiver(name="rx", position=rx_position)],
            frequency=3.0e9,
            device="cuda",
        )

    iso = _solve_los(_scene("iso", (10.0, 0.0, 0.0))).coeff_tensor()
    broadside = _solve_los(_scene("tr38901", (10.0, 0.0, 0.0))).coeff_tensor()
    zenith = _solve_los(_scene("tr38901", (0.0, 0.0, 10.0))).coeff_tensor()

    assert float(torch.abs(broadside).max().item()) > 2.0 * float(torch.abs(iso).max().item())
    assert float(torch.abs(zenith).max().item()) < 0.2 * float(torch.abs(broadside).max().item())


def test_path_solver_applies_tr38901_horizontal_cut():
    def _scene(rx_position: tuple[float, float, float]) -> wc.Scene:
        return wc.Scene(
            transmitters=[
                wc.Transmitter(
                    name="tx",
                    position=(0.0, 0.0, 0.0),
                    array=wc.AntennaArray(element_positions=[(0.0, 0.0, 0.0)], pattern="tr38901"),
                )
            ],
            receivers=[wc.Receiver(name="rx", position=rx_position)],
            frequency=3.0e9,
            device="cuda",
        )

    broadside = torch.abs(_solve_los(_scene((10.0, 0.0, 0.0))).coeff_tensor()).max()
    horizontal_side = torch.abs(_solve_los(_scene((0.0, 10.0, 0.0))).coeff_tensor()).max()

    assert float(horizontal_side.item()) < 0.2 * float(broadside.item())


def test_path_solver_applies_short_dipole_pattern():
    def _scene(pattern: str, rx_position: tuple[float, float, float]) -> wc.Scene:
        return wc.Scene(
            transmitters=[
                wc.Transmitter(
                    name="tx",
                    position=(0.0, 0.0, 0.0),
                    array=wc.AntennaArray(element_positions=[(0.0, 0.0, 0.0)], pattern=pattern),
                )
            ],
            receivers=[wc.Receiver(name="rx", position=rx_position)],
            frequency=3.0e9,
            device="cuda",
        )

    iso_broadside = torch.abs(_solve_los(_scene("iso", (10.0, 0.0, 0.0))).coeff_tensor()).max()
    dipole_broadside = torch.abs(_solve_los(_scene("dipole", (10.0, 0.0, 0.0))).coeff_tensor()).max()
    iso_zenith = torch.abs(_solve_los(_scene("iso", (0.0, 0.0, 10.0))).coeff_tensor()).max()
    dipole_zenith = torch.abs(_solve_los(_scene("dipole", (0.0, 0.0, 10.0))).coeff_tensor()).max()

    broadside_ratio = dipole_broadside / iso_broadside
    assert float(torch.abs(broadside_ratio - math.sqrt(1.5)).item()) < 1.0e-3
    assert float(dipole_zenith.item()) < 1.0e-4 * float(iso_zenith.item())


def test_path_solver_uses_element_orientations_for_pattern_response():
    def _scene(element_orientations=None) -> wc.Scene:
        return wc.Scene(
            transmitters=[
                wc.Transmitter(
                    name="tx",
                    position=(0.0, 0.0, 0.0),
                    array=wc.AntennaArray(
                        element_positions=[(0.0, 0.0, 0.0)],
                        element_orientations=element_orientations,
                        pattern="tr38901",
                    ),
                )
            ],
            receivers=[wc.Receiver(name="rx", position=(10.0, 0.0, 0.0))],
            frequency=3.0e9,
            device="cuda",
        )

    boresight = torch.abs(_solve_los(_scene()).coeff_tensor()).max()
    tilted = torch.abs(_solve_los(_scene([(math.pi / 2.0, 0.0, 0.0)])).coeff_tensor()).max()

    assert float(tilted.item()) < 0.2 * float(boresight.item())


def test_explicit_array_mode_rotates_element_positions_by_endpoint_orientation():
    scene = wc.Scene(
        transmitters=[
            wc.Transmitter(
                name="tx",
                position=(0.0, 0.0, 0.0),
                orientation=(math.pi / 2.0, 0.0, 0.0),
                array=wc.ULA(num_elements=2, spacing=1.0, axis="x"),
            )
        ],
        receivers=[wc.Receiver(name="rx", position=(10.0, 0.0, 0.0))],
        frequency=3.0e9,
        device="cuda",
    )

    result = _solve_los(scene, synthetic_array=False)
    tau = drjit_to_torch_view(result.tau, dtype=torch.float32)
    per_tx_antenna_tau = tau[0, 0, 0, :, 0]

    assert float(torch.abs(per_tx_antenna_tau[0] - per_tx_antenna_tau[1]).item()) < 1.0e-11
