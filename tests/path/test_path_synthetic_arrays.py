import math

import pytest
import torch

from witwin.channel import (
    AntennaArray,
    AntennaPattern,
    ReceiverPoint,
    Scene,
    Transmitter,
    capabilities,
)
from witwin.channel.path import (
    Config,
    PathResult,
    RaggedPathSoA,
    pack_synthetic_arrays,
    solve,
)
from witwin.channel.path import solver as path_solver
from witwin.channel.core.antenna import apply_endpoint_weights


def _centre_result() -> PathResult:
    ragged = RaggedPathSoA.from_flat(
        num_rx=1,
        num_rx_ant=1,
        num_tx=1,
        num_tx_ant=1,
        rx_id=torch.zeros(1, dtype=torch.int32),
        tx_id=torch.zeros(1, dtype=torch.int32),
        field=torch.ones((1, 1), dtype=torch.complex64),
        delay_s=torch.tensor([1.0e-9]),
        theta_t=torch.tensor([math.pi / 2.0]),
        phi_t=torch.zeros(1),
        theta_r=torch.tensor([math.pi / 2.0]),
        phi_r=torch.zeros(1),
        interaction_type=torch.empty((1, 0), dtype=torch.int32),
        primitive_id=torch.empty((1, 0), dtype=torch.int32),
        material_id=torch.empty((1, 0), dtype=torch.int32),
        position=torch.empty((1, 0, 3)),
        normal=torch.empty((1, 0, 3)),
    )
    return PathResult.from_ragged(ragged)


def test_synthetic_array_packing_preserves_paths_and_adds_steering_phase():
    result = pack_synthetic_arrays(
        _centre_result(),
        frequency_hz=299_792_458.0,
        transmitters=[
            Transmitter(
                position=torch.zeros(3),
                array=AntennaArray.ula(2, 0.5),
            )
        ],
        receivers=[ReceiverPoint(position=torch.ones(3))],
    )

    assert result.a.shape == (1, 1, 1, 2, 1, 1)
    torch.testing.assert_close(
        result.a[0, 0, 0, :, 0, 0],
        torch.tensor([-1.0j, 1.0j], dtype=torch.complex64),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert result.num_paths.tolist() == [[[[1, 1]]]]
    assert result.metadata["array_semantics"] == "synthetic_far_field_phase_weighting"


def test_synthetic_ura_is_row_major_and_uses_both_local_axes():
    result = pack_synthetic_arrays(
        _centre_result(),
        frequency_hz=299_792_458.0,
        transmitters=[
            Transmitter(
                position=torch.zeros(3),
                array=AntennaArray.ura(2, 2, (0.5, 0.5)),
            )
        ],
        receivers=[ReceiverPoint(position=torch.ones(3))],
    )

    torch.testing.assert_close(
        result.a[0, 0, 0, :, 0, 0],
        torch.tensor([-1.0j, -1.0j, 1.0j, 1.0j], dtype=torch.complex64),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_rotated_directional_pattern_is_evaluated_in_endpoint_local_frame():
    unrotated = pack_synthetic_arrays(
        _centre_result(),
        frequency_hz=1.0e9,
        transmitters=[
            Transmitter(
                position=torch.zeros(3),
                pattern=AntennaPattern("horizontal"),
            )
        ],
        receivers=[ReceiverPoint(position=torch.ones(3))],
    )
    rotated = pack_synthetic_arrays(
        _centre_result(),
        frequency_hz=1.0e9,
        transmitters=[
            Transmitter(
                position=torch.zeros(3),
                orientation=torch.tensor([math.pi / 2.0, 0.0, 0.0]),
                pattern=AntennaPattern("horizontal"),
            )
        ],
        receivers=[ReceiverPoint(position=torch.ones(3))],
    )

    torch.testing.assert_close(unrotated.a, torch.zeros_like(unrotated.a))
    torch.testing.assert_close(
        rotated.a, torch.ones_like(rotated.a), atol=1.0e-6, rtol=1.0e-6
    )


def test_multi_endpoint_weights_beamform_cir_cfr_and_taps_without_mutating_raw_h():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for end-to-end array topology")
    tx_weights = (
        torch.tensor([1.0, 0.5j], dtype=torch.complex64),
        torch.tensor([-0.25j, 1.0], dtype=torch.complex64),
    )
    rx_weights = (
        torch.tensor([1.0, -0.5j], dtype=torch.complex64),
        torch.tensor([0.25j, 0.75], dtype=torch.complex64),
    )
    scene = Scene(
        structures=[],
        transmitters=[
            Transmitter(
                position=torch.tensor([0.0, float(index), 0.0]),
                array=AntennaArray.ula(2, 0.05, axis="y"),
                precoding=tx_weights[index],
            )
            for index in range(2)
        ],
        receivers=[
            ReceiverPoint(
                position=torch.tensor([20.0, float(index), 0.0]),
                array=AntennaArray.ura(1, 2, (0.05, 0.05), axes=("y", "z")),
                combining=rx_weights[index],
            )
            for index in range(2)
        ],
        frequency=1.0e9,
    )

    result = solve(scene, Config(components={"los"}))
    raw = result.a.clone()
    beamformed = result.beamform()
    frequencies = torch.tensor([0.0, 1.0e6], device=result.a.device)
    raw_cfr = result.cfr(frequencies, normalize_delays=False)
    expected_cfr = apply_endpoint_weights(
        raw_cfr,
        tx_weights=torch.stack(tx_weights).to(result.a.device),
        rx_weights=torch.stack(rx_weights).to(result.a.device),
    )

    torch.testing.assert_close(
        beamformed.cfr(frequencies, normalize_delays=False), expected_cfr
    )
    coefficient, tau = beamformed.cir(normalize_delays=False)
    assert coefficient.shape == (2, 2, 4, 1)
    assert tau.shape == (2, 2, 4)
    torch.testing.assert_close(coefficient.sum(dim=2), expected_cfr[..., 0])
    taps = beamformed.taps(1.0e9, 8, normalize_delays=True)
    torch.testing.assert_close(
        taps.sum(dim=-1),
        beamformed.cfr(torch.zeros(1, device=result.a.device))[..., 0],
    )
    torch.testing.assert_close(result.a, raw)


def test_explicit_array_traces_exact_per_element_free_space_distance():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for explicit path topology")
    scene = Scene(
        structures=[],
        transmitters=[
            Transmitter(
                position=torch.zeros(3),
                array=AntennaArray.ula(2, 0.5),
                synthetic_array=False,
            )
        ],
        receivers=[
            ReceiverPoint(
                position=torch.tensor([0.0, 2.0, 0.0]),
                array=AntennaArray.ula(2, 0.5),
                synthetic_array=False,
            )
        ],
        frequency=1.0e9,
    )

    result = solve(scene, Config(components={"los"}))

    assert result.a.shape == (1, 2, 1, 2, 1, 1)
    expected_distance = torch.tensor(
        [[2.0, math.sqrt(4.25)], [math.sqrt(4.25), 2.0]],
        device=result.tau.device,
    )
    torch.testing.assert_close(
        result.tau[0, :, 0, :, 0] * 299_792_458.0,
        expected_distance,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert result.metadata["array_semantics"] == "explicit_per_element_topology"
    assert capabilities()["solvers"]["path"]["supports_arrays"]


def test_synthetic_and_explicit_arrays_converge_in_the_far_field():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for explicit path topology")
    array = AntennaArray.ula(2, 0.01, axis="y")

    def solve_mode(synthetic: bool) -> PathResult:
        return solve(
            Scene(
                structures=[],
                transmitters=[
                    Transmitter(
                        position=torch.zeros(3),
                        array=array,
                        synthetic_array=synthetic,
                    )
                ],
                receivers=[
                    ReceiverPoint(position=torch.tensor([1000.0, 100.0, 0.0]))
                ],
                frequency=1.0e9,
            ),
            Config(components={"los"}),
        )

    synthetic = solve_mode(True)
    explicit = solve_mode(False)

    torch.testing.assert_close(synthetic.a, explicit.a, rtol=2.0e-3, atol=1.0e-8)
    assert torch.unique(synthetic.tau).numel() == 1
    assert torch.unique(explicit.tau).numel() == 2


def test_unsupported_synthetic_layout_fails_before_native_solve(monkeypatch):
    scene = Scene(
        structures=[],
        transmitters=[
            Transmitter(position=torch.zeros(3), array=AntennaArray.single()),
            Transmitter(
                position=torch.ones(3),
                array=AntennaArray.ula(2, 0.5),
            ),
        ],
        receivers=[ReceiverPoint(position=torch.tensor([0.0, 2.0, 0.0]))],
        frequency=1.0e9,
    )
    monkeypatch.setattr(
        path_solver,
        "_solve_base",
        lambda *_args, **_kwargs: pytest.fail("native solve ran before array preflight"),
    )

    with pytest.raises(ValueError, match="same antenna count"):
        solve(scene, Config(components={"los"}))


@pytest.mark.parametrize("synthetic_array", [True, False])
def test_partial_endpoint_weights_fail_before_native_solve(
    monkeypatch, synthetic_array
):
    scene = Scene(
        structures=[],
        transmitters=[
            Transmitter(
                position=torch.zeros(3),
                array=AntennaArray.ula(2, 0.5),
                synthetic_array=synthetic_array,
                precoding=torch.ones(2, dtype=torch.complex64),
            ),
            Transmitter(
                position=torch.ones(3),
                array=AntennaArray.ula(2, 0.5),
                synthetic_array=synthetic_array,
            ),
        ],
        receivers=[ReceiverPoint(position=torch.tensor([0.0, 2.0, 0.0]))],
        frequency=1.0e9,
    )
    monkeypatch.setattr(
        path_solver,
        "_solve_base",
        lambda *_args, **_kwargs: pytest.fail("native solve ran before weight preflight"),
    )

    with pytest.raises(ValueError, match="precoding must be configured on every"):
        solve(scene, Config(components={"los"}))
