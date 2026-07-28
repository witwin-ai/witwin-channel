import math

import pytest
import torch

from tests.support.core_world import make_receiver, make_transmitter
from witwin.channel import capabilities
from witwin.channel.path import (
    Config,
    PathResult,
    RaggedPathSoA,
    solve,
)
from witwin.channel.path.arrays import pack_synthetic_arrays
from witwin.channel.path import solver as path_solver
from witwin.channel.scene.endpoints import apply_endpoint_weights
from witwin.channel.scene.endpoints import _ReceiverPointView, _TransmitterView
from witwin.core import AntennaPattern, Scene


def _ula(num_antennas: int, spacing_m: float, *, axis: str = "x") -> torch.Tensor:
    positions = torch.zeros((num_antennas, 3), dtype=torch.float32)
    offsets = torch.arange(num_antennas, dtype=torch.float32)
    offsets -= 0.5 * (num_antennas - 1)
    positions[:, {"x": 0, "y": 1, "z": 2}[axis]] = offsets * spacing_m
    return positions


def _ura(
    rows: int,
    columns: int,
    spacing_m: tuple[float, float],
    *,
    axes: tuple[str, str] = ("x", "y"),
) -> torch.Tensor:
    positions = torch.zeros((rows * columns, 3), dtype=torch.float32)
    row = torch.arange(rows, dtype=torch.float32) - 0.5 * (rows - 1)
    column = torch.arange(columns, dtype=torch.float32) - 0.5 * (columns - 1)
    row_grid, column_grid = torch.meshgrid(row, column, indexing="ij")
    axis_ids = {"x": 0, "y": 1, "z": 2}
    positions[:, axis_ids[axes[0]]] = row_grid.reshape(-1) * spacing_m[0]
    positions[:, axis_ids[axes[1]]] = column_grid.reshape(-1) * spacing_m[1]
    return positions


def _solver_transmitter(position, **kwargs) -> _TransmitterView:
    return _TransmitterView(make_transmitter(position, **kwargs))


def _solver_receiver(position, **kwargs) -> _ReceiverPointView:
    return _ReceiverPointView(make_receiver(position, **kwargs))


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
            _solver_transmitter(
                torch.zeros(3),
                element_positions=_ula(2, 0.5),
            )
        ],
        receivers=[_solver_receiver(torch.ones(3))],
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
            _solver_transmitter(
                torch.zeros(3),
                element_positions=_ura(2, 2, (0.5, 0.5)),
            )
        ],
        receivers=[_solver_receiver(torch.ones(3))],
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
            _solver_transmitter(
                torch.zeros(3),
                pattern=AntennaPattern("horizontal"),
            )
        ],
        receivers=[_solver_receiver(torch.ones(3))],
    )
    rotated = pack_synthetic_arrays(
        _centre_result(),
        frequency_hz=1.0e9,
        transmitters=[
            _solver_transmitter(
                torch.zeros(3),
                orientation=torch.tensor([math.pi / 2.0, 0.0, 0.0]),
                pattern=AntennaPattern("horizontal"),
            )
        ],
        receivers=[_solver_receiver(torch.ones(3))],
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
        endpoints=[
            make_transmitter(
                torch.tensor([0.0, float(index), 0.0]),
                element_positions=_ula(2, 0.05, axis="y"),
                weights=tx_weights[index],
            )
            for index in range(2)
        ]
        + [
            make_receiver(
                torch.tensor([20.0, float(index), 0.0]),
                element_positions=_ura(1, 2, (0.05, 0.05), axes=("y", "z")),
                weights=rx_weights[index],
            )
            for index in range(2)
        ],
    )

    result = solve(scene, Config(components={"los"}), reference_frequency_hz=1.0e9)
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
        endpoints=[
            make_transmitter(
                torch.zeros(3),
                element_positions=_ula(2, 0.5),
                synthetic_array=False,
            ),
            make_receiver(
                torch.tensor([0.0, 2.0, 0.0]),
                element_positions=_ula(2, 0.5),
                synthetic_array=False,
            ),
        ],
    )

    result = solve(scene, Config(components={"los"}), reference_frequency_hz=1.0e9)

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
    element_positions = _ula(2, 0.01, axis="y")

    def solve_mode(synthetic: bool) -> PathResult:
        return solve(
            Scene(
                structures=[],
                endpoints=[
                    make_transmitter(
                        torch.zeros(3),
                        element_positions=element_positions,
                        synthetic_array=synthetic,
                    ),
                    make_receiver(torch.tensor([1000.0, 100.0, 0.0])),
                ],
            ),
            Config(components={"los"}),
            reference_frequency_hz=1.0e9,
        )

    synthetic = solve_mode(True)
    explicit = solve_mode(False)

    torch.testing.assert_close(synthetic.a, explicit.a, rtol=2.0e-3, atol=1.0e-8)
    assert torch.unique(synthetic.tau).numel() == 1
    assert torch.unique(explicit.tau).numel() == 2


def test_unsupported_synthetic_layout_fails_before_native_solve(monkeypatch):
    scene = Scene(
        structures=[],
        endpoints=[
            make_transmitter(torch.zeros(3), element_positions=torch.zeros((1, 3))),
            make_transmitter(
                torch.ones(3),
                element_positions=_ula(2, 0.5),
            ),
            make_receiver(torch.tensor([0.0, 2.0, 0.0])),
        ],
    )
    monkeypatch.setattr(
        path_solver,
        "_solve_base",
        lambda *_args, **_kwargs: pytest.fail(
            "native solve ran before array preflight"
        ),
    )

    with pytest.raises(ValueError, match="same antenna count"):
        solve(
            scene,
            Config(components={"los"}),
            reference_frequency_hz=1.0e9,
        )


@pytest.mark.parametrize("synthetic_array", [True, False])
def test_partial_endpoint_weights_fail_before_native_solve(
    monkeypatch, synthetic_array
):
    scene = Scene(
        structures=[],
        endpoints=[
            make_transmitter(
                torch.zeros(3),
                element_positions=_ula(2, 0.5),
                synthetic_array=synthetic_array,
                weights=torch.ones(2, dtype=torch.complex64),
            ),
            make_transmitter(
                torch.ones(3),
                element_positions=_ula(2, 0.5),
                synthetic_array=synthetic_array,
            ),
            make_receiver(torch.tensor([0.0, 2.0, 0.0])),
        ],
    )
    monkeypatch.setattr(
        path_solver,
        "_solve_base",
        lambda *_args, **_kwargs: pytest.fail(
            "native solve ran before weight preflight"
        ),
    )

    with pytest.raises(ValueError, match="precoding must be configured on every"):
        solve(
            scene,
            Config(components={"los"}),
            reference_frequency_hz=1.0e9,
        )
