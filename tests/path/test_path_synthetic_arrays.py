import math

import pytest
import torch

from witwin.channel_native import (
    AntennaArray,
    ReceiverPoint,
    Scene,
    Transmitter,
    capabilities,
)
from witwin.channel_native.path import (
    Config,
    PathResultV2,
    RaggedPathSoA,
    pack_synthetic_arrays,
    solve_v2,
)


def _centre_result() -> PathResultV2:
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
    return PathResultV2.from_ragged(ragged)


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

    result = solve_v2(scene, Config(components={"los"}))

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
