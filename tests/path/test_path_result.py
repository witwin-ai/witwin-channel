from dataclasses import replace

import pytest
import torch

from tests.support.scenes import empty_space_los_scene
import witwin.channel.path as path_api
from witwin.channel.path import (
    Config,
    InteractionType,
    PathResult,
    RaggedPathSoA,
    solve,
)


def test_path_public_api_is_single_versionless_contract():
    assert set(path_api.__all__) == {
        "Config",
        "InteractionType",
        "PathResult",
        "RaggedPathSoA",
        "solve",
    }
    assert not hasattr(path_api, "pack_synthetic_arrays")
    assert not hasattr(path_api, "explicit_array_scene")
    assert not hasattr(path_api, "pack_explicit_arrays")


def _ragged(
    *, count: int, rx_id=None, tx_id=None, field=None, depth: int = 2, max_paths=None
):
    rx_id = torch.tensor(rx_id if rx_id is not None else [0] * count, dtype=torch.int32)
    tx_id = torch.tensor(tx_id if tx_id is not None else [0] * count, dtype=torch.int32)
    field = (
        field if field is not None else torch.ones((count, 1), dtype=torch.complex64)
    )
    types = torch.zeros((count, depth), dtype=torch.int32)
    if count and depth:
        types[:, 0] = int(InteractionType.REFLECTION)
    return RaggedPathSoA.from_flat(
        num_rx=2,
        num_rx_ant=1,
        num_tx=2,
        num_tx_ant=1,
        rx_id=rx_id,
        tx_id=tx_id,
        field=field,
        delay_s=torch.arange(count, dtype=torch.float32) * 1.0e-9,
        theta_t=torch.zeros(count),
        phi_t=torch.zeros(count),
        theta_r=torch.zeros(count),
        phi_r=torch.zeros(count),
        interaction_type=types,
        primitive_id=torch.full((count, depth), -1, dtype=torch.int32),
        material_id=torch.full((count, depth), -1, dtype=torch.int32),
        position=torch.zeros((count, depth, 3)),
        normal=torch.zeros((count, depth, 3)),
        max_paths_per_pair=max_paths,
    )


def test_zero_path_result_has_real_zero_width_path_dimension():
    result = PathResult.from_ragged(_ragged(count=0))

    assert result.a.shape == (2, 1, 2, 1, 0, 1)
    assert result.valid.shape == (2, 1, 2, 1, 0)
    assert result.num_paths.shape == (2, 1, 2, 1)
    assert result.num_paths.sum() == 0


def test_ragged_packing_is_stable_per_pair_and_uses_per_pair_limit():
    field = torch.tensor([[10], [20], [30], [40], [50]], dtype=torch.complex64)
    ragged = _ragged(
        count=5,
        rx_id=[1, 0, 1, 0, 1],
        tx_id=[0, 1, 0, 1, 0],
        field=field,
        max_paths=2,
    )
    result = PathResult.from_ragged(ragged, max_paths_per_pair=2)

    assert result.metadata["max_paths_scope"] == "per_pair"
    assert result.num_paths[0, 0, 1, 0] == 2
    assert result.num_paths[1, 0, 0, 0] == 2
    assert result.a[0, 0, 1, 0, :, 0].real.tolist() == [20.0, 40.0]
    assert result.a[1, 0, 0, 0, :, 0].real.tolist() == [10.0, 30.0]


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("rx_id", [-1]),
        ("rx_ant_id", [1]),
        ("tx_id", [2]),
        ("tx_ant_id", [1]),
    ],
)
def test_ragged_rejects_each_invalid_endpoint_dimension(field, values):
    kwargs = {
        "rx_id": torch.tensor([0], dtype=torch.int32),
        "rx_ant_id": torch.tensor([0], dtype=torch.int32),
        "tx_id": torch.tensor([0], dtype=torch.int32),
        "tx_ant_id": torch.tensor([0], dtype=torch.int32),
    }
    kwargs[field] = torch.tensor(values, dtype=torch.int32)
    with pytest.raises(ValueError, match=field):
        RaggedPathSoA.from_flat(
            num_rx=2,
            num_rx_ant=1,
            num_tx=2,
            num_tx_ant=1,
            field=torch.ones((1, 1), dtype=torch.complex64),
            delay_s=torch.zeros(1),
            theta_t=torch.zeros(1),
            phi_t=torch.zeros(1),
            theta_r=torch.zeros(1),
            phi_r=torch.zeros(1),
            interaction_type=torch.empty((1, 0), dtype=torch.int32),
            primitive_id=torch.empty((1, 0), dtype=torch.int32),
            material_id=torch.empty((1, 0), dtype=torch.int32),
            position=torch.empty((1, 0, 3)),
            normal=torch.empty((1, 0, 3)),
            **kwargs,
        )


def test_path_result_validates_shape_contract():
    result = PathResult.from_ragged(_ragged(count=1))

    with pytest.raises(ValueError, match="tau"):
        replace(result, tau=torch.zeros(2))


def test_solve_los_shape_ids_lengths_and_per_pair_count():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solve")
    result = solve(
        empty_space_los_scene(),
        Config(max_depth=0, components={"los"}),
        reference_frequency_hz=3.0e9,
    )

    assert result.a.shape == (2, 1, 2, 1, 1, 1)
    assert torch.all(result.num_paths == 1)
    assert result.tx_id.shape == result.valid.shape
    assert result.rx_id.shape == result.valid.shape
    assert result.path_length_m.shape == result.valid.shape
    torch.testing.assert_close(
        result.path_length_m[result.valid], result.tau[result.valid] * 299_792_458.0
    )
    assert result.metadata["schema"] == "PathResult"
    assert result.metadata["max_paths_scope"] == "per_pair"
