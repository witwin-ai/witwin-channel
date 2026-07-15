from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import pytest
import torch

from witwin.channel_native.core.path_topology import TopologyBatch
from witwin.channel_native.propagation.models.adapters import (
    PathExecutionStats,
    TopologyBatchSidecars,
    evaluated_paths_from_topology_batch,
)


_TOPOLOGY_FIELDS = (
    "valid",
    "tx_id",
    "rx_id",
    "depth",
    "component_id",
    "primitive_id",
    "edge_id",
    "material_id",
    "primitive_sequence",
    "material_sequence",
    "interaction_type",
)
_GEOMETRY_FIELDS = (
    "path_length_m",
    "delay_s",
    "field_direction",
    "interaction_position",
    "interaction_normal",
    "interaction_positions",
    "interaction_normals",
)
_PATH_FIELDS = ("path_gain", "path_field", "field_xyz", "coefficient")


def _float_vector(rows: int) -> torch.Tensor:
    base = torch.arange(rows * 2, dtype=torch.float32, requires_grad=True) * 1.0
    return base[::2]


def _float_xyz(rows: int) -> torch.Tensor:
    base = torch.arange(rows * 6, dtype=torch.float32, requires_grad=True) * 1.0
    return base.reshape(rows, 6)[:, ::2]


def _float_sequence(rows: int, width: int) -> torch.Tensor:
    base = torch.arange(rows * width * 6, dtype=torch.float32, requires_grad=True) * 1.0
    return base.reshape(rows, width, 6)[:, :, ::2]


def _complex_vector(rows: int) -> torch.Tensor:
    real = _float_vector(rows)
    return torch.complex(real, real)


def _complex_xyz(rows: int) -> torch.Tensor:
    real = _float_xyz(rows)
    return torch.complex(real, real)


def _batch(rows: int = 3, width: int = 2) -> TopologyBatch:
    id_storage = torch.arange(rows * 4, dtype=torch.int32).reshape(rows, 4)
    tx_id = id_storage[:, 0]
    rx_id = id_storage[:, 1]
    ids = id_storage[:, 2]
    sequence_storage = torch.arange(rows * width * 6, dtype=torch.int32).reshape(
        rows, width, 6
    )
    primitive_sequence = sequence_storage[:, :, 0]
    material_sequence = sequence_storage[:, :, 2]
    interaction_type = sequence_storage[:, :, 4]
    valid = torch.ones(rows, dtype=torch.bool)
    path_length_m = _float_vector(rows)
    delay_s = _float_vector(rows)
    field_direction = _float_xyz(rows)
    interaction_position = _float_xyz(rows)
    interaction_normal = _float_xyz(rows)
    interaction_positions = _float_sequence(rows, width)
    interaction_normals = _float_sequence(rows, width)
    path_gain = _float_vector(rows)
    path_field = _complex_vector(rows)
    field_xyz = _complex_xyz(rows)
    coefficient = _complex_vector(rows)
    diffraction_vector_field = _complex_xyz(rows)
    return TopologyBatch(
        valid=valid,
        tx_id=tx_id,
        rx_id=rx_id,
        depth=ids,
        component_id=ids,
        primitive_id=ids,
        edge_id=ids,
        path_length_m=path_length_m,
        delay_s=delay_s,
        path_gain=path_gain,
        path_field=path_field,
        field_xyz=field_xyz,
        coefficient=coefficient,
        field_direction=field_direction,
        interaction_position=interaction_position,
        interaction_normal=interaction_normal,
        material_id=ids,
        primitive_sequence=primitive_sequence,
        material_sequence=material_sequence,
        interaction_type=interaction_type,
        interaction_positions=interaction_positions,
        interaction_normals=interaction_normals,
        launch_count=11,
        visibility_rejection_count=12,
        selected_edge_count=13,
        candidate_count=14,
        guardrail_count=15,
        diffraction_vector_field=diffraction_vector_field,
        ad_companion_launches=16,
        ad_tape_bytes=17,
    )


def _assert_exact_tensor(observed: torch.Tensor, expected: torch.Tensor) -> None:
    assert observed is expected
    assert observed.data_ptr() == expected.data_ptr()
    assert observed.untyped_storage()._cdata == expected.untyped_storage()._cdata
    assert (
        observed.untyped_storage().data_ptr() == expected.untyped_storage().data_ptr()
    )
    assert observed.storage_offset() == expected.storage_offset()
    assert observed.stride() == expected.stride()
    assert observed.dtype == expected.dtype
    assert observed.device == expected.device
    assert observed.requires_grad == expected.requires_grad
    assert observed.grad_fn is expected.grad_fn


def test_adapter_preserves_all_22_tensor_objects_and_metadata():
    batch = _batch()

    evaluated, _ = evaluated_paths_from_topology_batch(batch)

    for contract, names in (
        (evaluated.topology, _TOPOLOGY_FIELDS),
        (evaluated.geometry, _GEOMETRY_FIELDS),
        (evaluated.fields, _PATH_FIELDS),
    ):
        for name in names:
            _assert_exact_tensor(getattr(contract, name), getattr(batch, name))


def test_adapter_reuses_one_row_identity_and_preserves_shared_views():
    batch = _batch()
    evaluated, _ = evaluated_paths_from_topology_batch(batch)

    assert evaluated.geometry.row_identity is evaluated.topology.row_identity
    assert evaluated.fields.row_identity is evaluated.topology.row_identity
    assert evaluated.row_identity is evaluated.topology.row_identity
    assert evaluated.topology.tx_id is not evaluated.topology.rx_id
    assert (
        evaluated.topology.tx_id.untyped_storage()._cdata
        == evaluated.topology.rx_id.untyped_storage()._cdata
    )
    assert (
        evaluated.topology.tx_id.storage_offset()
        != evaluated.topology.rx_id.storage_offset()
    )
    assert (
        evaluated.topology.primitive_sequence
        is not evaluated.topology.material_sequence
    )
    assert (
        evaluated.topology.primitive_sequence.untyped_storage()._cdata
        == evaluated.topology.interaction_type.untyped_storage()._cdata
    )
    assert (
        evaluated.topology.primitive_sequence.storage_offset()
        != evaluated.topology.interaction_type.storage_offset()
    )


@pytest.mark.parametrize(("rows", "width"), [(0, 0), (0, 4), (3, 0)])
def test_adapter_accepts_empty_rows_and_zero_width(rows: int, width: int):
    batch = _batch(rows=rows, width=width)

    evaluated, _ = evaluated_paths_from_topology_batch(batch)

    assert evaluated.row_count == rows
    assert evaluated.topology.sequence_width == width
    assert evaluated.geometry.interaction_positions.shape == (rows, width, 3)


@pytest.mark.parametrize(
    ("field_name", "bad_value", "error"),
    [
        ("valid", torch.empty((2, 1), dtype=torch.bool), "valid must have rank 1"),
        (
            "interaction_positions",
            torch.empty((3, 1, 3), dtype=torch.float32),
            "interaction_positions must have shape",
        ),
        (
            "field_xyz",
            torch.empty((3, 2), dtype=torch.complex64),
            "field_xyz must have shape",
        ),
    ],
)
def test_adapter_fails_loudly_on_invalid_source_shape(
    field_name: str, bad_value: torch.Tensor, error: str
):
    batch = replace(_batch(), **{field_name: bad_value})

    with pytest.raises(ValueError, match=error):
        evaluated_paths_from_topology_batch(batch)


def test_adapter_preserves_execution_and_optional_sidecars_exactly():
    batch = _batch()

    _, sidecars = evaluated_paths_from_topology_batch(batch)

    assert sidecars.execution == PathExecutionStats(
        launch_count=11,
        visibility_rejection_count=12,
        selected_edge_count=13,
        candidate_count=14,
        guardrail_count=15,
        ad_companion_launches=16,
        ad_tape_bytes=17,
    )
    assert sidecars.diffraction_vector_field is batch.diffraction_vector_field
    _assert_exact_tensor(
        sidecars.diffraction_vector_field, batch.diffraction_vector_field
    )


def test_adapter_sidecars_are_frozen_and_slotted():
    batch = _batch()
    _, sidecars = evaluated_paths_from_topology_batch(batch)

    for contract in (sidecars.execution, sidecars):
        assert not hasattr(contract, "__dict__")
        with pytest.raises(FrozenInstanceError):
            setattr(contract, fields(contract)[0].name, object())
    assert [item.name for item in fields(TopologyBatchSidecars)] == [
        "execution",
        "diffraction_vector_field",
    ]


def test_adapter_calls_no_tensor_allocation_or_transform_api(monkeypatch):
    batch = _batch()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("adapter must not allocate or transform tensors")

    for name in ("clone", "empty", "zeros", "as_tensor", "stack", "cat"):
        monkeypatch.setattr(torch, name, forbidden)
    for name in ("clone", "contiguous", "to", "detach", "cpu", "cuda"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)

    evaluated, sidecars = evaluated_paths_from_topology_batch(batch)

    assert evaluated.topology.valid is batch.valid
    assert sidecars.diffraction_vector_field is batch.diffraction_vector_field
