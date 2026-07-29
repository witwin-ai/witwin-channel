# Copyright Xingyu Chen.
# Tests contracts.

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from witwin.channel.propagation import (
    EvaluatedPaths,
    PathFields,
    PathGeometry,
    PathTopology,
)


def _topology_inputs(rows: int = 3, width: int = 2) -> dict[str, torch.Tensor]:
    ids = torch.arange(rows * 2, dtype=torch.int32)[::2]
    sequence = torch.arange(rows * width * 2, dtype=torch.int32).reshape(
        rows, width * 2
    )[:, ::2]
    return {
        "valid": torch.ones(rows, dtype=torch.bool),
        "tx_id": ids,
        "rx_id": ids,
        "depth": ids,
        "component_id": ids,
        "primitive_id": ids,
        "edge_id": ids,
        "material_id": ids,
        "primitive_sequence": sequence,
        "material_sequence": sequence,
        "interaction_type": sequence,
    }


def _geometry_inputs(rows: int = 3, width: int = 2) -> dict[str, torch.Tensor]:
    scalar = torch.arange(rows * 2, dtype=torch.float32)[::2].requires_grad_()
    vector = (
        torch.arange(rows * 6, dtype=torch.float32)
        .reshape(rows, 6)[:, ::2]
        .requires_grad_()
    )
    sequence = (
        torch.arange(rows * width * 6, dtype=torch.float32)
        .reshape(rows, width, 6)[:, :, ::2]
        .requires_grad_()
    )
    return {
        "path_length_m": scalar,
        "delay_s": scalar,
        "field_direction": vector,
        "interaction_position": vector,
        "interaction_normal": vector,
        "interaction_positions": sequence,
        "interaction_normals": sequence,
    }


def _field_inputs(rows: int = 3) -> dict[str, torch.Tensor]:
    gain = torch.arange(rows * 2, dtype=torch.float32)[::2].requires_grad_()
    scalar = torch.complex(gain, gain)
    vector_base = (
        torch.arange(rows * 6, dtype=torch.float32).reshape(rows, 6).requires_grad_()
    )
    vector = torch.complex(vector_base, vector_base)[:, ::2]
    return {
        "path_gain": gain,
        "path_field": scalar,
        "field_xyz": vector,
        "coefficient": scalar,
    }


def _contracts(
    rows: int = 3, width: int = 2
) -> tuple[PathTopology, PathGeometry, PathFields, EvaluatedPaths]:
    topology = PathTopology(**_topology_inputs(rows, width))
    geometry = PathGeometry(
        row_identity=topology.row_identity, **_geometry_inputs(rows, width)
    )
    path_fields = PathFields(row_identity=topology.row_identity, **_field_inputs(rows))
    return (
        topology,
        geometry,
        path_fields,
        EvaluatedPaths(topology=topology, geometry=geometry, fields=path_fields),
    )


def test_contracts_are_frozen_slotted_and_resource_free():
    topology, geometry, path_fields, evaluated = _contracts()
    forbidden = {
        "scene",
        "compiled_scene",
        "native_handle",
        "runtime_handle",
        "cache",
        "runtime_cache",
        "workspace",
    }

    for contract in (topology, geometry, path_fields, evaluated):
        assert not hasattr(contract, "__dict__")
        contract_fields = fields(contract)
        assert not (forbidden & {item.name for item in contract_fields})
        with pytest.raises(FrozenInstanceError):
            setattr(contract, contract_fields[0].name, object())


def test_construction_is_zero_copy_and_preserves_tensor_metadata():
    topology_inputs = _topology_inputs()
    topology = PathTopology(**topology_inputs)
    geometry_inputs = _geometry_inputs()
    geometry = PathGeometry(row_identity=topology.row_identity, **geometry_inputs)
    field_inputs = _field_inputs()
    path_fields = PathFields(row_identity=topology.row_identity, **field_inputs)

    for contract, inputs in (
        (topology, topology_inputs),
        (geometry, geometry_inputs),
        (path_fields, field_inputs),
    ):
        for name, tensor in inputs.items():
            observed = getattr(contract, name)
            assert observed is tensor
            assert observed.data_ptr() == tensor.data_ptr()
            assert observed.stride() == tensor.stride()
            assert observed.requires_grad == tensor.requires_grad


def test_evaluated_paths_requires_exact_shared_row_identity():
    first_topology, first_geometry, _, _ = _contracts()
    second_topology, _, second_fields, _ = _contracts()

    with pytest.raises(ValueError, match="fields must share topology row_identity"):
        EvaluatedPaths(
            topology=first_topology,
            geometry=first_geometry,
            fields=second_fields,
        )
    assert first_topology.row_identity is not second_topology.row_identity


def test_row_count_device_and_empty_rows_are_consistent():
    topology, geometry, path_fields, evaluated = _contracts(rows=0, width=0)

    assert topology.row_count == geometry.row_count == path_fields.row_count == 0
    assert evaluated.row_count == 0
    assert topology.sequence_width == 0
    assert topology.device == geometry.device == path_fields.device == evaluated.device


@pytest.mark.parametrize(
    ("contract", "error"),
    [
        ("topology_dtype", "tx_id must use torch.int32"),
        ("geometry_shape", "path_length_m must have shape"),
        ("geometry_width", "interaction_positions must have shape"),
        ("fields_dtype", "field_xyz must use torch.complex64"),
        ("fields_device", "field_xyz must be on cpu"),
    ],
)
def test_contracts_reject_misaligned_tensor_metadata(contract: str, error: str):
    topology_inputs = _topology_inputs()
    if contract == "topology_dtype":
        topology_inputs["tx_id"] = topology_inputs["tx_id"].to(torch.int64)
        with pytest.raises(ValueError, match=error):
            PathTopology(**topology_inputs)
        return

    topology = PathTopology(**topology_inputs)
    if contract.startswith("geometry"):
        geometry_inputs = _geometry_inputs()
        if contract == "geometry_shape":
            geometry_inputs["path_length_m"] = torch.empty(2, dtype=torch.float32)
        else:
            geometry_inputs["interaction_positions"] = torch.empty(
                (3, 1, 3), dtype=torch.float32
            )
        with pytest.raises(ValueError, match=error):
            PathGeometry(row_identity=topology.row_identity, **geometry_inputs)
        return

    field_inputs = _field_inputs()
    if contract == "fields_dtype":
        field_inputs["field_xyz"] = field_inputs["field_xyz"].to(torch.complex128)
    else:
        field_inputs["field_xyz"] = torch.empty(
            (3, 3), dtype=torch.complex64, device="meta"
        )
    with pytest.raises(ValueError, match=error):
        PathFields(row_identity=topology.row_identity, **field_inputs)