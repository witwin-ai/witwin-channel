from __future__ import annotations

import torch

from witwin.channel_native.propagation.geometry.endpoints import ReceiverLayout
from witwin.channel_native.deterministic import accumulation
from witwin.channel_native.propagation.models import (
    EvaluatedPaths,
    PathFields,
    PathGeometry,
    PathTopology,
)


def _evaluated_paths() -> EvaluatedPaths:
    topology = PathTopology(
        valid=torch.ones(1, dtype=torch.bool),
        tx_id=torch.zeros(1, dtype=torch.int32),
        rx_id=torch.zeros(1, dtype=torch.int32),
        depth=torch.zeros(1, dtype=torch.int32),
        component_id=torch.zeros(1, dtype=torch.int32),
        primitive_id=torch.full((1,), -1, dtype=torch.int32),
        edge_id=torch.full((1,), -1, dtype=torch.int32),
        material_id=torch.full((1,), -1, dtype=torch.int32),
        primitive_sequence=torch.empty((1, 0), dtype=torch.int32),
        material_sequence=torch.empty((1, 0), dtype=torch.int32),
        interaction_type=torch.empty((1, 0), dtype=torch.int32),
    )
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        path_length_m=torch.ones(1, dtype=torch.float32),
        delay_s=torch.ones(1, dtype=torch.float32),
        field_direction=torch.ones((1, 3), dtype=torch.float32),
        interaction_position=torch.zeros((1, 3), dtype=torch.float32),
        interaction_normal=torch.zeros((1, 3), dtype=torch.float32),
        interaction_positions=torch.empty((1, 0, 3), dtype=torch.float32),
        interaction_normals=torch.empty((1, 0, 3), dtype=torch.float32),
    )
    fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=torch.ones(1, dtype=torch.float32),
        path_field=torch.ones(1, dtype=torch.complex64),
        field_xyz=torch.ones((1, 3), dtype=torch.complex64),
        coefficient=torch.ones(1, dtype=torch.complex64),
    )
    return EvaluatedPaths(topology=topology, geometry=geometry, fields=fields)


def test_accumulation_consumes_split_evaluated_path_contract(monkeypatch):
    paths = _evaluated_paths()
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_accumulate(**kwargs):
        captured.update(kwargs)
        scalar = torch.ones(1)
        complex_scalar = torch.ones(1, dtype=torch.complex64)
        return scalar, complex_scalar, {"los": scalar}, {"los": complex_scalar}

    monkeypatch.setattr(accumulation, "accumulate_flat_components", fake_accumulate)
    monkeypatch.setattr(
        accumulation,
        "apply_layout_to_accumulation",
        lambda **_kwargs: sentinel,
    )

    result = accumulation.accumulate_path_result(
        paths,
        frequency_hz=1.0,
        num_tx=1,
        num_rx=1,
        layout=ReceiverLayout(kind="point", receiver_count=1),
        coherent=True,
        return_field=True,
    )

    assert result is sentinel
    assert captured["tx_id"] is paths.topology.tx_id
    assert captured["rx_id"] is paths.topology.rx_id
    assert captured["component_id"] is paths.topology.component_id
    assert captured["path_gain"] is paths.fields.path_gain
    assert captured["path_field"] is paths.fields.path_field
