from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from witwin.channel_native.propagation.geometry import coupled


def test_coupled_query_names_raw_geometry_and_preserves_identity(monkeypatch):
    source = torch.zeros((2, 3))
    receiver = torch.ones((2, 3))
    face_id = torch.tensor([0, 1], dtype=torch.int32)
    face_anchor = torch.zeros((2, 3))
    face_normal = torch.ones((2, 3))
    edge_id = torch.tensor([2, 3], dtype=torch.int32)
    edge_position = torch.zeros((2, 3))
    edge_direction = torch.ones((2, 3))
    edge_t_min = torch.zeros(2)
    edge_t_max = torch.ones(2)
    surface_group_id = torch.tensor([0, 1], dtype=torch.int32)
    surface_group_size = torch.tensor([1, 1], dtype=torch.int32)
    surface_group_members = torch.arange(6, dtype=torch.int32).reshape(3, 2).t()
    raw = {
        "valid": torch.tensor([True, False]),
        "interaction_type_sequence": torch.zeros((2, 2), dtype=torch.int32),
        "primitive_sequence": torch.zeros((2, 2), dtype=torch.int32),
        "edge_sequence": torch.zeros((2, 2), dtype=torch.int32),
        "face_id": face_id,
        "edge_id": edge_id,
        "interaction_positions": torch.zeros((2, 2, 3)),
        "interaction_normals": torch.ones((2, 2, 3)),
        "reflection_position": torch.zeros((2, 3)),
        "reflection_normal": torch.ones((2, 3)),
        "edge_position": edge_position,
        "edge_direction": edge_direction,
        "path_length_m": torch.ones(2),
        "delay_s": torch.ones(2),
    }
    calls = []

    def fake_forward(*args):
        calls.append(args)
        return raw

    monkeypatch.setattr(
        coupled.geometry_bridge,
        "raydn_coupled_rd_geometry_forward",
        fake_forward,
    )
    query = coupled.CoupledGeometryQuery(
        raydn_handle=23,
        source=source,
        receiver=receiver,
        face_id=face_id,
        face_anchor=face_anchor,
        face_normal=face_normal,
        edge_id=edge_id,
        edge_position=edge_position,
        edge_direction=edge_direction,
        edge_t_min=edge_t_min,
        edge_t_max=edge_t_max,
        surface_group_id=surface_group_id,
        surface_group_size=surface_group_size,
        surface_group_members=surface_group_members,
        reverse=True,
    )

    result = coupled.query_coupled_geometry(query)

    assert [field.name for field in fields(query)] == [
        "raydn_handle",
        "source",
        "receiver",
        "face_id",
        "face_anchor",
        "face_normal",
        "edge_id",
        "edge_position",
        "edge_direction",
        "edge_t_min",
        "edge_t_max",
        "surface_group_id",
        "surface_group_size",
        "surface_group_members",
        "reverse",
    ]
    assert [field.name for field in fields(result)] == list(raw)
    assert calls == [
        (
            23,
            source,
            receiver,
            face_id,
            face_anchor,
            face_normal,
            edge_id,
            edge_position,
            edge_direction,
            edge_t_min,
            edge_t_max,
            surface_group_id,
            surface_group_size,
            surface_group_members,
            True,
        )
    ]
    for name, value in raw.items():
        assert getattr(result, name) is value
    assert query.surface_group_members is surface_group_members
    assert query.surface_group_members.stride() == surface_group_members.stride()
    with pytest.raises(FrozenInstanceError):
        result.valid = torch.ones_like(result.valid)
