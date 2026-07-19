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
        "coupled_rd_geometry_forward",
        fake_forward,
    )
    query = coupled.CoupledGeometryQuery(
        rayd_resource=23,
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
        "rayd_resource",
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


def test_coupled_dd_query_names_raw_geometry_and_preserves_identity(monkeypatch):
    source = torch.zeros((2, 3))
    receiver = torch.ones((2, 3))
    edge1_id = torch.tensor([2, 3], dtype=torch.int32)
    edge1_position = torch.zeros((2, 3))
    edge1_direction = torch.ones((2, 3))
    edge1_t_min = torch.zeros(2)
    edge1_t_max = torch.ones(2)
    edge2_id = torch.tensor([4, 5], dtype=torch.int32)
    edge2_position = torch.ones((2, 3))
    edge2_direction = torch.zeros((2, 3))
    edge2_t_min = torch.zeros(2)
    edge2_t_max = torch.ones(2)
    raw = {
        "valid": torch.tensor([True, False]),
        "interaction_type_sequence": torch.full((2, 2), 2, dtype=torch.int32),
        "primitive_sequence": torch.full((2, 2), -1, dtype=torch.int32),
        "edge_sequence": torch.zeros((2, 2), dtype=torch.int32),
        "edge1_id": edge1_id,
        "edge2_id": edge2_id,
        "interaction_positions": torch.zeros((2, 2, 3)),
        "interaction_normals": torch.full((2, 2, 3), float("nan")),
        "edge1_position": torch.zeros((2, 3)),
        "edge2_position": torch.ones((2, 3)),
        "path_length_m": torch.ones(2),
        "delay_s": torch.ones(2),
    }
    calls = []

    def fake_forward(*args):
        calls.append(args)
        return raw

    monkeypatch.setattr(
        coupled.geometry_bridge,
        "coupled_dd_geometry_forward",
        fake_forward,
    )
    query = coupled.CoupledDdGeometryQuery(
        rayd_resource=42,
        source=source,
        receiver=receiver,
        edge1_id=edge1_id,
        edge1_position=edge1_position,
        edge1_direction=edge1_direction,
        edge1_t_min=edge1_t_min,
        edge1_t_max=edge1_t_max,
        edge2_id=edge2_id,
        edge2_position=edge2_position,
        edge2_direction=edge2_direction,
        edge2_t_min=edge2_t_min,
        edge2_t_max=edge2_t_max,
    )

    result = coupled.query_coupled_dd_geometry(query)

    assert [field.name for field in fields(query)] == [
        "rayd_resource",
        "source",
        "receiver",
        "edge1_id",
        "edge1_position",
        "edge1_direction",
        "edge1_t_min",
        "edge1_t_max",
        "edge2_id",
        "edge2_position",
        "edge2_direction",
        "edge2_t_min",
        "edge2_t_max",
    ]
    assert [field.name for field in fields(result)] == list(raw)
    assert calls == [
        (
            42,
            source,
            receiver,
            edge1_id,
            edge1_position,
            edge1_direction,
            edge1_t_min,
            edge1_t_max,
            edge2_id,
            edge2_position,
            edge2_direction,
            edge2_t_min,
            edge2_t_max,
        )
    ]
    for name, value in raw.items():
        assert getattr(result, name) is value
    with pytest.raises(FrozenInstanceError):
        result.valid = torch.ones_like(result.valid)
