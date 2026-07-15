from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from witwin.channel_native.propagation.geometry import (
    reflection as geometry_reflection,
)


class _Raydn:
    def require_handle(self) -> int:
        return 23


def test_reflection_epc_query_names_raw_geometry_and_preserves_identity(monkeypatch):
    source = torch.zeros((2, 3))
    receiver = torch.ones((2, 3))
    active = torch.tensor([True, False])
    expected_prim_ids = torch.tensor([[0], [1]], dtype=torch.int32)
    direct_plane_points = torch.zeros((2, 1, 3))
    direct_plane_normals = torch.ones((2, 1, 3))
    surface_group_id = torch.tensor([0, 1], dtype=torch.int32)
    surface_group_size = torch.tensor([1, 1], dtype=torch.int32)
    surface_group_members = torch.arange(6, dtype=torch.int32).reshape(3, 2).t()
    raw = (
        torch.tensor([True, False]),
        torch.tensor([2.0, 3.0]),
        expected_prim_ids,
        torch.tensor([[0], [1]], dtype=torch.int32),
        torch.zeros((2, 1, 3)),
        torch.ones((2, 1, 3)),
    )
    calls = []

    def fake_forward(*args):
        calls.append(args)
        return raw

    monkeypatch.setattr(
        geometry_reflection.geometry_bridge,
        "raydn_reflection_epc_paths_forward",
        fake_forward,
    )
    query = geometry_reflection.ReflectionEpcQuery(
        raydn=_Raydn(),
        source=source,
        receiver=receiver,
        active=active,
        expected_prim_ids=expected_prim_ids,
        direct_plane_points=direct_plane_points,
        direct_plane_normals=direct_plane_normals,
        surface_group_id=surface_group_id,
        surface_group_size=surface_group_size,
        surface_group_members=surface_group_members,
        max_bounces=2,
        visibility_ignore_mode=1,
    )

    result = geometry_reflection.query_reflection_epc(query)

    assert [field.name for field in fields(query)] == [
        "raydn",
        "source",
        "receiver",
        "active",
        "expected_prim_ids",
        "direct_plane_points",
        "direct_plane_normals",
        "surface_group_id",
        "surface_group_size",
        "surface_group_members",
        "max_bounces",
        "visibility_ignore_mode",
    ]
    assert [field.name for field in fields(result)] == [
        "visible",
        "path_length_m",
        "resolved_prim_ids",
        "surface_group_ids",
        "hit_positions",
        "normals",
    ]
    assert calls == [
        (
            23,
            source,
            receiver,
            active,
            expected_prim_ids,
            direct_plane_points,
            direct_plane_normals,
            surface_group_id,
            surface_group_size,
            surface_group_members,
            2,
            1,
        )
    ]
    assert result.visible is raw[0]
    assert result.path_length_m is raw[1]
    assert result.resolved_prim_ids is raw[2]
    assert result.surface_group_ids is raw[3]
    assert result.hit_positions is raw[4]
    assert result.normals is raw[5]
    assert query.surface_group_members is surface_group_members
    assert query.surface_group_members.stride() == surface_group_members.stride()
    with pytest.raises(FrozenInstanceError):
        result.visible = torch.ones_like(result.visible)
