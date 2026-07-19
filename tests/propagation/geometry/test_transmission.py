from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from witwin.channel_native.propagation.geometry import transmission


def test_closest_hit_query_names_raw_result_without_copies(monkeypatch):
    origin = torch.arange(6, dtype=torch.float32).reshape(3, 2).t()
    direction = torch.ones((2, 3), dtype=torch.float32)
    ray_tmax = torch.tensor([2.0, 3.0])
    active = torch.tensor([True, False])
    raw = {
        "t": torch.tensor([1.0, float("inf")]),
        "p": torch.arange(6, dtype=torch.float32).reshape(3, 2).t(),
        "geo_n": torch.ones((2, 3)),
        "global_prim_id": torch.tensor([4, -1], dtype=torch.int32),
    }
    calls = []

    def fake_forward(handle, ray_o, ray_d, actual_tmax, actual_active, *, flags):
        calls.append((handle, ray_o, ray_d, actual_tmax, actual_active, flags))
        return raw

    monkeypatch.setattr(
        transmission.geometry_bridge,
        "rayd_intersect_forward",
        fake_forward,
    )
    query = transmission.TransmissionClosestHitQuery(
        handle=37,
        origin=origin,
        direction=direction,
        ray_tmax=ray_tmax,
        active=active,
        flags=7,
    )

    result = transmission.query_transmission_closest_hit(query)

    assert [field.name for field in fields(query)] == [
        "handle",
        "origin",
        "direction",
        "ray_tmax",
        "active",
        "flags",
    ]
    assert [field.name for field in fields(result)] == [
        "t",
        "position",
        "geometric_normal",
        "global_primitive_id",
    ]
    assert calls == [(37, origin, direction, ray_tmax, active, 7)]
    assert query.origin is origin
    assert query.origin.stride() == origin.stride()
    assert result.t is raw["t"]
    assert result.position is raw["p"]
    assert result.position.stride() == raw["p"].stride()
    assert result.geometric_normal is raw["geo_n"]
    assert result.global_primitive_id is raw["global_prim_id"]
    with pytest.raises(FrozenInstanceError):
        result.t = torch.zeros_like(result.t)
