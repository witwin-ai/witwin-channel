# Copyright Xingyu Chen.
# Tests visibility.

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from witwin.channel.propagation import geometry as visibility


class _Rayd:
    available = True

    def require_resource(self) -> int:
        return 17


def test_typed_visibility_query_consumes_raw_native_tuple(monkeypatch):
    start = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    end = start + 1.0
    active = torch.tensor([True, False])
    visible = torch.tensor([False, True])
    calls = []

    def fake_forward(handle, actual_start, actual_end, actual_active):
        calls.append((handle, actual_start, actual_end, actual_active))
        return visible, torch.tensor([99])

    monkeypatch.setattr(
        visibility.geometry_kernels,
        "rayd_visibility_forward",
        fake_forward,
    )
    query = visibility.VisibilityQuery(
        rayd=_Rayd(),
        start=start,
        end=end,
        active=active,
    )

    result = visibility.run_visibility_query(query)

    assert isinstance(result, visibility.VisibilityResult)
    assert [field.name for field in fields(query)] == [
        "rayd",
        "start",
        "end",
        "active",
    ]
    assert [field.name for field in fields(result)] == ["visible"]
    assert result.visible is visible
    assert calls == [(17, start, end, active)]
    with pytest.raises(FrozenInstanceError):
        result.visible = torch.ones_like(visible)


def test_legacy_visibility_helpers_keep_contiguous_and_gating_semantics(monkeypatch):
    start = torch.arange(6, dtype=torch.float32).reshape(3, 2).t()
    end = (start + 1.0).contiguous()
    visible = torch.tensor([True, False])
    calls = []

    def fake_forward(handle, actual_start, actual_end, active):
        calls.append((handle, actual_start, actual_end, active))
        return (visible,)

    monkeypatch.setattr(
        visibility.geometry_kernels,
        "rayd_visibility_forward",
        fake_forward,
    )

    assert visibility._los_visibility_mask(
        _Rayd(),
        start,
        end,
        has_structures=False,
    ) is None
    result = visibility._los_visibility_mask(
        _Rayd(),
        start,
        end,
        has_structures=True,
    )
    assert result is visible
    assert calls[0][0] == 17
    assert calls[0][1].is_contiguous()
    assert calls[0][2].is_contiguous()
    assert calls[0][3] is None
