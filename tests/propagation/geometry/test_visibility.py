# Copyright Xingyu Chen.
# Tests visibility.

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
import hashlib
from pathlib import Path

import pytest
import torch

from witwin.channel.propagation import geometry as visibility


# ``_rayd_visibility_mask`` re-pinned when the geometry kernel facades moved to
# ``witwin.channel.kernels.geometry``: the only AST difference is the module
# alias the native visibility call is spelled through (``geometry_bridge`` ->
# ``geometry_kernels``). Control flow, arguments and evaluation order are
# unchanged.
_DIGESTS = {
    "_rayd_visibility_mask": (
        "169a295d4086227ef85e28ac87d3de79781d4cc52ae6687876f58de621b05f41"
    ),
    "_los_visibility_mask": (
        "95411fd7b1c9d50f50e40d1d2af9e55d171c71c904e4139686148a5602c00878"
    ),
}


class _Rayd:
    available = True

    def require_resource(self) -> int:
        return 17


def _digest(name: str) -> str:
    tree = ast.parse(Path(visibility.__file__).read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return hashlib.sha256(
        ast.dump(definition, include_attributes=False).encode()
    ).hexdigest()


def test_visibility_helpers_preserve_frozen_bodies():
    for name, digest in _DIGESTS.items():
        assert _digest(name) == digest


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