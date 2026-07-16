from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from witwin.channel_native.runtime import native_handles


def test_native_handle_normalizer_is_a_pure_stdlib_runtime_owner():
    source_path = Path(native_handles.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = [
        node.module for node in tree.body if isinstance(node, ast.ImportFrom)
    ]
    definitions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    assert imports == ["__future__"]
    assert not any(isinstance(node, ast.Import) for node in tree.body)
    assert definitions == {"_raydn_scene_handle_id"}
    assert native_handles._raydn_scene_handle_id.__module__ == native_handles.__name__
    assert native_handles.__all__ == ["_raydn_scene_handle_id"]


@pytest.mark.parametrize(
    ("handle", "expected"),
    (
        (7, 7),
        (SimpleNamespace(handle=11), 11),
        (SimpleNamespace(handle=lambda: 13), 13),
    ),
)
def test_native_handle_normalizer_preserves_supported_forms(handle, expected):
    assert native_handles._raydn_scene_handle_id(handle) == expected


def test_native_handle_normalizer_rejects_unsupported_forms():
    with pytest.raises(TypeError, match="must be an int or expose handle"):
        native_handles._raydn_scene_handle_id(SimpleNamespace(handle="invalid"))
