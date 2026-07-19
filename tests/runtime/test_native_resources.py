from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from witwin.channel_native.runtime import native_resources


def test_native_resource_normalizer_is_a_pure_stdlib_runtime_owner():
    source_path = Path(native_resources.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
    definitions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    assert imports == ["__future__"]
    assert not any(isinstance(node, ast.Import) for node in tree.body)
    assert definitions == {"_rayd_scene_resource"}
    assert native_resources.__all__ == ["_rayd_scene_resource"]


def test_native_resource_normalizer_unwraps_typed_wrapper():
    resource = object()
    wrapper = SimpleNamespace(require_resource=lambda: resource)

    assert native_resources._rayd_scene_resource(wrapper) is resource
    assert native_resources._rayd_scene_resource(resource) is resource


def test_native_resource_normalizer_rejects_integer_handles():
    with pytest.raises(TypeError, match="typed scene resource"):
        native_resources._rayd_scene_resource(7)
