# Copyright Xingyu Chen.
# Tests native resources.

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from witwin.channel import runtime


def test_native_resource_normalizer_is_a_pure_stdlib_runtime_owner():
    """The normalizer reaches for builtins and the caller's object, nothing else.

    This used to be read off the whole file, back when the normalizer had a
    module to itself. The merged runtime module cannot state it that way, so
    the same claim is now read off the one function it was ever about, which
    is also the tighter reading: an import or a torch call anywhere in the
    body fails here, and a second definition site fails the unpacking.

    The old ``__all__ == ["_rayd_scene_resource"]`` half described a module that
    held one function and cannot survive the merge; the unpacking above carries
    the part of it that mattered, and the name stays private.
    """

    source_path = Path(runtime.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    (owner,) = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_rayd_scene_resource"
    ]

    assert not [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    assert {
        ast.unparse(call.func) for call in ast.walk(owner) if isinstance(call, ast.Call)
    } == {"isinstance", "TypeError", "getattr", "callable", "require"}
    assert runtime._rayd_scene_resource.__module__ == "witwin.channel.runtime"
    assert "_rayd_scene_resource" not in runtime.__all__


def test_native_resource_normalizer_unwraps_typed_wrapper():
    resource = object()
    wrapper = SimpleNamespace(require_resource=lambda: resource)

    assert runtime._rayd_scene_resource(wrapper) is resource
    assert runtime._rayd_scene_resource(resource) is resource


def test_native_resource_normalizer_rejects_integer_handles():
    with pytest.raises(TypeError, match="typed scene resource"):
        runtime._rayd_scene_resource(7)