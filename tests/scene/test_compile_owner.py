# Copyright Xingyu Chen.
# Tests compile owner.

import importlib
import inspect

import pytest

from witwin.channel.scene import compile as compile_entry
from witwin.core import Scene
from witwin.core import scene as core_scene

compile_module = importlib.import_module("witwin.channel.scene.compiler")


def test_compile_has_one_channel_runtime_owner():
    assert compile_entry.__module__ == "witwin.channel.scene"
    assert compile_module.compile.__module__ == compile_module.__name__
    assert not hasattr(core_scene, "compile")
    assert not hasattr(Scene, "compile")


def test_importing_internal_compiler_keeps_public_compile_callable():
    scene_package = importlib.import_module("witwin.channel.scene")
    importlib.import_module("witwin.channel.scene.compiler")

    assert scene_package.compile is compile_entry
    assert callable(scene_package.compile)


def test_compile_requires_explicit_reference_frequency():
    signature = inspect.signature(compile_entry)
    parameter = signature.parameters["reference_frequency_hz"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty

    with pytest.raises(TypeError, match="reference_frequency_hz"):
        compile_entry(Scene())


def test_core_scene_has_no_solver_runtime_resource_facades():
    scene = Scene()
    for legacy_name in ("rayd_scene", "frequency", "with_frequency"):
        assert not hasattr(scene, legacy_name)