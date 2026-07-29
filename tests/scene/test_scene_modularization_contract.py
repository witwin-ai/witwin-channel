# Copyright Xingyu Chen.
# Tests scene modularization contract.

import importlib.util

import pytest

import witwin.channel as channel
import witwin.channel.scene as channel_scene
import witwin.core as core
from witwin.channel.scene.compiler import CompiledScene


def test_public_logical_world_contracts_are_core_owned():
    """Core owns the world model, and Channel does not republish it."""

    for name in (
        "AntennaPattern",
        "AntennaState",
        "MaterialLayer",
        "PhaseScreen",
        "PhysicalMaterial",
        "ReceiverGrid",
        "Scene",
        "SceneSnapshot",
        "Structure",
        "SurfaceRoughness",
    ):
        assert getattr(core, name).__module__.startswith("witwin.core")
        assert not hasattr(channel, name), name
        assert name not in channel.__all__, name


def test_channel_root_exports_only_channel_owned_names():
    assert sorted(channel.__all__) == [
        "Complex3State",
        "JonesState",
        "build_info",
        "capabilities",
        "pipeline_cache_key",
        "runtime_diagnostics",
    ]


def test_channel_scene_package_owns_only_compile_runtime_contracts():
    compile_module = importlib.import_module("witwin.channel.scene.compiler")
    assert callable(compile_module.compile)
    assert channel_scene.CompiledScene is CompiledScene
    for legacy_name in (
        "Scene",
        "Transmitter",
        "ReceiverPoint",
        "ReceiverGrid",
        "GeometryStore",
        "MaterialStore",
        "AssignmentStore",
        "SurfaceAssignment",
    ):
        assert legacy_name not in channel_scene.__all__


def test_core_scene_has_no_channel_compile_or_frequency_facades():
    for legacy_name in (
        "compile",
        "rayd_scene",
        "with_frequency",
        "frequency",
        "transmitters",
        "receivers",
        "add",
    ):
        assert not hasattr(core.Scene, legacy_name)


def test_deleted_legacy_owner_modules_do_not_resolve():
    """The whole ``witwin.channel.core`` namespace is gone, not just its leaves.

    ``find_spec`` on a child of a missing package raises rather than returning
    ``None``, so assert the parent is absent and that every historical leaf is
    unreachable through it.
    """

    assert importlib.util.find_spec("witwin.channel.core") is None
    for module_name in (
        "scene",
        "scene_loader",
        "materials",
        "material_runtime",
        "runtime.compiled_scene",
        "runtime.geometry",
        "runtime.material_store",
        "runtime.assignments",
        "kernels.extension",
        "kernels.ops",
        "path_topology",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.util.find_spec(f"witwin.channel.core.{module_name}")


def test_deleted_public_logical_facades_are_not_reintroduced():
    for legacy_name in (
        "Dielectric",
        "PerfectConductor",
        "Transmitter",
        "ReceiverPoint",
        "SurfaceAssignment",
    ):
        assert not hasattr(channel, legacy_name)