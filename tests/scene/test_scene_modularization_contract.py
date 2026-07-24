import importlib.util
import importlib

import witwin.channel as channel
import witwin.channel.scene as channel_scene
import witwin.core as core
from witwin.channel.scene.compiled import CompiledScene


def test_public_logical_world_contracts_are_core_owned():
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
        assert getattr(channel, name) is getattr(core, name)
        assert getattr(channel, name).__module__.startswith("witwin.core")


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
    for module_name in (
        "witwin.channel.core.scene",
        "witwin.channel.core.scene_loader",
        "witwin.channel.core.materials",
        "witwin.channel.core.material_runtime",
        "witwin.channel.core.runtime.compiled_scene",
        "witwin.channel.core.runtime.geometry",
        "witwin.channel.core.runtime.material_store",
        "witwin.channel.core.runtime.assignments",
    ):
        assert importlib.util.find_spec(module_name) is None


def test_deleted_public_logical_facades_are_not_reintroduced():
    for legacy_name in (
        "Dielectric",
        "PerfectConductor",
        "Transmitter",
        "ReceiverPoint",
        "SurfaceAssignment",
    ):
        assert not hasattr(channel, legacy_name)
