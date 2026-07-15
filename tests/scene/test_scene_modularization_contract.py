from __future__ import annotations

import gc
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass, replace
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import get_args, get_origin, get_type_hints
import weakref

import pytest
import torch

import witwin.channel_native as public
from tests.support.scenes import empty_space_los_scene
from witwin.channel_native import materials as public_materials
from witwin.channel_native import scattering
from witwin.channel_native.core import materials as legacy_materials
from witwin.channel_native.core import objects as legacy_objects
from witwin.channel_native.core import scene as legacy_scene
from witwin.channel_native.core.runtime import assignments as legacy_assignments
from witwin.channel_native.core.runtime import compiled_scene as legacy_compiled
from witwin.channel_native.core.runtime import geometry as legacy_geometry
from witwin.channel_native.core.runtime import material_store as legacy_material_store
from witwin.channel_native.core.runtime import raydn as legacy_raydn
from witwin.channel_native.deterministic.result import Result as DeterministicResult
from witwin.channel_native.montecarlo.basic.result import Result as BasicResult
from witwin.channel_native.montecarlo.bdpt.result import Result as BDPTResult
from witwin.channel_native.path.result import BeamformedPathResult, PathResult
from witwin.channel_native.propagation.models.evaluated import EvaluatedPaths
from witwin.channel_native.propagation.models.fields import PathFields
from witwin.channel_native.propagation.models.geometry import PathGeometry
from witwin.channel_native.propagation.models.topology import PathTopology
from witwin.channel_native.scene import compiled as canonical_compiled


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


_LEGACY_CLASS_CASES = (
    ("witwin.channel_native.core.scene", "Scene", public.Scene),
    ("witwin.channel_native.core.objects", "Structure", public.Structure),
    ("witwin.channel_native.core.objects", "Transmitter", public.Transmitter),
    ("witwin.channel_native.core.objects", "ReceiverPoint", public.ReceiverPoint),
    ("witwin.channel_native.core.objects", "ReceiverGrid", public.ReceiverGrid),
    (
        "witwin.channel_native.core.runtime.compiled_scene",
        "CompiledScene",
        legacy_compiled.CompiledScene,
    ),
    (
        "witwin.channel_native.core.runtime.geometry",
        "GeometryStore",
        legacy_geometry.GeometryStore,
    ),
    (
        "witwin.channel_native.core.runtime.material_store",
        "MaterialStore",
        legacy_material_store.MaterialStore,
    ),
    (
        "witwin.channel_native.core.runtime.assignments",
        "AssignmentStore",
        legacy_assignments.AssignmentStore,
    ),
    (
        "witwin.channel_native.core.runtime.raydn",
        "RayDNScene",
        legacy_raydn.RayDNScene,
    ),
)

_FORBIDDEN_RESOURCE_FIELD_NAMES = {
    "cache",
    "compiled_scene",
    "native_handle",
    "runtime_cache",
    "runtime_handle",
    "scene",
    "workspace",
}
_RESOURCE_TYPES = (
    legacy_scene.Scene,
    legacy_compiled.CompiledScene,
    legacy_raydn.RayDNScene,
)


def _pickle_global(module: str, name: str) -> bytes:
    return f"c{module}\n{name}\n.".encode()


def _triangle_scene() -> public.Scene:
    return public.Scene(
        structures=[
            public.Structure(
                vertices=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=torch.float32,
                ),
                faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
                material=public.Dielectric(eps_r=2.0),
                name="wall",
            )
        ],
        transmitters=[public.Transmitter(position=torch.tensor([0.0, 0.0, 1.0]))],
        receivers=[public.ReceiverPoint(position=torch.tensor([1.0, 0.0, 1.0]))],
        frequency=3.5e9,
    )


def _annotation_types(annotation: object):
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            yield annotation
        return
    for argument in get_args(annotation):
        yield from _annotation_types(argument)


def _assert_dataclass_schema_resource_free(
    contract_type: type[object], seen: set[type[object]]
) -> None:
    if contract_type in seen:
        return
    seen.add(contract_type)
    hints = get_type_hints(contract_type)
    for item in fields(contract_type):
        assert item.name not in _FORBIDDEN_RESOURCE_FIELD_NAMES
        for nested_type in _annotation_types(hints[item.name]):
            assert not issubclass(nested_type, _RESOURCE_TYPES)
            if is_dataclass(nested_type):
                _assert_dataclass_schema_resource_free(nested_type, seen)


def _assert_object_graph_resource_free(value: object, seen: set[int]) -> None:
    if value is None or isinstance(
        value,
        (bool, int, float, complex, str, bytes, torch.Tensor, torch.device),
    ):
        return
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    assert not isinstance(value, _RESOURCE_TYPES)
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            assert item.name not in _FORBIDDEN_RESOURCE_FIELD_NAMES
            _assert_object_graph_resource_free(getattr(value, item.name), seen)
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str):
                assert key not in _FORBIDDEN_RESOURCE_FIELD_NAMES
            _assert_object_graph_resource_free(nested, seen)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for nested in value:
            _assert_object_graph_resource_free(nested, seen)


def test_public_and_legacy_scene_class_identity_and_pickle_replay():
    assert legacy_objects.Structure is public.Structure is legacy_scene.Structure
    assert legacy_objects.Transmitter is public.Transmitter is legacy_scene.Transmitter
    assert (
        legacy_objects.ReceiverPoint
        is public.ReceiverPoint
        is legacy_scene.ReceiverPoint
    )
    assert legacy_objects.ReceiverGrid is public.ReceiverGrid is legacy_scene.ReceiverGrid
    assert (
        canonical_compiled.CompiledScene
        is legacy_scene.CompiledScene
        is legacy_compiled.CompiledScene
    )
    assert canonical_compiled.CompiledScene.__module__ == (
        "witwin.channel_native.core.runtime.compiled_scene"
    )
    assert legacy_scene.GeometryStore is legacy_geometry.GeometryStore
    assert legacy_scene.MaterialStore is legacy_material_store.MaterialStore
    assert legacy_scene.AssignmentStore is legacy_assignments.AssignmentStore

    for module, name, owner in _LEGACY_CLASS_CASES:
        assert pickle.loads(_pickle_global(module, name)) is owner
        assert pickle.loads(pickle.dumps(owner)) is owner


def test_compiled_scene_dataclass_schema_and_type_hints_are_exact():
    owner = canonical_compiled.CompiledScene
    schema = fields(owner)

    assert tuple(item.name for item in schema) == (
        "geometry",
        "materials",
        "assignments",
        "raydn",
        "workspace",
        "geometry_version",
        "material_version",
        "assignment_version",
        "_kirchhoff_tables_cache",
        "_phase_screen_runtimes_cache",
    )
    assert all(item.default is MISSING for item in schema[:8])
    assert all(item.default_factory is MISSING for item in schema)
    assert tuple(item.default for item in schema[8:]) == (None, None)
    assert all(not item.repr and not item.compare for item in schema[8:])
    assert get_type_hints(owner) == {
        "geometry": legacy_geometry.GeometryStore,
        "materials": legacy_material_store.MaterialStore,
        "assignments": legacy_assignments.AssignmentStore,
        "raydn": legacy_raydn.RayDNScene,
        "workspace": object | None,
        "geometry_version": int,
        "material_version": int,
        "assignment_version": int,
        "_kirchhoff_tables_cache": dict[int, object] | None,
        "_phase_screen_runtimes_cache": dict[int, object] | None,
    }


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.scene import compiled as canonical; "
            "from witwin.channel_native.core.runtime import compiled_scene as legacy; "
            "from witwin.channel_native.core import scene as core_scene"
        ),
        (
            "from witwin.channel_native.core.runtime import compiled_scene as legacy; "
            "from witwin.channel_native.core import scene as core_scene; "
            "from witwin.channel_native.scene import compiled as canonical"
        ),
        (
            "from witwin.channel_native.core import scene as core_scene; "
            "from witwin.channel_native.scene import compiled as canonical; "
            "from witwin.channel_native.core.runtime import compiled_scene as legacy"
        ),
    ),
)
def test_compiled_scene_fresh_import_order_and_legacy_pickle_replay(imports: str):
    source = (
        f"{imports}; import pickle; "
        "owner = canonical.CompiledScene; "
        "assert owner is legacy.CompiledScene is core_scene.CompiledScene; "
        "assert owner.__module__ == "
        "'witwin.channel_native.core.runtime.compiled_scene'; "
        "assert pickle.loads("
        "b'cwitwin.channel_native.core.runtime.compiled_scene\\nCompiledScene\\n.'"
        ") is owner; "
        "assert pickle.loads(pickle.dumps(owner)) is owner"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(REPOSITORY_ROOT / "src"),
            environment.get("PYTHONPATH"),
        )
        if value
    )

    subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )


def test_public_material_identity_and_legacy_pickle_replay():
    for name in public_materials.__all__:
        owner = getattr(legacy_materials, name)
        assert getattr(public_materials, name) is owner
        assert pickle.loads(
            _pickle_global("witwin.channel_native.core.materials", name)
        ) is owner
        assert pickle.loads(pickle.dumps(owner)) is owner


def test_compile_cache_invalidation_and_material_storage_are_exact():
    scene = _triangle_scene()
    compiled = scene.compile()

    assert scene.compile() is compiled
    assert scene.raydn_scene() is compiled.raydn
    assert compiled.materials.layer_offset.tolist() == [0]
    assert compiled.materials.layer_count.tolist() == [1]
    for item in fields(compiled.materials):
        value = getattr(compiled.materials, item.name)
        if isinstance(value, torch.Tensor):
            repeated = getattr(scene.compile().materials, item.name)
            assert repeated is value
            assert repeated.untyped_storage().data_ptr() == (
                value.untyped_storage().data_ptr()
            )
    assert compiled.materials.eps_r.untyped_storage().data_ptr() != (
        compiled.materials.layer_eps_r.untyped_storage().data_ptr()
    )

    geometry = scene.with_structure_vertices(
        0,
        torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
    ).compile()
    material = scene.with_structure_material(0, public.Dielectric(eps_r=3.0)).compile()
    frequency = scene.with_frequency(5.0e9).compile()
    screen = public_materials.PhaseScreen(
        height=[[0.0, 1.0], [1.0, 0.0]], height_scale_m=1.0e-3
    )
    assignment = scene.with_structure_material(
        0,
        public_materials.SurfaceAssignment(
            material=scene.structures[0].material,
            phase_screen=screen,
        ),
    ).compile()

    assert geometry is not compiled
    assert geometry.geometry_version == compiled.geometry_version + 1
    assert geometry.material_version == compiled.material_version
    assert geometry.assignment_version == compiled.assignment_version
    assert material is not compiled
    assert material.material_version == compiled.material_version + 1
    assert material.geometry_version == compiled.geometry_version
    assert material.assignment_version == compiled.assignment_version
    assert frequency is not compiled
    assert frequency.material_version == compiled.material_version + 1
    assert frequency.geometry_version == compiled.geometry_version
    assert frequency.materials.cache_token != compiled.materials.cache_token
    assert assignment is not compiled
    assert assignment.material_version == compiled.material_version + 1
    assert assignment.assignment_version == compiled.assignment_version
    assert assignment.materials.cache_token != compiled.materials.cache_token
    assert assignment.assignments.structure_phase_screens == {0: screen}


def test_compiled_scattering_resources_are_built_once_on_first_access(monkeypatch):
    base = public.Scene(
        structures=[], transmitters=[], receivers=[], frequency=3.5e9
    ).compile()
    materials = replace(
        base.materials,
        scatter_model_id=torch.ones_like(base.materials.scatter_model_id),
        rough_sigma_h_m=torch.full_like(base.materials.rough_sigma_h_m, 1.0e-3),
        rough_corr_x_m=torch.full_like(base.materials.rough_corr_x_m, 0.1),
        rough_corr_y_m=torch.full_like(base.materials.rough_corr_y_m, 0.2),
        cache_token="rough-material",
    )
    screen = public_materials.PhaseScreen(
        height=[[0.0, 1.0], [1.0, 0.0]], height_scale_m=1.0e-3
    )
    assignments = replace(
        base.assignments,
        surface_material_id=torch.tensor([0], dtype=torch.int32),
        structure_material_id=torch.tensor([0], dtype=torch.int32),
        structure_phase_screens={0: screen},
    )
    compiled = replace(base, materials=materials, assignments=assignments)
    table_resource = object()
    screen_resource = object()
    events: list[str] = []

    def build_table(*args, **kwargs):
        del args, kwargs
        events.append("table")
        return table_resource

    def build_screen(*args, **kwargs):
        del args, kwargs
        events.append("screen")
        return screen_resource

    monkeypatch.setattr(scattering, "build_kirchhoff_table", build_table)
    monkeypatch.setattr(scattering, "PhaseScreenRuntime", build_screen)

    assert events == []
    tables = compiled.kirchhoff_tables
    assert tables == {0: table_resource}
    assert compiled.kirchhoff_tables is tables
    assert events == ["table"]
    runtimes = compiled.phase_screen_runtimes
    assert runtimes == {0: screen_resource}
    assert compiled.phase_screen_runtimes is runtimes
    assert events == ["table", "screen"]


@pytest.mark.parametrize(
    "contract_type",
    (
        PathTopology,
        PathGeometry,
        PathFields,
        EvaluatedPaths,
        PathResult,
        BeamformedPathResult,
        DeterministicResult,
        BasicResult,
        BDPTResult,
    ),
)
def test_propagation_and_public_result_schemas_are_recursively_resource_free(
    contract_type: type[object],
):
    _assert_dataclass_schema_resource_free(contract_type, set())


class _FakeNativeOwner:
    pass


def test_solver_results_do_not_retain_compiled_scene_or_native_owner():
    from witwin.channel_native.deterministic import Config as DeterministicConfig
    from witwin.channel_native.deterministic import solve as solve_deterministic
    from witwin.channel_native.montecarlo.basic import Config as BasicConfig
    from witwin.channel_native.montecarlo.basic import solve as solve_basic
    from witwin.channel_native.montecarlo.bdpt import Config as BDPTConfig
    from witwin.channel_native.montecarlo.bdpt import solve as solve_bdpt
    from witwin.channel_native.path import Config as PathConfig
    from witwin.channel_native.path import solve as solve_path

    assert torch.cuda.is_available()
    scene = empty_space_los_scene()
    results = [
        solve_path(scene, PathConfig(max_depth=0, components={"los"})),
        solve_deterministic(
            scene,
            DeterministicConfig(
                max_depth=0,
                max_diffraction_order=0,
                components={"los"},
            ),
        ),
        solve_basic(
            scene,
            BasicConfig(samples=4, max_depth=0, seed=7, components={"los"}),
        ),
        solve_bdpt(
            scene,
            BDPTConfig(
                samples=4,
                max_depth=0,
                max_diffraction_order=0,
                seed=7,
                components={"los"},
            ),
        ),
    ]
    for result in results:
        _assert_object_graph_resource_free(result, set())

    compiled = scene.compile()
    owner = _FakeNativeOwner()
    owner_ref = weakref.ref(owner)
    compiled.raydn = legacy_raydn.RayDNScene(handle=17, owner=owner)
    del owner
    gc.collect()
    assert owner_ref() is not None

    released_result = results.pop()
    del released_result
    gc.collect()
    assert owner_ref() is not None

    del scene
    del compiled
    gc.collect()
    assert owner_ref() is None
    for result in results:
        _assert_object_graph_resource_free(result, set())
