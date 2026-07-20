from __future__ import annotations

import gc
from types import SimpleNamespace
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
from witwin.channel_native.scene.kernels import rayd_scene as legacy_rayd
from witwin.channel_native.deterministic.result import Result as DeterministicResult
from witwin.channel_native.montecarlo.basic.result import Result as BasicResult
from witwin.channel_native.montecarlo.bdpt.result import Result as BDPTResult
from witwin.channel_native.path.result import BeamformedPathResult, PathResult
from witwin.channel_native.propagation.models.evaluated import EvaluatedPaths
from witwin.channel_native.propagation.models.fields import PathFields
from witwin.channel_native.propagation.models.geometry import PathGeometry
from witwin.channel_native.propagation.models.topology import PathTopology
from witwin.channel_native.scene import models as canonical_models
from witwin.channel_native.scene import compiled as canonical_compiled
from witwin.channel_native.scene import scattering_resources
from witwin.channel_native.scene.kernels import rayd_scene as canonical_rayd
from witwin.channel_native.scene.stores import _validation as canonical_validation
from witwin.channel_native.scene.stores import assignments as canonical_assignments
from witwin.channel_native.scene.stores import geometry as canonical_geometry
from witwin.channel_native.scene.stores import materials as canonical_material_stores


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
        canonical_geometry.GeometryStore,
    ),
    (
        "witwin.channel_native.core.runtime.material_store",
        "MaterialStore",
        canonical_material_stores.MaterialStore,
    ),
    (
        "witwin.channel_native.core.runtime.assignments",
        "AssignmentStore",
        canonical_assignments.AssignmentStore,
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
    legacy_rayd.RayDSceneResource,
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


def _triangle_uv_scene() -> public.Scene:
    scene = _triangle_scene()
    structure = scene.structures[0]
    return public.Scene(
        structures=[
            replace(
                structure,
                uv=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                face_uv=structure.faces.clone(),
            )
        ],
        transmitters=scene.transmitters,
        receivers=scene.receivers,
        frequency=scene.frequency,
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
    assert canonical_models.Scene is legacy_scene.Scene is public.Scene
    assert canonical_models.Scene.__module__ == "witwin.channel_native.core.scene"
    assert canonical_models.Scene.compile.__globals__ is vars(canonical_models)
    assert "__all__" not in vars(legacy_scene)
    assert (
        canonical_models.Structure
        is legacy_objects.Structure
        is public.Structure
        is legacy_scene.Structure
    )
    assert (
        canonical_models.Transmitter
        is legacy_objects.Transmitter
        is public.Transmitter
        is legacy_scene.Transmitter
    )
    assert (
        canonical_models.ReceiverPoint
        is legacy_objects.ReceiverPoint
        is public.ReceiverPoint
        is legacy_scene.ReceiverPoint
    )
    assert (
        canonical_models.ReceiverGrid
        is legacy_objects.ReceiverGrid
        is public.ReceiverGrid
        is legacy_scene.ReceiverGrid
    )
    assert canonical_models.Material is legacy_objects.Material
    assert canonical_models.planar_uv is legacy_objects.planar_uv
    assert "__all__" not in vars(legacy_objects)
    assert (
        canonical_compiled.CompiledScene
        is legacy_scene.CompiledScene
        is legacy_compiled.CompiledScene
    )
    assert canonical_compiled.CompiledScene.__module__ == (
        "witwin.channel_native.core.runtime.compiled_scene"
    )
    assert (
        canonical_geometry.GeometryStore
        is legacy_scene.GeometryStore
        is legacy_geometry.GeometryStore
    )
    assert canonical_geometry.GeometryStore.__module__ == (
        "witwin.channel_native.core.runtime.geometry"
    )
    assert (
        canonical_material_stores.MaterialStore
        is legacy_scene.MaterialStore
        is legacy_material_store.MaterialStore
        is canonical_compiled.MaterialStore
    )
    assert canonical_material_stores.MaterialStore.__module__ == (
        "witwin.channel_native.core.runtime.material_store"
    )
    assert (
        canonical_assignments.AssignmentStore
        is legacy_scene.AssignmentStore
        is legacy_assignments.AssignmentStore
        is canonical_compiled.AssignmentStore
    )
    assert canonical_assignments.AssignmentStore.__module__ == (
        "witwin.channel_native.core.runtime.assignments"
    )

    for module, name, owner in _LEGACY_CLASS_CASES:
        assert pickle.loads(_pickle_global(module, name)) is owner
        assert pickle.loads(pickle.dumps(owner)) is owner


def test_geometry_store_schema_type_hints_and_validation_owner_are_exact():
    owner = canonical_geometry.GeometryStore

    assert tuple(item.name for item in fields(owner)) == (
        "vertices",
        "faces",
        "face_normals",
        "edges",
        "edge_adj_faces",
        "edge_param_range",
        "face_structure_id",
        "face_surface_id",
        "version",
    )
    assert get_type_hints(owner) == {
        "vertices": torch.Tensor,
        "faces": torch.Tensor,
        "face_normals": torch.Tensor,
        "edges": torch.Tensor,
        "edge_adj_faces": torch.Tensor,
        "edge_param_range": torch.Tensor,
        "face_structure_id": torch.Tensor,
        "face_surface_id": torch.Tensor,
        "version": int,
    }
    from witwin.channel_native.core.runtime import _validation as legacy_validation

    assert canonical_validation.require_tensor is legacy_validation.require_tensor


def test_material_store_schema_type_hints_and_defaults_are_exact():
    owner = canonical_material_stores.MaterialStore
    schema = fields(owner)

    assert tuple(item.name for item in schema) == (
        "material_id",
        "eps_r",
        "mu_r",
        "sigma_e",
        "gain",
        "model_id",
        "thickness_m",
        "scattering_coefficient",
        "xpd_coefficient",
        "layer_offset",
        "layer_count",
        "layer_thickness_m",
        "layer_eps_r",
        "layer_sigma_e",
        "layer_mu_r",
        "rough_sigma_h_m",
        "rough_corr_x_m",
        "rough_corr_y_m",
        "rough_axis_rad",
        "geometry_mode_id",
        "scatter_model_id",
        "material_keys",
        "frequency_hz",
        "abi_version",
        "cache_token",
        "version",
        "frequency_dependent",
    )
    assert all(item.default is MISSING for item in schema[:-1])
    assert all(item.default_factory is MISSING for item in schema)
    assert schema[-1].default == ()
    assert get_type_hints(owner) == {
        "material_id": torch.Tensor,
        "eps_r": torch.Tensor,
        "mu_r": torch.Tensor,
        "sigma_e": torch.Tensor,
        "gain": torch.Tensor,
        "model_id": torch.Tensor,
        "thickness_m": torch.Tensor,
        "scattering_coefficient": torch.Tensor,
        "xpd_coefficient": torch.Tensor,
        "layer_offset": torch.Tensor,
        "layer_count": torch.Tensor,
        "layer_thickness_m": torch.Tensor,
        "layer_eps_r": torch.Tensor,
        "layer_sigma_e": torch.Tensor,
        "layer_mu_r": torch.Tensor,
        "rough_sigma_h_m": torch.Tensor,
        "rough_corr_x_m": torch.Tensor,
        "rough_corr_y_m": torch.Tensor,
        "rough_axis_rad": torch.Tensor,
        "geometry_mode_id": torch.Tensor,
        "scatter_model_id": torch.Tensor,
        "material_keys": tuple[str, ...],
        "frequency_hz": float,
        "abi_version": int,
        "cache_token": str,
        "version": int,
        "frequency_dependent": tuple[str, ...],
    }


def test_assignment_store_schema_type_hints_and_defaults_are_exact():
    owner = canonical_assignments.AssignmentStore
    schema = fields(owner)

    assert tuple(item.name for item in schema) == (
        "face_material_id",
        "edge_material_id0",
        "edge_material_id1",
        "surface_material_id",
        "structure_material_id",
        "num_faces",
        "num_edges",
        "version",
        "structure_phase_screens",
    )
    assert all(item.default is MISSING for item in schema[:-1])
    assert schema[-1].default is MISSING
    assert schema[-1].default_factory is dict
    assert get_type_hints(owner) == {
        "face_material_id": torch.Tensor,
        "edge_material_id0": torch.Tensor,
        "edge_material_id1": torch.Tensor,
        "surface_material_id": torch.Tensor,
        "structure_material_id": torch.Tensor,
        "num_faces": int,
        "num_edges": int,
        "version": int,
        "structure_phase_screens": dict[int, public_materials.PhaseScreen],
    }


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.scene.kernels import rayd_scene as canonical; "
            "from witwin.channel_native.scene.kernels import rayd_scene as legacy; "
            "from witwin.channel_native.core import scene as core_scene; "
            "from witwin.channel_native.scene import compiled; "
            "from witwin.channel_native.core.runtime import compiled_scene as legacy_compiled"
        ),
        (
            "from witwin.channel_native.scene.kernels import rayd_scene as legacy; "
            "from witwin.channel_native.scene import compiled; "
            "from witwin.channel_native.scene.kernels import rayd_scene as canonical; "
            "from witwin.channel_native.core.runtime import compiled_scene as legacy_compiled; "
            "from witwin.channel_native.core import scene as core_scene"
        ),
        (
            "from witwin.channel_native.core import scene as core_scene; "
            "from witwin.channel_native.core.runtime import compiled_scene as legacy_compiled; "
            "from witwin.channel_native.scene.kernels import rayd_scene as legacy; "
            "from witwin.channel_native.scene.kernels import rayd_scene as canonical; "
            "from witwin.channel_native.scene import compiled"
        ),
    ),
)
def test_rayd_scene_fresh_import_order_type_hints_and_legacy_pickle_replay(
    imports: str,
):
    source = (
        f"{imports}; import pickle; import torch; from typing import get_type_hints; "
        "scene_owner = canonical.RayDSceneResource; "
        "edge_owner = canonical.RayDEdgeRecords; "
        "assert scene_owner is legacy.RayDSceneResource is core_scene.RayDSceneResource; "
        "assert scene_owner is compiled.RayDSceneResource is legacy_compiled.RayDSceneResource; "
        "assert edge_owner is legacy.RayDEdgeRecords; "
        "assert canonical.build_scene_from_structures is "
        "legacy.build_scene_from_structures is core_scene.build_scene_from_structures; "
        "assert scene_owner.__module__ == "
        "'witwin.channel_native.scene.kernels.rayd_scene'; "
        "assert edge_owner.__module__ == "
        "'witwin.channel_native.scene.kernels.rayd_scene'; "
        "assert get_type_hints(edge_owner)['vertices'] is torch.Tensor; "
        "assert get_type_hints(scene_owner)['mesh_tensors'] == "
        "tuple[tuple[torch.Tensor, ...], ...]; "
        "assert pickle.loads(pickle.dumps(scene_owner)) is scene_owner; "
        "assert pickle.loads(pickle.dumps(edge_owner)) is edge_owner"
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


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.scene.stores import geometry as canonical; "
            "from witwin.channel_native.core.runtime import geometry as legacy; "
            "from witwin.channel_native.core import scene as core_scene; "
            "from witwin.channel_native.scene import compiled"
        ),
        (
            "from witwin.channel_native.core.runtime import geometry as legacy; "
            "from witwin.channel_native.scene import compiled; "
            "from witwin.channel_native.core import scene as core_scene; "
            "from witwin.channel_native.scene.stores import geometry as canonical"
        ),
        (
            "from witwin.channel_native.core import scene as core_scene; "
            "from witwin.channel_native.scene.stores import geometry as canonical; "
            "from witwin.channel_native.core.runtime import geometry as legacy; "
            "from witwin.channel_native.scene import compiled"
        ),
    ),
)
def test_geometry_store_fresh_import_order_and_legacy_pickle_replay(imports: str):
    source = (
        f"{imports}; import pickle; from typing import get_type_hints; import torch; "
        "owner = canonical.GeometryStore; "
        "assert owner is legacy.GeometryStore is core_scene.GeometryStore; "
        "assert owner is compiled.GeometryStore; "
        "assert owner.__module__ == 'witwin.channel_native.core.runtime.geometry'; "
        "assert get_type_hints(owner)['vertices'] is torch.Tensor; "
        "assert pickle.loads("
        "b'cwitwin.channel_native.core.runtime.geometry\\nGeometryStore\\n.'"
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


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.scene.stores import materials as canonical; "
            "from witwin.channel_native.core.runtime import material_store as legacy; "
            "from witwin.channel_native.core import scene as core_scene; "
            "from witwin.channel_native.scene import compiled"
        ),
        (
            "from witwin.channel_native.core.runtime import material_store as legacy; "
            "from witwin.channel_native.scene import compiled; "
            "from witwin.channel_native.core import scene as core_scene; "
            "from witwin.channel_native.scene.stores import materials as canonical"
        ),
        (
            "from witwin.channel_native.core import scene as core_scene; "
            "from witwin.channel_native.scene.stores import materials as canonical; "
            "from witwin.channel_native.core.runtime import material_store as legacy; "
            "from witwin.channel_native.scene import compiled"
        ),
    ),
)
def test_material_store_fresh_import_order_and_legacy_pickle_replay(imports: str):
    source = (
        f"{imports}; import pickle; from typing import get_type_hints; import torch; "
        "owner = canonical.MaterialStore; "
        "assert owner is legacy.MaterialStore is core_scene.MaterialStore; "
        "assert owner is compiled.MaterialStore; "
        "assert owner.__module__ == "
        "'witwin.channel_native.core.runtime.material_store'; "
        "assert get_type_hints(owner)['material_id'] is torch.Tensor; "
        "assert pickle.loads("
        "b'cwitwin.channel_native.core.runtime.material_store\\nMaterialStore\\n.'"
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


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.scene.stores import assignments as canonical; "
            "from witwin.channel_native.core.runtime import assignments as legacy; "
            "from witwin.channel_native.core import scene as core_scene; "
            "from witwin.channel_native.scene import compiled"
        ),
        (
            "from witwin.channel_native.core.runtime import assignments as legacy; "
            "from witwin.channel_native.scene import compiled; "
            "from witwin.channel_native.core import scene as core_scene; "
            "from witwin.channel_native.scene.stores import assignments as canonical"
        ),
        (
            "from witwin.channel_native.core import scene as core_scene; "
            "from witwin.channel_native.scene.stores import assignments as canonical; "
            "from witwin.channel_native.core.runtime import assignments as legacy; "
            "from witwin.channel_native.scene import compiled"
        ),
    ),
)
def test_assignment_store_fresh_import_order_and_legacy_pickle_replay(imports: str):
    source = (
        f"{imports}; import pickle; from typing import get_type_hints; import torch; "
        "from witwin.channel_native.core.materials import PhaseScreen; "
        "owner = canonical.AssignmentStore; "
        "assert owner is legacy.AssignmentStore is core_scene.AssignmentStore; "
        "assert owner is compiled.AssignmentStore; "
        "assert owner.__module__ == "
        "'witwin.channel_native.core.runtime.assignments'; "
        "hints = get_type_hints(owner); "
        "assert hints['face_material_id'] is torch.Tensor; "
        "assert hints['structure_phase_screens'] == dict[int, PhaseScreen]; "
        "assert pickle.loads("
        "b'cwitwin.channel_native.core.runtime.assignments\\nAssignmentStore\\n.'"
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


def test_rayd_scene_lifecycle_identity_schema_type_hints_and_defaults_are_exact():
    edge_owner = canonical_rayd.RayDEdgeRecords
    edge_schema = fields(edge_owner)

    assert edge_owner is legacy_rayd.RayDEdgeRecords
    assert edge_owner.__module__ == "witwin.channel_native.scene.kernels.rayd_scene"
    assert tuple(item.name for item in edge_schema) == (
        "vertices",
        "faces",
        "face_normals",
        "edge_v0",
        "edge_v1",
        "face0",
        "face1",
        "shape_id",
        "local_edge_id",
        "opposite",
    )
    assert all(item.default is MISSING for item in edge_schema)
    assert all(item.default_factory is MISSING for item in edge_schema)
    assert get_type_hints(edge_owner) == {
        name: torch.Tensor for name in (item.name for item in edge_schema)
    }

    scene_owner = canonical_rayd.RayDSceneResource
    scene_schema = fields(scene_owner)
    assert (
        scene_owner
        is legacy_rayd.RayDSceneResource
        is legacy_scene.RayDSceneResource
        is canonical_compiled.RayDSceneResource
        is legacy_compiled.RayDSceneResource
    )
    assert scene_owner.__module__ == "witwin.channel_native.scene.kernels.rayd_scene"
    assert tuple(item.name for item in scene_schema) == (
        "resource",
        "mesh_tensors",
        "reason",
        "runtime_cache",
    )
    assert tuple(item.default for item in scene_schema[:-1]) == (None, (), None)
    assert all(item.default_factory is MISSING for item in scene_schema[:-1])
    assert scene_schema[-1].default is MISSING
    assert scene_schema[-1].default_factory is dict
    assert not scene_schema[-1].compare and not scene_schema[-1].repr
    assert get_type_hints(scene_owner) == {
        "resource": object | None,
        "mesh_tensors": tuple[tuple[torch.Tensor, ...], ...],
        "reason": str | None,
        "runtime_cache": dict[str, object],
    }
    assert legacy_scene.build_scene_from_structures is (
        canonical_rayd.build_scene_from_structures
    )
    assert legacy_rayd.build_scene_from_structures is (
        canonical_rayd.build_scene_from_structures
    )


def test_compiled_scene_dataclass_schema_and_type_hints_are_exact():
    owner = canonical_compiled.CompiledScene
    schema = fields(owner)

    assert legacy_compiled.MaterialStore is canonical_material_stores.MaterialStore
    assert legacy_compiled.AssignmentStore is canonical_assignments.AssignmentStore
    assert tuple(item.name for item in schema) == (
        "geometry",
        "materials",
        "assignments",
        "rayd",
        "geometry_version",
        "material_version",
        "assignment_version",
        "_kirchhoff_resources_cache",
        "_phase_screen_resources_cache",
    )
    assert all(item.default is MISSING for item in schema[:7])
    assert all(item.default_factory is MISSING for item in schema)
    assert tuple(item.default for item in schema[7:]) == (None, None)
    assert all(not item.repr and not item.compare for item in schema[7:])
    assert get_type_hints(owner) == {
        "geometry": legacy_geometry.GeometryStore,
        "materials": legacy_material_store.MaterialStore,
        "assignments": legacy_assignments.AssignmentStore,
        "rayd": legacy_rayd.RayDSceneResource,
        "geometry_version": int,
        "material_version": int,
        "assignment_version": int,
        "_kirchhoff_resources_cache": (
            scattering_resources.KirchhoffRuntimeResources | None
        ),
        "_phase_screen_resources_cache": (
            scattering_resources.PhaseScreenRuntimeResources | None
        ),
    }


def test_phase_screen_scene_resource_schema_is_exact():
    assert tuple(
        item.name for item in fields(scattering_resources.PhaseScreenStructureResource)
    ) == (
        "structure_index",
        "material_index",
        "face_range",
        "first_face",
        "face_count",
        "uv_vertex_count",
        "runtime",
        "uv_vertices",
        "face_uv",
        "uv_tris",
        "face_areas_m2",
        "uv_world_scale_m",
        "rms_slope",
    )
    assert tuple(
        item.name for item in fields(scattering_resources.PhaseScreenRuntimeResources)
    ) == ("key", "structures")


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
    assert scene.rayd_scene() is compiled.rayd
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
    base = _triangle_uv_scene().compile()
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
    # The resource build also derives the ADR-010 KirchhoffTableStack from
    # each table's f_te/f_tm, so the sentinel must carry real tensors.
    table_resource = SimpleNamespace(
        f_te=torch.zeros((2, 2, 2, 2), dtype=torch.float32),
        f_tm=torch.zeros((2, 2, 2, 2), dtype=torch.float32),
    )
    screen_resource = SimpleNamespace(
        device=torch.device("cuda"),
        heights_m=torch.zeros((2, 2), device="cuda", dtype=torch.float32),
        screen=screen,
    )
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
    assert compiled._kirchhoff_resources_cache is None
    assert compiled._phase_screen_resources_cache is None
    kirchhoff_resources = compiled.kirchhoff_resources
    assert compiled.kirchhoff_resources is kirchhoff_resources
    assert kirchhoff_resources.key == scattering_resources.ScatteringResourceKey(
        material_cache_token="rough-material",
        assignment_version=compiled.assignment_version,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    tables = kirchhoff_resources.tables
    assert tables == {0: table_resource}
    assert compiled.kirchhoff_tables is tables
    materials_by_index = kirchhoff_resources.materials
    assert compiled.rough_material_runtimes is materials_by_index
    assert materials_by_index[0].table is table_resource
    from witwin.channel_native.montecarlo.events import scattering as scattering_events

    assert scattering_events.RoughMaterialRuntime is (
        scattering_resources.RoughMaterialRuntime
    )
    assert pickle.loads(
        _pickle_global(
            "witwin.channel_native.montecarlo.events.scattering",
            "RoughMaterialRuntime",
        )
    ) is scattering_resources.RoughMaterialRuntime
    assert pickle.loads(pickle.dumps(scattering_events.RoughMaterialRuntime)) is (
        scattering_resources.RoughMaterialRuntime
    )
    assert scattering_events.rough_material_runtimes(compiled) is materials_by_index
    assert events == ["table"]
    assert compiled._phase_screen_resources_cache is None
    phase_screen_resources = compiled.phase_screen_resources
    assert compiled.phase_screen_resources is phase_screen_resources
    resources = phase_screen_resources.structures
    assert set(resources) == {0}
    assert resources[0].runtime is screen_resource
    assert resources[0].face_range == (0, 1)
    assert resources[0].material_index == 0
    assert events == ["table", "screen"]


def test_compiled_scattering_resource_failures_are_retryable(monkeypatch):
    base = _triangle_uv_scene().compile()
    compiled = replace(
        base,
        materials=replace(
            base.materials,
            scatter_model_id=torch.ones_like(base.materials.scatter_model_id),
            rough_sigma_h_m=torch.full_like(base.materials.rough_sigma_h_m, 1.0e-3),
            rough_corr_x_m=torch.full_like(base.materials.rough_corr_x_m, 0.1),
            rough_corr_y_m=torch.full_like(base.materials.rough_corr_y_m, 0.2),
        ),
        assignments=replace(
            base.assignments,
            structure_material_id=torch.tensor([0], dtype=torch.int32),
            structure_phase_screens={
                0: public_materials.PhaseScreen(height=[[0.0]], height_scale_m=1.0)
            },
        ),
    )
    table_calls = 0
    screen_calls = 0

    def build_table(*args, **kwargs):
        nonlocal table_calls
        del args, kwargs
        table_calls += 1
        if table_calls == 1:
            raise RuntimeError("table build failed")
        return SimpleNamespace(
            f_te=torch.zeros((2, 2, 2, 2), dtype=torch.float32),
            f_tm=torch.zeros((2, 2, 2, 2), dtype=torch.float32),
        )

    def build_screen(*args, **kwargs):
        nonlocal screen_calls
        screen_calls += 1
        if screen_calls == 1:
            raise RuntimeError("screen build failed")
        return SimpleNamespace(
            device=torch.device(kwargs.get("device", "cuda")),
            heights_m=torch.zeros((1, 1), device="cuda", dtype=torch.float32),
            screen=args[0],
        )

    monkeypatch.setattr(scattering, "build_kirchhoff_table", build_table)
    monkeypatch.setattr(scattering, "PhaseScreenRuntime", build_screen)

    with pytest.raises(RuntimeError, match="table build failed"):
        compiled.kirchhoff_resources
    assert compiled._kirchhoff_resources_cache is None
    assert compiled.kirchhoff_resources is compiled.kirchhoff_resources
    assert table_calls == 2
    assert screen_calls == 0

    with pytest.raises(RuntimeError, match="screen build failed"):
        compiled.phase_screen_resources
    assert compiled._phase_screen_resources_cache is None
    assert compiled.phase_screen_resources is compiled.phase_screen_resources
    assert screen_calls == 2


def test_zero_face_phase_screen_resource_keeps_runtime_without_requiring_uv(
    monkeypatch,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for resident phase-screen resources")
    base = public.Scene(
        structures=[], transmitters=[], receivers=[], frequency=3.5e9
    ).compile()
    screen = public_materials.PhaseScreen(height=[[0.0]], height_scale_m=1.0)
    assignments = replace(
        base.assignments,
        structure_material_id=torch.tensor([0], dtype=torch.int32),
        structure_phase_screens={0: screen},
    )
    empty_vertices = torch.empty((0, 3), device="cuda", dtype=torch.float32)
    empty_faces = torch.empty((0, 3), device="cuda", dtype=torch.int32)
    empty_uv = torch.empty((0, 2), device="cuda", dtype=torch.float32)
    empty_face_uv = torch.empty((0, 3), device="cuda", dtype=torch.int32)
    fake_rayd = SimpleNamespace(
        available=True,
        mesh_tensors=((empty_vertices, empty_faces, empty_uv, empty_face_uv),),
        edge_records=lambda: SimpleNamespace(
            vertices=empty_vertices, faces=empty_faces
        ),
    )
    runtime = SimpleNamespace(
        device=torch.device("cuda"),
        heights_m=torch.zeros((1, 1), device="cuda", dtype=torch.float32),
        screen=screen,
    )
    monkeypatch.setattr(scattering, "PhaseScreenRuntime", lambda *args, **kwargs: runtime)
    key = scattering_resources.PhaseScreenResourceKey(
        material_cache_token=base.materials.cache_token,
        geometry_version=0,
        assignment_version=0,
        phase_screen_token=((0, id(screen)),),
        device=torch.device("cuda"),
    )

    resources = scattering_resources.build_phase_screen_resources(
        base.materials, assignments, fake_rayd, key
    )
    resource = resources.structures[0]
    assert resource.runtime is runtime
    assert resource.face_range == (0, 0)
    assert resource.face_count == 0
    assert resource.uv_vertex_count == 0
    assert resource.uv_vertices.shape == (0, 2)
    assert resource.face_uv.shape == (0, 3)
    assert resource.uv_tris.shape == (0, 3, 2)
    assert resource.face_areas_m2.shape == (0,)
    assert resource.uv_world_scale_m == 1.0
    assert resource.rms_slope == 0.0


def test_phase_screen_resource_identity_invalidation_and_scene_isolation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for resident phase-screen resources")
    screen = public_materials.PhaseScreen(
        height=torch.zeros((2, 2)), height_scale_m=1.0
    )

    def bind(scene: public.Scene):
        compiled = scene.compile()
        return replace(
            compiled,
            assignments=replace(
                compiled.assignments,
                structure_phase_screens={0: screen},
            ),
        )

    compiled = bind(_triangle_uv_scene())
    first = compiled.phase_screen_resources
    assert compiled.phase_screen_resources is first
    first_resource = first.structures[0]
    assert compiled.phase_screen_resources.structures[0].uv_tris is (
        first_resource.uv_tris
    )

    assert isinstance(screen.height, torch.Tensor)
    screen.height.add_(1.0)
    rebuilt = compiled.phase_screen_resources
    assert rebuilt is not first
    assert rebuilt.key.phase_screen_token != first.key.phase_screen_token
    torch.testing.assert_close(first_resource.runtime.heights_m, torch.zeros((2, 2), device="cuda"))
    torch.testing.assert_close(rebuilt.structures[0].runtime.heights_m, torch.ones((2, 2), device="cuda"))

    other = bind(_triangle_uv_scene()).phase_screen_resources.structures[0]
    assert other.runtime.heights_m.untyped_storage().data_ptr() != (
        rebuilt.structures[0].runtime.heights_m.untyped_storage().data_ptr()
    )
    assert other.uv_tris.untyped_storage().data_ptr() != (
        rebuilt.structures[0].uv_tris.untyped_storage().data_ptr()
    )


def test_smooth_compiled_scene_persists_independent_empty_resources():
    compiled = public.Scene(
        structures=[], transmitters=[], receivers=[], frequency=3.5e9
    ).compile()

    tables = compiled.kirchhoff_tables
    assert tables == {}
    assert compiled.kirchhoff_tables is tables
    assert compiled.rough_material_runtimes == {}
    assert compiled._phase_screen_resources_cache is None

    resources = compiled.phase_screen_resources
    assert resources.structures == {}
    assert compiled.phase_screen_resources is resources


def test_compiled_scattering_resources_rebuild_when_key_changes():
    compiled = public.Scene(
        structures=[], transmitters=[], receivers=[], frequency=3.5e9
    ).compile()
    first_kirchhoff = compiled.kirchhoff_resources
    first_screens = compiled.phase_screen_resources

    changed = replace(
        compiled,
        materials=replace(compiled.materials, cache_token="changed-material"),
        assignment_version=compiled.assignment_version + 1,
    )

    assert changed._kirchhoff_resources_cache is first_kirchhoff
    assert changed._phase_screen_resources_cache is first_screens
    assert changed.kirchhoff_resources is not first_kirchhoff
    assert changed.phase_screen_resources is not first_screens
    assert changed.kirchhoff_resources.key.material_cache_token == "changed-material"
    assert changed.phase_screen_resources.key.assignment_version == (
        compiled.assignment_version + 1
    )


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
    compiled.rayd = legacy_rayd.RayDSceneResource(resource=owner)
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
