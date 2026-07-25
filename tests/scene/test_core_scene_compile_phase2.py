from __future__ import annotations

import importlib

import pytest
import torch

import witwin.channel as channel
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure
from witwin.channel.scene.kernels.rayd_scene import (
    RayDEdgeRecords,
    RayDSceneResource,
)


compile_module = importlib.import_module("witwin.channel.scene.compiler")


def _scene(*, eps_r: float = 2.5) -> Scene:
    geometry = Mesh(
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        torch.tensor([[0, 1, 2]], dtype=torch.int32),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    material = PhysicalMaterial(
        eps_r=eps_r,
        material_id=701,
        name="wall",
    )
    structure = Structure(
        geometry,
        material,
        structure_id=501,
        material_id=701,
        assignment_id=801,
        surface_id=601,
        primitive_ids=(901,),
    )
    return Scene(structures=(structure,))


def _fake_rayd(structures) -> RayDSceneResource:
    vertices = torch.cat(tuple(item.vertices for item in structures), dim=0)
    faces = torch.cat(tuple(item.faces for item in structures), dim=0)
    records = RayDEdgeRecords(
        vertices=vertices,
        faces=faces,
        face_normals=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        edge_v0=torch.empty(0, dtype=torch.int32),
        edge_v1=torch.empty(0, dtype=torch.int32),
        face0=torch.empty(0, dtype=torch.int32),
        face1=torch.empty(0, dtype=torch.int32),
        shape_id=torch.empty(0, dtype=torch.int32),
        local_edge_id=torch.empty(0, dtype=torch.int32),
        opposite=torch.empty(0, dtype=torch.int32),
    )
    return RayDSceneResource(
        reason="test resource",
        runtime_cache={"edge_records": records},
    )


def _install_cpu_compile_seams(monkeypatch):
    builds: list[tuple[object, ...]] = []

    def build(structures):
        builds.append(structures)
        return _fake_rayd(structures)

    monkeypatch.setattr(compile_module, "build_scene_from_structures", build)
    monkeypatch.setattr(
        compile_module.topology_primitives,
        "core_pack_int2",
        lambda left, right: torch.stack((left, right), dim=1).to(torch.int32),
    )
    monkeypatch.setattr(
        compile_module,
        "bdpt_zero_matrix",
        lambda reference, *, rows, cols: torch.zeros(
            (rows, cols), dtype=torch.float32, device=reference.device
        ),
    )
    return builds


def test_root_does_not_republish_the_core_world_model():
    """Each world type has exactly one import path, and it is ``witwin.core``."""

    for name in ("Scene", "SceneSnapshot", "Structure", "PhysicalMaterial",
                 "ReceiverGrid", "AntennaState"):
        assert not hasattr(channel, name), name
        assert name not in channel.__all__, name


def test_compile_maps_stable_ids_to_dense_runtime_rows(monkeypatch):
    _install_cpu_compile_seams(monkeypatch)
    compile_module.clear_compile_cache()

    compiled = compile_module.compile(_scene(), reference_frequency_hz=3.5e9)

    assert compiled.geometry.face_structure_id.tolist() == [501]
    assert compiled.geometry.face_surface_id.tolist() == [601]
    assert compiled.geometry.face_primitive_id.tolist() == [901]
    assert compiled.materials.material_id.tolist() == [701]
    assert compiled.assignments.assignment_id.tolist() == [801]
    assert compiled.assignments.face_material_id.tolist() == [0]
    assert compiled.assignments.structure_material_id.tolist() == [0]
    assert compiled.assignments.face_material_id.dtype == torch.int32


def test_four_domain_cache_reuses_only_unaffected_resources(monkeypatch):
    builds = _install_cpu_compile_seams(monkeypatch)
    compile_module.clear_compile_cache()
    scene = _scene()

    first = compile_module.compile(scene, reference_frequency_hz=3.5e9)
    assert compile_module.compile(scene, reference_frequency_hz=3.5e9) is first

    replacement = PhysicalMaterial(
        eps_r=4.0,
        material_id=701,
        name="wall",
    )
    material_scene = scene.with_material(701, replacement)
    material_compiled = compile_module.compile(
        material_scene, reference_frequency_hz=3.5e9
    )

    assert len(builds) == 1
    assert material_compiled.rayd is first.rayd
    assert material_compiled.geometry is first.geometry
    assert material_compiled.materials is not first.materials
    assert material_compiled.assignments is first.assignments

    frequency_compiled = compile_module.compile(
        material_scene, reference_frequency_hz=5.0e9
    )
    assert len(builds) == 1
    assert frequency_compiled.rayd is material_compiled.rayd
    assert frequency_compiled.geometry is material_compiled.geometry
    assert frequency_compiled.materials is not material_compiled.materials
    assert frequency_compiled.assignments is material_compiled.assignments


def test_compile_cache_does_not_cross_distinct_world_ids(monkeypatch):
    builds = _install_cpu_compile_seams(monkeypatch)
    compile_module.clear_compile_cache()

    first = _scene()
    second_structure = Structure(
        first.structures[0].geometry,
        PhysicalMaterial(eps_r=2.5, material_id=1701),
        structure_id=1501,
        material_id=1701,
        assignment_id=1801,
        surface_id=1601,
        primitive_ids=(1901,),
    )
    second = Scene(structures=(second_structure,))

    left = compile_module.compile(first, reference_frequency_hz=3.5e9)
    right = compile_module.compile(second, reference_frequency_hz=3.5e9)

    assert len(builds) == 2
    assert left.rayd is not right.rayd


def test_explicit_pec_and_frequency_gate(monkeypatch):
    _install_cpu_compile_seams(monkeypatch)
    compile_module.clear_compile_cache()
    scene = _scene()
    pec = PhysicalMaterial.perfect_conductor(material_id=701)
    compiled = compile_module.compile(
        scene.with_material(701, pec),
        reference_frequency_hz=3.5e9,
    )

    assert compiled.materials.model_id.tolist() == [2]
    assert compiled.materials.sigma_e.tolist() == [1.0e9]
    compiled.require_reference_frequency(3.5e9)
    with pytest.raises(ValueError, match="does not exactly match"):
        compiled.require_reference_frequency(3.6e9)
