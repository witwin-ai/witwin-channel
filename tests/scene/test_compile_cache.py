# Copyright Xingyu Chen.
# Tests compile cache keys, replacement, byte bound, ID columns and stale-cache safety.

from __future__ import annotations

import importlib

import pytest
import torch

from witwin.core import Box, Mesh, PhysicalMaterial, Scene, Structure

from tests.support.core_world import make_receiver, make_transmitter

compile_module = importlib.import_module("witwin.channel.scene.compiler")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

_FREQUENCY = 3.5e9
_WALL = [[2.5, -3.0, -1.0], [2.5, 3.0, -1.0], [2.5, -3.0, 2.0], [2.5, 3.0, 2.0]]
_FLOOR = [[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [-10.0, 10.0, 0.0], [10.0, 10.0, 0.0]]


def _mesh(vertices, faces=((0, 1, 2), (1, 3, 2))) -> Mesh:
    return Mesh(
        torch.tensor(vertices, dtype=torch.float32),
        torch.tensor(faces, dtype=torch.int64),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )


def _scene(*, wall: Mesh | None = None, eps_r: float | torch.Tensor = 4.0) -> Scene:
    material = PhysicalMaterial(eps_r=eps_r, sigma_e=0.01)
    return Scene(
        structures=[
            Structure(_mesh(_FLOOR), material),
            Structure(_mesh(_WALL) if wall is None else wall, material),
        ],
        endpoints=[
            make_transmitter(torch.tensor([0.0, -1.0, 0.5])),
            make_receiver(torch.tensor([0.0, 1.0, 0.5])),
        ],
    )


@pytest.fixture
def builds(monkeypatch):
    """Count native scene builds while the real builder runs."""

    calls: list[tuple[object, ...]] = []
    real = compile_module.build_scene_from_structures

    def counting(structures, **kwargs):
        calls.append(kwargs.get("dynamic", ()))
        return real(structures, **kwargs)

    monkeypatch.setattr(compile_module, "build_scene_from_structures", counting)
    compile_module.clear_compile_cache()
    yield calls
    compile_module.clear_compile_cache()


def _compile(scene, frequency=_FREQUENCY):
    return compile_module.compile(scene, reference_frequency_hz=frequency)


def _deterministic(scene):
    from witwin.channel import deterministic

    config = deterministic.Config(max_depth=1, components=("los", "reflection"))
    return deterministic.solve(scene, config, reference_frequency_hz=_FREQUENCY)


def _assert_identical(left, right) -> None:
    """Every result tensor equal bit for bit, including the exported path table."""

    assert torch.equal(left.path_gain, right.path_gain)
    assert torch.equal(left.field, right.field)
    assert left.component_power.keys() == right.component_power.keys()
    for name, value in left.component_power.items():
        assert torch.equal(value, right.component_power[name]), name
    for name, value in left.component_fields.items():
        assert torch.equal(value, right.component_fields[name]), name
    assert (left.paths is None) == (right.paths is None)
    if left.paths is not None:
        for name in left.paths.__dataclass_fields__:
            assert torch.equal(getattr(left.paths, name), getattr(right.paths, name)), name


def _same_endpoints(target: Scene, source: Scene) -> Scene:
    with torch.no_grad():
        for mine, theirs in zip(target.endpoints, source.endpoints):
            mine.position.copy_(theirs.position)
    return target


@cuda
def test_moving_a_scene_antenna_reuses_the_compiled_runtime(builds):
    scene = _scene()
    first = _compile(scene)
    with torch.no_grad():
        scene.endpoints[0].position[0] += 0.25
    again = _compile(scene)

    assert again is first
    assert len(builds) == 1
    assert len(compile_module._REGISTRY) == 1


@cuda
def test_an_antenna_move_does_not_make_the_consumer_refuse_the_compiled_scene(builds):
    from witwin.channel.propagation import consumer

    from tests.propagation.consumer.test_consumer_input_guards import _request

    scene = _scene()
    compiled = _compile(scene)
    with torch.no_grad():
        scene.endpoints[0].position[1] += 0.5
    assert compiled.stale_domain() is None
    assert consumer.evaluate(compiled, _request()).paths.path_count > 0


@cuda
@pytest.mark.parametrize("edit", ["in_place", "data_swap", "attribute"])
def test_a_vertex_edit_recompiles_and_matches_a_fresh_scene(builds, edit):
    wall = _mesh(_WALL)
    scene = _scene(wall=wall)
    first = _compile(scene)
    moved = torch.tensor(_WALL) + torch.tensor([0.5, 0.0, 0.0])
    if edit == "in_place":
        with torch.no_grad():
            wall.vertices.copy_(moved)
    elif edit == "data_swap":
        wall.vertices.data = moved.clone()
    else:
        wall._vertices_tensor = moved.clone()

    assert first.stale_domain() == "geometry_version"
    second = _compile(scene)
    assert second is not first
    assert second.rayd is not first.rayd
    assert torch.equal(second.geometry.vertices, _compile(_scene(wall=_mesh(moved.tolist()))).geometry.vertices)
    assert len([entry for entry in compile_module._REGISTRY._compiled.values() if entry.source is scene]) == 1


@cuda
def test_repeated_vertex_edits_refit_in_place_and_stay_bit_identical(builds):
    wall = _mesh(_WALL)
    scene = _scene(wall=wall)
    first = _compile(scene)
    results = []
    compiled = [first]
    for _ in range(4):
        with torch.no_grad():
            wall.vertices[:, 0] += 0.25
        results.append(_deterministic(scene))
        compiled.append(_compile(scene))
    # The first vertex-only change rebuilds with the wall refittable; later ones refit it.
    assert builds == [(), [False, True]]
    assert compiled[1].rayd is not first.rayd
    assert all(item.rayd is compiled[1].rayd for item in compiled[2:])
    assert compiled[-1].rayd.generation == 3
    assert all(item.stale_domain() == "geometry_version" for item in compiled[:-1])
    assert compiled[-1].stale_domain() is None
    assert len(compile_module._REGISTRY) == 1

    reference = _same_endpoints(_scene(wall=_mesh(wall.vertices.tolist())), scene)
    compile_module.clear_compile_cache()
    _assert_identical(results[-1], _deterministic(reference))


@cuda
def test_a_stale_compiled_scene_is_refused_after_a_refit(builds):
    from witwin.channel.propagation import consumer

    from tests.propagation.consumer.test_consumer_input_guards import _request

    wall = _mesh(_WALL)
    scene = _scene(wall=wall)
    _compile(scene)
    with torch.no_grad():
        wall.vertices[:, 0] += 0.25
    old = _compile(scene)
    with torch.no_grad():
        wall.vertices[:, 0] += 0.25
    new = _compile(scene)
    assert new.rayd is old.rayd
    with pytest.raises(ValueError, match="the compiled scene is stale"):
        consumer.evaluate(old, _request())
    assert consumer.evaluate(new, _request()).paths.path_count > 0


@cuda
def test_material_and_frequency_changes_reuse_the_rayd_scene(builds):
    eps = torch.tensor(4.0)
    scene = _scene(eps_r=eps)
    first = _compile(scene)
    with torch.no_grad():
        eps.add_(1.0)
    assert first.stale_domain() == "material_version"
    material = _compile(scene)
    frequency = _compile(scene, 5.0e9)

    assert len(builds) == 1
    assert material is not first and frequency is not material
    assert material.rayd is first.rayd and frequency.rayd is first.rayd
    assert material.geometry is first.geometry
    assert float(material.materials.eps_r[0]) == 5.0
    assert material.materials is not first.materials
    assert frequency.materials is not material.materials


@cuda
def test_trace_backend_and_device_are_part_of_the_key(builds, monkeypatch):
    from witwin.channel import runtime

    scene = _scene()
    monkeypatch.setenv(runtime.TRACE_BACKEND_ENV, "cuda")
    cuda_backend = _compile(scene)
    monkeypatch.setenv(runtime.TRACE_BACKEND_ENV, "auto")
    auto_backend = _compile(scene)
    assert auto_backend.rayd is not cuda_backend.rayd
    assert len(builds) == 2

    key = (id(scene), ("float", _FREQUENCY.hex()), torch.cuda.current_device(), "auto",
           compile_module.MATERIAL_ABI_VERSION)
    state = compile_module.world_state(scene)
    assert compile_module._REGISTRY.get(key, scene, state) is auto_backend
    other_device = (key[0], key[1], key[2] + 1, *key[3:])
    assert compile_module._REGISTRY.get(other_device, scene, state) is None
    assert compile_module._REGISTRY.candidates(key[2] + 1) == ()


@cuda
def test_a_moved_geometry_tensor_device_recompiles(builds):
    wall = _mesh(_WALL)
    scene = _scene(wall=wall)
    first = _compile(scene)
    wall.vertices.data = wall.vertices.data.to("cuda")
    assert first.stale_domain() == "geometry_version"
    assert _compile(scene) is not first


@cuda
def test_32_moved_transmitter_solves_keep_one_runtime(builds):
    scene = _scene()
    _deterministic(scene)
    single = compile_module._REGISTRY.resident_bytes()
    for _ in range(32):
        with torch.no_grad():
            scene.endpoints[0].position[0] += 0.01
        _deterministic(scene)
    assert len(builds) == 1
    assert len(compile_module._REGISTRY) == 1
    assert compile_module._REGISTRY.resident_bytes() <= 2 * single


@cuda
def test_the_cache_is_bounded_by_bytes(builds, monkeypatch):
    scenes = [_scene() for _ in range(4)]
    for scene in scenes:
        _compile(scene)
    one = compile_module._REGISTRY.resident_bytes() // 4
    monkeypatch.setattr(compile_module, "_CACHE_BYTE_LIMIT", 2 * one + one // 2)
    _compile(_scene())
    assert len(compile_module._REGISTRY) == 2
    monkeypatch.setattr(compile_module, "_CACHE_BYTE_LIMIT", 0)
    newest = _scene()
    compiled = _compile(newest)
    assert len(compile_module._REGISTRY) == 1
    assert _compile(newest) is compiled


def test_warm_compile_hit_does_not_hash_the_core_world(monkeypatch):
    from tests.scene.test_core_scene_compile import _install_cpu_compile_seams, _scene as one_wall

    _install_cpu_compile_seams(monkeypatch)
    compile_module.clear_compile_cache()
    scene = one_wall()
    first = compile_module.compile(scene, reference_frequency_hz=_FREQUENCY)

    def refuse(*args, **kwargs):
        raise AssertionError("a warm hit must not read the Core version domains")

    monkeypatch.setattr(compile_module, "_versions", refuse)
    assert compile_module.compile(scene, reference_frequency_hz=_FREQUENCY) is first
    compile_module.clear_compile_cache()


def _fake_rayd(structures):
    from witwin.channel.scene.resources import RayDEdgeRecords, RayDSceneResource

    vertices = torch.cat(tuple(item.vertices for item in structures), dim=0)
    offsets, faces, total = [], [], 0
    for item in structures:
        faces.append(item.faces + total)
        total += int(item.vertices.shape[0])
    faces = torch.cat(faces, dim=0)
    empty = torch.empty(0, dtype=torch.int32)
    records = RayDEdgeRecords(
        vertices=vertices, faces=faces, face_normals=torch.zeros_like(faces, dtype=torch.float32),
        edge_v0=empty, edge_v1=empty, face0=empty, face1=empty, shape_id=empty,
        local_edge_id=empty, opposite=empty,
    )
    del offsets
    return RayDSceneResource(reason="test resource", runtime_cache={"edge_records": records})


def test_face_id_columns_match_core_primitive_ids(monkeypatch):
    from tests.scene.test_core_scene_compile import _install_cpu_compile_seams

    _install_cpu_compile_seams(monkeypatch)
    monkeypatch.setattr(
        compile_module, "build_scene_from_structures", lambda structures, **_: _fake_rayd(structures)
    )
    compile_module.clear_compile_cache()
    material = PhysicalMaterial(eps_r=3.0)
    authored = _mesh(_WALL, faces=((0, 1, 2), (0, 2, 1), (1, 3, 2), (1, 1, 2)))
    dropped = _mesh(_FLOOR, faces=((0, 1, 2), (2, 1, 0), (1, 3, 2)))
    plain = Box(position=(0.0, 0.0, 5.0), size=(1.0, 1.0, 1.0))
    scene = Scene(
        structures=[
            Structure(authored, material, primitive_ids=(11, 12, 13, 14)),
            Structure(dropped, material),
            Structure(plain, material),
        ]
    )
    compiled = compile_module.compile(scene, reference_frequency_hz=_FREQUENCY)

    expected: list[int] = []
    expected_structures: list[int] = []
    expected_surfaces: list[int] = []
    for structure, kept in zip(scene.structures, ((0, 2), (0, 2), tuple(range(12)))):
        expected.extend(int(structure.primitive_id(index)) for index in kept)
        expected_structures.extend(int(structure.structure_id) for _ in kept)
        expected_surfaces.extend(int(structure.surface_id) for _ in kept)
    assert compiled.geometry.face_primitive_id.tolist() == expected
    assert compiled.geometry.face_structure_id.tolist() == expected_structures
    assert compiled.geometry.face_surface_id.tolist() == expected_surfaces
    assert compiled.assignments.face_material_id.tolist() == [0] * len(expected)
    compile_module.clear_compile_cache()
