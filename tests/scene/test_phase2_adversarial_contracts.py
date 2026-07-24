from __future__ import annotations

import importlib

import pytest
import torch

from witwin.core import (
    AntennaPattern,
    AntennaState,
    Mesh,
    PhysicalMaterial,
    ReceiverGrid,
    Scene,
    Structure,
)
from witwin.channel.scene.antenna import pattern_field_response
from witwin.channel.montecarlo.basic import Config as MonteCarloBasicConfig
from witwin.channel.montecarlo.basic import solver as montecarlo_basic_solver
from witwin.channel.scene.endpoints import (
    _endpoint_views,
    _validate_scalar_endpoint_boundary,
)
from witwin.channel.scene.kernels.rayd_scene import (
    RayDEdgeRecords,
    RayDSceneResource,
)


compile_module = importlib.import_module("witwin.channel.scene.compiler")


def _world(
    *,
    x_offset: float,
    eps_r: float,
    geometry: Mesh | None = None,
) -> Scene:
    resolved_geometry = geometry or Mesh(
        torch.tensor(
            [
                [x_offset, 0.0, 0.0],
                [x_offset + 1.0, 0.0, 0.0],
                [x_offset, 1.0, 0.0],
            ]
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
    return Scene(
        structures=(
            Structure(
                resolved_geometry,
                material,
                structure_id=501,
                material_id=701,
                assignment_id=801,
                surface_id=601,
                primitive_ids=(901,),
            ),
        )
    )


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


def _install_compile_seams(monkeypatch):
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


def test_cache_never_equates_unrelated_worlds_with_the_same_stable_ids(
    monkeypatch,
):
    builds = _install_compile_seams(monkeypatch)
    compile_module.clear_compile_cache()

    left = compile_module.compile(
        _world(x_offset=0.0, eps_r=2.5),
        reference_frequency_hz=3.5e9,
    )
    right = compile_module.compile(
        _world(x_offset=10.0, eps_r=7.0),
        reference_frequency_hz=3.5e9,
    )

    assert len(builds) == 2
    assert right.rayd is not left.rayd
    assert right.geometry is not left.geometry
    assert right.materials is not left.materials
    assert torch.equal(
        right.geometry.vertices[0],
        torch.tensor([10.0, 0.0, 0.0]),
    )
    assert right.materials.eps_r.tolist() == [7.0]


def test_cache_reuses_shared_geometry_across_immutable_material_update(
    monkeypatch,
):
    builds = _install_compile_seams(monkeypatch)
    compile_module.clear_compile_cache()
    geometry = _world(x_offset=0.0, eps_r=2.5).structures[0].geometry
    scene = _world(x_offset=0.0, eps_r=2.5, geometry=geometry)

    left = compile_module.compile(scene, reference_frequency_hz=3.5e9)
    right = compile_module.compile(
        scene.with_material(
            701,
            PhysicalMaterial(eps_r=4.0, material_id=701, name="wall"),
        ),
        reference_frequency_hz=3.5e9,
    )

    assert len(builds) == 1
    assert right.rayd is left.rayd
    assert right.geometry is left.geometry
    assert right.materials is not left.materials
    assert right.assignments is left.assignments


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("isotropic", 1.0),
        ("vertical", 1.0),
        ("horizontal", 0.0),
    ],
)
def test_pattern_helper_consumes_the_canonical_core_contract(kind, expected):
    direction = torch.tensor([[1.0, 0.0, 0.0]])

    response = pattern_field_response(AntennaPattern(kind), direction)

    assert response.dtype == torch.complex64
    assert response.tolist() == [complex(expected)]


def test_pattern_helper_preserves_core_custom_callable():
    pattern = AntennaPattern(
        "custom",
        lambda direction: direction[..., 1].to(torch.complex64) * (1.0 + 2.0j),
    )

    response = pattern_field_response(
        pattern,
        torch.tensor([[0.0, 1.0, 0.0]]),
    )

    assert response.tolist() == [1.0 + 2.0j]


def test_point_endpoint_boundary_preserves_live_tensor_identity():
    endpoint = AntennaState(
        12,
        "tx",
        torch.tensor([0.0, 0.0, 0.0], requires_grad=True),
    )
    views = _endpoint_views(Scene(endpoints=(endpoint,)))

    _validate_scalar_endpoint_boundary(views)

    assert views[0].position is endpoint.position
    assert views[0].position.requires_grad


def test_grid_spacing_rejects_silent_gradient_detach():
    grid = ReceiverGrid(
        13,
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (2, 3),
        torch.tensor([0.5, 0.25], requires_grad=True),
    )
    views = _endpoint_views(Scene(endpoints=(grid,)))

    with pytest.raises(RuntimeError, match="tensor-native endpoint ABI"):
        _validate_scalar_endpoint_boundary(views)


def test_mc_basic_rejects_endpoint_array_before_scene_compile(monkeypatch):
    scene = Scene(
        endpoints=(
            AntennaState(
                14,
                "tx",
                (0.0, 0.0, 0.0),
                element_positions=(
                    (-0.05, 0.0, 0.0),
                    (0.05, 0.0, 0.0),
                ),
            ),
            AntennaState(15, "rx", (1.0, 0.0, 0.0)),
        )
    )
    compile_called = False

    def unexpected_compile(*args, **kwargs):
        nonlocal compile_called
        compile_called = True
        raise AssertionError("scene compiled before endpoint preflight")

    monkeypatch.setattr(
        montecarlo_basic_solver,
        "compile_scene",
        unexpected_compile,
    )

    with pytest.raises(ValueError, match="does not support antenna arrays"):
        montecarlo_basic_solver.solve(
            scene,
            MonteCarloBasicConfig(samples=1, components={"los"}),
            reference_frequency_hz=1.0e9,
        )

    assert not compile_called
