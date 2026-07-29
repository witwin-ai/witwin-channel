# Copyright Xingyu Chen.
# Structure UV fields, planar_uv helper, and RayD forwarding.

"""Structure UV fields, planar_uv helper, and RayD forwarding."""

import math

import pytest
import torch

from witwin.core import (
    MaterialLayer,
    PhaseScreen,
    PhysicalMaterial,
    Mesh,
    Scene,
    Structure,
    SurfaceRoughness,
)
from witwin.channel.scene import compile as compile_scene
from tests.support.core_world import (
    make_receiver,
    make_transmitter,
    planar_uv,
)


def _source_linked_rayd_available() -> bool:
    from witwin.channel.deployment import build_info

    try:
        return build_info()["rayd_integration"] == "source-linked"
    except ModuleNotFoundError:
        return False


def _wall_vertices() -> torch.Tensor:
    return torch.tensor(
        [
            [2.5, -2.0, -1.0],
            [2.5, 2.0, -1.0],
            [2.5, -2.0, 2.0],
            [2.5, 2.0, 2.0],
        ]
    )


def _wall_faces() -> torch.Tensor:
    return torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32)


def _wall_structure(with_uv: bool) -> Structure:
    vertices = _wall_vertices()
    faces = _wall_faces()
    kwargs = {}
    if with_uv:
        uv = planar_uv(
            vertices,
            axis_u=torch.tensor([0.0, 1.0, 0.0]),
            axis_v=torch.tensor([0.0, 0.0, 1.0]),
            origin=torch.tensor([2.5, -2.0, -1.0]),
            scale=0.25,
        )
        kwargs = {"uv": uv, "face_uv": faces.clone()}
    return Structure(
        Mesh(
            vertices,
            faces,
            recenter=False,
            fill_mode="surface",
            topology_diagnostics=False,
        ),
        PhysicalMaterial(eps_r=4.0, sigma_e=0.01),
        name="uv-wall",
        surface_id=1,
        **kwargs,
    )


def test_planar_uv_projects_onto_axes():
    vertices = _wall_vertices()
    uv = planar_uv(
        vertices,
        axis_u=torch.tensor([0.0, 1.0, 0.0]),
        axis_v=torch.tensor([0.0, 0.0, 1.0]),
        origin=torch.tensor([2.5, -2.0, -1.0]),
        scale=0.25,
    )
    expected = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.75], [1.0, 0.75]]
    )
    assert uv.dtype == torch.float32
    torch.testing.assert_close(uv, expected)


def test_planar_uv_defaults_to_world_origin():
    vertices = torch.tensor([[1.0, 2.0, 3.0]])
    uv = planar_uv(
        vertices,
        axis_u=torch.tensor([1.0, 0.0, 0.0]),
        axis_v=torch.tensor([0.0, 1.0, 0.0]),
        origin=torch.zeros(3),
        scale=1.0,
    )
    torch.testing.assert_close(uv, torch.tensor([[1.0, 2.0]]))


def test_structure_uv_validation():
    vertices = _wall_vertices()
    faces = _wall_faces()
    material = PhysicalMaterial(eps_r=2.0)
    geometry = Mesh(
        vertices,
        faces,
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    uv = torch.zeros((4, 2))
    with pytest.raises(ValueError, match="together"):
        Structure(geometry, material, uv=uv)
    with pytest.raises(ValueError, match=r"uv must have shape \(T, 2\)"):
        Structure(
            geometry,
            material,
            uv=torch.zeros((4, 3)),
            face_uv=faces.clone(),
        )
    with pytest.raises(ValueError, match="one row per mesh face"):
        Structure(
            geometry,
            material,
            uv=uv,
            face_uv=faces[:1].clone(),
        )
    structure = Structure(
        geometry, material, uv=uv, face_uv=faces.clone()
    )
    assert structure.uv.dtype == torch.float32
    assert structure.face_uv.dtype == torch.int32
    # Structures without UV keep None fields (empty per-mesh UV in RayD).
    bare = Structure(geometry, material)
    assert bare.uv is None and bare.face_uv is None


def _uv_scene(with_uv: bool) -> Scene:
    return Scene(
        structures=[_wall_structure(with_uv)],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, 0.0, 0.0])),
            make_receiver(position=torch.tensor([5.0, 0.0, 0.0])),
        ],
    )


@pytest.mark.parametrize("with_uv", [True, False])
def test_scene_with_uv_builds_and_traces(with_uv):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RayD native scene construction")
    if not _source_linked_rayd_available():
        pytest.skip("RayD native extension is not built")

    from witwin.channel.kernels.geometry import (
        rayd_intersect_forward,
    )

    scene = _uv_scene(with_uv)
    compiled = compile_scene(scene, reference_frequency_hz=3.0e9)
    rayd = compiled.rayd
    assert rayd.available
    ray_o = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda")
    ray_d = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device="cuda")
    ray_tmax = torch.tensor([10.0], dtype=torch.float32, device="cuda")
    active = torch.tensor([True], dtype=torch.bool, device="cuda")
    hit = rayd_intersect_forward(rayd, ray_o, ray_d, ray_tmax, active, flags=7)
    torch.testing.assert_close(
        hit["t"].cpu(), torch.tensor([2.5], dtype=torch.float32)
    )
    # Scene compile keeps working on top of the UV-carrying RayD scene.
    assert compiled.rayd is rayd


def test_compiled_scene_lazy_scattering_caches():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RayD native scene construction")
    if not _source_linked_rayd_available():
        pytest.skip("RayD native extension is not built")

    frequency = 6.0e9
    sigma_e = 0.1 * 2.0 * math.pi * frequency * 8.8541878128e-12
    material = PhysicalMaterial(
        layers=(
            MaterialLayer(
                thickness_m=0.5,
                eps_r=4.0,
                sigma_e=sigma_e,
            ),
        ),
        roughness_front=SurfaceRoughness(
            rms_height_m=1e-3,
            correlation_length_x_m=0.1,
            correlation_length_y_m=0.1,
        ),
        name="rough-wall",
    )
    screen = PhaseScreen(height=torch.zeros(16, 16), height_scale_m=1e-3)
    structure = _wall_structure(with_uv=True)
    scene = Scene(
        structures=[
            Structure(
                structure.geometry,
                material,
                phase_screen=screen,
                uv=structure.uv,
                face_uv=structure.face_uv,
            )
        ],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, 0.0, 0.0])),
            make_receiver(position=torch.tensor([5.0, 0.0, 0.0])),
        ],
    )
    compiled = compile_scene(scene, reference_frequency_hz=frequency)
    assert int(compiled.materials.scatter_model_id[0]) == 1

    tables = compiled.kirchhoff_tables
    assert set(tables.keys()) == {0}
    table = tables[0]
    assert table.f_te.shape == (32, 1, 32, 64)
    assert table.frequency_hz == frequency
    # Lazy cache: second access returns the same objects.
    assert compiled.kirchhoff_tables is tables

    resources = compiled.phase_screen_resources
    assert set(resources.structures) == {0}
    resource = resources.structures[0]
    assert resource.runtime.heights_m.shape == (16, 16)
    assert resource.face_range == (0, 2)
    assert resource.first_face == 0
    assert resource.face_count == 2
    assert resource.uv_vertex_count == 4
    assert resource.uv_vertices.shape == (4, 2)
    assert resource.face_uv.shape == (2, 3)
    assert resource.uv_tris.shape == (2, 3, 2)
    assert resource.face_areas_m2.shape == (2,)
    assert resource.uv_world_scale_m > 0.0
    assert resource.rms_slope == 0.0
    assert compiled.phase_screen_resources is resources