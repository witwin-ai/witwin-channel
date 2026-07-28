import sys

import pytest
import torch

from witwin.channel import runtime
from witwin.channel.scene import compile as compile_scene
from witwin.core import Scene
from tests.support.core_world import (
    make_mesh_structure,
    make_receiver,
    make_transmitter,
)
from witwin.core import PhysicalMaterial


def _source_linked_rayd_available() -> bool:
    from witwin.channel.deployment import build_info

    try:
        return build_info()["rayd_integration"] == "source-linked"
    except ModuleNotFoundError:
        return False


def test_rayd_scene_wrapper_does_not_import_python_rayd():
    sys.modules.pop("rayd", None)

    from witwin.channel.scene.resources import RayDSceneResource

    assert RayDSceneResource.__name__ == "RayDSceneResource"
    assert "rayd" not in sys.modules


def test_rayd_scene_exposes_typed_resource():
    from witwin.channel.scene.resources import RayDSceneResource

    resource = object()
    scene = RayDSceneResource(resource)

    assert scene.resource is resource


def test_validated_native_loader_does_not_import_python_rayd():
    sys.modules.pop("rayd", None)

    runtime.native_extension()

    assert "rayd" not in sys.modules


def test_compile_builds_rayd_scene_handle_when_backend_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RayD native scene construction")

    if not _source_linked_rayd_available():
        pytest.skip("RayD native extension is not built")

    scene = Scene(
        structures=[
            make_mesh_structure(
                vertices=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=torch.float32,
                ),
                faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
                material=PhysicalMaterial(eps_r=2.0),
            )
        ],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, 0.0, 1.0])),
            make_receiver(position=torch.tensor([1.0, 0.0, 1.0])),
        ],
    )

    compiled = compile_scene(scene, reference_frequency_hz=3.5e9)

    assert compiled.rayd.available
    assert compiled.rayd.require_resource() is not None
    assert "rayd" not in sys.modules


def test_scene_reuses_cached_rayd_scene_handle_when_backend_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RayD native scene construction")

    if not _source_linked_rayd_available():
        pytest.skip("RayD native extension is not built")

    scene = Scene(
        structures=[
            make_mesh_structure(
                vertices=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=torch.float32,
                ),
                faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
                material=PhysicalMaterial(eps_r=2.0),
            )
        ],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, 0.0, 1.0])),
            make_receiver(position=torch.tensor([1.0, 0.0, 1.0])),
        ],
    )

    first = compile_scene(scene, reference_frequency_hz=3.5e9)
    second = compile_scene(scene, reference_frequency_hz=3.5e9)

    assert first is second
    assert first.rayd is second.rayd


def test_rayd_scene_exports_non_manifold_edge_records_when_backend_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RayD native scene construction")

    if not _source_linked_rayd_available():
        pytest.skip("RayD native extension is not built")

    scene = Scene(
        structures=[
            make_mesh_structure(
                vertices=torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                faces=torch.tensor(
                    [
                        [0, 1, 2],
                        [1, 0, 3],
                        [0, 1, 4],
                    ],
                    dtype=torch.int32,
                ),
                material=PhysicalMaterial(eps_r=2.0),
            )
        ],
    )

    records = compile_scene(
        scene, reference_frequency_hz=3.5e9
    ).rayd.edge_records()

    assert records.edge_v0.is_cuda
    assert records.edge_v0.dtype == torch.int32
    assert records.edge_v0.shape == (9,)
    central = (records.edge_v0.cpu() == 0) & (records.edge_v1.cpu() == 1)
    assert int(central.sum().item()) == 3
    assert bool((records.face1.cpu()[central] >= 0).all())


def test_rayd_intersect_forward_uses_native_rayd_scene_bridge_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RayD native intersection")

    from witwin.channel.kernels.geometry import (
        rayd_intersect_forward,
    )

    if not _source_linked_rayd_available():
        pytest.skip("RayD native extension is not built")

    scene = Scene(
        structures=[
            make_mesh_structure(
                vertices=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=torch.float32,
                ),
                faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
                material=PhysicalMaterial(eps_r=2.0),
            )
        ],
    )
    rayd = compile_scene(scene, reference_frequency_hz=3.5e9).rayd
    ray_o = torch.tensor([[0.25, 0.25, 1.0]], dtype=torch.float32, device="cuda")
    ray_d = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device="cuda")
    ray_tmax = torch.tensor([10.0], dtype=torch.float32, device="cuda")
    active = torch.tensor([True], dtype=torch.bool, device="cuda")

    hit = rayd_intersect_forward(rayd, ray_o, ray_d, ray_tmax, active, flags=7)

    torch.testing.assert_close(hit["t"].cpu(), torch.tensor([1.0], dtype=torch.float32))
    torch.testing.assert_close(hit["p"].cpu(), torch.tensor([[0.25, 0.25, 0.0]], dtype=torch.float32))
    assert int(hit["prim_id"].cpu()[0].item()) == 0
    assert int(hit["global_prim_id"].cpu()[0].item()) == 0
    assert "rayd" not in sys.modules
