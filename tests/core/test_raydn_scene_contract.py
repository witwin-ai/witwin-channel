import sys

import pytest
import torch

from witwin.channel_native import ReceiverPoint, Scene, Structure, Transmitter
from witwin.channel_native.core.materials import Dielectric


def test_raydn_scene_wrapper_does_not_import_python_raydn():
    sys.modules.pop("raydn", None)

    from witwin.channel_native.core.runtime.raydn import RayDNScene

    assert RayDNScene.__name__ == "RayDNScene"
    assert "raydn" not in sys.modules


def test_raydn_scene_exposes_opaque_handle():
    from witwin.channel_native.core.runtime.raydn import RayDNScene

    handle = object()
    scene = RayDNScene(handle)

    assert scene.handle is handle


def test_raydn_backend_loader_does_not_import_python_raydn():
    sys.modules.pop("raydn", None)

    from witwin.channel_native.core.kernels import raydn_backend

    raydn_backend.native_extension()

    assert "raydn" not in sys.modules


def test_compile_builds_raydn_scene_handle_when_backend_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RayDN native scene construction")

    from witwin.channel_native.core.kernels import raydn_backend

    if raydn_backend.native_extension() is None:
        pytest.skip("RayDN native extension is not built")

    scene = Scene(
        structures=[
            Structure(
                vertices=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=torch.float32,
                ),
                faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
                material=Dielectric(eps_r=2.0),
            )
        ],
        transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 1.0]))],
        receivers=[ReceiverPoint(position=torch.tensor([1.0, 0.0, 1.0]))],
        frequency=3.5e9,
    )

    compiled = scene.compile()

    assert compiled.raydn.available
    assert compiled.raydn.require_handle() is not None
    assert "raydn" not in sys.modules


def test_scene_reuses_cached_raydn_scene_handle_when_backend_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RayDN native scene construction")

    from witwin.channel_native.core.kernels import raydn_backend

    if raydn_backend.native_extension() is None:
        pytest.skip("RayDN native extension is not built")

    scene = Scene(
        structures=[
            Structure(
                vertices=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=torch.float32,
                ),
                faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
                material=Dielectric(eps_r=2.0),
            )
        ],
        transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 1.0]))],
        receivers=[ReceiverPoint(position=torch.tensor([1.0, 0.0, 1.0]))],
        frequency=3.5e9,
    )

    first = scene.raydn_scene()
    second = scene.raydn_scene()
    compiled = scene.compile()

    assert first is second
    assert compiled.raydn is first


def test_raydn_scene_exports_non_manifold_edge_records_when_backend_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RayDN native scene construction")

    from witwin.channel_native.core.kernels import raydn_backend

    if raydn_backend.native_extension() is None:
        pytest.skip("RayDN native extension is not built")

    scene = Scene(
        structures=[
            Structure(
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
                material=Dielectric(eps_r=2.0),
            )
        ],
        transmitters=[],
        receivers=[],
        frequency=3.5e9,
    )

    records = scene.compile().raydn.edge_records()

    assert records.edge_v0.is_cuda
    assert records.edge_v0.dtype == torch.int32
    assert records.edge_v0.shape == (9,)
    central = (records.edge_v0.cpu() == 0) & (records.edge_v1.cpu() == 1)
    assert int(central.sum().item()) == 3
    assert bool((records.face1.cpu()[central] >= 0).all())


def test_bdpt_intersect_forward_uses_native_raydn_scene_bridge_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RayDN native intersection")

    from witwin.channel_native.core.kernels import raydn_backend
    from witwin.channel_native.core.kernels.ops import bdpt_intersect_forward

    if raydn_backend.native_extension() is None:
        pytest.skip("RayDN native extension is not built")

    scene = Scene(
        structures=[
            Structure(
                vertices=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=torch.float32,
                ),
                faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
                material=Dielectric(eps_r=2.0),
            )
        ],
        transmitters=[],
        receivers=[],
        frequency=3.5e9,
    )
    raydn = scene.raydn_scene()
    ray_o = torch.tensor([[0.25, 0.25, 1.0]], dtype=torch.float32, device="cuda")
    ray_d = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device="cuda")
    ray_tmax = torch.tensor([10.0], dtype=torch.float32, device="cuda")
    active = torch.tensor([True], dtype=torch.bool, device="cuda")

    hit = bdpt_intersect_forward(raydn, ray_o, ray_d, ray_tmax, active, flags=7)

    torch.testing.assert_close(hit["t"].cpu(), torch.tensor([1.0], dtype=torch.float32))
    torch.testing.assert_close(hit["p"].cpu(), torch.tensor([[0.25, 0.25, 0.0]], dtype=torch.float32))
    assert int(hit["prim_id"].cpu()[0].item()) == 0
    assert int(hit["global_prim_id"].cpu()[0].item()) == 0
    assert "raydn" not in sys.modules
