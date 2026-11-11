# Copyright Xingyu Chen.
# Second-order AD through the RayD geometry companions is refused before any native launch.

"""Second-order AD through the RayD geometry companions is refused before any native launch."""

from __future__ import annotations

import pytest
import torch

from witwin.core import Mesh, PhysicalMaterial, Scene, Structure
from witwin.channel.kernels import geometry as ops
from witwin.channel.scene import compile as compile_scene

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for RayD geometry AD"
)

_TILTED_TRIANGLE_VERTICES = (
    (0.0, 0.0, 0.0),
    (2.0, 0.0, 0.3),
    (0.0, 2.0, 0.2),
)
_TRIANGLE_FACES = ((0, 1, 2),)


def _source_linked_rayd_available() -> bool:
    from witwin.channel.deployment import build_info

    try:
        return build_info()["rayd_integration"] == "source-linked"
    except ModuleNotFoundError:
        return False


def _triangle_scene() -> object:
    if not _source_linked_rayd_available():
        pytest.skip("RayD native extension is not built")
    scene = Scene(
        structures=(
            Structure(
                Mesh(
                    torch.tensor(_TILTED_TRIANGLE_VERTICES, dtype=torch.float32),
                    torch.tensor(_TRIANGLE_FACES, dtype=torch.int32),
                    recenter=False,
                    fill_mode="surface",
                    topology_diagnostics=False,
                ),
                PhysicalMaterial(eps_r=2.0),
            ),
        ),
        endpoints=(),
    )
    return compile_scene(scene, reference_frequency_hz=3.5e9).rayd


def test_double_backward_raises():
    """create_graph=True through the once-differentiable face-normal backward must raise.

 The raise fires inside the backward that ``create_graph=True`` asked to be
 differentiable, before any native companion launches, and names the owner.
 """

    rayd = _triangle_scene()
    records = rayd.edge_records()
    vertices = records.vertices.detach().clone().requires_grad_(True)
    table = ops.rayd_face_normals_ad(rayd, vertices, records.face_normals.contiguous())
    with pytest.raises(NotImplementedError, match="first-order only") as raised:
        torch.autograd.grad(table.sum(), vertices, create_graph=True)
    assert "_RaydFaceNormalsAdFunction.backward" in str(raised.value)
