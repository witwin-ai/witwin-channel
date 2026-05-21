"""Low-level coverage of channel_scene helpers using the post-consolidation API.

Class wrappers (``ArrayOps`` / ``Broadcast`` / ``ComplexOps`` / ``EvalOps`` /
``TorchBridge``) and the deleted ``forward_ad`` / ``wrap_forward_ad`` helpers
have been removed; this file exercises the equivalent module-level functions
from :mod:`witwin.channel.core`.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import drjit as dr
import witwin.channel as wt

from witwin.channel.core.scene import Mesh, Scene
from witwin.channel.core.scene.builder import SceneBuilder
from witwin.channel.core.numerics.arrays import (
    barrier,
    broadcast_complex,
    broadcast_float,
    broadcast_int,
    broadcast_point,
    broadcast_vector,
    broadcast_vector_dict,
    complex_abs_sqr,
    complex_zero,
    concat_arrays,
    empty_complex,
    empty_point3,
    empty_vector3,
    empty_vector3u,
    eval_and_sync,
    eval_complex,
    eval_nested,
    gather_point3,
    mask_count,
    safe_normalize,
    scalar,
    timing_enabled,
    zeros_point3,
    zeros_vector3,
)
from witwin.channel.core.numerics.tensors import to_torch_view


def _tolist(value) -> list:
    return np.asarray(value).reshape(-1).tolist()


def test_drjit_array_helpers_cover_init_concat_gather_and_normalize() -> None:
    point = wt.Point3f(wt.Float([1.0, 2.0]), wt.Float([3.0, 4.0]), wt.Float([5.0, 6.0]))
    vector = wt.Vector3f(wt.Float([3.0, 0.0]), wt.Float([4.0, 0.0]), wt.Float([0.0, 0.0]))

    assert scalar(wt.Float([2.0])) == pytest.approx(2.0)
    assert mask_count(wt.Bool([True, False, True])) == 2
    assert dr.width(empty_complex().real) == 0
    assert dr.width(empty_point3().x) == 0
    assert dr.width(empty_vector3().x) == 0
    assert dr.width(empty_vector3u().x) == 0
    assert _tolist(zeros_point3(2).x) == [0.0, 0.0]
    assert _tolist(zeros_vector3(2).y) == [0.0, 0.0]
    assert _tolist(concat_arrays(wt.Float, [wt.Float(), wt.Float([1.0, 2.0])])) == [1.0, 2.0]
    assert _tolist(gather_point3(point, wt.UInt32([1, 0])).x) == [2.0, 1.0]
    assert _tolist(safe_normalize(vector).x) == pytest.approx([0.6, 0.0])
    assert _tolist(safe_normalize(vector).y) == pytest.approx([0.8, 0.0])


def test_broadcast_helpers_cover_scalar_complex_point_and_vector_dict() -> None:
    complex_value = wt.Complex2f(wt.Float([1.0]), wt.Float([2.0]))

    assert _tolist(broadcast_float(wt.Float(2.0), 3)) == [2.0, 2.0, 2.0]
    assert _tolist(broadcast_complex(complex_value, 2).real) == [1.0, 1.0]
    assert _tolist(broadcast_int(wt.UInt32(7), 2)) == [7, 7]
    assert _tolist(broadcast_point(wt.Point3f(1.0, 2.0, 3.0), 2).z) == [3.0, 3.0]
    assert _tolist(broadcast_vector(wt.Vector3f(4.0, 5.0, 6.0), 2).x) == [4.0, 4.0]

    broadcast_dict = broadcast_vector_dict(
        {"x": complex_value, "y": complex_value, "z": complex_value}, 2,
    )
    assert _tolist(broadcast_dict["y"].imag) == [2.0, 2.0]


def test_complex_zero_eval_and_abs_sqr_cover_field_helpers() -> None:
    complex_field = wt.Complex2f(wt.Float([3.0, 4.0]), wt.Float([4.0, 3.0]))

    assert _tolist(complex_zero(2).real) == [0.0, 0.0]
    assert _tolist(eval_complex(complex_field).imag) == [4.0, 3.0]
    assert _tolist(complex_abs_sqr(complex_field)) == [25.0, 25.0]


def test_eval_helpers_traverse_nested_structures_without_state() -> None:
    point = wt.Point3f(wt.Float([1.0, 2.0]), wt.Float([3.0, 4.0]), wt.Float([5.0, 6.0]))
    complex_field = wt.Complex2f(wt.Float([3.0, 4.0]), wt.Float([4.0, 3.0]))
    nested = {
        "point": point,
        "complex": complex_field,
        "seq": [wt.Float([1.0]), (wt.Float([2.0]),)],
    }

    assert timing_enabled() is False
    assert eval_nested(nested) is nested
    assert _tolist(eval_and_sync(wt.Float([9.0]))) == [9.0]
    assert _tolist(barrier(wt.Float([8.0]))) == [8.0]


def test_to_torch_view_returns_detached_tensor() -> None:
    viewed = to_torch_view(wt.Float([1.0, 2.0]), detach=True)
    assert isinstance(viewed, torch.Tensor)
    assert viewed.tolist() == pytest.approx([1.0, 2.0])


def test_mesh_validation_and_update_error_paths(triangle_vertices: torch.Tensor, triangle_faces: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="shape \\(N, 3\\) with N > 0"):
        Mesh(torch.zeros((0, 3), dtype=torch.float32), triangle_faces)

    with pytest.raises(ValueError, match="shape \\(M, 3\\) with M > 0"):
        Mesh(triangle_vertices, torch.zeros((0, 3), dtype=torch.int32))

    mesh = Mesh(triangle_vertices, triangle_faces)
    moved_vertices, moved_faces = mesh.to_mesh(device="cpu")

    assert moved_vertices.dtype == torch.float32
    assert moved_faces.dtype == torch.int32

    with pytest.raises(ValueError, match="Expected 3 vertices"):
        mesh.update_vertices(torch.zeros((2, 3), dtype=torch.float32))


def test_rayd_and_builder_error_paths_are_explicit() -> None:
    with pytest.raises(ValueError, match="no meshes"):
        SceneBuilder.build_rayd_scene([])

    broken_runtime = SimpleNamespace(
        _structure_meshes=[{"vertices": torch.zeros((1, 2), dtype=torch.float32), "faces": torch.zeros((1, 3), dtype=torch.int32)}],
        _rayd_scene="sentinel",
    )
    with pytest.raises(ValueError, match="shape"):
        SceneBuilder.configure_runtime_backends(broken_runtime)
    assert broken_runtime._rayd_scene is None

    assert SceneBuilder._build_triangle_material_data([], 0) is None
    assert SceneBuilder._material_is_specified(SimpleNamespace(eps_r=1.0, sigma_e=0.0)) is False
    assert SceneBuilder._material_is_specified(SimpleNamespace(eps_r=2.0, sigma_e=0.0)) is True
    assert _tolist(SceneBuilder._material_array(1.5, 2, name="scalar")) == [1.5, 1.5]
    assert _tolist(SceneBuilder._material_array(torch.tensor([2.0]), 2, name="tensor")) == [2.0, 2.0]
    assert _tolist(SceneBuilder._material_array(wt.Float([3.0]), 2, name="drjit")) == [3.0, 3.0]

    with pytest.raises(ValueError, match="length 2"):
        SceneBuilder._material_array(wt.Float([1.0, 2.0, 3.0]), 2, name="bad-drjit")

    with pytest.raises(ValueError, match="length 2"):
        SceneBuilder._material_array(torch.tensor([1.0, 2.0, 3.0]), 2, name="bad-tensor")

    with pytest.raises(TypeError, match="float-compatible"):
        SceneBuilder._material_array(object(), 1, name="bad-object")


def test_scene_add_validation() -> None:
    scene = Scene(device="cpu")

    with pytest.raises(TypeError, match="Scene.add expects"):
        scene.add(object())
