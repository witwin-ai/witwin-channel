from __future__ import annotations

import ast
import gc
import hashlib
import inspect
from types import SimpleNamespace
import weakref

import pytest
import torch

from witwin.channel_native.core.kernels import ops
from witwin.channel_native.core import scene as core_scene
from witwin.channel_native.core.runtime import compiled_scene as legacy_compiled
from witwin.channel_native.core.runtime import raydn as legacy_raydn
from witwin.channel_native.runtime import symbols
from witwin.channel_native.scene import kernels
from witwin.channel_native.scene import compiled
from witwin.channel_native.scene.kernels import rayd_scene


_SCENE_KERNEL_NAMES = (
    "_raydn_scene_handle_id",
    "raydn_scene_create",
    "raydn_scene_edge_records",
)

_RAYDN_LIFECYCLE_AST_DIGESTS = {
    "RayDNEdgeRecords": "16dd24f8e2a79f43454e4a609d68c608ed64546e5f03ad0d30db4b9580682b50",
    "RayDNScene": "7d11e7c30592c857b6468cf6a9d8741565295b5dd58d9bf0fc65b5aa79c87b5e",
    "_empty_tensor": "23843cfd3570ca0ed7fc050e97f9cc27c5b24af88ae0cf079bd0d60ea0a609c2",
    "_mesh_flags": "ab687287bfec1c541820f1eb9f115be95a63ecc04d48e6d2e982012e09dbdd7b",
    "build_scene_from_structures": "eab42344fde4f7b8af977298bb07bb5e32655a87a0c201c2ceb8154fea32f051",
}


def _body_ast(function: object) -> str:
    node = ast.parse(inspect.getsource(function)).body[0]
    assert isinstance(node, ast.FunctionDef)
    return ast.dump(
        ast.Module(body=node.body, type_ignores=[]), include_attributes=False
    )


def _definition_ast_digest(definition: object) -> str:
    module = ast.parse(inspect.getsource(rayd_scene))
    node = next(
        item
        for item in module.body
        if isinstance(item, (ast.FunctionDef, ast.ClassDef))
        and item.name == definition.__name__
    )
    assert isinstance(node, (ast.FunctionDef, ast.ClassDef))
    source = ast.dump(node, include_attributes=False).encode()
    return hashlib.sha256(source).hexdigest()


def test_raydn_scene_lifecycle_has_one_canonical_owner_and_legacy_identity():
    assert (
        rayd_scene.RayDNScene
        is legacy_raydn.RayDNScene
        is core_scene.RayDNScene
        is compiled.RayDNScene
        is legacy_compiled.RayDNScene
    )
    assert rayd_scene.RayDNEdgeRecords is legacy_raydn.RayDNEdgeRecords
    assert rayd_scene.build_scene_from_structures is (
        legacy_raydn.build_scene_from_structures
    )
    assert rayd_scene.build_scene_from_structures is (
        core_scene.build_scene_from_structures
    )
    assert rayd_scene.RayDNScene.__module__ == legacy_raydn.__name__
    assert rayd_scene.RayDNEdgeRecords.__module__ == legacy_raydn.__name__


@pytest.mark.parametrize("name, expected", _RAYDN_LIFECYCLE_AST_DIGESTS.items())
def test_raydn_scene_lifecycle_move_preserves_frozen_definition_ast(
    name: str,
    expected: str,
):
    assert _definition_ast_digest(getattr(rayd_scene, name)) == expected


@pytest.mark.parametrize("name", _SCENE_KERNEL_NAMES)
def test_scene_kernel_body_matches_compatibility_facade(name: str):
    assert _body_ast(getattr(rayd_scene, name)) == _body_ast(getattr(ops, name))


@pytest.mark.parametrize("name", _SCENE_KERNEL_NAMES)
def test_scene_kernel_package_reexports_canonical_owner(name: str):
    owner = getattr(rayd_scene, name)

    assert owner.__module__ == rayd_scene.__name__
    assert getattr(kernels, name) is owner


def test_scene_kernel_uses_canonical_required_symbol():
    assert rayd_scene._required_native_op is symbols.required_symbol


def test_raydn_edge_records_preserve_order_cache_identity_and_owner_lifetime(
    monkeypatch: pytest.MonkeyPatch,
):
    values = tuple(torch.tensor([index]) for index in range(12))
    packed = torch.tensor([[2.0, 3.0, 4.0]])
    calls: list[int] = []
    pack_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def edge_records(handle: int) -> tuple[torch.Tensor, ...]:
        calls.append(handle)
        return values

    def pack(
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        pack_calls.append((x, y, z))
        return packed

    class Owner:
        pass

    owner = Owner()
    owner_ref = weakref.ref(owner)
    scene = rayd_scene.RayDNScene(handle=17, owner=owner)
    del owner
    gc.collect()
    assert owner_ref() is not None

    monkeypatch.setattr(rayd_scene, "raydn_scene_edge_records", edge_records)
    monkeypatch.setattr(rayd_scene, "mc_pack_vec3", pack)
    first = scene.edge_records()
    second = scene.edge_records()

    assert second is first
    assert scene.runtime_cache["edge_records"] is first
    assert calls == [17]
    assert pack_calls == [(values[2], values[3], values[4])]
    assert (
        first.vertices,
        first.faces,
        first.face_normals,
        first.edge_v0,
        first.edge_v1,
        first.face0,
        first.face1,
        first.shape_id,
        first.local_edge_id,
        first.opposite,
    ) == (
        values[0],
        values[1],
        packed,
        values[5],
        values[6],
        values[7],
        values[8],
        values[9],
        values[10],
        values[11],
    )

    del scene, first, second
    gc.collect()
    assert owner_ref() is None


def test_raydn_scene_builder_preserves_native_order_flags_uv_and_keepalive(
    monkeypatch: pytest.MonkeyPatch,
):
    real_device = torch.device
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    face_uv = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    structures = (
        SimpleNamespace(
            vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
            uv=uv,
            face_uv=face_uv,
        ),
        SimpleNamespace(
            vertices=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
            faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        ),
    )
    captured: list[tuple[object, ...]] = []
    native_owner = object()

    def create(*args: object) -> tuple[int, object]:
        captured.append(args)
        return 23, native_owner

    monkeypatch.setattr(rayd_scene.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rayd_scene.torch, "device", lambda _name: real_device("cpu"))
    monkeypatch.setattr(rayd_scene, "raydn_scene_create", create)

    scene = rayd_scene.build_scene_from_structures(structures)
    vertices, faces, exported_uv, exported_face_uv, left, right, flags = captured[0]

    assert scene.handle == 23
    assert scene.owner is native_owner
    assert len(captured) == 1
    assert flags == [2, 2]
    assert len(scene.mesh_tensors) == 2
    for index, keepalive in enumerate(scene.mesh_tensors):
        assert keepalive == (
            vertices[index],
            faces[index],
            exported_uv[index],
            exported_face_uv[index],
            left[index],
            right[index],
        )
    assert (
        exported_uv[0].untyped_storage().data_ptr() == uv.untyped_storage().data_ptr()
    )
    assert exported_face_uv[0].untyped_storage().data_ptr() == (
        face_uv.untyped_storage().data_ptr()
    )
    assert exported_uv[1].shape == (0, 2)
    assert exported_face_uv[1].shape == (0, 3)
    assert left[0].shape == right[0].shape == (0, 4)


def test_raydn_scene_builder_preserves_unavailable_reasons(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(rayd_scene.torch.cuda, "is_available", lambda: False)

    assert (
        rayd_scene.build_scene_from_structures(()).reason == "scene has no structures"
    )
    assert rayd_scene.build_scene_from_structures((object(),)).reason == (
        "CUDA is unavailable"
    )


def test_scene_create_preserves_native_argument_order(monkeypatch: pytest.MonkeyPatch):
    captured: list[tuple[object, ...]] = []
    owner = object()

    def required_symbol(name: str):
        assert name == "raydn_scene_create"

        def create(*args: object) -> tuple[int, object]:
            captured.append(args)
            return 7, owner

        return create

    monkeypatch.setattr(rayd_scene, "_required_native_op", required_symbol)
    inputs = ([torch.empty(0)] for _ in range(6))
    vertices, faces, uv, face_uv, to_world_left, to_world_right = inputs
    mesh_flags = [3]

    result = rayd_scene.raydn_scene_create(
        vertices,
        faces,
        uv,
        face_uv,
        to_world_left,
        to_world_right,
        mesh_flags,
    )

    assert result == (7, owner)
    assert captured == [
        (
            vertices,
            faces,
            uv,
            face_uv,
            to_world_left,
            to_world_right,
            mesh_flags,
        )
    ]


def test_scene_edge_records_normalizes_handle_and_tuple(
    monkeypatch: pytest.MonkeyPatch,
):
    record = torch.empty(0)
    calls: list[tuple[object, ...]] = []

    def required_symbol(name: str):
        assert name == "raydn_scene_edge_records"

        def edge_records(*args: object) -> list[torch.Tensor]:
            calls.append(args)
            return [record]

        return edge_records

    class Handle:
        def handle(self) -> int:
            return 11

    monkeypatch.setattr(rayd_scene, "_required_native_op", required_symbol)

    assert rayd_scene.raydn_scene_edge_records(Handle()) == (record,)
    assert calls == [(11,)]
